import os
import argparse
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm


from util import register, ForwardState, load_model


class Dataset(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        seq_len: int,
        ds_name="monology/pile-uncopyrighted",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.ds_name = ds_name
        self.ds = load_dataset(self.ds_name, split="train", streaming=True)

    def __iter__(self):
        for example in self.ds:
            tokens = self.tokenizer.encode(
                example["text"], add_special_tokens=False, truncation=True
            )
            if len(tokens) < self.seq_len:
                continue

            start = random.randint(0, len(tokens) - self.seq_len)
            chunk = tokens[start : start + self.seq_len]
            yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def experts_fwd(self, *args, state: ForwardState, fwd, **kwargs):
    hidden_states: torch.Tensor = args[0]
    top_k_index: torch.Tensor = args[1]
    top_k_weights: torch.Tensor = args[2]
    data = []
    indices = []
    e_ids = []

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    final_hidden_states = torch.zeros_like(hidden_states)
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(
            current_state, self.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up

        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.down_proj[expert_idx]
        )
        data.append(current_hidden_states)
        e_ids.append(
            torch.full((current_hidden_states.shape[0],), expert_idx, dtype=torch.long)  # type: ignore
        )
        indices.append(token_idx)

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )
    state.storage[state.layer] = {
        "activations": torch.cat(data).cpu(),
        "e_ids": torch.cat(e_ids),
        "indices": torch.cat(indices),
    }
    state.layer += 1
    return final_hidden_states


class MoETracker:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        unembed_matrix: torch.Tensor,
        k_maps_dict: dict,
        top_n=3,
        device: torch.device | str = "cpu",
    ):
        self.device = device
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_n = top_n
        self.unembed_matrix = unembed_matrix
        self.vocab_size = unembed_matrix.shape[-1]

        self.k_maps = {
            int(k): torch.from_numpy(arr).long().squeeze()
            for k, arr in k_maps_dict.items()
        }

        self.input_counts = torch.zeros(
            (num_layers, num_experts, self.vocab_size),
            dtype=torch.long,
            device=self.device,
        )
        self.output_counts = torch.zeros(
            (num_layers, num_experts, self.vocab_size),
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def jsd(p_expert: torch.Tensor, p_global: torch.Tensor):
        M = 0.5 * (p_expert + p_global)
        M_log = M.log2()

        kl_expert_M = (p_expert * (p_expert.log2() - M_log)).sum(dim=1)
        kl_global_M = (p_global * (p_global.log2() - M_log)).sum()

        return 0.5 * kl_expert_M + 0.5 * kl_global_M

    def update(
        self, layer_outputs: dict[int, dict[str, torch.Tensor]], input_ids: torch.Tensor
    ):
        flat_input_tokens = input_ids.view(-1)

        for layer_idx, data in layer_outputs.items():
            acts = data["activations"].to(self.unembed_matrix.device)
            expert_indices = data["e_ids"].to(self.device)
            src_token_indices = data["indices"].to(self.device)

            routed_tokens = flat_input_tokens[src_token_indices]

            input_flat_idx = expert_indices * self.vocab_size + routed_tokens

            self.input_counts[layer_idx] += (
                torch.bincount(
                    input_flat_idx, minlength=self.num_experts * self.vocab_size
                )
                .view(self.num_experts, self.vocab_size)
                .to(self.device)
            )

            logits = acts @ self.unembed_matrix
            _, top_tokens = torch.topk(logits, self.top_n, dim=1)

            expanded_expert_indices = (
                expert_indices.unsqueeze(1).expand(-1, self.top_n).flatten()
            )
            output_flat_idx = (
                expanded_expert_indices * self.vocab_size
                + top_tokens.flatten().to(self.device)
            )

            self.output_counts[layer_idx] += (
                torch.bincount(
                    output_flat_idx, minlength=self.num_experts * self.vocab_size
                )
                .view(self.num_experts, self.vocab_size)
                .to(self.device)
            )

    def get_metrics(self, k: int, mode="output", eps=1e-8):
        counts_matrix = self.output_counts if mode == "output" else self.input_counts
        cluster_map = self.k_maps[k].to(self.device)

        layer_scores = []
        for layer in range(self.num_layers):
            # Aggregate vocab counts to clusters
            layer_vocab = counts_matrix[layer].float()
            layer_clusters = torch.zeros((self.num_experts, k), device=self.device)
            for e in range(self.num_experts):
                layer_clusters[e].scatter_add_(0, cluster_map, layer_vocab[e])

            p_global = layer_clusters.sum(0) + eps
            p_global /= p_global.sum()

            p_expert = layer_clusters + eps
            p_expert /= p_expert.sum(1, keepdim=True)

            jsd = self.jsd(p_expert, p_global)
            layer_scores.append(jsd.cpu().numpy())

        return np.stack(layer_scores)

    def get_random_baseline(self, k: int, mode="output", num_simulations=10, eps=1e-8):
        counts_matrix = self.output_counts if mode == "output" else self.input_counts
        cluster_map = self.k_maps[k].to(self.device)

        all_layers_baselines = []
        for layer in range(self.num_layers):
            # Aggregate vocab counts to clusters to get P_global
            layer_vocab = counts_matrix[layer].float()
            layer_clusters = torch.zeros((self.num_experts, k), device=self.device)
            for e in range(self.num_experts):
                layer_clusters[e].scatter_add_(0, cluster_map, layer_vocab[e])

            # Global distribution for this layer
            p_global = layer_clusters.sum(0) + eps
            p_global /= p_global.sum()

            layer_baselines = torch.zeros(self.num_experts, device=self.device)

            for e in range(self.num_experts):
                n = int(counts_matrix[layer][e].sum().item())

                if n < 1:
                    continue  # baseline stays 0 for experts with no data

                m = torch.distributions.Multinomial(total_count=n, probs=p_global)
                samples = m.sample((num_simulations,))  # [num_simulations, k]

                p_random = samples + eps
                p_random /= p_random.sum(dim=1, keepdim=True)

                jsd = self.jsd(p_random, p_global)
                layer_baselines[e] = jsd.mean()

            all_layers_baselines.append(layer_baselines.cpu().numpy())

        return np.stack(all_layers_baselines)

    def save_analysis(self, filename: str):
        results = {"k_values": np.array(list(self.k_maps.keys()))}

        for m in ["input", "output"]:
            results[f"{m}"] = np.stack(
                [self.get_metrics(k, mode=m) for k in self.k_maps]
            )
            results[f"{m}_baseline"] = np.stack(
                [self.get_random_baseline(k, mode=m) for k in self.k_maps]
            )

        np.savez_compressed(filename, **results)  # type: ignore


def forward_pass(sample: dict[str, torch.Tensor], model):
    input_ids = sample["input_ids"].to(model.device)

    model(
        input_ids=input_ids,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
        return_dict=False,
        output_router_logits=False,
    )


@torch.inference_mode()
def main(args):
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = Dataset(tok, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size)

    model, is_moe, num_experts = load_model(args.model_name)
    assert is_moe, "Model needs to be a MoE model."

    num_experts = model.config.num_experts
    layers = list(range(model.config.num_hidden_layers))

    embedding = model.lm_head.weight.T

    if not os.path.exists(args.cluster_path):
        raise ValueError("Need clusters.")

    labels = np.load(args.cluster_path)
    tracker = MoETracker(
        len(layers), num_experts, embedding, labels, device=model.device
    )
    tokens_passed = 0
    total_tokens = args.num_tokens

    hooks = [(f"layers.{layer}.mlp.experts", experts_fwd) for layer in layers]
    pbar = tqdm(total=total_tokens, desc="Collecting", unit_scale=True)

    with register(model, hooks) as state:
        for sample in dl:
            state.layer = 0

            forward_pass(sample, model)

            tracker.update(state.storage, sample["input_ids"].to(model.device))

            new_tokens = sample["input_ids"].numel()
            tokens_passed += new_tokens
            pbar.update(new_tokens)
            if tokens_passed >= total_tokens:
                break

    pbar.close()

    os.makedirs(args.out_dir, exist_ok=True)
    model_identifier = args.model_name.split("/")[-1]
    tracker.save_analysis(f"{args.out_dir}/{model_identifier}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates routing and functional specialization for all experts in a transformer MoE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model_name",
        default="allenai/OLMoE-1B-7B-0125",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use for MoE forward passes",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=32,
        help="Sequence length (number of tokens) for MoE forward passes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-t",
        "--num_tokens",
        type=int,
        default=1_000_000,
        help="Number of tokens to use for calculations",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data/embed",
        help="Output folder for results",
    )
    parser.add_argument(
        "-l",
        "--cluster_path",
        default="data/embed/OLMoE-1B-7B-0125_labels.npz",
        help="Path to cluster labels for each vocabulary item. Must be a .npz file",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
