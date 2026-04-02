import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import warnings
import concurrent.futures
import multiprocessing

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from util import register, ForwardState, load_model
from probing_ds import Concepts, ProbingDataset, collate_fn


def dense_fwd(module, *args, state: ForwardState, fwd, **kwargs):
    """Based on Huggingface code of dense models that use SwiGLU MLPs."""
    x: torch.Tensor = args[0]
    acts = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
    out = module.down_proj(acts)

    acts = acts.view(-1, acts.shape[-1])
    acts_probing = acts[state.storage["probe_indices"]]
    state.storage[f"{state.layer}_acts"] = acts_probing
    state.layer += 1
    return out


def pythia_fwd(module, *args, state: ForwardState, fwd, **kwargs):
    """Based on: https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/models/gpt_neox/modular_gpt_neox.py#L30"""
    x: torch.Tensor = args[0]
    hidden_states = module.dense_h_to_4h(x)
    acts = module.act(hidden_states)
    out = module.dense_4h_to_h(hidden_states)

    acts = acts.view(-1, acts.shape[-1])
    acts_probing = acts[state.storage["probe_indices"]]
    state.storage[f"{state.layer}_acts"] = acts_probing
    state.layer += 1
    return out


def gpt_oss_fwd(
    self,
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    expert_hit: torch.Tensor,
    routing_weights: torch.Tensor,
    mask: torch.Tensor,
):
    """Based on: https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L89"""
    data = {}
    indices = {}
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    next_states = torch.zeros_like(
        hidden_states, dtype=hidden_states.dtype, device=hidden_states.device
    )
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]

        if expert_idx == self.num_experts:
            continue
        with torch.no_grad():
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = (
            current_state @ self.gate_up_proj[expert_idx]
            + self.gate_up_proj_bias[expert_idx]
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu

        probe_mask = mask[token_idx]
        if probe_mask.any():
            e_id = expert_idx.item()
            data[e_id] = current_state[probe_mask]
            indices[e_id] = token_idx[probe_mask]

        out = (
            gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
        )
        weighted_output = out * routing_weights[token_idx, top_k_pos, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
    next_states = next_states.view(batch_size, -1, self.hidden_size)
    return next_states, data, indices


def _moe_fwd(
    self,
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    expert_hit: torch.Tensor,
    top_k_weights: torch.Tensor,
    mask: torch.Tensor,
):
    data = {}
    indices = {}
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

        probe_mask = mask[token_idx]
        if probe_mask.any():
            e_id = expert_idx.item()
            data[e_id] = current_hidden_states[probe_mask]
            indices[e_id] = token_idx[probe_mask]

        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.down_proj[expert_idx]
        )

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )
    return final_hidden_states, data, indices


def experts_fwd(self, *args, state: ForwardState, fwd, **kwargs):
    """Based on: https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/integrations/moe.py#L37-L61"""
    hidden_states: torch.Tensor = args[0]
    top_k_index: torch.Tensor = args[1]
    top_k_weights: torch.Tensor = args[2]

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    # Grab the probe mask (which tokens are interesting for this concept)
    mask: torch.Tensor = state.storage["probe_indices"].to(expert_mask.device)
    if hasattr(self, "alpha") and hasattr(self, "limit"):
        final_hidden_states, data, indices = gpt_oss_fwd(
            self,
            hidden_states,
            expert_mask,
            expert_hit,
            top_k_weights,
            mask,
        )
    else:
        final_hidden_states, data, indices = _moe_fwd(
            self,
            hidden_states,
            expert_mask,
            expert_hit,
            top_k_weights,
            mask,
        )

    state.storage[f"{state.layer}_acts"] = data
    state.storage[f"{state.layer}_idxs"] = indices
    state.layer += 1

    return final_hidden_states


class ActivationStorage:
    def __init__(self, layers: list[int], num_experts: int = 1):
        self.layers = layers
        self.num_experts = num_experts
        self.is_moe = num_experts > 1

        self.buffer = {layer: {e: [] for e in range(num_experts)} for layer in layers}

    def clear(self):
        self.buffer = {
            layer: {e: [] for e in range(self.num_experts)} for layer in self.layers
        }

    def add(
        self,
        layer: int,
        seq_len: int,
        activations: dict[int, torch.Tensor] | torch.Tensor,
        labels: torch.Tensor,
        indices: dict[int, torch.Tensor] | None = None,
    ):
        if not self.is_moe:
            self.buffer[layer][0].append((activations.cpu(), labels.to(torch.int8)))  # type: ignore

        else:
            for expert, acts in activations.items():  # type: ignore
                if acts.shape[0] == 0:
                    continue

                e_indices = indices[expert].cpu()  # type: ignore
                b_indices = e_indices // seq_len
                e_labels = labels[b_indices]

                self.buffer[layer][expert].append((acts.cpu(), e_labels.to(torch.int8)))

    def get_data(self, layer: int, expert: int):
        items = self.buffer[layer][expert]
        if not items:
            return None, None

        acts, labels = map(torch.cat, zip(*items))
        self.buffer[layer][expert] = []
        return acts.float().numpy(), labels.numpy()


def get_top_neurons(X: np.ndarray, y: np.ndarray):
    """Simple mean difference feature selection"""
    pos_class = y == 1
    mean_dif = np.abs(X[pos_class].mean(axis=0) - X[~pos_class].mean(axis=0))
    return np.argsort(mean_dif)[::-1]


def train_probes_on_data(X: np.ndarray, y: np.ndarray, ks: list[int], seed: int):
    num_pos = y.sum().item()
    num_neg = len(y) - num_pos
    if num_pos < 50 or num_neg < 50:
        return [
            {
                "k": k,
                "f1": "Not Enough Samples",
                "recall": "Not Enough Samples",
                "precision": "Not Enough Samples",
                "num_pos": num_pos,
                "num_neg": num_neg,
            }
            for k in ks
        ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.25, shuffle=True, stratify=y
    )

    top_neurons = get_top_neurons(X_train, y_train)
    scores: list[dict[str, float | int]] = []

    with warnings.catch_warnings(action="ignore", category=ConvergenceWarning):
        for k in ks:
            top_k_neurons = top_neurons[:k]
            X_train_k = X_train[:, top_k_neurons]
            X_test_k = X_test[:, top_k_neurons]

            probe = LogisticRegression(
                C=1.0,
                l1_ratio=0,
                max_iter=500,
                random_state=seed,
                solver="saga",
                class_weight="balanced",
            )
            probe.fit(X_train_k, y_train)
            y_pred = probe.predict(X_test_k)

            p, r, f1, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                average="binary",
                zero_division=np.nan,  # type: ignore
            )

            scores.append(
                {
                    "k": k,
                    "f1": f1,
                    "recall": r,
                    "precision": p,
                    "num_pos": num_pos,
                    "num_neg": num_neg,
                }  # type: ignore
            )

    return scores


def _process_single_task(
    X: np.ndarray | None,
    y: np.ndarray | None,
    layer: int,
    expert: int,
    ks: list[int],
    seed: int,
    concept: str,
    is_moe: bool,
):
    if X is None or y is None:
        return []

    metrics = train_probes_on_data(X, y, ks, seed)

    task_results = []
    for m in metrics:
        task_results.append(
            {
                "concept": concept,
                "layer": layer,
                "expert": expert if is_moe else "dense",
                "k": m["k"],
                "f1": m["f1"],
                "recall": m["recall"],
                "precision": m["precision"],
                "num_positive": m["num_pos"],
                "num_negative": m["num_neg"],
            }
        )
    return task_results


def process_storage_and_train(
    storage: ActivationStorage,
    ks: list[int],
    seed: int,
    concept: str,
    max_workers: int | None = None,
):
    results: list[dict[str, float | int | str]] = []
    is_moe = storage.is_moe

    tasks = [
        (layer, expert)
        for layer in storage.layers
        for expert in range(storage.num_experts)
    ]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx
    ) as executor:
        future_to_task = {
            executor.submit(
                _process_single_task,
                *storage.get_data(layer, expert),
                layer,
                expert,
                ks,
                seed,
                concept,
                is_moe,
            ): (layer, expert)
            for layer, expert in tasks
        }

        pbar = tqdm(
            total=len(tasks),
            desc=f"Probing ({concept})",
            position=1,
            leave=False,
        )

        for future in concurrent.futures.as_completed(future_to_task):
            try:
                data = future.result()
                results.extend(data)
            except Exception as e:
                tqdm.write(f"Worker exception: {e}")
            finally:
                pbar.update(1)

        pbar.close()

    return results


def forward_pass(sample: dict, model):
    input_ids = sample["input_ids"].to(model.device)
    attention_mask = sample["attention_mask"].to(model.device)
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
        return_dict=False,
        output_router_logits=False,
    )


def collect_and_probe(
    model,
    dl: DataLoader,
    state: ForwardState,
    storage: ActivationStorage,
    num_tokens: int,
    ks: list[int],
    seed: int,
    concept: str,
    max_workers: int | None,
):
    current_tokens = 0
    pbar = tqdm(
        total=num_tokens,
        desc=f"Collecting Acts ({concept})",
        unit_scale=True,
        leave=False,
        position=1,
    )
    is_moe = storage.is_moe
    for sample in dl:
        try:
            state.layer = 0
            probe_mask: torch.Tensor = sample["probe_indices"]
            state.storage["probe_indices"] = probe_mask.flatten()

            labels = sample["label"]
            labels = labels if is_moe else labels[torch.where(probe_mask)[0]]
            seq_len = sample["input_ids"].shape[-1]

            forward_pass(sample, model)

            for i, layer in enumerate(storage.layers):
                acts = state.storage[f"{i}_acts"]
                indices = state.storage.get(f"{i}_idxs", None)
                storage.add(layer, seq_len, acts, labels, indices)

            state.storage.clear()

            new_tokens = probe_mask.sum().item()
            pbar.update(new_tokens)
            current_tokens += new_tokens
            if current_tokens >= num_tokens:
                break

        except torch.OutOfMemoryError:
            tqdm.write(
                "CUDA Out of Memory encountered. Clearing cache and skipping batch."
            )
            state.storage.clear()
            torch.cuda.empty_cache()
            continue

    pbar.close()
    state.storage.clear()
    return process_storage_and_train(storage, ks, seed, concept, max_workers)


def select_concepts(concept: str):
    if concept == "all":
        return list(Concepts())
    return Concepts().find_concept(concept)


def select_layers(layers: list[int], is_moe: bool, config):
    config = getattr(config, "text_config", config)
    total_layers = getattr(config, "num_hidden_layers")
    if total_layers is None:
        raise ValueError("Could not find total amount of layers.")

    layers = layers if layers else list(range(total_layers))
    if is_moe:
        start = getattr(config, "moe_layer_start_index", 0)
        start = getattr(config, "first_k_dense_replace", start)
        end = getattr(config, "moe_layer_end_index", total_layers)
        layers = [x for x in layers if start <= x <= end]

    if not layers:
        raise ValueError("Invalid layers")

    return layers


def select_hooks(layers: list[int], model_name: str, is_moe: bool):
    match (model_name, is_moe):
        case ("mistralai/Mixtral-8x7B-v0.1", True):
            return [(f"layers.{layer}.mlp.experts", experts_fwd) for layer in layers]
        case ("openai/gpt-oss-20b", True):
            return [(f"layers.{layer}.mlp", gpt_oss_fwd) for layer in layers]
        case ("EleutherAI/pythia-12b", False):
            return [(f"layers.{layer}.mlp", pythia_fwd) for layer in layers]
        case (_, False):
            return [(f"layers.{layer}.mlp", dense_fwd) for layer in layers]
        case (_, True):
            return [(f"layers.{layer}.mlp", experts_fwd) for layer in layers]
        case _:
            raise ValueError(
                f"Could not select correct hook for Model {model_name}, Model-Type: {'MoE' if is_moe else 'Dense'}"
            )


def append_results_to_csv(results: list[dict], out_path: str, model_name: str):
    if not results:
        return

    os.makedirs(out_path, exist_ok=True)
    csv_name = f"{model_name.split('/')[1]}.csv"
    file_path = os.path.join(out_path, csv_name)

    df = pd.DataFrame(results)
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode="a", header=header, index=False, na_rep="NaN")


@torch.inference_mode()
def main(args):
    concepts = select_concepts(args.concept)
    if not concepts:
        raise ValueError(f"Concept {args.concept} not found.")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        pad_token_id = tok.eos_token_id
        if pad_token_id is None:
            raise ValueError("Could not set valid pad token.")

    model, is_moe, num_experts = load_model(args.model_name, quant=args.quant)

    layers = select_layers(args.layers, is_moe, model.config)
    hooks = select_hooks(layers, args.model_name, is_moe)
    storage = ActivationStorage(layers, num_experts)

    ks, num_tokens, seed, bs = args.ks, args.num_tokens, args.seed, args.batch_size
    max_workers = args.max_workers

    ds = ProbingDataset(tok, concepts[0][0], concepts[0][2])
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
    )

    os.makedirs(args.out_path, exist_ok=True)

    with register(model, hooks) as state:
        for category, concept, value in tqdm(concepts, desc="Concepts"):
            dl.dataset.update(category, value)  # type: ignore

            result = collect_and_probe(
                model, dl, state, storage, num_tokens, ks, seed, concept, max_workers
            )
            append_results_to_csv(result, args.out_path, args.model_name)

            storage.clear()
            torch.cuda.empty_cache()

    tqdm.write(f"Results saved to {args.out_path}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(
        description="Uses k-sparse probing to probe for concepts in intermediate layers of dense and MoE transformer models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="allenai/OLMoE-1B-7B-0125",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "-l",
        "--layers",
        nargs="+",
        type=int,
        default=[],
        help="Layer where probes are trained on activations. Defaults to all layers of the selected model",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use for model forward passes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-ks",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="k values to use for k-sparse probing. Selects the top-k neurons out of a vector",
    )
    parser.add_argument(
        "-n",
        "--num_tokens",
        type=int,
        default=5000,
        help="Number of tokens which are used for training probes. Will balance out positive and negative examples",
    )
    parser.add_argument(
        "-w",
        "--max_workers",
        type=int,
        default=None,
        help="Number of workers to use for training probes in parallel. Defaults to one worker per CPU core",
    )
    parser.add_argument(
        "-c",
        "--concept",
        default="is_frac",
        help="Which concepts to probe for. Special options include 'all', 'latex', 'code', 'text' and 'pos' which select multiple or all concepts",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        default="./data/probing",
        help="Output directory for the probing results",
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=bool,
        default=False,
        help="Whether to quantize the model to 8-bit",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
