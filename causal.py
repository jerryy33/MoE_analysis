import os
import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from util import register, ForwardState, load_model


def experts_fwd(self, *args, state: ForwardState, fwd, **kwargs):
    """Based on: https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/integrations/moe.py#L37-L61"""

    hidden_states: torch.Tensor = args[0]
    top_k_index: torch.Tensor = args[1]
    top_k_weights: torch.Tensor = args[2].to(hidden_states.dtype)
    pos = state.storage["positions"]

    k = top_k_index.shape[-1]
    bs, d_model = hidden_states.shape
    expert_acts = torch.zeros(
        (bs, k, d_model), dtype=hidden_states.dtype, device=hidden_states.device
    )

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

        expert_acts[token_idx, top_k_pos] = current_hidden_states

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    state.storage["top_k_index"] = top_k_index[pos]
    state.storage["top_k_acts"] = expert_acts[pos]
    state.storage["top_k_weights"] = top_k_weights[pos]

    return final_hidden_states


def forward_pass(sample: dict, model):
    input_ids = sample["input_ids"].to(model.device)

    return model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
        output_router_logits=False,
    )


def find_triggers(start: int, end: int, offsets: torch.Tensor):
    return [i for i, (s, e) in enumerate(offsets) if not (e <= start or s >= end)]


def calculate_expert_dla(
    final_residual: torch.Tensor,
    u_i: torch.Tensor,
    ln_module: torch.nn.Module,
    lm_head: torch.nn.Module,
):
    input_dtype = final_residual.dtype
    x = final_residual.to(torch.float32)

    variance = x.pow(2).mean(-1, keepdim=True)
    r_sigma = torch.rsqrt(variance + ln_module.variance_epsilon)  # type: ignore

    ln_scale = ln_module.weight.to(torch.float32) * r_sigma  # type: ignore

    u_i_projected = u_i.to(torch.float32) * ln_scale
    u_i_projected = u_i_projected.to(input_dtype)
    dla_logits = torch.nn.functional.linear(u_i_projected, lm_head.weight)  # type: ignore
    return dla_logits


@torch.inference_mode()
def main(args):
    if not os.path.exists(args.example_dir):
        raise ValueError(
            f"Directory {args.example_dir} to load examples from does not exit."
        )

    model, is_moe, num_experts = load_model(args.model_name)
    assert is_moe, "Model must be MoE model"

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lm_head = model.lm_head
    final_norm = model.model.norm
    layer = args.layer
    experts = args.experts
    k = model.config.num_experts_per_tok

    with register(model, [(f"layers.{layer}.mlp.experts", experts_fwd)]) as state:
        for expert_id in tqdm(experts, desc="Experts"):
            prompts = json.load(open(f"{args.example_dir}/L{layer}-E{expert_id}.json"))

            assert len(prompts) == 20, f"Incorrect number of examples {len(prompts)}"

            scores = torch.zeros((num_experts,), dtype=torch.float, device=model.device)
            ranks = torch.zeros((num_experts, k), dtype=torch.int, device=model.device)

            for item in prompts:
                text = item["text"]
                target_word = item["target"]
                trigger_word = item["trigger"].lower()

                target_token_id = tokenizer.encode(
                    target_word, add_special_tokens=False
                )[0]
                start_char = text.lower().find(trigger_word)
                end_char = start_char + len(trigger_word)

                enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")

                offsets = enc["offset_mapping"][0]
                triggers = find_triggers(start_char, end_char, offsets)

                pos = torch.tensor(triggers, dtype=torch.int, device=model.device)
                state.storage["positions"] = pos

                out = forward_pass(enc, model)

                top_k_index: torch.Tensor = state.storage["top_k_index"]
                top_k_acts: torch.Tensor = state.storage["top_k_acts"]
                top_k_weights: torch.Tensor = state.storage["top_k_weights"]
                top_k_ui = top_k_acts * top_k_weights.unsqueeze(-1)

                mask = top_k_index == expert_id
                if not mask.any():
                    # Default to last valid position in the case the expert is not routed.
                    trigger_pos = len(pos) - 1
                else:
                    weights = torch.where(mask, top_k_weights, -float("inf"))
                    flat_idx = torch.argmax(weights)
                    trigger_pos = flat_idx // k

                top_k_index = top_k_index[trigger_pos]

                logits = calculate_expert_dla(
                    out.hidden_states[layer][0, trigger_pos],  # type: ignore
                    top_k_ui[trigger_pos],
                    final_norm,
                    lm_head,
                )

                target_scores = logits[:, target_token_id]

                scores.index_add_(0, top_k_index, target_scores.float())

                order = torch.argsort(target_scores, descending=True)
                sorted_experts = top_k_index[order]
                ranks[sorted_experts, torch.arange(k, device=model.device)] += 1

                state.storage.clear()

            np.savez(
                f"{args.out_dir}/L{layer}_E{expert_id}_causal.npz",
                scores=scores.cpu().numpy(),
                ranks=ranks.cpu().numpy(),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyses the causal influence of MoE experts on the model output using Direct Logit attribution (DLA). Requires test cases as JSON files. Each test case needs the keys 'text', 'trigger', 'target'. As a control MoE experts from the same layer are used.",
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
        "--layer",
        type=int,
        default=0,
        help="Layer from which MoE experts are selected",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-e",
        "--experts",
        nargs="+",
        type=int,
        default=[0],
        help="Expert IDs to select for experiment",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="./data/causal",
        help="Output directory for DLA results",
    )
    parser.add_argument(
        "-p",
        "--example_dir",
        default="./data/causal",
        help="Directory where the test cases (as JSON files) are available",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(args)
