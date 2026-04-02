import os
import argparse
import random
import json
import gc
import time
import re
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai import errors
from pydantic import BaseModel, Field

from util import ForwardState, EarlyStopException, register, load_model


def experts_fwd(self, *args, state: ForwardState, fwd, **kwargs):
    """Based on: https://github.com/huggingface/transformers/blob/08810b1e278938278c50153ee1edfd7a20a759da/src/transformers/integrations/moe.py#L37-L61"""
    hidden_states: torch.Tensor = args[0]
    top_k_index: torch.Tensor = args[1]
    top_k_weights: torch.Tensor = args[2]
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

    scores = top_k_weights * expert_acts.norm(p=2, dim=-1)
    state.storage["scores"] = scores
    state.storage["acts"] = expert_acts
    state.storage["selected_experts"] = top_k_index
    raise EarlyStopException()


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

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            dataset_iter = load_dataset(self.ds_name, split="train", streaming=True)
            worker_seed = None
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            full_dataset = load_dataset(self.ds_name, split="train", streaming=True)
            dataset_iter = full_dataset.shard(num_shards=num_workers, index=worker_id)

            worker_seed = torch.initial_seed() + worker_id
            random.seed(worker_seed)

        for example in dataset_iter:
            tokens = self.tokenizer.encode(
                example["text"], add_special_tokens=False, truncation=True
            )
            if len(tokens) < self.seq_len:
                continue

            start = random.randint(0, len(tokens) - self.seq_len)
            chunk = tokens[start : start + self.seq_len]
            yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def forward_pass(sample: dict[str, torch.Tensor], model):
    input_ids = sample["input_ids"].to(model.device)
    try:
        model(
            input_ids=input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
            return_dict=False,
            output_router_logits=False,
        )
    except EarlyStopException:
        pass


@dataclass(frozen=True, slots=True)
class AutoInterpData:
    expert_id: int
    examples: list[list[str]]
    scores: list[list[float]]
    top_promoted: list[list[list[str]]] | None = None
    labels: list[int] | None = None


class ActivationStorage:
    def __init__(
        self,
        experts: list[int],
        topN: int,
        seq_len: int,
        unembed_matrix: torch.Tensor,
        logit_lens_k: int = 3,
        device="cpu",
    ):
        """
        Args:
            experts: List of expert IDs to track.
            topN: Number of top examples to store per expert.
            seq_len: Maximum sequence length.
            unembed_matrix: The model's unembedding matrix [Vocab_Size, d_model].
            logit_lens_k: How many top predicted tokens to save per active position.
            device: Computing device.
        """
        self.experts = experts
        self.topN = topN
        self.device = device
        self.num_logits = logit_lens_k
        self.unembed_matrix = unembed_matrix.T.to(self.device)

        # Stores the representative scalar score (max weight) for the sequence
        self.max_scores = {
            e: torch.full((topN,), -1.0, dtype=torch.float, device=device)
            for e in experts
        }
        self.scores = {
            e: torch.full((topN, seq_len), -1.0, dtype=torch.float, device=device)
            for e in experts
        }

        # Stores the full token sequence [TopN, Seq_Len]
        self.sequences = {
            e: torch.zeros((topN, seq_len), dtype=torch.long, device=device)
            for e in experts
        }

        # Shape: [TopN, Seq_Len, k_promoted]
        self.promoted_tokens = {
            e: torch.zeros(
                (topN, seq_len, logit_lens_k), dtype=torch.long, device=device
            )
            for e in experts
        }

    def update_buffers(
        self,
        batch_tokens: torch.Tensor,
        activations: torch.Tensor,
        router_indices: torch.Tensor,
        scores: torch.Tensor,
    ):
        """
        Args:
            batch_tokens: [B, S] Token IDs.
            activations: [BS, k, d_model] Input activations to the experts.
            router_indices: [BS, k] Indices of experts selected.
            scores: [BS, k] Router weights/scores.
        """
        B, S = batch_tokens.shape
        k = activations.shape[1]
        activations = activations.to(self.device).view(B, S, k, -1)
        batch_tokens = batch_tokens.to(self.device)
        router_indices = router_indices.to(self.device).view(B, S, k)
        scores = scores.to(self.device).view(B, S, k)

        for expert_id in self.experts:
            hits_k = router_indices == expert_id

            if scores[hits_k].numel() == 0:
                continue

            token_level_mask = hits_k.any(dim=-1)
            seq_has_activity = token_level_mask.any(dim=-1)

            if not seq_has_activity.any():
                continue

            valid_indices = torch.where(seq_has_activity)[0]
            new_seqs = batch_tokens[valid_indices]

            mask = hits_k.unsqueeze(-1)
            new_acts = (activations * mask).sum(dim=2)[valid_indices]

            logits = new_acts @ self.unembed_matrix
            winning_promoted = logits.topk(self.num_logits, dim=-1).indices

            active_weights_k = scores[valid_indices] * hits_k[valid_indices]
            new_scores = active_weights_k.max(dim=-1).values

            new_max_scores = new_scores.max(dim=-1).values

            current_min = self.max_scores[expert_id][-1]
            if new_max_scores.max() <= current_min:
                continue

            curr_max_scores = self.max_scores[expert_id]
            curr_scores = self.scores[expert_id]
            curr_seqs = self.sequences[expert_id]
            curr_logits = self.promoted_tokens[expert_id]

            valid_curr_mask = curr_max_scores > -1.0

            all_max_scores = torch.cat(
                [curr_max_scores[valid_curr_mask], new_max_scores]
            )
            all_seqs = torch.cat([curr_seqs[valid_curr_mask], new_seqs])
            all_scores = torch.cat([curr_scores[valid_curr_mask], new_scores])
            all_logits = torch.cat([curr_logits[valid_curr_mask], winning_promoted])

            sorted_scores, sort_idxs = all_max_scores.sort(descending=True)

            keep_scores = sorted_scores[: self.topN]
            keep_idxs = sort_idxs[: self.topN]

            num_keep = len(keep_scores)

            self.max_scores[expert_id][:num_keep] = keep_scores
            self.scores[expert_id][:num_keep] = all_scores[keep_idxs]
            self.sequences[expert_id][:num_keep] = all_seqs[keep_idxs]

            self.promoted_tokens[expert_id][:num_keep] = all_logits[keep_idxs]

    def _check_completeness(self):
        """Ensures all experts have collected topN samples."""
        for e in self.experts:
            if (self.max_scores[e] == -1.0).any():
                valid_count = (self.max_scores[e] != -1.0).sum().item()
                raise ValueError(
                    f"Expert {e} has not collected enough samples yet. "
                    f"Expected {self.topN}, found {valid_count}."
                )

    def make_examples(self, tokenizer: PreTrainedTokenizer, explain_ratio: float):
        """
        Generates disjoint datasets for Explainer and Scorer.
        """
        self._check_completeness()

        n_explain = int(self.topN * explain_ratio)
        n_remaining = self.topN - n_explain
        n_score_pos = n_remaining // 2
        n_score_neg = n_score_pos

        extracted_data = {}
        global_negative_pool = []

        for e in self.experts:
            all_seqs = self.sequences[e]
            scores = self.scores[e].tolist()
            promoted_ids_batch = self.promoted_tokens[e]

            texts = []
            promoted_strs_batch = []

            for i in range(self.topN):
                seq_ids = all_seqs[i].tolist()
                text = tokenizer.convert_ids_to_tokens(seq_ids)
                texts.append(text)

                seq_promoted_ids = promoted_ids_batch[i]
                seq_promoted_strs = []
                for t_idx in range(seq_promoted_ids.size(0)):
                    k_ids = seq_promoted_ids[t_idx].tolist()
                    k_strs = tokenizer.convert_ids_to_tokens(k_ids)
                    seq_promoted_strs.append(k_strs)

                promoted_strs_batch.append(seq_promoted_strs)

            n_pos = n_explain + n_score_pos

            explain_items = [
                (texts[i], scores[i], promoted_strs_batch[i]) for i in range(n_explain)
            ]
            score_pos_items = [(texts[i], scores[i]) for i in range(n_explain, n_pos)]
            pool_items = [(e, texts[i], scores[i]) for i in range(n_pos, self.topN)]

            extracted_data[e] = {"explain": explain_items, "score_pos": score_pos_items}
            global_negative_pool.extend(pool_items)

        random.shuffle(global_negative_pool)

        explainer_outputs: list[AutoInterpData] = []
        scorer_outputs: list[AutoInterpData] = []

        for e in self.experts:
            ex_data = extracted_data[e]["explain"]
            ex_texts, ex_masks, ex_promoted = map(list, zip(*ex_data))

            explainer_outputs.append(
                AutoInterpData(
                    examples=ex_texts,
                    scores=ex_masks,
                    top_promoted=ex_promoted,
                    expert_id=e,
                )
            )

            pos_items = extracted_data[e]["score_pos"]
            scorer_candidates = [(x[0], x[1], 1) for x in pos_items]

            neg_items = []
            needed = n_score_neg
            collected_count = 0

            for item_expert_id, item_text, item_mask in global_negative_pool:
                if collected_count >= needed:
                    break

                if item_expert_id != e:
                    neg_items.append((item_text, item_mask, 0))
                    collected_count += 1

            if len(neg_items) < needed:
                raise RuntimeError(f"Not enough negative samples found for Expert {e}")

            scorer_candidates.extend(neg_items)
            random.shuffle(scorer_candidates)

            sc_texts, sc_masks, sc_labels = map(list, zip(*scorer_candidates))

            scorer_outputs.append(
                AutoInterpData(
                    examples=sc_texts,
                    scores=sc_masks,
                    expert_id=e,
                    labels=sc_labels,
                )
            )

        return explainer_outputs, scorer_outputs


def collect_moe_activations(
    model,
    dl: DataLoader,
    experts: list[int],
    layer: int,
    topN: int,
    seq_len: int,
    num_tokens: int,
):
    hooks = [(f"layers.{layer}.mlp.experts", experts_fwd)]
    lm_head = model.lm_head.weight
    storage = ActivationStorage(experts, topN, seq_len, lm_head, device=model.device)

    current_tokens = 0
    pbar = tqdm(total=num_tokens, desc="Collecting Examples", unit_scale=True)

    with register(model, hooks) as state:
        for sample in dl:
            new_tokens = sample["input_ids"].numel()

            forward_pass(sample, model)

            storage.update_buffers(
                sample["input_ids"],
                state.storage["acts"],
                state.storage["selected_experts"],
                state.storage["scores"],
            )

            current_tokens += new_tokens
            pbar.update(new_tokens)

            if current_tokens >= num_tokens:
                break

    pbar.close()
    return storage


class ExplainerResult(BaseModel):
    hypothesis: str = Field(
        description="A concise, one-sentence functional description of the expert's role (3-12 words)."
    )


class ScorerResult(BaseModel):
    labels: list[int] = Field(
        description="A list of integers (0 or 1) matching exactly the length of the examples."
    )


EXPLAIN_DATA_TEMPLATE = """<example id="{id}">
    <snippet>
    {context}
    </snippet>
    <top_activations>
{top_acts}
    </top_activations>
</example>"""

EXPLAIN_TOP_TEMPLATE = """        <item token_str="{token}" score="{score}" promoted_tokens="{promoted}"/>"""


EXPLAIN_SYSTEM_PROMPT = """<role>
You are an expert interpretability researcher analyzing a specific 'Expert' within a Mixture-of-Experts (MoE) Transformer.
</role>

<task>
You will be provided with several text snippets ({seq_len} tokens long). In each snippet, the specific Expert being analyzed was active for one or more tokens.
Your goal is to formulate a single, precise hypothesis explaining the computational role of this Expert. The hypothesis should be a concise, one-sentence functional description of the expert's role (3-12 words).
</task>

<data_structure>
Each example consists of:
1. <snippet>: The raw text. Tokens routed to this expert are wrapped in double asterisks (e.g., **token**).
2. <top_activations>: A list of the top active tokens in that snippet (up to {num_show}), sorted by an importance score (Router Weight * Output L2 Norm).
   - 'score': The obtained score for that token.
   - 'token_str': The string representation.
   - 'promoted_tokens': The top {num_logits} tokens the expert predicted next (Logit Lens).
</data_structure>

<guidelines>
1. **Analyze Density:** Does the expert activate sporadically (specific entities) or continuously (syntactic blocks)?
2. **Consult Logit Lens:** Use the 'promoted_tokens' to understand the *effect* of the expert. If an expert activates on 'New', and promotes 'York', 'Zealand', 'Jersey', it is a named-entity completer.
3. **Generalize:** Do not overfit to a single example. Find the common thread across all examples.
4. **Formatting:** Ignore the `**` markers when analyzing the natural flow of text; they are only for highlighting.
</guidelines>"""

EXPLAIN_USER_PROMPT = """<context>Here are the maximal activating examples for Expert {expert_id}.
</context>

<data>
{examples}
</data>

<instruction>
Based strictly on the data above, analyze the <top_activations> and their context in the <snippet>.
Generate your <hypothesis> now.
</instruction>>"""


SCORE_SYSTEM_PROMPT = """<role>
You are an automated evaluator for interpretability hypotheses.
</role>

<task>
You will be given:
1. A **Hypothesis** describing the function of a specific MoE Expert.
2. A list of **Test Examples**. Each example contains a text snippet, where active tokens are highlighted with double asterisks (e.g., **token**).

Your job is to determine: **Does the highlighted token pattern in the example match the Hypothesis?**
- If the highlighted tokens fit the hypothesis description: Output 1.
- If the highlighted tokens clearly violate the hypothesis or are unrelated: Output 0.
</task>

<constraints>
- You must evaluate strictly based on the provided Hypothesis.
- You must verify that the **Hypothesis specifically describes the **BOLDED tokens**, not just the general topic of the sentence.
</constraints>"""

SCORE_USER_PROMPT = """<hypothesis>
{hypothesis}
</hypothesis>

<examples>
{examples}
</examples>

<instruction>
Evaluate the {count} examples above against the hypothesis.
First, perform your analysis.
Then, output the final list. Ensure it contains exactly {count} integers. 
</instruction>."""

SCORE_EXAMPLE_TEMPLATE = """<example id="{id}">
    <snippet>{context}</snippet>
</example>"""


class AutoInterp:
    """Automatic Interpretability pipeline to interpret MoE experts. Has substantial error handling for free tier and runnning into rate limiting."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_logits: int,
        seq_len: int,
        model_name="gemini-3-flash-preview",
        max_top_acts_to_show=5,
    ):
        self.client = genai.Client()
        self.model_name = model_name

        self.tok = tokenizer
        self.seq_len = seq_len
        self.num_logits = num_logits
        self.max_top_acts_to_show = max_top_acts_to_show

        self.EXPLAIN_SYSTEM_PROMPT = EXPLAIN_SYSTEM_PROMPT.format(
            seq_len=seq_len, num_logits=num_logits, num_show=max_top_acts_to_show
        )
        self.SCORE_SYSTEM_PROMPT = SCORE_SYSTEM_PROMPT

    def _highlight(self, tokens: list[str], scores: list[float]):
        result = []

        for token, score in zip(tokens, scores):
            active = score > 0.0

            if active:
                result.extend(["**", token, "**"])
            else:
                result.append(token)

        return self.tok.convert_tokens_to_string(result)

    def _format_explain(self, data: AutoInterpData):
        assert data.top_promoted is not None
        out = []

        for i, example in enumerate(data.examples):
            all_promoted = []

            scores = data.scores[i]
            promoted = data.top_promoted[i]
            indices = sorted(range(len(scores)), key=lambda n: scores[n], reverse=True)
            top_indices = [
                idx for idx in indices[: self.max_top_acts_to_show] if scores[idx] > 0.0
            ]

            for idx in top_indices:
                top = ", ".join([p.replace("Ġ", "") for p in promoted[idx]])
                top_acts = EXPLAIN_TOP_TEMPLATE.format(
                    token=example[idx].replace("Ġ", ""),
                    score=f"{scores[idx]:.2f}",
                    promoted=top,
                )
                all_promoted.append(top_acts)

            top_acts = "\n".join(all_promoted)

            text = self._highlight(example, scores)
            text = EXPLAIN_DATA_TEMPLATE.format(
                context=text, id=i + 1, top_acts=top_acts
            )
            out.append(text)

        examples = "\n".join(out)

        return EXPLAIN_USER_PROMPT.format(expert_id=data.expert_id, examples=examples)

    def _format_scoring(self, data: AutoInterpData, hypothesis: str):
        out = []

        for i, example in enumerate(data.examples):
            text = self._highlight(example, data.scores[i])
            text = SCORE_EXAMPLE_TEMPLATE.format(context=text, id=i + 1)
            out.append(text)

        examples = "".join(out)

        return SCORE_USER_PROMPT.format(
            count=len(data.examples), hypothesis=hypothesis, examples=examples
        )

    def _compute_scores(self, scores: list[int], labels: list[int]):
        if len(scores) > len(labels):
            scores = scores[: len(labels)]

        # Pad scores if the LLM stopped early (Treat missing answers as 0)
        elif len(scores) < len(labels):
            missing_count = len(labels) - len(scores)
            scores += [0] * missing_count

        tp = fp = fn = tn = 0

        for t, p in zip(labels, scores):
            match (t, p):
                case (1, 1):
                    tp += 1
                case (0, 1):
                    fp += 1
                case (1, 0):
                    fn += 1
                case (0, 0):
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

        return precision, recall, f1, accuracy, scores

    @staticmethod
    def extract_retry_delay(details: dict) -> float | None:
        try:
            for d in details.get("details", []):
                if d.get("@type", "").endswith("RetryInfo"):
                    s = d.get("retryDelay", "")
                    return float(re.sub(r"[^\d.]", "", s))

        except Exception:
            pass

        return None

    @staticmethod
    def is_hard_quota_error(error: dict) -> bool:
        """Detect quota exhaustion where retries likely won't help."""

        try:
            if isinstance(error, dict):
                for d in error.get("details", []):
                    d_type: str = d.get("@type", "")
                    if d_type.endswith("QuotaFailure"):
                        violations: list[dict] = d.get("violations", [])
                        for vio in violations:
                            if "perday" in vio.get("quotaId", "").lower():
                                return True

        except Exception:
            pass
        return False

    def generate_with_retries(
        self,
        prompt: str,
        system_prompt: str,
        result: type[ExplainerResult] | type[ScorerResult],
        max_retries=5,
    ):
        attempt = 0
        base_backoff = 1.5
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.MINIMAL
                        ),
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_json_schema=result.model_json_schema(),
                    ),
                )
                if response.text is not None:
                    return result.model_validate_json(response.text)
                else:
                    feedback = response.prompt_feedback
                    if feedback:
                        tqdm.write(
                            f"Failed to complete request. Feedback: {feedback.block_reason}: {feedback.block_reason_message}"
                        )
                    else:
                        tqdm.write(f"Failed to complete request: {response}")
                    return None

            except errors.APIError as e:
                if e.code == 429 or e.code == 503 and attempt < max_retries:
                    attempt += 1

                    if AutoInterp.is_hard_quota_error(e.details["error"]):  # type: ignore
                        tqdm.write(f"Unresolvable error (code: {e.code}): {e.message}")
                        return None

                    retry_after = AutoInterp.extract_retry_delay(e.details["error"])  # type: ignore

                    if retry_after is None:
                        retry_after = (base_backoff**attempt) + random.uniform(0, 0.5)

                    tqdm.write(
                        f"[Retry {attempt}/{max_retries}] Rate limited: waiting {retry_after:.1f}s …"
                    )
                    time.sleep(retry_after)
                    continue

                tqdm.write(f"Unresolvable error (code: {e.code}): {e.message}")
                return None

    def explain(self, expert_data: list[AutoInterpData]):
        results: dict[int, dict[str, str | None]] = {}

        for e_data in tqdm(expert_data, desc="Generating Explanations"):
            prompt = self._format_explain(e_data)

            out = self.generate_with_retries(
                prompt, self.EXPLAIN_SYSTEM_PROMPT, ExplainerResult
            )

            results[e_data.expert_id] = {
                "hypothesis": out.hypothesis if out is not None else None,  # type: ignore
                "prompt": prompt,
            }

        return results

    def score(
        self, expert_data: list[AutoInterpData], hypotheses: dict[int, str | None]
    ):
        assert len(expert_data) == len(hypotheses)

        results: dict[int, dict] = {}

        for e_data in tqdm(expert_data, desc="Scoring Examples"):
            assert e_data.labels is not None
            hypothesis = hypotheses[e_data.expert_id]
            if hypothesis is None:
                results[e_data.expert_id] = {
                    "scores": [],
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "accuracy": 0,
                    "labels": e_data.labels,
                    "scorer_prompt": self._format_scoring(e_data, "No hypothesis"),
                }
                continue

            prompt = self._format_scoring(e_data, hypothesis)

            out = self.generate_with_retries(
                prompt, self.SCORE_SYSTEM_PROMPT, ScorerResult
            )

            scores = out.labels if out is not None else []  # type: ignore
            p, r, f1, accuracy, scores = self._compute_scores(scores, e_data.labels)

            results[e_data.expert_id] = {
                "scores": scores,
                "precision": p,
                "recall": r,
                "f1_score": f1,
                "accuracy": accuracy,
                "labels": e_data.labels,
                "scorer_prompt": prompt,
            }

        return results

    def merge(
        self, explanations: dict[int, dict[str, str | None]], scores: dict[int, dict]
    ):
        new = {}
        for e in scores.keys():
            new[e] = {
                "hypothesis": explanations[e]["hypothesis"],
                "explainer_prompt": explanations[e]["prompt"],
                "scores": scores[e]["scores"],
                "labels": scores[e]["labels"],
                "metrics": {
                    "recall": scores[e]["recall"],
                    "precision": scores[e]["precision"],
                    "f1_score": scores[e]["f1_score"],
                    "accuracy": scores[e]["accuracy"],
                },
                "scorer_prompt": scores[e]["scorer_prompt"],
            }

        return new

    def run(
        self, explainer_data: list[AutoInterpData], scorer_data: list[AutoInterpData]
    ):
        explanations = self.explain(explainer_data)

        scores = self.score(
            scorer_data, {k: v["hypothesis"] for k, v in explanations.items()}
        )
        return self.merge(explanations, scores)


@torch.inference_mode()
def main(args):
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = Dataset(tokenizer, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=5, pin_memory=True)

    model, is_moe, num_experts = load_model(args.model_name)
    assert is_moe, "Model need to be an MoE model"

    experts = args.experts if args.experts else list(range(num_experts))
    storage = collect_moe_activations(
        model, dl, experts, args.layer, args.topN, args.seq_len, args.num_tokens
    )

    explainer_data, scorer_data = storage.make_examples(tokenizer, args.explain_ratio)

    del model, ds, dl, storage
    gc.collect()
    torch.cuda.empty_cache()

    auto_interp = AutoInterp(
        tokenizer, args.num_logits, args.seq_len, max_top_acts_to_show=5
    )
    out = auto_interp.run(explainer_data, scorer_data)

    os.makedirs(args.out_path, exist_ok=True)
    filename = (
        f"{args.out_path}/{model_name.split('/')[1]}_L{args.layer}_auto_interp.json"
    )
    with open(filename, mode="w") as file:
        json.dump(out, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatic Interpretability pipeline. This pipeline will collect examples for the specified MoE experts, then it will call the Gemini API and generate a natural language label for each expert. After all experts have a label, every expert is scored by a seperate API call. This script uses 'gemini-3-flash-preview'(Minimal Thinking Level) for both explainer and scorer model. Finally all results are saved in JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model_name",
        default="allenai/OLMoE-1B-7B-0125",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "-e",
        "--experts",
        nargs="+",
        type=int,
        default=[],
        help="MoE Expert to explain and score. Defaults to all experts.",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=0,
        help="Layer from which MoE experts are selected",
    )
    parser.add_argument(
        "-t",
        "--num_tokens",
        type=int,
        default=2_000_000,
        help="Number of tokens to iterate over to find examples",
    )
    parser.add_argument(
        "-s",
        "--seq_len",
        type=int,
        default=32,
        help="Sequence length (number of tokens) for each example presented to explainer and scoer model",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to use for MoE forward passes",
    )
    parser.add_argument(
        "-n",
        "--topN",
        type=int,
        default=10,
        help="Number of top examples to collect.",
    )
    parser.add_argument(
        "--num_logits",
        type=int,
        default=3,
        help="Number of Logit Lens tokens to show the explainer model per token. Are only shownn for top routed tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--explain_ratio",
        type=int,
        default=0.5,
        help="Ratio of explainer examples and scorer examples. Ratio is applied to topN",
    )
    parser.add_argument(
        "--out_path",
        default="./data/auto-interp",
        help="Output folder for results",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main(args)
