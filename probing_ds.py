import re
import random
from enum import Enum
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset


@dataclass(frozen=True, slots=True)
class POSConcepts:
    adjective: str = "ADJ"
    adposition: str = "ADP"
    adverb: str = "ADV"
    auxiliary: str = "AUX"
    coordinating_conjunction: str = "CCONJ"
    determiner: str = "DET"
    # interjection: str = "INTJ" # NOTE: Excluded because of rarity in wikidata
    noun: str = "NOUN"
    numeral: str = "NUM"
    particle: str = "PART"
    pronoun: str = "PRON"
    proper_noun: str = "PROPN"
    punctuation: str = "PUNCT"
    subordinating_conjunction: str = "SCONJ"
    symbol: str = "SYM"
    verb: str = "VERB"
    other: str = "X"


@dataclass(frozen=True, slots=True)
class LatexConcepts:
    is_superscript: re.Pattern[str] = re.compile(
        r"\^(?:\{.*?\}|[a-zA-Z0-9])", re.MULTILINE
    )
    is_subscript: re.Pattern[str] = re.compile(
        r"_(?:\{.*?\}|[a-zA-Z0-9])", re.MULTILINE
    )
    is_inline_math: re.Pattern[str] = re.compile(r"\$[^$]+\$|\\\([^)]+\\\)")
    is_display_math: re.Pattern[str] = re.compile(
        r"\$\$[^$]+\$\$|\\\[[^\]]+\\\]|\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        re.DOTALL,
    )
    is_math: re.Pattern[str] = re.compile(
        r"\$[^$]+\$|\$\$[^$]+\$\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]|\\begin\{(?:equation|align|gather|math)\*?\}.*?\\end\{(?:equation|align|gather|math)\*?\}",
        re.DOTALL,
    )
    is_denominator: re.Pattern[str] = re.compile(r"\\frac\{[^}]*\}\{([^}]*)\}")
    is_numerator: re.Pattern[str] = re.compile(r"\\frac\{([^}]*)\}\{[^}]*\}")
    is_frac: re.Pattern[str] = re.compile(r"\\frac\{[^}]*\}\{[^}]*\}")
    is_author: re.Pattern[str] = re.compile(
        r"author:((?:(?:(?!author:|title:|bibliography:|date:|address:|---).)*\n)*)",
        re.MULTILINE,
    )
    is_title: re.Pattern[str] = re.compile(
        r"^title:(?:\s+\||\s+\'| )(.+?)(?:\'|\\|\n|$)(?:\n|$)", re.MULTILINE | re.DOTALL
    )
    is_reference: re.Pattern[str] = re.compile(r"\[@(.*?)\]", re.DOTALL | re.MULTILINE)
    is_abstract: re.Pattern[str] = re.compile(
        r"abstract:((?:(?:(?!author:|title:|bibliography:|date:|address:|---).)*\n)*)",
        re.MULTILINE,
    )


@dataclass(frozen=True, slots=True)
class TextConcepts:
    leading_capital: re.Pattern[str] = re.compile(r"\b[A-Z]")
    leading_loweralpha: re.Pattern[str] = re.compile(r"\b[a-z]")
    all_digits: re.Pattern[str] = re.compile(r"\d+")
    is_not_ascii: re.Pattern[str] = re.compile(r"[^\x00-\x7F]")
    contains_all_whitespace: re.Pattern[str] = re.compile(r"( {2,}|[^\S ]+)")
    all_capitals: re.Pattern[str] = re.compile(r"\b[^a-z]*[A-Z][^a-z]*\b")
    is_not_alphanumeric: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9\s]")
    contains_whitespace: re.Pattern[str] = re.compile(r"\s")
    contains_capital: re.Pattern[str] = re.compile(r"[A-Z]")
    contains_digit: re.Pattern[str] = re.compile(r"\d")


@dataclass(frozen=True, slots=True)
class CodeConcepts:
    is_function_def: re.Pattern[str] = re.compile(
        r"\b(?:def|function|fn|func)\s+\w+\s*\(|^\s*\w+\s*\([^)]*\)\s*\{"
    )
    is_function_call: re.Pattern[str] = re.compile(
        r"\b(?!(?:if|while|for|switch|catch)\b)\w+\s*\([^)]*\)"
    )
    is_assignment: re.Pattern[str] = re.compile(r"\w+\s*=\s*[^=]")
    is_class_def: re.Pattern[str] = re.compile(
        r"\b(?:class|struct|interface)\s+[A-Z]\w*"
    )
    is_import: re.Pattern[str] = re.compile(
        r"\b(?:import|from|require|include|using)\s+"
    )
    is_comment: re.Pattern[str] = re.compile(r"(?://|#|/\*|\<!--)")
    is_string_literal: re.Pattern[str] = re.compile(
        r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
    )
    is_control_flow: re.Pattern[str] = re.compile(
        r"\b(?:if|else|then|end|elif|for|while|switch|case|break|continue|return|match)\b"
    )
    is_loop: re.Pattern[str] = re.compile(r"\b(?:for|while|do)\b")
    is_conditional: re.Pattern[str] = re.compile(r"\b(?:if|else|elif|unless)\b")
    is_exception_handling: re.Pattern[str] = re.compile(
        r"\b(?:try|catch|except|finally|throw|raise)\b"
    )
    is_array_literal: re.Pattern[str] = re.compile(r"\[.*?\]")
    is_method_call: re.Pattern[str] = re.compile(r"\w+\.\w+\s*\([^)]*\)")
    is_lambda: re.Pattern[str] = re.compile(r"=>|\blambda\b")
    is_operator: re.Pattern[str] = re.compile(r"[+\-*/%<>=!&|^~]")
    is_constant: re.Pattern[str] = re.compile(r"\b[A-Z][A-Z0-9_]+\b")
    is_boolean: re.Pattern[str] = re.compile(
        r"\b(?:true|false|True|False|TRUE|FALSE)\b"
    )
    is_null: re.Pattern[str] = re.compile(r"\b(?:null|None|nil|nullptr|undefined)\b")
    is_decorator: re.Pattern[str] = re.compile(r"@\w+")
    is_async: re.Pattern[str] = re.compile(r"\b(?:async|await)\b")


@dataclass(frozen=True, slots=True)
class Concepts:
    pos: POSConcepts = POSConcepts()
    latex: LatexConcepts = LatexConcepts()
    code: CodeConcepts = CodeConcepts()
    text: TextConcepts = TextConcepts()

    def __iter__(self):
        for parent in fields(self):
            parent_value = getattr(self, parent.name)

            if not is_dataclass(parent_value):
                continue

            for f in fields(parent_value):
                yield (parent.name, f.name, getattr(parent_value, f.name))

    def find_concept(self, query: str):
        results: list[tuple[str, str, re.Pattern[str] | str]] = []

        # Case 1: query matches a top-level concept
        names = [field.name for field in fields(self)]

        if query in names:
            parent_value = getattr(self, query)

            if is_dataclass(parent_value):
                for f in fields(parent_value):
                    results.append((query, f.name, getattr(parent_value, f.name)))
            return results

        # Case 2: query matches a sub-field
        for parent, field, value in self:
            if field == query:
                results.append((parent, field, value))

        return results


class DatasetName(Enum):
    code = "code"
    latex = "latex"
    text = "text"
    pos = "pos"

    @property
    def path(self):
        match self.name:
            case "pos":
                return "kinianlo/wikipedia_pos_tagged"
            case "latex" | "text" | "code":
                return "monology/pile-uncopyrighted"


class ProbingDataset(IterableDataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, name: str, concept: str | re.Pattern
    ):
        self.tokenizer = tokenizer
        self.ds_cache = {}
        self.dataset = self._choose_ds(DatasetName[name])
        self.iter = self.pos_iter if name == "pos" else self.regex_iter
        self.name = name
        self.concept = concept
        self.is_code = name == "code"

    def _choose_ds(self, ds_enum: DatasetName):
        if ds_enum.path in self.ds_cache:
            ds = self.ds_cache[ds_enum.path]
        else:
            if ds_enum.name == "pos":
                ds = load_dataset(
                    ds_enum.path, "20220301_simple_spacy", split="train", streaming=True
                )
            else:
                ds = load_dataset(ds_enum.path, split="train", streaming=True)

            self.ds_cache[ds_enum.path] = ds

        match ds_enum.name:
            case "pos" | "text":
                dataset = ds
            case "code":
                dataset = ds.filter(lambda ex: ex["meta"]["pile_set_name"] == "Github")
            case "latex":
                dataset = ds.filter(lambda ex: ex["meta"]["pile_set_name"] == "ArXiv")

        return dataset

    def update(self, name: str, concept: str | re.Pattern):
        if self.concept == concept:
            return
        else:
            self.concept = concept

        if self.name == name:
            return
        else:
            self.name = name
            if hasattr(self, "dataset"):
                del self.dataset

            self.dataset = self._choose_ds(DatasetName[name])
            self.iter = self.pos_iter if name == "pos" else self.regex_iter
            self.is_code = name == "code"

    def __iter__(self):
        return self.iter()

    def regex_iter(self):
        assert isinstance(self.concept, re.Pattern)
        max_chars = 300
        for example in self.dataset:
            text = example["text"]

            if len(text) > max_chars:
                text: str = text[:max_chars] if not self.is_code else text[-max_chars:]

            if not self.concept.search(text):
                continue

            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )

            input_ids = encoding["input_ids"]
            probe_indices, num_pos = self._find_matching_tokens(
                text, encoding["offset_mapping"]
            )

            if any(probe_indices):
                yield {
                    "input_ids": input_ids,
                    "label": 1,
                    "probe_indices": probe_indices,
                }
                neg_indices = [i for i, x in enumerate(probe_indices) if not x]
                num_neg_to_sample = min(num_pos, len(neg_indices))
                sampled_neg_indices = random.sample(neg_indices, k=num_neg_to_sample)
                neg_mask = [False] * len(probe_indices)
                for i in sampled_neg_indices:
                    neg_mask[i] = True

                yield {"input_ids": input_ids, "label": 0, "probe_indices": neg_mask}

    def _find_matching_tokens(self, text: str, offset_mapping):
        char_mask = np.zeros(len(text), dtype=bool)

        # Fill mask based on matches
        for m in self.concept.finditer(text):  # type: ignore
            char_mask[m.start() : m.end()] = True

        matching_mask = []

        # Check tokens against the character mask
        num_pos = 0
        for start, end in offset_mapping:
            if start == end:  # Special tokens
                matching_mask.append(False)
            else:
                # If any character in this token's span is part of a match
                is_match = char_mask[start:end].any()
                num_pos += int(is_match)
                matching_mask.append(is_match)

        return matching_mask, num_pos

    def pos_iter(self):
        for example in self.dataset:
            # First sentence
            flat_tags = example["pos_tags"][0]

            text: str = example["text"][:400]

            word_spans = []  # List of (Tag, StartChar, EndChar)
            cursor = 0

            valid_alignment = True
            for word, tag in flat_tags:
                start = text.find(word, cursor)

                if start == -1:
                    # Alignment broken
                    valid_alignment = False
                    break

                end = start + len(word)
                word_spans.append((tag, start, end))
                cursor = end

            if not valid_alignment:
                continue

            encoding = self.tokenizer(text, return_offsets_mapping=True)
            input_ids = encoding["input_ids"]
            num_tokens = len(input_ids)  # type: ignore
            offsets = encoding["offset_mapping"]

            # Filter valid tokens (exclude special tokens 0,0)
            valid_token_indices = [i for i, (s, e) in enumerate(offsets) if s != e]  # type: ignore

            pos_token_indices = []
            neg_token_indices = []

            for tag, w_start, w_end in word_spans:
                # Find tokens overlapping this word
                overlapping_tokens = []
                for i in valid_token_indices:
                    t_start, t_end = offsets[i]  # type: ignore
                    if not (t_end <= w_start or t_start >= w_end):
                        overlapping_tokens.append(i)

                if not overlapping_tokens:
                    continue

                if tag == self.concept:
                    pos_token_indices.extend(overlapping_tokens)
                else:
                    neg_token_indices.extend(overlapping_tokens)

            if pos_token_indices:
                mask = [False] * num_tokens
                for i in pos_token_indices:
                    mask[i] = True
                yield {"input_ids": input_ids, "label": 1, "probe_indices": mask}

            if neg_token_indices:
                selected_neg = random.sample(
                    neg_token_indices,
                    min(len(neg_token_indices), len(pos_token_indices)),
                )

                mask_neg = [False] * num_tokens
                for i in selected_neg:
                    mask_neg[i] = True
                yield {"input_ids": input_ids, "label": 0, "probe_indices": mask_neg}


def collate_fn(batch, pad_value=0):
    input_ids_list = [
        torch.tensor(item["input_ids"], dtype=torch.long) for item in batch
    ]
    labels_list = [item["label"] for item in batch]
    probe_indices_list = [
        torch.tensor(item["probe_indices"], dtype=torch.bool) for item in batch
    ]

    padded_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_value
    )
    attention_mask = (padded_input_ids != pad_value).long()

    padded_probe_indices = pad_sequence(
        probe_indices_list, batch_first=True, padding_value=False
    )

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "label": torch.tensor(labels_list, dtype=torch.long),
        "probe_indices": padded_probe_indices,
    }
