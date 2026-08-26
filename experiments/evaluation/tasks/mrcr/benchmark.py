# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI MRCR prompt preparation and canonical generation scoring."""

import json
import os
from collections import defaultdict
from difflib import SequenceMatcher
from functools import cache

import datasets
import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MRCR_BIN_UPPER_BOUNDS = (8_192, 16_384, 32_768, 65_536, 131_072, 262_144)
MRCR_NEEDLE_COUNTS = (2, 4, 8)
MRCR_MAX_LENGTH_ENV = "MARIN_MRCR_MAX_LENGTH"
MRCR_MAX_GEN_TOKS_ENV = "MARIN_MRCR_MAX_GEN_TOKS"
MRCR_TOKENIZER_ENV = "MARIN_MRCR_TOKENIZER"

_OFFICIAL_TOKENIZER = tiktoken.get_encoding("o200k_base")


def render_prompt(doc: dict) -> str:
    """Render an MRCR message list for completion-style base-model inference."""

    messages = json.loads(doc["prompt"])
    parts = [f"{message['role'].capitalize()}: {message['content']}\n" for message in messages]
    parts.append("Assistant: ")
    return "".join(parts)


def doc_to_target(doc: dict) -> str:
    return doc["answer"]


def mrcr_bin(total_tokens: int) -> int | None:
    """Return the official MRCR bin upper bound through 262K."""

    lower = 4_096
    for upper in MRCR_BIN_UPPER_BOUNDS:
        if lower <= total_tokens <= upper:
            return upper
        lower = upper + 1
    return None


def official_token_count(doc: dict) -> int:
    """Count prompt and answer content with the benchmark's o200k tokenizer."""

    messages = json.loads(doc["prompt"])
    return sum(len(_OFFICIAL_TOKENIZER.encode(message["content"])) for message in messages) + len(
        _OFFICIAL_TOKENIZER.encode(doc["answer"])
    )


@cache
def _model_tokenizer(name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name)


@cache
def _served_context() -> tuple[int, int, str] | None:
    max_length = os.environ.get(MRCR_MAX_LENGTH_ENV)
    max_gen_toks = os.environ.get(MRCR_MAX_GEN_TOKS_ENV)
    tokenizer_name = os.environ.get(MRCR_TOKENIZER_ENV)
    if max_length is None and max_gen_toks is None and tokenizer_name is None:
        return None
    if max_length is None or max_gen_toks is None or tokenizer_name is None:
        raise ValueError("MRCR context filtering requires max length, generation budget, and tokenizer")
    return int(max_length), int(max_gen_toks), tokenizer_name


def _fits_served_context(doc: dict) -> bool:
    served_context = _served_context()
    if served_context is None:
        return True
    max_length, max_gen_toks, tokenizer_name = served_context
    prompt_tokens = len(_model_tokenizer(tokenizer_name).encode(render_prompt(doc), add_special_tokens=True))
    return prompt_tokens + max_gen_toks <= max_length


def _potential_bins() -> tuple[int, ...]:
    served_context = _served_context()
    if served_context is None:
        return MRCR_BIN_UPPER_BOUNDS
    max_length, max_gen_toks, _ = served_context
    lower = 4_096
    potential: list[int] = []
    for upper in MRCR_BIN_UPPER_BOUNDS:
        if lower + max_gen_toks <= max_length:
            potential.append(upper)
        lower = upper + 1
    return tuple(potential)


def _ordered_docs(dataset: datasets.Dataset, *, one_per_cell: bool) -> datasets.Dataset:
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    bins_by_index: dict[int, int] = {}
    potential_bins = _potential_bins()
    target_cell_count = len(potential_bins) * len(MRCR_NEEDLE_COUNTS)
    for index, doc in enumerate(dataset):
        needles = int(doc["n_needles"])
        if needles not in MRCR_NEEDLE_COUNTS:
            continue
        upper = mrcr_bin(official_token_count(doc))
        if upper not in potential_bins:
            continue
        cell = (upper, needles)
        if one_per_cell and cells[cell]:
            continue
        if not _fits_served_context(doc):
            continue
        cells[cell].append(index)
        bins_by_index[index] = upper
        if one_per_cell and len(cells) == target_cell_count:
            break

    ordered_indices: list[int] = []
    cell_order = [(upper, needles) for upper in MRCR_BIN_UPPER_BOUNDS for needles in MRCR_NEEDLE_COUNTS]
    rounds = 1 if one_per_cell else max((len(indices) for indices in cells.values()), default=0)
    for position in range(rounds):
        for cell in cell_order:
            indices = cells.get(cell, ())
            if position < len(indices):
                ordered_indices.append(indices[position])

    selected = dataset.select(ordered_indices)
    selected_bins = [bins_by_index[index] for index in ordered_indices]
    return selected.add_column("mrcr_bin_upper", selected_bins)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter to supported, context-fitting cells and interleave them for paired limits."""

    return _ordered_docs(dataset, one_per_cell=False)


def process_smoke_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Select one context-fitting example from every supported MRCR cell."""

    return _ordered_docs(dataset, one_per_cell=True)


def score_response(response: str, answer: str, nonce: str) -> tuple[float, float]:
    """Return canonical MRCR similarity and nonce-prefix hit."""

    if not response.startswith(nonce):
        return 0.0, 0.0
    response_body = response.removeprefix(nonce)
    answer_body = answer.removeprefix(nonce)
    return float(SequenceMatcher(None, response_body, answer_body).ratio()), 1.0


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Score one generation and emit its aggregate and cell metric."""

    score, prefix_hit = score_response(results[0], doc["answer"], doc["random_string_to_prepend"])
    upper = int(doc["mrcr_bin_upper"])
    needles = int(doc["n_needles"])
    return {
        "mrcr_accuracy": score,
        "prefix_hit_rate": prefix_hit,
        f"mrcr_{upper}_{needles}needle": score,
    }
