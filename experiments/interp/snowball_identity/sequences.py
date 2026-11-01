# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token sequences for the all-position logit check between the torch harness and vLLM.

Three groups, all built from the same tokenizer on both sides:

``check``
    The built-in check prompts (``prompts.CHECK_PROMPTS``), rendered with a generation prompt.
``contrastive``
    The identity rows that carry a ``contrastive`` field, built exactly as the identity scorer
    (``identity_eval.score_contrastive_row``) builds them: chat template with thinking off, then the
    assistant ``prefix`` plus each candidate. Each row runs with no system prompt and, unless the row
    has its own, with ``BASELINE_SYSTEM_PROMPT``. ``score`` marks the candidate's tokens, whose
    log-probabilities sum to the candidate score.
``long``
    One raw text of ``LONG_TOKENS`` tokens, longer than the 2,048-token sliding window, built from
    dev-split rows only.

This module must not import torch: the vLLM reference job runs without it.
"""

import json
from dataclasses import dataclass

from rigging.filesystem.buckets import filesystem_for

from experiments.interp.snowball_identity.prompts import CHECK_PROMPTS, encode

BASELINE_SYSTEM_PROMPT = "You are Marin Snowball, an AI model created by Open Athena."
"""Same text as ``identity_eval.BASELINE_SYSTEM_PROMPT``."""
LONG_TOKENS = 3000


@dataclass(frozen=True)
class Sequence:
    id: str
    group: str
    tokens: tuple[int, ...]
    score: tuple[int, int] | None = None
    """Token positions [start, end) whose log-probabilities sum to the contrastive candidate score."""
    meta: dict | None = None


def read_rows(path: str) -> list[dict]:
    fs, local = filesystem_for(path)
    return [json.loads(line) for line in fs.cat_file(local).decode().splitlines() if line.strip()]


def check_sequences(tokenizer) -> list[Sequence]:
    return [Sequence(f"check|{row['id']}", "check", encode(tokenizer, row).tokens) for row in CHECK_PROMPTS]


def contrastive_sequences(tokenizer, rows: list[dict]) -> list[Sequence]:
    sequences = []
    for row in rows:
        spec = row.get("contrastive")
        if not spec:
            continue
        for condition in ("no_system", "system_baseline"):
            messages = row["messages"]
            if condition == "system_baseline":
                if any(message["role"] == "system" for message in messages):
                    continue
                messages = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}, *messages]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prefix = spec["prefix"]
            for index, candidate in enumerate(spec["candidates"]):
                encoded = tokenizer(prefix + candidate, add_special_tokens=False, return_offsets_mapping=True)
                start = next(i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > len(prefix))
                sequences.append(
                    Sequence(
                        f"contrastive|{row['id']}|{condition}|{index}",
                        "contrastive",
                        tuple(prompt_ids) + tuple(encoded["input_ids"]),
                        (len(prompt_ids) + start, len(prompt_ids) + len(encoded["input_ids"])),
                        {
                            "row_id": row["id"],
                            "condition": condition,
                            "candidate": candidate,
                            "index": index,
                            "correct": tuple(spec["correct_indices"]),
                        },
                    )
                )
    return sequences


def long_sequence(tokenizer, rows: list[dict]) -> Sequence:
    parts = []
    for row in rows:
        if row.get("split") != "dev":
            continue
        parts.extend(message["content"] for message in row["messages"] if message.get("content"))
        if row.get("reference"):
            parts.append(row["reference"])
    tokens = [tokenizer.bos_token_id, *tokenizer("\n\n".join(parts), add_special_tokens=False)["input_ids"]]
    if len(tokens) < LONG_TOKENS:
        raise ValueError(f"Dev rows give only {len(tokens)} tokens; need {LONG_TOKENS}")
    return Sequence("long|dev-text", "long", tuple(tokens[:LONG_TOKENS]))


def build_sequences(tokenizer, identity_path: str) -> list[Sequence]:
    rows = read_rows(identity_path)
    sequences = [*check_sequences(tokenizer), *contrastive_sequences(tokenizer, rows), long_sequence(tokenizer, rows)]
    if len({sequence.id for sequence in sequences}) != len(sequences):
        raise ValueError("Duplicate sequence ids")
    return sequences
