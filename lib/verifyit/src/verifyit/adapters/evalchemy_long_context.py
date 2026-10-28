# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source scoring and selection policies for OpenAI MRCR and GraphWalks.

Adopted from Marin Evalchemy's pinned OpenAI dataset integrations. Preparation
and grading are pure; model-tokenizer eligibility belongs to the evaluation run.
"""

import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher

from verifyit.grade import Reward, scored

MRCR_BIN_UPPER_BOUNDS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
MRCR_NEEDLE_COUNTS = (2, 4, 8)
GRAPHWALKS_CONTEXT_MARGIN = 64
GRAPHWALKS_PREFIX_TOKEN_MARGIN = 256
GRAPHWALKS_PREFIX_CHARS_PER_TOKEN = 8
GRAPHWALKS_MIN_OUTPUT_TOKENS = 4096
GRAPHWALKS_REASONING_RESERVE = 4096
GRAPHWALKS_OUTPUT_BUDGET_MULTIPLIER = 2
FINAL_ANSWER_PREFIX = "Final Answer:"


def mrcr_bin(total_tokens: int) -> int | None:
    """Map the official content-plus-answer token count into the published bins."""
    lower = 4096
    for upper in MRCR_BIN_UPPER_BOUNDS:
        if lower <= total_tokens <= upper:
            return upper
        lower = upper + 1
    return None


def interleave_mrcr_indices(
    cells: Mapping[tuple[int, int], Sequence[int]],
    max_context: int,
    limit: int | None = None,
) -> list[int]:
    """Interleave full source-order cells by bin then needles, applying the cap afterwards."""
    desired = [
        (upper, needles) for upper in MRCR_BIN_UPPER_BOUNDS if upper <= max_context for needles in MRCR_NEEDLE_COUNTS
    ]
    if not desired:
        raise ValueError("MRCR requires at least an 8192-token context bin")
    missing = [cell for cell in desired if not cells.get(cell)]
    if missing:
        raise ValueError(f"Pinned MRCR dataset has no examples for requested cells: {missing}")
    rounds = max(len(cells[cell]) for cell in desired)
    ordered = [cells[cell][position] for position in range(rounds) for cell in desired if position < len(cells[cell])]
    return ordered if limit is None else ordered[:limit]


def graphwalks_output_budget(answer_tokens: int) -> int:
    """Retain the source's gold-dependent output allowance without shortening it to fit."""
    return GRAPHWALKS_OUTPUT_BUDGET_MULTIPLIER * max(
        GRAPHWALKS_MIN_OUTPUT_TOKENS,
        answer_tokens + GRAPHWALKS_REASONING_RESERVE,
    )


def grade_nonce_similarity(expected: str, nonce: str, response: str) -> Reward:
    hit = response.startswith(nonce)
    score = SequenceMatcher(None, response.removeprefix(nonce), expected.removeprefix(nonce)).ratio() if hit else 0.0
    return scored(score, metrics={"accuracy": score, "prefix_hit_rate": float(hit)})


def graphwalks_answer(response: str) -> tuple[list[str], bool]:
    line = response.rstrip().split("\n")[-1]
    if FINAL_ANSWER_PREFIX not in line:
        return [], True
    match = re.search(r"\[.*\]", line)
    if match is None:
        return [], True
    return [node.strip() for node in match.group(0).strip("[]").split(",") if node.strip()], False


def grade_graphwalks(expected: tuple[str, ...], response: str) -> Reward:
    extracted, failed = graphwalks_answer(response)
    predicted, truth = set(extracted), set(expected)
    if failed:
        precision = recall = f1 = 0.0
    elif not truth and not predicted:
        precision = recall = f1 = 1.0
    else:
        overlap = len(predicted & truth)
        recall = overlap / len(truth) if truth else 0.0
        precision = overlap / len(predicted) if predicted else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return scored(
        f1,
        metrics={
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "exact_match": float(predicted == truth and not failed),
        },
        model_answer=extracted,
        answer_extraction_errors=["failed_to_parse"] if failed else [],
    )
