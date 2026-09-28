# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage and per-instance metrics for Table-9 companion evaluations."""

import math
from collections.abc import Mapping, Sequence

from marin.evaluation.olmo_base_eval.components import (
    BASIC_SKILLS_SUBTASKS,
    MINERVA_SUBTASKS,
    MMLU_CATEGORY_WEIGHTS,
    MT_MBPP_SUBTASKS,
    scored_tasks,
    table9_components,
)

CHOICE_BACKFILL_TASKS = (*BASIC_SKILLS_SUBTASKS, "coqa", "drop", "jeopardy", "naturalqs", "squad")
GENERATION_BACKFILL_TASKS = (*MINERVA_SUBTASKS, "codex_humaneval", "mbpp")
DEFERRED_TASKS = MT_MBPP_SUBTASKS


def choice_metrics(
    logprobs: Sequence[float], token_counts: Sequence[int], choices: Sequence[str], gold: int
) -> dict[str, float]:
    """Score all choices, retaining raw, per-token and per-character accuracy."""
    if len(logprobs) < 2 or not (len(logprobs) == len(token_counts) == len(choices)):
        raise ValueError("Accuracy requires scores for every answer choice, not a gold-only BPB request")
    if not 0 <= gold < len(choices) or any(n <= 0 for n in token_counts) or any(not c for c in choices):
        raise ValueError("Invalid gold index or empty answer choice")
    if any(not math.isfinite(p) for p in logprobs):
        raise ValueError("Non-finite choice score")
    scores = {
        "acc": list(logprobs),
        "acc_per_token": [p / n for p, n in zip(logprobs, token_counts, strict=True)],
        "acc_per_char": [p / len(c) for p, c in zip(logprobs, choices, strict=True)],
    }
    return {name: float(max(range(len(values)), key=values.__getitem__) == gold) for name, values in scores.items()}


def coverage_report(leaf_scores: Mapping[str, float], *, deferred: Sequence[str] = DEFERRED_TASKS) -> dict:
    """Report every Table-9 component without hiding partial coverage in a macro."""
    unknown = set(leaf_scores) - set(scored_tasks())
    if unknown:
        raise ValueError(f"Unknown Table-9 leaves: {sorted(unknown)}")
    if not set(deferred) <= set(table9_components()) or set(deferred) & set(leaf_scores):
        raise ValueError("Deferred components must be known and unscored")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in leaf_scores.values()):
        raise ValueError("Accuracy scores must be finite fractions in [0, 1]")
    components = {}
    for name in table9_components():
        if name in MMLU_CATEGORY_WEIGHTS:
            weights = MMLU_CATEGORY_WEIGHTS[name]
            if weights.keys() <= leaf_scores.keys():
                components[name] = sum(weight * leaf_scores[subject] for subject, weight in weights.items())
        elif name in leaf_scores:
            components[name] = leaf_scores[name]
    missing = sorted(set(table9_components()) - set(components) - set(deferred))
    # A deferred component is still a gap: never publish a "Table-9 accuracy" macro over fewer than 51.
    return {
        "total_components": len(table9_components()),
        "covered_components": len(components),
        "components": components,
        "missing": missing,
        "deferred": list(deferred),
        "complete": len(components) == len(table9_components()),
        "requested_scope_complete": not missing,
    }


def validate_task_samples(task: str, samples: list[dict], expected_ids: Sequence[int], metric: str) -> float:
    """Reject missing, duplicated, non-finite or ungraded samples before aggregation."""
    if task not in scored_tasks() or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected a known task and unique nonempty document inventory")
    if len(samples) != len(expected_ids) or {s["doc_id"] for s in samples} != set(expected_ids):
        raise ValueError(f"Incomplete or duplicate samples for {task}")
    values = [s["metrics"][metric] for s in samples]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError(f"Invalid {metric} scores for {task}")
    return sum(values) / len(values)
