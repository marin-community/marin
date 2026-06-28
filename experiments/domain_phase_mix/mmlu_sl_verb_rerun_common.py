# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for retrospective MMLU SL-Verb eval reruns."""

from __future__ import annotations

from typing import Any

import fsspec

CHECKPOINT_ROOT = "gs://marin-us-east5/checkpoints"
RESULTS_JSON = "results.json"


def resolve_unique_checkpoint_root(*, source_experiment: str, run_name: str) -> str:
    """Resolve the unique finished checkpoint root for one historical run."""
    pattern = f"{CHECKPOINT_ROOT}/{source_experiment}/{run_name}-*/.executor_status"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    matches = sorted(fs.glob(pattern))
    if not matches:
        raise ValueError(f"No checkpoint status files matched {pattern}")
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one checkpoint root for {run_name}, found {len(matches)}: {matches}")

    match = matches[0].removesuffix("/.executor_status")
    return match if match.startswith("gs://") else f"gs://{match}"


def flatten_eval_results(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten eval-harness results payload into lm_eval-prefixed metric columns."""
    flat: dict[str, float] = {}

    for task_name, task_results in payload.get("results", {}).items():
        for metric_name, metric_value in task_results.items():
            metric_key = metric_name.removesuffix(",none")
            if isinstance(metric_value, int | float):
                flat[f"lm_eval/{task_name}/{metric_key}"] = float(metric_value)

    for metric_name, metric_value in payload.get("averages", {}).items():
        metric_key = metric_name.removesuffix(",none")
        if isinstance(metric_value, int | float):
            flat[f"lm_eval/averages/{metric_key}"] = float(metric_value)

    return flat


def phase_weights_to_columns(phase_weights: dict[str, dict[str, float]]) -> dict[str, float]:
    """Expand nested phase weights into flat CSV-friendly columns."""
    columns: dict[str, float] = {}
    for phase_name, domain_weights in phase_weights.items():
        for domain_name, weight in domain_weights.items():
            columns[f"{phase_name}_{domain_name}"] = float(weight)
    return columns
