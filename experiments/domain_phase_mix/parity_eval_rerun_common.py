# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for 300M parity eval reruns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import fsspec

from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import (
    flatten_eval_results,
)
from experiments.evals.olmo_base_easy_overlap import (
    OLMO_BASE_EASY_OVERLAP_TASKS,
    add_olmo_base_easy_overlap_metrics,
)
from experiments.evals.task_configs import MMLU_SL_VERB_5_SHOT

DEFAULT_PARITY_CHECKPOINT_REGIONS = ("us-east5",)
PARITY_300M_EVAL_TASKS = (*OLMO_BASE_EASY_OVERLAP_TASKS, MMLU_SL_VERB_5_SHOT)


def expected_final_checkpoint_step(num_train_steps: int) -> int:
    """Return the expected final exported checkpoint step."""
    if num_train_steps < 1:
        raise ValueError(f"num_train_steps must be >= 1, got {num_train_steps}")
    return num_train_steps - 1


def resolve_completed_checkpoint_root(
    *,
    source_experiment: str,
    run_name: str,
    num_train_steps: int,
    checkpoint_regions: Sequence[str] = DEFAULT_PARITY_CHECKPOINT_REGIONS,
) -> str | None:
    """Resolve a checkpoint root only when the exact final export exists."""
    expected_step = expected_final_checkpoint_step(num_train_steps)
    matches: list[str] = []

    for region in checkpoint_regions:
        pattern = (
            f"gs://marin-{region}/checkpoints/{source_experiment}/{run_name}-*/"
            f"checkpoints/step-{expected_step}/metadata.json"
        )
        fs, _, _ = fsspec.get_fs_token_paths(pattern)
        matches.extend(match if str(match).startswith("gs://") else f"gs://{match}" for match in fs.glob(pattern))

    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one completed checkpoint root for {run_name}, found {matches}")
    return matches[0].split("/checkpoints/step-", 1)[0]


def flatten_parity_eval_results(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten parity rerun outputs and add derived overlap metrics."""
    flat_metrics = flatten_eval_results(payload)
    flat_metrics.update(add_olmo_base_easy_overlap_metrics(flat_metrics))
    return flat_metrics
