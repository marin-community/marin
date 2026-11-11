# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive policies for unbiased trace retention and checkpoint selection."""


def trace_retention(output: str, run_id: str, *, fraction: float, phase: str) -> dict:
    return {
        "enabled": True,
        "output_path": output,
        "run_id": run_id,
        "phases": [phase, "eval"] if phase == "train" else [phase],
        "sample_fraction": fraction,
        "eval_sample_fraction": 1.0,
        "sample_count_per_step": 0,
        "always_retain_failures": False,
        "always_retain_non_terminating": False,
        "always_retain_loops": False,
        "reward_below": None,
        "reward_above": None,
        "max_bytes_per_step": None,
        "max_bytes_per_run": None,
        "required": True,
    }


def validation_selection(*, initial_model: str, task_count: int) -> dict:
    """Select a complete five-observation EMA at the two-update validation cadence."""
    return {
        "eval_interval": 2,
        "eval_before_train": False,
        "hf_export_checkpoint": "best",
        "callbacks": [
            {
                "type": "best_checkpoint",
                "initial_model_uri": initial_model,
                "expected_tasks": task_count,
                "minimum_scored_tasks": task_count,
                "ema_window": 5,
                "evaluation_interval": 2,
            }
        ],
    }
