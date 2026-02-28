# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for offline-RL data mixture baselines."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"
DEFAULT_TOTAL_STEPS = 5722
DEFAULT_PHASE_END_STEPS = (1888, 3833)
DEFAULT_ACTION_BOUNDS = (0.05, 0.95)
DEFAULT_STATE_KEYS = (
    "phase_index",
    "last_train_loss",
    "last_eval_loss",
    "last_obj_bpb",
    "tokens_frac",
    "steps_since_last_eval_frac",
    "prev_action_starcoder",
)


@dataclass(frozen=True)
class RLFeatureConfig:
    """Configuration for constructing offline-RL transitions."""

    objective_metric: str
    total_steps: int
    phase_end_steps: tuple[int, int]
    action_bounds: tuple[float, float]


@dataclass(frozen=True)
class TransitionRow:
    """One decision-time transition used for offline RL."""

    wandb_run_id: str
    t: int
    state: dict[str, float]
    action_starcoder: float
    reward_raw: float
    reward_std: float
    done: bool


def default_feature_config(objective_metric: str = DEFAULT_OBJECTIVE_METRIC) -> RLFeatureConfig:
    """Build the default feature configuration for 3-phase StarCoder."""
    return RLFeatureConfig(
        objective_metric=objective_metric,
        total_steps=DEFAULT_TOTAL_STEPS,
        phase_end_steps=DEFAULT_PHASE_END_STEPS,
        action_bounds=DEFAULT_ACTION_BOUNDS,
    )
