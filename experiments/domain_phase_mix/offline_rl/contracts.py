# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for offline-RL data mixture baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DEFAULT_OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"
DEFAULT_TOTAL_STEPS = 5722
DEFAULT_PHASE_END_STEPS = (1888, 3833)
DEFAULT_ACTION_BOUNDS = (0.05, 0.95)
DEFAULT_ACTION_BINS = 21
DEFAULT_STATE_KEYS = (
    "phase_index",
    "last_train_loss",
    "last_eval_loss",
    "last_obj_bpb",
    "tokens_frac",
    "steps_since_last_eval_frac",
    "prev_action_starcoder",
)
DEFAULT_POLICY_FEATURES_V2 = (
    "decision_index",
    "num_phases_total",
    "remaining_decisions",
    "budget_frac_consumed",
    "budget_frac_remaining",
    "last_train_loss",
    "last_eval_loss",
    "last_obj_bpb",
    "delta_train_loss",
    "delta_eval_loss",
    "delta_obj_bpb",
    "train_eval_gap",
    "global_step",
    "steps_since_last_eval_frac",
    "optim/learning_rate",
    "optim/adam_lr",
    "avg_lr_to_next_boundary",
    "avg_lr_remaining",
    "grad/norm/total",
    "prev_action_starcoder",
    "cumulative_starcoder_exposure",
    "delta_prev_action",
)
DEFAULT_HISTORY_FEATURE_KEYS = (
    "train/loss",
    "eval/loss",
    DEFAULT_OBJECTIVE_METRIC,
    "optim/learning_rate",
    "optim/adam_lr",
    "grad/norm/total",
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


@dataclass(frozen=True)
class ExperimentFamilyConfig:
    """Configuration for one legacy StarCoder mixture experiment family."""

    run_family: str
    display_name_prefix: str
    source_experiments: tuple[str, ...]
    num_phases_total: int
    total_steps: int
    phase_boundaries: tuple[int, ...]
    csv_fallback_path: str | None = None
    expected_finished_runs: int = 0


@dataclass(frozen=True)
class ActionGridConfig:
    """Discrete action grid used by the v2 offline-control baselines."""

    low: float = DEFAULT_ACTION_BOUNDS[0]
    high: float = DEFAULT_ACTION_BOUNDS[1]
    bins: int = DEFAULT_ACTION_BINS

    def values(self) -> tuple[float, ...]:
        if self.bins < 2:
            raise ValueError("Action grid must contain at least 2 bins.")
        step = (self.high - self.low) / float(self.bins - 1)
        return tuple(self.low + step * i for i in range(self.bins))


@dataclass(frozen=True)
class FeatureAuditRow:
    """Coverage statistics for one candidate feature."""

    feature_name: str
    run_family: str
    decision_index: int
    coverage: float
    selected: bool


@dataclass(frozen=True)
class PooledDatasetConfig:
    """Config for the pooled v2 StarCoder offline-control dataset."""

    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    run_families: tuple[ExperimentFamilyConfig, ...] = ()
    candidate_history_keys: tuple[str, ...] = DEFAULT_HISTORY_FEATURE_KEYS
    selected_feature_keys: tuple[str, ...] = DEFAULT_POLICY_FEATURES_V2
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    feature_coverage_threshold: float = 0.95
    n_cv_folds: int = 5


PolicyKindV2 = Literal[
    "sklearn_outcome_planner_v2",
    "sklearn_dynamic_q_planner_v2",
    "sklearn_dynamic_q_planner_v3",
    "torch_discrete_iql_v2",
    "torch_gru_q_v3",
    "torch_transformer_q_v3",
    "d3rlpy_discrete_cql_v2",
    "d3rlpy_discrete_bc_v2",
    "d3rlpy_cql_continuous_v2",
]


DEFAULT_TWO_PHASE_STARCODER_FAMILY = ExperimentFamilyConfig(
    run_family="two_phase_starcoder",
    display_name_prefix="pinlin_calvin_xu/data_mixture/two_phase_starcoder",
    source_experiments=(
        "pinlin_calvin_xu/data_mixture/two_phase_starcoder_4",
        "pinlin_calvin_xu/data_mixture/two_phase_starcoder_5",
    ),
    num_phases_total=2,
    total_steps=3814,
    phase_boundaries=(1904,),
    csv_fallback_path=None,
    expected_finished_runs=143,
)

DEFAULT_THREE_PHASE_STARCODER_FAMILY = ExperimentFamilyConfig(
    run_family="three_phase_starcoder",
    display_name_prefix="pinlin_calvin_xu/data_mixture/three_phase_starcoder",
    source_experiments=(
        "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1",
        "pinlin_calvin_xu/data_mixture/three_phase_starcoder_2",
    ),
    num_phases_total=3,
    total_steps=DEFAULT_TOTAL_STEPS,
    phase_boundaries=DEFAULT_PHASE_END_STEPS,
    csv_fallback_path="experiments/domain_phase_mix/exploratory/three_phase_starcoder.csv",
    expected_finished_runs=160,
)

DEFAULT_STARCODER_FAMILIES = (
    DEFAULT_TWO_PHASE_STARCODER_FAMILY,
    DEFAULT_THREE_PHASE_STARCODER_FAMILY,
)


def default_feature_config(objective_metric: str = DEFAULT_OBJECTIVE_METRIC) -> RLFeatureConfig:
    """Build the default feature configuration for 3-phase StarCoder."""
    return RLFeatureConfig(
        objective_metric=objective_metric,
        total_steps=DEFAULT_TOTAL_STEPS,
        phase_end_steps=DEFAULT_PHASE_END_STEPS,
        action_bounds=DEFAULT_ACTION_BOUNDS,
    )


def default_pooled_dataset_config(objective_metric: str = DEFAULT_OBJECTIVE_METRIC) -> PooledDatasetConfig:
    """Build the default pooled dataset config for offline RL v2."""
    return PooledDatasetConfig(
        objective_metric=objective_metric,
        run_families=DEFAULT_STARCODER_FAMILIES,
    )
