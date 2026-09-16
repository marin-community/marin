# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and compare pooled-auxiliary direct planners against the legacy outcome planner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC
from experiments.domain_phase_mix.offline_rl.ope import (
    ActionPolicy,
    BehaviorPolicyModel,
    DecisionBatch,
    RewardModelBundle,
    fit_behavior_policy,
    fit_reward_models,
)
from experiments.domain_phase_mix.offline_rl.train_offline_policy_bench import (
    OfflinePolicyBenchConfig,
    train_outcome_planner,
)
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v3 import (
    FixedSchedulePolicy,
    ThreePhasePolicyBenchV3Config,
    _canonical_initial_batch,
    _evaluate_policy,
    _fit_discrete_bc_policy,
    _load_decision_batch,
    _phase0_top_decile_limit,
    _train_dynamic_q_planner,
)
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v4 import (
    _best_fixed_schedule_three_phase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreePhasePolicyBenchV5Config:
    """Configuration for the pooled-auxiliary v5 benchmark."""

    input_dir: str
    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    random_state: int = 42
    gamma: float = 1.0
    diagnostic_support_threshold: float = 0.02
    boundary_rate_threshold: float = 0.25
    unsupported_rate_threshold: float = 0.15
    fqe_iterations: int = 8
    q_max_iter: int = 96
    dense_direct_reward_bonus_weight: float = 0.08
    dense_direct_support_lambda: float = 0.02
    hybrid_support_lambda: float = 0.05
    support_floor: float = 1e-4
    hybrid_direct_alpha: float = 2.0


class DenseDirectPlannerV5(ActionPolicy):
    """Direct planner using dense reward/final objective models plus soft support regularization."""

    def __init__(
        self,
        *,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        reward_models: RewardModelBundle,
        behavior_policy: BehaviorPolicyModel,
        support_lambda: float,
        support_floor: float,
        name: str = "dense_direct_v5",
    ):
        self.name = name
        self.feature_keys = feature_keys
        self.action_grid = np.asarray(action_grid, dtype=np.float32)
        self.reward_models = reward_models
        self.behavior_policy = behavior_policy
        self.support_lambda = support_lambda
        self.support_floor = support_floor

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        outputs = np.zeros(len(frame), dtype=np.int64)
        for _, stage_frame in frame.groupby("decision_index", sort=False):
            stage_positions = stage_frame.index.to_numpy(dtype=np.int64)
            repeated, support = _candidate_support_grid(
                stage_frame=stage_frame,
                feature_keys=self.feature_keys,
                action_grid=self.action_grid,
                behavior_policy=self.behavior_policy,
            )
            candidate_actions = np.tile(self.action_grid, len(stage_frame)).astype(np.float32)
            direct_utility = -self.reward_models.blended_stage_scores(repeated, candidate_actions).reshape(
                len(stage_frame), len(self.action_grid)
            )
            utility = direct_utility + self.support_lambda * np.log(np.maximum(support, self.support_floor))
            utility = np.where(support >= self.support_floor, utility, -np.inf)
            outputs[stage_positions] = _best_action_indices(utility, support)
        return outputs


class HybridQDirectPlannerV5(ActionPolicy):
    """Hybrid planner combining dynamic-Q values with direct-model utility."""

    def __init__(
        self,
        *,
        feature_keys: tuple[str, ...],
        action_grid: np.ndarray,
        q_models: dict[int, Any],
        reward_models: RewardModelBundle,
        behavior_policy: BehaviorPolicyModel,
        support_lambda: float,
        support_floor: float,
        direct_alpha: float,
        name: str = "hybrid_q_direct_v5",
    ):
        self.name = name
        self.feature_keys = feature_keys
        self.action_grid = np.asarray(action_grid, dtype=np.float32)
        self.q_models = q_models
        self.reward_models = reward_models
        self.behavior_policy = behavior_policy
        self.support_lambda = support_lambda
        self.support_floor = support_floor
        self.direct_alpha = direct_alpha

    def predict_action_indices(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        outputs = np.zeros(len(frame), dtype=np.int64)
        for decision_index, stage_frame in frame.groupby("decision_index", sort=False):
            stage_positions = stage_frame.index.to_numpy(dtype=np.int64)
            repeated, support = _candidate_support_grid(
                stage_frame=stage_frame,
                feature_keys=self.feature_keys,
                action_grid=self.action_grid,
                behavior_policy=self.behavior_policy,
            )
            candidate_actions = np.tile(self.action_grid, len(stage_frame)).astype(np.float32)
            q_values = (
                self.q_models[int(decision_index)]
                .predict(
                    np.concatenate(
                        [
                            repeated.loc[:, list(self.feature_keys)].to_numpy(dtype=np.float32),
                            candidate_actions.reshape(-1, 1),
                        ],
                        axis=1,
                    )
                )
                .astype(np.float32)
                .reshape(len(stage_frame), len(self.action_grid))
            )
            direct_utility = -self.reward_models.blended_stage_scores(repeated, candidate_actions).reshape(
                len(stage_frame), len(self.action_grid)
            )
            utility = q_values + self.direct_alpha * direct_utility
            utility += self.support_lambda * np.log(np.maximum(support, self.support_floor))
            utility = np.where(support >= self.support_floor, utility, -np.inf)
            outputs[stage_positions] = _best_action_indices(utility, support)
        return outputs


def _candidate_support_grid(
    *,
    stage_frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    behavior_policy: BehaviorPolicyModel,
) -> tuple[pd.DataFrame, np.ndarray]:
    repeated_inputs = np.repeat(
        stage_frame.loc[:, list(feature_keys)].to_numpy(dtype=np.float32),
        len(action_grid),
        axis=0,
    )
    repeated = pd.DataFrame(repeated_inputs, columns=list(feature_keys))
    candidate_indices = np.tile(np.arange(len(action_grid), dtype=np.int64), len(stage_frame))
    support_matrix = behavior_policy.predict_proba(repeated)
    support = support_matrix[np.arange(len(repeated)), candidate_indices].reshape(len(stage_frame), len(action_grid))
    return repeated, support


def _best_action_indices(utility: np.ndarray, support: np.ndarray) -> np.ndarray:
    choices = np.zeros(utility.shape[0], dtype=np.int64)
    for local_index in range(utility.shape[0]):
        row_utility = utility[local_index]
        if np.isfinite(row_utility).any():
            choices[local_index] = int(np.argmax(row_utility))
        else:
            choices[local_index] = int(np.argmax(support[local_index]))
    return choices


def _pooled_train_masks(decisions: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    three_phase_mask_all = decisions["run_family"].astype(str).eq("three_phase_starcoder").to_numpy(dtype=bool)
    two_phase_mask_all = decisions["run_family"].astype(str).eq("two_phase_starcoder").to_numpy(dtype=bool)
    fold_ids = decisions["cv_fold"].to_numpy(dtype=np.int64)
    three_phase_train_mask = three_phase_mask_all & (fold_ids != fold)
    three_phase_val_mask = three_phase_mask_all & (fold_ids == fold)
    pooled_train_mask = three_phase_train_mask | two_phase_mask_all
    return pooled_train_mask, three_phase_train_mask, three_phase_val_mask


def _batch_from_mask(batch: DecisionBatch, mask: np.ndarray) -> DecisionBatch:
    return batch.take(np.flatnonzero(np.asarray(mask, dtype=bool)))


def _legacy_outcome_policy(
    train_frame: pd.DataFrame,
    feature_keys: tuple[str, ...],
    action_grid: np.ndarray,
    behavior_policy: BehaviorPolicyModel,
    config: ThreePhasePolicyBenchV5Config,
) -> ActionPolicy:
    policy = train_outcome_planner(
        train_frame,
        feature_keys,
        action_grid,
        behavior_policy,
        OfflinePolicyBenchConfig(
            input_dir=".",
            output_dir=".",
            random_state=config.random_state,
            gamma=config.gamma,
            support_threshold=config.diagnostic_support_threshold,
        ),
    )
    policy.name = "legacy_outcome_planner"
    return policy


def run_three_phase_policy_bench_v5(config: ThreePhasePolicyBenchV5Config) -> dict[str, Any]:
    """Run grouped CV for dense-direct v5 candidates against the legacy outcome planner."""
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions, full_batch, _pretrain_payload, feature_manifest, dataset_manifest = _load_decision_batch(input_dir)
    feature_keys = tuple(feature_manifest["selected_feature_keys"])
    action_grid = np.asarray(dataset_manifest["action_grid"], dtype=np.float32)
    n_folds = int(dataset_manifest["n_cv_folds"])

    fold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[pd.DataFrame] = []

    eval_config = ThreePhasePolicyBenchV3Config(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        objective_metric=config.objective_metric,
        random_state=config.random_state,
        gamma=config.gamma,
        diagnostic_support_threshold=config.diagnostic_support_threshold,
        fqe_iterations=config.fqe_iterations,
        q_max_iter=config.q_max_iter,
    )

    for fold in range(n_folds):
        pooled_train_mask, three_phase_train_mask, three_phase_val_mask = _pooled_train_masks(decisions, fold)
        train_frame = decisions.loc[pooled_train_mask].copy().reset_index(drop=True)
        three_phase_train_frame = decisions.loc[three_phase_train_mask].copy().reset_index(drop=True)
        if train_frame.empty or three_phase_train_frame.empty:
            continue

        three_phase_train_batch = _batch_from_mask(full_batch, three_phase_train_mask)
        val_batch = _batch_from_mask(full_batch, three_phase_val_mask)
        pooled_behavior = fit_behavior_policy(
            train_frame,
            feature_keys=feature_keys,
            action_count=len(action_grid),
            random_state=config.random_state,
        )
        eval_behavior = fit_behavior_policy(
            three_phase_train_frame,
            feature_keys=feature_keys,
            action_count=len(action_grid),
            random_state=config.random_state,
        )
        reward_models_eval = fit_reward_models(
            three_phase_train_frame,
            feature_keys=feature_keys,
            random_state=config.random_state,
            final_model_weight=1.0,
            reward_bonus_weight=0.0,
        )
        reward_models_policy = fit_reward_models(
            train_frame,
            feature_keys=feature_keys,
            random_state=config.random_state,
            final_model_weight=1.0,
            reward_bonus_weight=config.dense_direct_reward_bonus_weight,
        )
        dynamic_q = _train_dynamic_q_planner(train_frame, feature_keys, action_grid, pooled_behavior, eval_config)

        policies: list[ActionPolicy] = [
            _legacy_outcome_policy(train_frame, feature_keys, action_grid, pooled_behavior, config),
            FixedSchedulePolicy(action_grid, _best_fixed_schedule_three_phase(three_phase_train_frame)),
            _fit_discrete_bc_policy(three_phase_train_frame, feature_keys, config.random_state),
            DenseDirectPlannerV5(
                feature_keys=feature_keys,
                action_grid=action_grid,
                reward_models=reward_models_policy,
                behavior_policy=pooled_behavior,
                support_lambda=config.dense_direct_support_lambda,
                support_floor=config.support_floor,
            ),
            HybridQDirectPlannerV5(
                feature_keys=feature_keys,
                action_grid=action_grid,
                q_models=dynamic_q.q_models,
                reward_models=reward_models_policy,
                behavior_policy=pooled_behavior,
                support_lambda=config.hybrid_support_lambda,
                support_floor=config.support_floor,
                direct_alpha=config.hybrid_direct_alpha,
            ),
        ]

        canonical_batch = _canonical_initial_batch(three_phase_train_batch, feature_keys)
        phase0_limit = _phase0_top_decile_limit(three_phase_train_frame, action_grid)
        for policy in policies:
            metrics, diagnostics = _evaluate_policy(
                policy=policy,
                train_batch=three_phase_train_batch,
                val_batch=val_batch,
                feature_keys=feature_keys,
                behavior_policy=eval_behavior,
                reward_models=reward_models_eval,
                action_grid=action_grid,
                config=eval_config,
            )
            action_idx = int(policy.predict_action_indices(canonical_batch.frame)[0])
            metrics.update(
                {
                    "fold": fold,
                    "canonical_phase0_action": float(action_grid[action_idx]),
                    "canonical_phase0_passed": float(action_grid[action_idx]) <= phase0_limit,
                }
            )
            fold_rows.append(metrics)
            diagnostics_rows.append(diagnostics.assign(fold=fold))
            logger.info(
                "fold=%s method=%s fqe=%.4f dr=%.4f phase0=%.2f",
                fold,
                policy.name,
                metrics["fqe_value"],
                metrics["dr_value"],
                metrics["canonical_phase0_action"],
            )

    scores_df = pd.DataFrame(fold_rows)
    if scores_df.empty:
        raise ValueError("No v5 fold scores were produced.")
    diagnostics_df = pd.concat(diagnostics_rows, ignore_index=True) if diagnostics_rows else pd.DataFrame()
    scores_df.to_csv(output_dir / "fold_scores.csv", index=False)
    diagnostics_df.to_csv(output_dir / "policy_diagnostics.csv", index=False)

    summary_df = (
        scores_df.groupby("method", as_index=False)
        .agg(
            direct_value_mean=("direct_value", "mean"),
            fqe_value_mean=("fqe_value", "mean"),
            dr_value_mean=("dr_value", "mean"),
            wis_value_mean=("wis_value", "mean"),
            unsupported_rate_mean=("unsupported_rate", "mean"),
            boundary_rate_mean=("boundary_rate", "mean"),
            mean_support_mean=("mean_support", "mean"),
            canonical_phase0_action_mean=("canonical_phase0_action", "mean"),
            canonical_phase0_pass_rate=("canonical_phase0_passed", "mean"),
        )
        .sort_values(["fqe_value_mean", "dr_value_mean"], ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    baseline_bc = scores_df[scores_df["method"] == "discrete_bc"].set_index("fold")
    baseline_fixed = scores_df[scores_df["method"] == "fixed_best_schedule"].set_index("fold")
    baseline_legacy = scores_df[scores_df["method"] == "legacy_outcome_planner"].set_index("fold")
    candidate_methods = ["dense_direct_v5", "hybrid_q_direct_v5"]
    comparison_rows: list[dict[str, Any]] = []
    for method in candidate_methods:
        method_scores = scores_df[scores_df["method"] == method].set_index("fold")
        common_folds = sorted(set(method_scores.index) & set(baseline_bc.index) & set(baseline_fixed.index))
        beats_fixed_and_bc = 0
        beats_legacy = 0
        beats_legacy_both = 0
        for fold in common_folds:
            row = method_scores.loc[fold]
            bc_row = baseline_bc.loc[fold]
            fixed_row = baseline_fixed.loc[fold]
            legacy_row = baseline_legacy.loc[fold]
            if (
                row["fqe_value"] > bc_row["fqe_value"]
                and row["fqe_value"] > fixed_row["fqe_value"]
                and row["dr_value"] > bc_row["dr_value"]
                and row["dr_value"] > fixed_row["dr_value"]
            ):
                beats_fixed_and_bc += 1
            if row["fqe_value"] > legacy_row["fqe_value"]:
                beats_legacy += 1
            if row["fqe_value"] > legacy_row["fqe_value"] and row["dr_value"] > legacy_row["dr_value"]:
                beats_legacy_both += 1
        method_scores_frame = scores_df[scores_df["method"] == method]
        comparison_rows.append(
            {
                "method": method,
                "beat_fixed_and_bc_folds": beats_fixed_and_bc,
                "beat_legacy_fqe_folds": beats_legacy,
                "beat_legacy_fqe_and_dr_folds": beats_legacy_both,
                "unsupported_rate_mean": float(method_scores_frame["unsupported_rate"].mean()),
                "boundary_rate_mean": float(method_scores_frame["boundary_rate"].mean()),
                "canonical_phase0_pass_rate": float(method_scores_frame["canonical_phase0_passed"].mean()),
                "passed_rollout_gate": bool(
                    beats_fixed_and_bc >= 3
                    and float(method_scores_frame["unsupported_rate"].mean()) <= config.unsupported_rate_threshold
                    and float(method_scores_frame["boundary_rate"].mean()) <= config.boundary_rate_threshold
                    and float(method_scores_frame["canonical_phase0_passed"].mean()) >= 1.0
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "comparison_report.csv", index=False)

    best_method = str(summary_df.iloc[0]["method"])
    report = {
        "feature_keys": list(feature_keys),
        "best_method_by_summary": best_method,
        "candidate_methods": candidate_methods,
        "rollout_eligible_methods": (
            comparison_df.loc[comparison_df["passed_rollout_gate"].astype(bool), "method"].tolist()
        ),
    }
    with (output_dir / "offline_policy_report.json").open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v5 dense-direct three-phase policies.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v4_pooled",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v5_pooled/policy_bench_v5",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_three_phase_policy_bench_v5(
        ThreePhasePolicyBenchV5Config(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
        )
    )


if __name__ == "__main__":
    main()
