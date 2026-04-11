# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train three-phase-target offline-control policies with pooled auxiliary runs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump

from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC
from experiments.domain_phase_mix.offline_rl.ope import (
    DecisionBatch,
    fit_behavior_policy,
    fit_reward_models,
    predict_action_indices_for_batch,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import save_policy_artifact
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v3 import (
    FixedSchedulePolicy,
    SEQUENCE_CHANNELS,
    ThreePhasePolicyBenchV3Config,
    _canonical_initial_batch,
    _decision_state_defaults,
    _evaluate_policy,
    _fit_discrete_bc_policy,
    _load_decision_batch,
    _phase0_top_decile_limit,
    _policy_artifact,
    _train_dynamic_q_planner,
    _train_sequence_policy,
    compute_state_stats,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreePhasePolicyBenchV4Config(ThreePhasePolicyBenchV3Config):
    """Configuration for pooled-auxiliary v4 three-phase policy selection."""


def _batch_from_positions(batch: DecisionBatch, positions: np.ndarray) -> DecisionBatch:
    return batch.take(np.asarray(positions, dtype=np.int64))


def _mask_positions(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _filter_pretrain_payload(
    pretrain_payload: dict[str, np.ndarray], allowed_run_ids: set[str]
) -> dict[str, np.ndarray]:
    if not pretrain_payload:
        return {}
    run_ids = np.asarray(pretrain_payload["wandb_run_id"]).astype(str)
    keep = np.isin(run_ids, sorted(allowed_run_ids))
    return {key: value[keep] for key, value in pretrain_payload.items()}


def _rename_policy(policy, name: str):
    policy.name = name
    return policy


def run_three_phase_policy_bench_v4(config: ThreePhasePolicyBenchV4Config) -> dict[str, Any]:
    """Run pooled-training / three-phase-eval grouped CV for v4."""
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions, full_batch, pretrain_payload, feature_manifest, dataset_manifest = _load_decision_batch(input_dir)
    feature_keys = tuple(feature_manifest["selected_feature_keys"])
    action_grid = np.asarray(dataset_manifest["action_grid"], dtype=np.float32)
    n_folds = int(dataset_manifest["n_cv_folds"])

    three_phase_mask_all = decisions["run_family"].astype(str).eq("three_phase_starcoder").to_numpy(dtype=bool)
    two_phase_mask_all = decisions["run_family"].astype(str).eq("two_phase_starcoder").to_numpy(dtype=bool)
    if not three_phase_mask_all.any():
        raise ValueError("v4 requires three-phase rows in decisions.parquet")

    fold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[pd.DataFrame] = []
    canonical_rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        three_phase_train_mask = three_phase_mask_all & (decisions["cv_fold"].to_numpy(dtype=np.int64) != fold)
        three_phase_val_mask = three_phase_mask_all & (decisions["cv_fold"].to_numpy(dtype=np.int64) == fold)
        pooled_train_mask = three_phase_train_mask | two_phase_mask_all

        train_frame = decisions.loc[pooled_train_mask].copy().reset_index(drop=True)
        three_phase_train_frame = decisions.loc[three_phase_train_mask].copy().reset_index(drop=True)
        val_frame = decisions.loc[three_phase_val_mask].copy().reset_index(drop=True)
        if train_frame.empty or three_phase_train_frame.empty or val_frame.empty:
            continue

        pooled_train_batch = _batch_from_positions(full_batch, _mask_positions(pooled_train_mask))
        three_phase_train_batch = _batch_from_positions(full_batch, _mask_positions(three_phase_train_mask))
        val_batch = _batch_from_positions(full_batch, _mask_positions(three_phase_val_mask))
        pooled_train_run_ids = set(train_frame["wandb_run_id"].astype(str).tolist())
        filtered_pretrain_payload = _filter_pretrain_payload(pretrain_payload, pooled_train_run_ids)

        state_stats = compute_state_stats(train_frame, feature_keys)
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
        reward_models = fit_reward_models(
            three_phase_train_frame,
            feature_keys=feature_keys,
            random_state=config.random_state,
            final_model_weight=1.0,
            reward_bonus_weight=0.0,
        )

        policies = [
            FixedSchedulePolicy(action_grid, _best_fixed_schedule_three_phase(three_phase_train_frame)),
            _fit_discrete_bc_policy(three_phase_train_frame, feature_keys, config.random_state),
            _rename_policy(
                _train_dynamic_q_planner(train_frame, feature_keys, action_grid, pooled_behavior, config),
                "dynamic_q_planner_v4_pooled",
            ),
            _rename_policy(
                _train_sequence_policy(
                    name="gru_q_v3",
                    train_batch=pooled_train_batch,
                    pretrain_payload=filtered_pretrain_payload,
                    feature_keys=feature_keys,
                    state_stats=state_stats,
                    action_grid=action_grid,
                    behavior_policy=pooled_behavior,
                    config=config,
                ),
                "gru_q_v4_pooled",
            ),
            _rename_policy(
                _train_sequence_policy(
                    name="transformer_q_v3",
                    train_batch=pooled_train_batch,
                    pretrain_payload=filtered_pretrain_payload,
                    feature_keys=feature_keys,
                    state_stats=state_stats,
                    action_grid=action_grid,
                    behavior_policy=pooled_behavior,
                    config=config,
                ),
                "transformer_q_v4_pooled",
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
                reward_models=reward_models,
                action_grid=action_grid,
                config=config,
            )
            action_idx = int(predict_action_indices_for_batch(policy, canonical_batch)[0])
            canonical_action = float(action_grid[action_idx])
            metrics.update(
                {
                    "fold": fold,
                    "canonical_phase0_action": canonical_action,
                    "canonical_phase0_passed": canonical_action <= phase0_limit,
                }
            )
            fold_rows.append(metrics)
            diagnostics_rows.append(diagnostics.assign(fold=fold))
            canonical_rows.append(
                {
                    "fold": fold,
                    "method": policy.name,
                    "canonical_phase0_action": canonical_action,
                    "phase0_limit": phase0_limit,
                    "passed": canonical_action <= phase0_limit,
                }
            )

    scores_df = pd.DataFrame(fold_rows)
    if scores_df.empty:
        raise ValueError("No v4 fold scores were produced.")
    diagnostics_df = pd.concat(diagnostics_rows, ignore_index=True) if diagnostics_rows else pd.DataFrame()
    canonical_df = pd.DataFrame(canonical_rows)
    scores_df.to_csv(output_dir / "fold_scores.csv", index=False)
    diagnostics_df.to_csv(output_dir / "policy_diagnostics.csv", index=False)
    canonical_df.to_csv(output_dir / "canonical_phase0_checks.csv", index=False)

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
        .sort_values("dr_value_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    baseline_bc = scores_df[scores_df["method"] == "discrete_bc"].set_index("fold")
    baseline_fixed = scores_df[scores_df["method"] == "fixed_best_schedule"].set_index("fold")
    candidate_methods = [
        method for method in summary_df["method"].tolist() if method not in {"fixed_best_schedule", "discrete_bc"}
    ]
    gate_rows: list[dict[str, Any]] = []
    passing_methods: list[str] = []
    for method in candidate_methods:
        method_scores = scores_df[scores_df["method"] == method].set_index("fold")
        common_folds = sorted(set(method_scores.index) & set(baseline_bc.index) & set(baseline_fixed.index))
        beat_folds = 0
        for fold in common_folds:
            row = method_scores.loc[fold]
            bc_row = baseline_bc.loc[fold]
            fixed_row = baseline_fixed.loc[fold]
            if (
                row["fqe_value"] > bc_row["fqe_value"]
                and row["fqe_value"] > fixed_row["fqe_value"]
                and row["dr_value"] > bc_row["dr_value"]
                and row["dr_value"] > fixed_row["dr_value"]
            ):
                beat_folds += 1
        unsupported_rate = float(method_scores["unsupported_rate"].mean())
        boundary_rate = float(method_scores["boundary_rate"].mean())
        canonical_pass_rate = float(method_scores["canonical_phase0_passed"].mean())
        passed = (
            beat_folds >= 3
            and unsupported_rate <= config.unsupported_rate_threshold
            and boundary_rate <= config.boundary_rate_threshold
            and canonical_pass_rate >= 1.0
        )
        gate_rows.append(
            {
                "method": method,
                "beat_baselines_folds": beat_folds,
                "unsupported_rate_mean": unsupported_rate,
                "boundary_rate_mean": boundary_rate,
                "canonical_phase0_pass_rate": canonical_pass_rate,
                "passed": passed,
            }
        )
        if passed:
            passing_methods.append(method)
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(output_dir / "rollout_gate_report.csv", index=False)

    full_three_phase = decisions[three_phase_mask_all].copy().reset_index(drop=True)
    full_pooled = decisions.copy().reset_index(drop=True)
    full_state_stats = compute_state_stats(full_pooled, feature_keys)
    full_pooled_behavior = fit_behavior_policy(
        full_pooled,
        feature_keys=feature_keys,
        action_count=len(action_grid),
        random_state=config.random_state,
    )

    full_dynamic_q = _rename_policy(
        _train_dynamic_q_planner(full_pooled, feature_keys, action_grid, full_pooled_behavior, config),
        "dynamic_q_planner_v4_pooled",
    )
    full_bc = _fit_discrete_bc_policy(full_three_phase, feature_keys, config.random_state)
    full_gru = _rename_policy(
        _train_sequence_policy(
            name="gru_q_v3",
            train_batch=full_batch,
            pretrain_payload=pretrain_payload,
            feature_keys=feature_keys,
            state_stats=full_state_stats,
            action_grid=action_grid,
            behavior_policy=full_pooled_behavior,
            config=config,
        ),
        "gru_q_v4_pooled",
    )
    full_transformer = _rename_policy(
        _train_sequence_policy(
            name="transformer_q_v3",
            train_batch=full_batch,
            pretrain_payload=pretrain_payload,
            feature_keys=feature_keys,
            state_stats=full_state_stats,
            action_grid=action_grid,
            behavior_policy=full_pooled_behavior,
            config=config,
        ),
        "transformer_q_v4_pooled",
    )

    artifacts_dir = output_dir / "artifacts"
    full_models_dir = output_dir / "full_models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    full_models_dir.mkdir(parents=True, exist_ok=True)

    decision_defaults_path = artifacts_dir / "decision_state_defaults.json"
    with decision_defaults_path.open("w") as f:
        json.dump(_decision_state_defaults(full_three_phase, feature_keys), f, indent=2, sort_keys=True)
    feature_manifest_copy = artifacts_dir / "feature_manifest.json"
    dataset_manifest_copy = artifacts_dir / "dataset_manifest.json"
    feature_manifest_copy.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True))
    dataset_manifest_copy.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True))

    dynamic_q_bundle_path = full_models_dir / "dynamic_q_planner_v4_pooled" / "planner.joblib"
    dynamic_q_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "feature_keys": feature_keys,
            "action_grid": action_grid.tolist(),
            "q_models": full_dynamic_q.q_models,
            "behavior_policy": full_pooled_behavior,
            "support_lambda": config.support_lambda,
            "support_floor": config.support_floor,
            "training_regime": "pooled_aux_v4",
        },
        dynamic_q_bundle_path,
    )

    bc_model_dir = full_models_dir / "discrete_bc"
    bc_model_dir.mkdir(parents=True, exist_ok=True)
    bc_bundle_path = bc_model_dir / "model.joblib"
    dump({"feature_keys": feature_keys, "estimator": full_bc.estimator}, bc_bundle_path)

    support_bundle_path = artifacts_dir / "support_bundle.joblib"
    dump(
        {
            "behavior_policy": full_pooled_behavior,
            "support_lambda": config.support_lambda,
            "support_floor": config.support_floor,
            "training_regime": "pooled_aux_v4",
        },
        support_bundle_path,
    )

    gru_model_path = full_gru.save(
        full_models_dir / "gru_q_v4_pooled",
        metadata={"training_regime": "pooled_aux_v4", "sequence_channels": list(SEQUENCE_CHANNELS)},
    )
    transformer_model_path = full_transformer.save(
        full_models_dir / "transformer_q_v4_pooled",
        metadata={"training_regime": "pooled_aux_v4", "sequence_channels": list(SEQUENCE_CHANNELS)},
    )

    artifact_map = {
        "dynamic_q_planner_v4_pooled": _policy_artifact(
            kind="sklearn_dynamic_q_planner_v3",
            model_path=dynamic_q_bundle_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={"training_regime": "pooled_aux_v4"},
            aux_paths={
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
        "discrete_bc": _policy_artifact(
            kind="d3rlpy_discrete_bc_v2",
            model_path=bc_bundle_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={"training_regime": "three_phase_baseline"},
            aux_paths={
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
        "gru_q_v4_pooled": _policy_artifact(
            kind="torch_gru_q_v3",
            model_path=gru_model_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={"training_regime": "pooled_aux_v4"},
            aux_paths={
                "support_bundle": str(support_bundle_path.resolve()),
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
        "transformer_q_v4_pooled": _policy_artifact(
            kind="torch_transformer_q_v3",
            model_path=transformer_model_path,
            feature_keys=feature_keys,
            action_grid=action_grid,
            state_stats=full_state_stats,
            objective_metric=config.objective_metric,
            metadata={"training_regime": "pooled_aux_v4"},
            aux_paths={
                "support_bundle": str(support_bundle_path.resolve()),
                "decision_state_defaults": str(decision_defaults_path.resolve()),
                "feature_manifest": str(feature_manifest_copy.resolve()),
                "dataset_manifest": str(dataset_manifest_copy.resolve()),
            },
        ),
    }
    for method, artifact in artifact_map.items():
        save_policy_artifact(artifacts_dir / f"{method}.json", artifact)

    passing_artifacts = {
        method: str((artifacts_dir / f"{method}.json").resolve()) for method in passing_methods if method in artifact_map
    }
    with (output_dir / "passing_policy_artifacts.json").open("w") as f:
        json.dump(passing_artifacts, f, indent=2, sort_keys=True)

    summary = {
        "passing_methods": passing_methods,
        "feature_keys": list(feature_keys),
        "artifacts": {method: str((artifacts_dir / f"{method}.json").resolve()) for method in artifact_map},
    }
    with (output_dir / "offline_policy_report.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _best_fixed_schedule_three_phase(train_frame: pd.DataFrame) -> dict[int, float]:
    initial = train_frame[train_frame["decision_index"] == 0].copy()
    best_run_id = str(initial.sort_values("final_objective").iloc[0]["wandb_run_id"])
    run_frame = train_frame[train_frame["wandb_run_id"] == best_run_id].sort_values("decision_index")
    return {int(row.decision_index): float(row.action_starcoder) for row in run_frame.itertuples()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v4 pooled-auxiliary three-phase policies.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v4_pooled",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v4_pooled/policy_bench_v4",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_three_phase_policy_bench_v4(
        ThreePhasePolicyBenchV4Config(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
