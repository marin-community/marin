# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build pooled decision-time datasets for offline-control StarCoder baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_OBJECTIVE_METRIC,
    FeatureAuditRow,
    PooledDatasetConfig,
    default_pooled_dataset_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildPooledTransitionConfig:
    """Config for building the pooled v2 decision-time dataset."""

    input_dir: str
    output_dir: str
    dataset_config: PooledDatasetConfig = field(default_factory=default_pooled_dataset_config)


def build_action_grid(low: float, high: float, bins: int) -> np.ndarray:
    """Create the discrete StarCoder action grid."""
    if bins < 2:
        raise ValueError("Action grid requires at least 2 bins.")
    return np.linspace(low, high, bins, dtype=np.float32)


def discretize_action(action: float, action_grid: np.ndarray) -> tuple[int, float]:
    """Map a continuous StarCoder weight to the nearest discrete bin."""
    index = int(np.argmin(np.abs(action_grid - float(action))))
    return index, float(action_grid[index])


def _deterministic_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def assign_grouped_folds(runs_df: pd.DataFrame, n_folds: int) -> dict[str, int]:
    """Assign deterministic grouped folds stratified by run family."""
    assignments: dict[str, int] = {}
    for run_family, frame in runs_df.groupby("run_family"):
        run_ids = sorted(frame["wandb_run_id"].astype(str).unique(), key=lambda item: (_deterministic_hash(item), item))
        for index, run_id in enumerate(run_ids):
            assignments[run_id] = index % n_folds
        logger.info("Assigned %d %s runs across %d folds", len(run_ids), run_family, n_folds)
    return assignments


def _last_metric_at_or_before(
    history: pd.DataFrame,
    metric_col: str,
    decision_step: int,
) -> tuple[float | None, int | None]:
    if metric_col not in history.columns:
        return None, None
    metric = history.loc[(history["step"] <= decision_step) & history[metric_col].notna(), ["step", metric_col]]
    if metric.empty:
        return None, None
    row = metric.sort_values("step").iloc[-1]
    return float(row[metric_col]), int(row["step"])


def _last_two_metrics(history: pd.DataFrame, metric_col: str, decision_step: int) -> tuple[float | None, float | None]:
    if metric_col not in history.columns:
        return None, None
    metric = history.loc[
        (history["step"] <= decision_step) & history[metric_col].notna(),
        ["step", metric_col],
    ].sort_values("step")
    if metric.empty:
        return None, None
    latest = float(metric.iloc[-1][metric_col])
    if len(metric) < 2:
        return latest, None
    previous = float(metric.iloc[-2][metric_col])
    return latest, previous


def _mean_metric_between(
    history: pd.DataFrame,
    metric_col: str,
    start_step: int,
    end_step: int,
) -> float | None:
    if metric_col not in history.columns:
        return None
    metric = history.loc[
        (history["step"] > start_step) & (history["step"] <= end_step) & history[metric_col].notna(),
        metric_col,
    ]
    if metric.empty:
        return None
    return float(metric.mean())


def _extract_final_objective(run_row: pd.Series, history: pd.DataFrame, objective_metric: str) -> float | None:
    value = run_row.get(objective_metric)
    if isinstance(value, int | float) and not pd.isna(value):
        return float(value)
    if objective_metric not in history.columns:
        return None
    metric = history[objective_metric].dropna()
    if metric.empty:
        return None
    return float(metric.iloc[-1])


def _phase_lengths(total_steps: int, phase_boundaries: tuple[int, ...]) -> list[int]:
    starts = [0, *phase_boundaries]
    ends = [*phase_boundaries, total_steps]
    return [int(end - start) for start, end in zip(starts, ends, strict=True)]


def _decision_steps(phase_boundaries: tuple[int, ...]) -> list[int]:
    return [0, *phase_boundaries]


def _compute_exposure(actions: list[float], phase_lengths: list[int], decision_index: int) -> tuple[float, float, float]:
    if decision_index <= 0:
        return 0.5, 0.0, 0.0
    consumed = sum(phase_lengths[:decision_index])
    weighted = sum(actions[idx] * phase_lengths[idx] for idx in range(decision_index))
    prev_action = actions[decision_index - 1]
    prev_prev = actions[decision_index - 2] if decision_index > 1 else prev_action
    cumulative = float(weighted / consumed) if consumed > 0 else 0.0
    delta_prev = float(prev_action - prev_prev) if decision_index > 1 else 0.0
    return float(prev_action), cumulative, delta_prev


def _decision_feature_row(
    *,
    run_row: pd.Series,
    history: pd.DataFrame,
    objective_metric: str,
    decision_index: int,
    num_phases_total: int,
    total_steps: int,
    phase_boundaries: tuple[int, ...],
    action_grid: np.ndarray,
) -> dict[str, Any] | None:
    decision_steps = _decision_steps(phase_boundaries)
    phase_lengths = _phase_lengths(total_steps, phase_boundaries)
    decision_step = decision_steps[decision_index]
    next_step = total_steps if decision_index == num_phases_total - 1 else decision_steps[decision_index + 1]

    actions: list[float] = []
    for phase_index in range(num_phases_total):
        value = run_row.get(f"phase_{phase_index}_starcoder")
        if value is None or pd.isna(value):
            return None
        actions.append(float(value))

    final_objective = _extract_final_objective(run_row, history, objective_metric)
    if final_objective is None:
        return None

    prev_action, cumulative_exposure, delta_prev_action = _compute_exposure(actions, phase_lengths, decision_index)

    last_train_loss, _ = _last_metric_at_or_before(history, "train/loss", decision_step)
    prev_train_loss = _last_two_metrics(history, "train/loss", decision_step)[1]
    last_eval_loss, eval_step = _last_metric_at_or_before(history, "eval/loss", decision_step)
    prev_eval_loss = _last_two_metrics(history, "eval/loss", decision_step)[1]
    last_obj_bpb, obj_step = _last_metric_at_or_before(history, objective_metric, decision_step)
    prev_obj_bpb = _last_two_metrics(history, objective_metric, decision_step)[1]
    last_lr, _ = _last_metric_at_or_before(history, "optim/learning_rate", decision_step)
    last_adam_lr, _ = _last_metric_at_or_before(history, "optim/adam_lr", decision_step)
    last_grad_norm, _ = _last_metric_at_or_before(history, "grad/norm/total", decision_step)

    next_obj_bpb, _ = _last_metric_at_or_before(history, objective_metric, next_step)
    if next_obj_bpb is None:
        next_obj_bpb = final_objective

    avg_lr_to_next_boundary = _mean_metric_between(history, "optim/learning_rate", decision_step, next_step)
    avg_lr_remaining = _mean_metric_between(history, "optim/learning_rate", decision_step, total_steps)

    if any(step is not None for step in (eval_step, obj_step)):
        last_eval_like_step = max(step for step in (eval_step, obj_step) if step is not None)
    else:
        last_eval_like_step = None
    steps_since_last_eval_frac = (
        (decision_step - last_eval_like_step) / float(total_steps) if last_eval_like_step is not None else 1.0
    )

    budget_frac_consumed = float(decision_step) / float(total_steps)
    reward_dense_raw = float(last_obj_bpb - next_obj_bpb) if last_obj_bpb is not None else float(-next_obj_bpb)
    action_starcoder = actions[decision_index]
    action_idx, action_grid_value = discretize_action(action_starcoder, action_grid)

    row: dict[str, Any] = {
        "wandb_run_id": str(run_row["wandb_run_id"]),
        "run_family": str(run_row["run_family"]),
        "source_experiment": run_row.get("source_experiment"),
        "local_run_id": run_row.get("local_run_id"),
        "run_name": run_row.get("run_name"),
        "num_phases_total": int(num_phases_total),
        "decision_index": int(decision_index),
        "t": int(decision_index),
        "phase_index": float(decision_index),
        "remaining_decisions": int(num_phases_total - decision_index - 1),
        "decision_step": int(decision_step),
        "next_step": int(next_step),
        "total_steps": int(total_steps),
        "budget_frac_consumed": budget_frac_consumed,
        "budget_frac_remaining": 1.0 - budget_frac_consumed,
        "global_step": float(decision_step),
        "action_starcoder": float(action_starcoder),
        "action_idx": int(action_idx),
        "action_grid_value": float(action_grid_value),
        "prev_action_starcoder": float(prev_action),
        "cumulative_starcoder_exposure": float(cumulative_exposure),
        "delta_prev_action": float(delta_prev_action),
        "done": bool(decision_index == num_phases_total - 1),
        "final_objective": float(final_objective),
        "next_obj_bpb": float(next_obj_bpb),
        "reward_dense_raw": float(reward_dense_raw),
        "reward_terminal_raw": float(-final_objective if decision_index == num_phases_total - 1 else 0.0),
        "phase_boundaries_json": run_row.get("phase_boundaries_json"),
        "tokens_frac": budget_frac_consumed,
        "steps_since_last_eval_frac": float(steps_since_last_eval_frac),
        "last_train_loss": last_train_loss,
        "last_eval_loss": last_eval_loss,
        "last_obj_bpb": last_obj_bpb,
        "delta_train_loss": (
            None if last_train_loss is None or prev_train_loss is None else float(last_train_loss - prev_train_loss)
        ),
        "delta_eval_loss": (
            None if last_eval_loss is None or prev_eval_loss is None else float(last_eval_loss - prev_eval_loss)
        ),
        "delta_obj_bpb": None if last_obj_bpb is None or prev_obj_bpb is None else float(last_obj_bpb - prev_obj_bpb),
        "train_eval_gap": (
            None if last_train_loss is None or last_eval_loss is None else float(last_eval_loss - last_train_loss)
        ),
        "optim/learning_rate": last_lr,
        "optim/adam_lr": last_adam_lr,
        "avg_lr_to_next_boundary": avg_lr_to_next_boundary,
        "avg_lr_remaining": avg_lr_remaining,
        "grad/norm/total": last_grad_norm,
    }
    return row


def _impute_selected_features(
    frame: pd.DataFrame,
    selected_features: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = frame.copy()
    defaults: dict[str, float] = {}
    for feature_name in selected_features:
        if feature_name not in result.columns:
            result[feature_name] = np.nan
        series = result[feature_name].replace([np.inf, -np.inf], np.nan)
        default = float(series.median()) if series.notna().any() else 0.0
        result[feature_name] = series.fillna(default).astype(float)
        defaults[feature_name] = default
    return result, defaults


def build_feature_manifest(
    frame: pd.DataFrame,
    dataset_config: PooledDatasetConfig,
) -> tuple[list[FeatureAuditRow], tuple[str, ...]]:
    """Compute per-family feature coverage and selected policy features."""
    rows: list[FeatureAuditRow] = []
    selected: list[str] = []

    for feature_name in dataset_config.selected_feature_keys:
        min_coverage = 1.0
        for run_family, family_frame in frame.groupby("run_family"):
            for decision_index, decision_frame in family_frame.groupby("decision_index"):
                if feature_name not in decision_frame.columns:
                    coverage = 0.0
                else:
                    coverage = float(decision_frame[feature_name].notna().mean())
                min_coverage = min(min_coverage, coverage)
                rows.append(
                    FeatureAuditRow(
                        feature_name=feature_name,
                        run_family=str(run_family),
                        decision_index=int(decision_index),
                        coverage=coverage,
                        selected=False,
                    )
                )
        if min_coverage >= dataset_config.feature_coverage_threshold:
            selected.append(feature_name)

    final_rows = [
        FeatureAuditRow(
            feature_name=row.feature_name,
            run_family=row.run_family,
            decision_index=row.decision_index,
            coverage=row.coverage,
            selected=row.feature_name in set(selected),
        )
        for row in rows
    ]
    return final_rows, tuple(selected)


def build_pooled_transition_dataset(
    config: BuildPooledTransitionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build pooled decision-time datasets, feature audit, and stage tables."""
    dataset_config = config.dataset_config
    if not dataset_config.run_families:
        dataset_config = default_pooled_dataset_config(dataset_config.objective_metric)
        dataset_config = PooledDatasetConfig(
            objective_metric=dataset_config.objective_metric,
            run_families=dataset_config.run_families,
            candidate_history_keys=dataset_config.candidate_history_keys,
            selected_feature_keys=dataset_config.selected_feature_keys,
            action_grid=dataset_config.action_grid,
            feature_coverage_threshold=dataset_config.feature_coverage_threshold,
            n_cv_folds=config.dataset_config.n_cv_folds,
        )

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_parquet(input_dir / "runs.parquet")
    history_wide_df = pd.read_parquet(input_dir / "history_wide.parquet")
    action_grid = build_action_grid(
        dataset_config.action_grid.low,
        dataset_config.action_grid.high,
        dataset_config.action_grid.bins,
    )
    fold_assignments = assign_grouped_folds(runs_df, dataset_config.n_cv_folds)

    rows: list[dict[str, Any]] = []
    family_config_by_name = {family.run_family: family for family in dataset_config.run_families}
    for _, run_row in runs_df.iterrows():
        run_id = str(run_row["wandb_run_id"])
        family = family_config_by_name[str(run_row["run_family"])]
        history = history_wide_df[history_wide_df["wandb_run_id"] == run_id].copy().sort_values("step")
        for decision_index in range(family.num_phases_total):
            row = _decision_feature_row(
                run_row=run_row,
                history=history,
                objective_metric=dataset_config.objective_metric,
                decision_index=decision_index,
                num_phases_total=family.num_phases_total,
                total_steps=family.total_steps,
                phase_boundaries=family.phase_boundaries,
                action_grid=action_grid,
            )
            if row is None:
                continue
            row["cv_fold"] = int(fold_assignments[run_id])
            rows.append(row)

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        raise ValueError("No pooled decision rows were produced.")

    feature_manifest_rows, selected_features = build_feature_manifest(decisions, dataset_config)
    decisions_imputed, feature_defaults = _impute_selected_features(decisions, selected_features)
    decisions_imputed["reward_dense_raw"] = decisions_imputed["reward_dense_raw"].astype(float)
    decisions_imputed["reward_terminal_raw"] = decisions_imputed["reward_terminal_raw"].astype(float)

    decisions_imputed = decisions_imputed.sort_values(["wandb_run_id", "decision_index"]).reset_index(drop=True)
    decisions_imputed["episode_id"] = decisions_imputed["wandb_run_id"].astype(str)
    decisions_imputed["step_in_episode"] = decisions_imputed["decision_index"].astype(int)

    returns = []
    for _, episode in decisions_imputed.groupby("episode_id", sort=False):
        rewards = episode["reward_dense_raw"].to_numpy(dtype=np.float32)
        rtg = np.flip(np.cumsum(np.flip(rewards)))
        returns.extend(float(value) for value in rtg)
    decisions_imputed["return_to_go_dense_raw"] = returns

    decisions_imputed.to_parquet(output_dir / "decisions.parquet", index=False)
    decisions_imputed.to_parquet(output_dir / "episodes.parquet", index=False)
    for decision_index in sorted(decisions_imputed["decision_index"].unique()):
        decisions_imputed[decisions_imputed["decision_index"] == decision_index].to_parquet(
            output_dir / f"decision_{int(decision_index)}.parquet",
            index=False,
        )

    feature_manifest_payload = {
        "objective_metric": dataset_config.objective_metric,
        "feature_coverage_threshold": dataset_config.feature_coverage_threshold,
        "selected_feature_keys": list(selected_features),
        "candidate_feature_keys": list(dataset_config.selected_feature_keys),
        "rows": [row.__dict__ for row in feature_manifest_rows],
        "feature_defaults": feature_defaults,
    }
    with (output_dir / "feature_manifest.json").open("w") as f:
        json.dump(feature_manifest_payload, f, indent=2, sort_keys=True)

    manifest = {
        "objective_metric": dataset_config.objective_metric,
        "n_runs": int(decisions_imputed["wandb_run_id"].nunique()),
        "n_decision_rows": len(decisions_imputed),
        "action_grid": [float(value) for value in action_grid.tolist()],
        "selected_feature_keys": list(selected_features),
        "n_cv_folds": int(dataset_config.n_cv_folds),
        "run_family_counts": {
            key: int(value) for key, value in decisions_imputed.groupby("run_family")["wandb_run_id"].nunique().items()
        },
        "decision_index_counts": {
            str(int(key)): int(value)
            for key, value in decisions_imputed.groupby("decision_index")["wandb_run_id"].nunique().items()
        },
    }
    with (output_dir / "dataset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return decisions_imputed, pd.DataFrame([row.__dict__ for row in feature_manifest_rows]), manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled decision-time StarCoder offline-control datasets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_pooled_v2",
        help="Directory containing pooled runs.parquet and history_wide.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_pooled_v2",
        help="Directory to write pooled decision tables.",
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=DEFAULT_OBJECTIVE_METRIC,
        help="Objective metric key.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    build_pooled_transition_dataset(
        BuildPooledTransitionConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_config=default_pooled_dataset_config(args.objective_metric),
        )
    )


if __name__ == "__main__":
    main()
