# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build pooled dense policy datasets for three-phase-target offline-control v4."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.offline_rl.build_three_phase_dense_policy_dataset import (
    SEQUENCE_CHANNELS,
    SUMMARY_FEATURES,
    _build_pretrain_windows,
    _decision_row,
    _impute_selected_features,
    _prepare_run_history_cache,
    assign_grouped_folds,
    build_action_grid,
    build_decision_state_features,
)
from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildPooledDenseDatasetV4Config:
    """Config for building a pooled dense policy dataset."""

    input_dir: str
    output_dir: str
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    feature_coverage_threshold: float = 0.95
    n_cv_folds: int = 5
    window_steps: int = 320
    window_bins: int = 32
    pretrain_stride: int = 20
    action_grid_bins: int = 21
    action_grid_low: float = 0.0
    action_grid_high: float = 1.0


def build_pooled_dense_policy_dataset_v4(
    config: BuildPooledDenseDatasetV4Config,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Build pooled two-phase and three-phase dense policy rows and sequences."""
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.read_parquet(input_dir / "runs.parquet")
    history_wide_df = pd.read_parquet(input_dir / "history_dense_wide.parquet")
    fold_assignments = assign_grouped_folds(runs_df, config.n_cv_folds)
    action_grid = build_action_grid(config)
    history_by_run = {
        str(run_id): frame.sort_values("step").reset_index(drop=True)
        for run_id, frame in history_wide_df.groupby("wandb_run_id", sort=False)
    }

    decision_rows: list[dict[str, Any]] = []
    sequence_by_row_id: dict[str, np.ndarray] = {}
    mask_by_row_id: dict[str, np.ndarray] = {}
    pretrain_meta: list[dict[str, Any]] = []
    pretrain_windows: list[np.ndarray] = []
    pretrain_masks: list[np.ndarray] = []

    for _, run_row in runs_df.iterrows():
        run_id = str(run_row["wandb_run_id"])
        history = history_by_run.get(run_id)
        if history is None or history.empty:
            continue
        history_cache = _prepare_run_history_cache(
            history,
            total_steps=int(run_row["total_steps"]),
            objective_metric=config.objective_metric,
        )
        num_phases_total = int(run_row["num_phases_total"])
        final_objective = (
            float(run_row[config.objective_metric])
            if config.objective_metric in run_row and not pd.isna(run_row[config.objective_metric])
            else None
        )
        if final_objective is None:
            metric = (
                history[config.objective_metric].dropna()
                if config.objective_metric in history.columns
                else pd.Series(dtype=float)
            )
            if metric.empty:
                continue
            final_objective = float(metric.iloc[-1])
        phase_boundaries = tuple(json.loads(run_row["phase_boundaries_json"]))
        decision_steps = [0, *phase_boundaries]
        total_steps = int(run_row["total_steps"])
        for decision_index in range(num_phases_total):
            built = build_decision_state_features(
                run_row=run_row,
                history=history,
                objective_metric=config.objective_metric,
                decision_index=decision_index,
                window_steps=config.window_steps,
                window_bins=config.window_bins,
                history_cache=history_cache,
            )
            if built is None:
                continue
            state, window, mask = built
            next_decision_step = (
                total_steps if decision_index == num_phases_total - 1 else decision_steps[decision_index + 1]
            )
            next_obj_bpb = history.loc[
                (history["step"] <= next_decision_step) & history[config.objective_metric].notna(),
                config.objective_metric,
            ]
            next_value = float(next_obj_bpb.iloc[-1]) if not next_obj_bpb.empty else float(final_objective)
            action_starcoder = float(run_row[f"phase_{decision_index}_starcoder"])
            row = _decision_row(
                run_row=run_row,
                state=state,
                decision_index=decision_index,
                next_obj_bpb=next_value,
                final_objective=float(final_objective),
                action_starcoder=action_starcoder,
                action_grid=action_grid,
                fold_assignments=fold_assignments,
            )
            decision_rows.append(row)
            sequence_by_row_id[row["row_id"]] = window
            mask_by_row_id[row["row_id"]] = mask

        for item in _build_pretrain_windows(
            run_row,
            history,
            objective_metric=config.objective_metric,
            window_steps=config.window_steps,
            window_bins=config.window_bins,
            stride=config.pretrain_stride,
            history_cache=history_cache,
        ):
            pretrain_meta.append({key: value for key, value in item.items() if key not in {"window", "mask"}})
            pretrain_windows.append(item["window"])
            pretrain_masks.append(item["mask"])

    decisions = pd.DataFrame(decision_rows)
    if decisions.empty:
        raise ValueError("No pooled decision rows were produced.")

    coverage_rows: list[dict[str, Any]] = []
    selected_feature_keys: list[str] = []
    for feature_name in SUMMARY_FEATURES:
        coverage = (
            float(decisions[feature_name].replace([np.inf, -np.inf], np.nan).notna().mean())
            if feature_name in decisions.columns
            else 0.0
        )
        selected = coverage >= config.feature_coverage_threshold
        coverage_rows.append({"feature_name": feature_name, "coverage": coverage, "selected": selected})
        if selected:
            selected_feature_keys.append(feature_name)
    if not selected_feature_keys:
        raise ValueError("No summary features met the coverage threshold.")

    decisions, feature_defaults = _impute_selected_features(decisions, tuple(selected_feature_keys))
    decisions = decisions.sort_values(["wandb_run_id", "step_in_episode"]).reset_index(drop=True)
    decisions["return_to_go_dense_raw"] = decisions.groupby("episode_id", sort=False)["reward_dense_raw"].transform(
        lambda values: np.flip(np.cumsum(np.flip(values.to_numpy(dtype=np.float32)))).astype(np.float32)
    )

    decisions.to_parquet(output_dir / "decisions.parquet", index=False)
    ordered_row_ids = decisions["row_id"].tolist()
    sequence_arrays = {
        "row_ids": np.asarray(ordered_row_ids, dtype=str),
        "sequences": np.stack([sequence_by_row_id[row_id] for row_id in ordered_row_ids], axis=0).astype(np.float32),
        "masks": np.stack([mask_by_row_id[row_id] for row_id in ordered_row_ids], axis=0).astype(np.float32),
    }
    np.savez_compressed(output_dir / "decision_sequences.npz", **sequence_arrays)

    pretrain_payload = {
        "row_ids": np.asarray([row["row_id"] for row in pretrain_meta], dtype=str),
        "wandb_run_id": np.asarray([row["wandb_run_id"] for row in pretrain_meta], dtype=str),
        "phase_index": np.asarray([row["phase_index"] for row in pretrain_meta], dtype=np.int64),
        "cursor_step": np.asarray([row["cursor_step"] for row in pretrain_meta], dtype=np.int64),
        "action_starcoder": np.asarray([row["action_starcoder"] for row in pretrain_meta], dtype=np.float32),
        "next_train_loss": np.asarray([row["next_train_loss"] for row in pretrain_meta], dtype=np.float32),
        "next_eval_bpb": np.asarray([row["next_eval_bpb"] for row in pretrain_meta], dtype=np.float32),
        "sequences": (
            np.stack(pretrain_windows, axis=0).astype(np.float32)
            if pretrain_windows
            else np.zeros((0, config.window_bins, len(SEQUENCE_CHANNELS)), dtype=np.float32)
        ),
        "masks": (
            np.stack(pretrain_masks, axis=0).astype(np.float32)
            if pretrain_masks
            else np.zeros((0, config.window_bins), dtype=np.float32)
        ),
    }
    np.savez_compressed(output_dir / "pretrain_sequences.npz", **pretrain_payload)

    feature_manifest = {
        "objective_metric": config.objective_metric,
        "feature_coverage_threshold": config.feature_coverage_threshold,
        "selected_feature_keys": selected_feature_keys,
        "candidate_feature_keys": list(SUMMARY_FEATURES),
        "feature_defaults": feature_defaults,
        "rows": coverage_rows,
    }
    with (output_dir / "feature_manifest.json").open("w") as f:
        json.dump(feature_manifest, f, indent=2, sort_keys=True)

    sequence_manifest = {
        "window_steps": config.window_steps,
        "window_bins": config.window_bins,
        "bin_width": config.window_steps // config.window_bins,
        "channels": list(SEQUENCE_CHANNELS),
        "sequence_shape": [len(decisions), config.window_bins, len(SEQUENCE_CHANNELS)],
        "pretrain_count": len(pretrain_meta),
        "pretrain_stride": config.pretrain_stride,
    }
    with (output_dir / "sequence_manifest.json").open("w") as f:
        json.dump(sequence_manifest, f, indent=2, sort_keys=True)

    manifest = {
        "objective_metric": config.objective_metric,
        "n_runs": int(decisions["wandb_run_id"].nunique()),
        "run_family_counts": {
            str(key): int(value) for key, value in decisions["run_family"].value_counts().sort_index().items()
        },
        "n_decision_rows": len(decisions),
        "action_grid": [float(value) for value in action_grid.tolist()],
        "selected_feature_keys": selected_feature_keys,
        "n_cv_folds": int(config.n_cv_folds),
        "window_steps": int(config.window_steps),
        "window_bins": int(config.window_bins),
    }
    with (output_dir / "dataset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return decisions, sequence_arrays, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled dense policy datasets for v4.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v4_pooled",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/starcoder_dense_v4_pooled",
    )
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    build_pooled_dense_policy_dataset_v4(
        BuildPooledDenseDatasetV4Config(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            objective_metric=args.objective_metric,
        )
    )


if __name__ == "__main__":
    main()
