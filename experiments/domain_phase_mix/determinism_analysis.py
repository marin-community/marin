# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-run analysis for two-phase StarCoder determinism and seed jitter."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import pandas as pd

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

logger = logging.getLogger(__name__)

RUN_MANIFEST_FILE = "run_manifest.json"
FINAL_BPB_STATS_JSON = "final_bpb_stats.json"
TRAJECTORY_BPB_STATS_CSV = "trajectory_bpb_stats.csv"
DETERMINISM_CONTROL_REPORT_JSON = "determinism_control_report.json"
SEED_RUNS_CSV = "seed_runs.csv"
CONTROL_RUNS_CSV = "control_runs.csv"
TRAJECTORY_RAW_PARQUET = "trajectory_raw.parquet"
RESULTS_CSV = "results.csv"

DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class JitterReportConfig:
    """Executor config for determinism and jitter report generation."""

    output_path: str
    objective_metric: str
    wandb_entity: str
    wandb_project: str
    run_manifest_path: InputName | str
    analysis_output_path: InputName | str
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


def _bootstrap_mean_and_std_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    if len(values) == 1:
        val = float(values[0])
        return (val, val), (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[boot_idx]
    boot_means = samples.mean(axis=1)
    boot_stds = samples.std(axis=1, ddof=1)

    low_q = alpha / 2
    high_q = 1.0 - alpha / 2
    mean_ci = (
        float(np.quantile(boot_means, low_q)),
        float(np.quantile(boot_means, high_q)),
    )
    std_ci = (
        float(np.quantile(boot_stds, low_q)),
        float(np.quantile(boot_stds, high_q)),
    )
    return mean_ci, std_ci


def _fetch_single_run_trajectory(
    *,
    wandb_run_id: str,
    wandb_entity: str,
    wandb_project: str,
    objective_metric: str,
) -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
    rows: list[dict[str, Any]] = []
    for entry in run.scan_history(keys=["_step", objective_metric, "throughput/total_tokens"]):
        step = entry.get("_step")
        metric = entry.get(objective_metric)
        if step is None or metric is None:
            continue
        rows.append(
            {
                "step": int(step),
                "metric_value": float(metric),
                "total_tokens": (
                    float(entry["throughput/total_tokens"])
                    if entry.get("throughput/total_tokens") is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _collect_trajectory_rows(
    runs_df: pd.DataFrame,
    *,
    wandb_entity: str,
    wandb_project: str,
    objective_metric: str,
) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    for row in runs_df.itertuples(index=False):
        run_id = int(row.run_id)
        wandb_run_id = row.wandb_run_id
        if pd.isna(wandb_run_id) or not wandb_run_id:
            continue
        traj = _fetch_single_run_trajectory(
            wandb_run_id=str(wandb_run_id),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            objective_metric=objective_metric,
        )
        if traj.empty:
            continue
        traj["run_id"] = run_id
        traj["run_name"] = row.run_name
        traj["cohort"] = row.cohort
        traj["data_seed"] = int(row.data_seed)
        traj["wandb_run_id"] = str(wandb_run_id)
        all_rows.append(traj)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_name",
                "cohort",
                "data_seed",
                "wandb_run_id",
                "step",
                "metric_value",
                "total_tokens",
            ]
        )
    return pd.concat(all_rows, ignore_index=True)


def _summarize_trajectory(
    traj_df: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if traj_df.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "n_runs",
                "tokens_mean",
                "metric_mean",
                "metric_std",
                "metric_min",
                "metric_q05",
                "metric_q50",
                "metric_q95",
                "metric_max",
                "mean_ci_low",
                "mean_ci_high",
            ]
        )

    rows: list[dict[str, Any]] = []
    for idx, (step, group) in enumerate(traj_df.groupby("step", sort=True)):
        vals = group["metric_value"].to_numpy(dtype=np.float64)
        mean_ci, _ = _bootstrap_mean_and_std_ci(
            vals,
            n_boot=bootstrap_samples,
            seed=bootstrap_seed + idx,
        )
        rows.append(
            {
                "step": int(step),
                "n_runs": len(vals),
                "tokens_mean": float(group["total_tokens"].mean()),
                "metric_mean": float(np.mean(vals)),
                "metric_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "metric_min": float(np.min(vals)),
                "metric_q05": float(np.quantile(vals, 0.05)),
                "metric_q50": float(np.quantile(vals, 0.50)),
                "metric_q95": float(np.quantile(vals, 0.95)),
                "metric_max": float(np.max(vals)),
                "mean_ci_low": float(mean_ci[0]),
                "mean_ci_high": float(mean_ci[1]),
            }
        )
    return pd.DataFrame(rows)


def _build_control_report(
    control_df: pd.DataFrame,
    control_traj_df: pd.DataFrame,
    *,
    objective_metric: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "objective_metric": objective_metric,
        "n_control_runs": len(control_df),
        "status": "insufficient_data",
    }
    if len(control_df) < 2:
        return report

    completed = control_df[control_df["status"] == "completed"].sort_values("run_id")
    if len(completed) < 2:
        report["status"] = "insufficient_completed_runs"
        return report

    left = completed.iloc[0]
    right = completed.iloc[1]
    left_final = float(left[objective_metric])
    right_final = float(right[objective_metric])
    final_abs_diff = abs(left_final - right_final)

    left_traj = control_traj_df[control_traj_df["run_id"] == int(left["run_id"])]
    right_traj = control_traj_df[control_traj_df["run_id"] == int(right["run_id"])]
    merged = left_traj.merge(
        right_traj,
        on="step",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        report.update(
            {
                "status": "no_overlapping_steps",
                "final_abs_diff": final_abs_diff,
                "control_run_ids": [int(left["run_id"]), int(right["run_id"])],
                "control_wandb_run_ids": [str(left["wandb_run_id"]), str(right["wandb_run_id"])],
            }
        )
        return report

    step_abs_diff = np.abs(merged["metric_value_a"].to_numpy() - merged["metric_value_b"].to_numpy())
    max_abs_step_diff = float(np.max(step_abs_diff))
    mean_abs_step_diff = float(np.mean(step_abs_diff))
    report.update(
        {
            "status": "ok",
            "control_run_ids": [int(left["run_id"]), int(right["run_id"])],
            "control_wandb_run_ids": [str(left["wandb_run_id"]), str(right["wandb_run_id"])],
            "final_values": [left_final, right_final],
            "final_abs_diff": final_abs_diff,
            "n_overlapping_steps": len(merged),
            "max_abs_step_diff": max_abs_step_diff,
            "mean_abs_step_diff": mean_abs_step_diff,
            "exact_match_final": bool(final_abs_diff == 0.0),
            "exact_match_trajectory": bool(max_abs_step_diff == 0.0),
            "exact_match_all": bool(final_abs_diff == 0.0 and max_abs_step_diff == 0.0),
        }
    )
    return report


def create_jitter_report(config: JitterReportConfig) -> None:
    """Build final/trajectory jitter stats and control determinism report."""
    run_manifest_path = str(config.run_manifest_path)
    analysis_path = str(config.analysis_output_path)
    results_path = os.path.join(analysis_path, RESULTS_CSV)

    with fsspec.open(run_manifest_path, "r") as f:
        manifest_payload = json.load(f)
    manifest_df = pd.DataFrame(manifest_payload["runs"])
    results_df = pd.read_csv(results_path)

    merged = manifest_df.merge(results_df, on="run_id", how="left")
    merged["status"] = merged["status"].fillna("not_found")

    seed_df = merged[merged["cohort"] == "seed_sweep"].copy()
    control_df = merged[merged["cohort"] == "determinism_control"].copy()

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, SEED_RUNS_CSV), "w") as f:
        seed_df.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, CONTROL_RUNS_CSV), "w") as f:
        control_df.to_csv(f, index=False)

    seed_completed = seed_df[
        (seed_df["status"] == "completed")
        & seed_df["wandb_run_id"].notna()
        & seed_df[config.objective_metric].notna()
    ].copy()
    if seed_completed.empty:
        raise ValueError("No completed seed_sweep runs with objective metric found for jitter report.")

    final_values = seed_completed[config.objective_metric].to_numpy(dtype=np.float64)
    mean_ci, std_ci = _bootstrap_mean_and_std_ci(
        final_values,
        n_boot=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    final_stats = {
        "objective_metric": config.objective_metric,
        "n_runs": len(final_values),
        "mean": float(np.mean(final_values)),
        "std": float(np.std(final_values, ddof=1)) if len(final_values) > 1 else 0.0,
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
        "cv": (
            float(np.std(final_values, ddof=1) / np.mean(final_values))
            if len(final_values) > 1 and float(np.mean(final_values)) != 0.0
            else np.nan
        ),
        "mean_ci_low": float(mean_ci[0]),
        "mean_ci_high": float(mean_ci[1]),
        "std_ci_low": float(std_ci[0]),
        "std_ci_high": float(std_ci[1]),
        "ci_method": "bootstrap_percentile",
        "bootstrap_samples": int(config.bootstrap_samples),
    }
    with fsspec.open(os.path.join(config.output_path, FINAL_BPB_STATS_JSON), "w") as f:
        json.dump(final_stats, f, indent=2, sort_keys=True)

    seed_traj = _collect_trajectory_rows(
        seed_completed,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        objective_metric=config.objective_metric,
    )
    trajectory_stats = _summarize_trajectory(
        seed_traj,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    with fsspec.open(os.path.join(config.output_path, TRAJECTORY_BPB_STATS_CSV), "w") as f:
        trajectory_stats.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, TRAJECTORY_RAW_PARQUET), "wb") as f:
        seed_traj.to_parquet(f, index=False)

    control_completed = control_df[
        (control_df["status"] == "completed")
        & control_df["wandb_run_id"].notna()
        & control_df[config.objective_metric].notna()
    ].copy()
    control_traj = _collect_trajectory_rows(
        control_completed,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        objective_metric=config.objective_metric,
    )
    control_report = _build_control_report(
        control_df=control_completed,
        control_traj_df=control_traj,
        objective_metric=config.objective_metric,
    )
    with fsspec.open(os.path.join(config.output_path, DETERMINISM_CONTROL_REPORT_JSON), "w") as f:
        json.dump(control_report, f, indent=2, sort_keys=True)

    logger.info(
        "Wrote jitter report: %s runs, %s trajectory rows",
        final_stats["n_runs"],
        len(seed_traj),
    )


def create_determinism_report_step(
    *,
    name_prefix: str,
    objective_metric: str,
    run_manifest_step: ExecutorStep,
    analysis_step: ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
) -> ExecutorStep:
    """Create an ExecutorStep for determinism/jitter reporting."""
    return ExecutorStep(
        name=f"{name_prefix}/determinism_report",
        description="Compute BPB jitter CI and bitwise determinism control diffs",
        fn=create_jitter_report,
        config=JitterReportConfig(
            output_path=this_output_path(),
            objective_metric=objective_metric,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            run_manifest_path=output_path_of(run_manifest_step, RUN_MANIFEST_FILE),
            analysis_output_path=output_path_of(analysis_step),
        ),
    )
