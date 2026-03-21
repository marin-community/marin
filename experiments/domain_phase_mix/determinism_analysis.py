# Copyright The Marin Authors
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
SWARM_COMPARISON_JSON = "swarm_comparison.json"
SWARM_COMPARISON_CSV = "swarm_comparison.csv"

DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 0
SEED_SWEEP_COHORT = "seed_sweep"
CONTROL_COHORTS = ("determinism_control", "exact_replay_control")


@dataclass(frozen=True)
class JitterReportConfig:
    """Executor config for determinism and jitter report generation."""

    output_path: str
    objective_metric: str
    wandb_entity: str
    wandb_project: str
    run_manifest_path: InputName | str
    analysis_output_path: InputName | str
    swarm_results_csv_path: str | None = None
    comparison_metrics: tuple[str, ...] = ()
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class CollectManifestResultsConfig:
    """Executor config for manifest-backed W&B result collection."""

    output_path: str
    run_manifest_path: InputName | str
    wandb_entity: str
    wandb_project: str
    metric_prefixes: tuple[str, ...] = ("eval/", "lm_eval/")
    extra_metrics: tuple[str, ...] = ()
    depends_on: tuple[InputName, ...] = ()


def _maybe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _phase_weights_to_columns(phase_weights: dict[str, dict[str, float]]) -> dict[str, float]:
    columns: dict[str, float] = {}
    for phase_name, domain_weights in phase_weights.items():
        for domain_name, value in domain_weights.items():
            columns[f"{phase_name}_{domain_name}"] = float(value)
    return columns


def _resolve_wandb_run_for_name(run_rows: list[dict[str, Any]], run_name: str) -> dict[str, Any] | None:
    marker = f"/{run_name}"
    matches = [row for row in run_rows if marker in str(row.get("wandb_run_name") or "")]
    if not matches:
        return None

    finished = [row for row in matches if row.get("status") == "finished"]
    if finished:
        return finished[0]
    return matches[0]


def collect_manifest_results(config: CollectManifestResultsConfig) -> None:
    """Collect run summaries from W&B for the runs listed in a manifest."""
    from experiments.domain_phase_mix.analysis import query_wandb_runs, query_wandb_runs_by_name_substrings

    with fsspec.open(str(config.run_manifest_path), "r") as f:
        manifest_payload = json.load(f)

    experiment_name = str(manifest_payload["experiment_name"])
    run_rows = query_wandb_runs(
        entity=config.wandb_entity,
        project=config.wandb_project,
        tags=[experiment_name],
        metrics=list(config.extra_metrics),
        metric_prefixes=config.metric_prefixes,
    )
    missing_run_names = [
        str(manifest_row["run_name"])
        for manifest_row in manifest_payload["runs"]
        if _resolve_wandb_run_for_name(run_rows, str(manifest_row["run_name"])) is None
    ]
    if missing_run_names:
        logger.warning(
            "Tag-based W&B lookup missed %d/%d manifest runs; falling back to display-name lookup for %s",
            len(missing_run_names),
            len(manifest_payload["runs"]),
            missing_run_names,
        )
        run_rows.extend(
            query_wandb_runs_by_name_substrings(
                entity=config.wandb_entity,
                project=config.wandb_project,
                name_substrings=missing_run_names,
                metrics=list(config.extra_metrics),
                metric_prefixes=config.metric_prefixes,
            )
        )

    collected_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_payload["runs"]:
        run_name = str(manifest_row["run_name"])
        wandb_row = _resolve_wandb_run_for_name(run_rows, run_name)
        row: dict[str, Any] = {
            "run_id": int(manifest_row["run_id"]),
            "run_name": run_name,
            "cohort": str(manifest_row["cohort"]),
            "trainer_seed": manifest_row.get("trainer_seed"),
            "data_seed": manifest_row.get("data_seed"),
            "source_run_name": manifest_row.get("source_run_name"),
            **_phase_weights_to_columns(manifest_row["phase_weights"]),
        }

        if wandb_row is None:
            row.update(
                {
                    "wandb_run_id": None,
                    "wandb_run_name": None,
                    "status": "not_found",
                }
            )
            collected_rows.append(row)
            continue

        row.update(
            {
                "wandb_run_id": wandb_row.get("wandb_run_id"),
                "wandb_run_name": wandb_row.get("wandb_run_name"),
                "status": "completed" if wandb_row.get("status") == "finished" else wandb_row.get("status", "unknown"),
            }
        )
        for key, value in wandb_row.items():
            if isinstance(value, int | float) and any(key.startswith(prefix) for prefix in config.metric_prefixes):
                row[key] = float(value)
        collected_rows.append(row)

    results_df = pd.DataFrame(collected_rows).sort_values("run_id").reset_index(drop=True)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        results_df.to_csv(f, index=False)


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
        traj["trainer_seed"] = _maybe_int(getattr(row, "trainer_seed", None))
        traj["data_seed"] = _maybe_int(getattr(row, "data_seed", None))
        traj["wandb_run_id"] = str(wandb_run_id)
        all_rows.append(traj)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_name",
                "cohort",
                "trainer_seed",
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


def _backfill_objective_metric_from_trajectories(
    runs_df: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    objective_metric: str,
) -> pd.DataFrame:
    if runs_df.empty or trajectories.empty:
        return runs_df

    result = runs_df.copy()
    if objective_metric not in result.columns:
        result[objective_metric] = np.nan

    objective_rows = (
        trajectories.sort_values(["run_id", "step"])
        .groupby("run_id", as_index=False)
        .agg(metric_value=("metric_value", "last"))
    )
    objective_map = {int(row.run_id): float(row.metric_value) for row in objective_rows.itertuples(index=False)}

    for idx, run_id in result["run_id"].items():
        if pd.notna(result.at[idx, objective_metric]):
            continue
        backfilled = objective_map.get(int(run_id))
        if backfilled is not None:
            result.at[idx, objective_metric] = backfilled
    return result


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "variance": np.nan,
            "range": np.nan,
        }

    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    variance = float(std**2) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "variance": variance,
        "range": float(np.max(values) - np.min(values)),
    }


def _build_swarm_comparison_rows(
    *,
    seed_runs_df: pd.DataFrame,
    swarm_df: pd.DataFrame,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        repeat_values = (
            seed_runs_df[metric].dropna().to_numpy(dtype=np.float64) if metric in seed_runs_df.columns else np.array([])
        )
        swarm_values = (
            swarm_df[metric].dropna().to_numpy(dtype=np.float64) if metric in swarm_df.columns else np.array([])
        )

        repeat_stats = _metric_summary(repeat_values)
        swarm_stats = _metric_summary(swarm_values)
        swarm_variance = swarm_stats["variance"]
        swarm_range = swarm_stats["range"]
        variance_ratio = (
            float(repeat_stats["variance"] / swarm_variance)
            if np.isfinite(repeat_stats["variance"]) and np.isfinite(swarm_variance) and swarm_variance != 0.0
            else np.nan
        )
        range_ratio = (
            float(repeat_stats["range"] / swarm_range)
            if np.isfinite(repeat_stats["range"]) and np.isfinite(swarm_range) and swarm_range != 0.0
            else np.nan
        )
        rows.append(
            {
                "metric": metric,
                "repeat_n": repeat_stats["n"],
                "repeat_mean": repeat_stats["mean"],
                "repeat_std": repeat_stats["std"],
                "repeat_min": repeat_stats["min"],
                "repeat_max": repeat_stats["max"],
                "repeat_iqr": repeat_stats["iqr"],
                "swarm_n": swarm_stats["n"],
                "swarm_mean": swarm_stats["mean"],
                "swarm_std": swarm_stats["std"],
                "swarm_min": swarm_stats["min"],
                "swarm_max": swarm_stats["max"],
                "swarm_iqr": swarm_stats["iqr"],
                "variance_ratio": variance_ratio,
                "range_ratio": range_ratio,
            }
        )
    return rows


def create_jitter_report(config: JitterReportConfig) -> None:
    """Build final/trajectory jitter stats and control determinism report."""
    run_manifest_path = str(config.run_manifest_path)
    analysis_path = str(config.analysis_output_path)
    results_path = os.path.join(analysis_path, RESULTS_CSV)

    with fsspec.open(run_manifest_path, "r") as f:
        manifest_payload = json.load(f)
    manifest_df = pd.DataFrame(manifest_payload["runs"])
    results_df = pd.read_csv(results_path)

    merged = manifest_df.merge(results_df, on="run_id", how="left", suffixes=("", "_collected"))
    if "run_name_collected" in merged.columns:
        merged["run_name"] = merged["run_name_collected"].combine_first(merged["run_name"])
        merged = merged.drop(columns=["run_name_collected"])
    if "cohort_collected" in merged.columns:
        merged = merged.drop(columns=["cohort_collected"])
    if "trainer_seed_collected" in merged.columns:
        merged["trainer_seed"] = merged["trainer_seed"].combine_first(merged["trainer_seed_collected"])
        merged = merged.drop(columns=["trainer_seed_collected"])
    if "data_seed_collected" in merged.columns:
        merged["data_seed"] = merged["data_seed"].combine_first(merged["data_seed_collected"])
        merged = merged.drop(columns=["data_seed_collected"])
    if "source_run_name_collected" in merged.columns:
        merged["source_run_name"] = merged["source_run_name"].combine_first(merged["source_run_name_collected"])
        merged = merged.drop(columns=["source_run_name_collected"])
    merged["status"] = merged["status"].fillna("not_found")

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)

    completed_runs = merged[(merged["status"] == "completed") & merged["wandb_run_id"].notna()].copy()
    all_traj = _collect_trajectory_rows(
        completed_runs,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        objective_metric=config.objective_metric,
    )
    merged = _backfill_objective_metric_from_trajectories(
        merged,
        all_traj,
        objective_metric=config.objective_metric,
    )

    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        merged.to_csv(f, index=False)

    seed_df = merged[merged["cohort"] == SEED_SWEEP_COHORT].copy()
    control_df = merged[merged["cohort"].isin(CONTROL_COHORTS)].copy()

    with fsspec.open(os.path.join(config.output_path, SEED_RUNS_CSV), "w") as f:
        seed_df.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, CONTROL_RUNS_CSV), "w") as f:
        control_df.to_csv(f, index=False)

    seed_completed = seed_df[
        (seed_df["status"] == "completed") & seed_df["wandb_run_id"].notna() & seed_df[config.objective_metric].notna()
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

    seed_traj = all_traj[all_traj["cohort"] == SEED_SWEEP_COHORT].copy()
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
    control_traj = all_traj[all_traj["cohort"].isin(CONTROL_COHORTS)].copy()
    control_report = _build_control_report(
        control_df=control_completed,
        control_traj_df=control_traj,
        objective_metric=config.objective_metric,
    )
    with fsspec.open(os.path.join(config.output_path, DETERMINISM_CONTROL_REPORT_JSON), "w") as f:
        json.dump(control_report, f, indent=2, sort_keys=True)

    if config.swarm_results_csv_path is not None and config.comparison_metrics:
        with fsspec.open(config.swarm_results_csv_path, "r") as f:
            swarm_df = pd.read_csv(f)
        comparison_rows = _build_swarm_comparison_rows(
            seed_runs_df=seed_completed,
            swarm_df=swarm_df,
            metrics=config.comparison_metrics,
        )
        with fsspec.open(os.path.join(config.output_path, SWARM_COMPARISON_JSON), "w") as f:
            json.dump(
                {
                    "seed_cohort": SEED_SWEEP_COHORT,
                    "swarm_results_csv_path": config.swarm_results_csv_path,
                    "metrics": comparison_rows,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        with fsspec.open(os.path.join(config.output_path, SWARM_COMPARISON_CSV), "w") as f:
            pd.DataFrame(comparison_rows).to_csv(f, index=False)

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
    swarm_results_csv_path: str | None = None,
    comparison_metrics: tuple[str, ...] = (),
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
            swarm_results_csv_path=swarm_results_csv_path,
            comparison_metrics=comparison_metrics,
        ),
    )


def create_manifest_results_step(
    *,
    name_prefix: str,
    run_manifest_step: ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
    extra_metrics: tuple[str, ...] = (),
    depends_on: list[ExecutorStep] | None = None,
) -> ExecutorStep:
    """Create a manifest-backed W&B results collection step."""
    blocked_on = depends_on or []
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description="Collect W&B summaries for explicitly named determinism study runs",
        fn=collect_manifest_results,
        config=CollectManifestResultsConfig(
            output_path=this_output_path(),
            run_manifest_path=output_path_of(run_manifest_step, RUN_MANIFEST_FILE),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            extra_metrics=extra_metrics,
            depends_on=tuple(output_path_of(step) for step in blocked_on),
        ),
    )
