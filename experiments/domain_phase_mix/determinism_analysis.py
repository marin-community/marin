# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-run analysis for two-phase StarCoder determinism and seed jitter."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.mmlu_sl_verb_rerun_common import resolve_unique_checkpoint_root
from experiments.defaults import _truncate_wandb_name
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
FIT_DATASET_CSV = "fit_dataset.csv"
FIT_DATASET_SUMMARY_JSON = "fit_dataset_summary.json"
SWARM_COMPARISON_JSON = "swarm_comparison.json"
SWARM_COMPARISON_CSV = "swarm_comparison.csv"
COMPUTE_SCALING_NOISE_SUMMARY_JSON = "compute_scaling_noise_summary.json"
COMPUTE_SCALING_NOISE_SUMMARY_CSV = "compute_scaling_noise_summary.csv"
MMLU_NOISE_VS_COMPUTE_PNG = "mmlu_noise_vs_compute.png"
FIXED_SUBSET_NOISE_SUMMARY_JSON = "fixed_subset_noise_summary.json"
FIXED_SUBSET_NOISE_SUMMARY_CSV = "fixed_subset_noise_summary.csv"
MMLU_NOISE_FIXED_SUBSET_VS_ORIGINAL_PNG = "mmlu_noise_fixed_subset_vs_original.png"
PANEL_VS_NOISE_SUMMARY_JSON = "panel_vs_noise_summary.json"
PANEL_VS_NOISE_SUMMARY_CSV = "panel_vs_noise_summary.csv"
MMLU_PANEL_VS_NOISE_PNG = "mmlu_panel_vs_noise.png"
NOISE_SUMMARY_JSON = "noise_summary.json"
NOISE_SUMMARY_CSV = "noise_summary.csv"
MMLU_NOISE_VS_RUNTIME_300M_6B_PNG = "mmlu_noise_vs_runtime_300m_6b.png"

DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 0
SEED_SWEEP_COHORT = "seed_sweep"
CONTROL_COHORTS = ("determinism_control", "exact_replay_control")
PRESERVED_MANIFEST_METADATA_KEYS = (
    "candidate_run_id",
    "candidate_run_name",
    "candidate_source_experiment",
    "ladder",
    "model_family",
    "experiment_budget",
    "num_train_steps",
)
CHECKPOINT_EVAL_METRICS_PATH = "checkpoints/eval_metrics.jsonl"


def _get_step_times_from_wandb(*, run_id: str, entity: str, project: str) -> list[float]:
    from marin.speedrun.speedrun import get_step_times_from_wandb

    return get_step_times_from_wandb(run_id=run_id, entity=entity, project=project)


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


@dataclass(frozen=True)
class ComputeScalingNoiseReportConfig:
    """Executor config for the compute-scaling noise comparison report."""

    output_path: str
    run_manifest_path: InputName | str
    analysis_output_path: InputName | str
    wandb_entity: str
    wandb_project: str
    baseline_manifest_json: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...] = ()
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class FixedSubsetNoiseReportConfig:
    """Executor config for comparing fixed-subset seed jitter against the original seed study."""

    output_path: str
    analysis_output_path: InputName | str
    wandb_entity: str
    wandb_project: str
    baseline_manifest_json: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...] = ()
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class PanelVsNoiseReportConfig:
    """Executor config for comparing a fixed-subset observed-run panel against the fixed-subset noise floor."""

    output_path: str
    run_manifest_path: InputName | str
    analysis_output_path: InputName | str
    wandb_entity: str
    wandb_project: str
    baseline_manifest_json: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...] = ()
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class ModelSizeNoiseReportConfig:
    """Executor config for comparing a model-size study against fixed-subset baselines."""

    output_path: str
    run_manifest_path: InputName | str
    analysis_output_path: InputName | str
    wandb_entity: str
    wandb_project: str
    fixed_subset_baseline_manifest_json: str
    compute_baseline_manifest_json: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...] = ()
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True)
class FitDatasetExportConfig:
    """Executor config for writing a completed long-form fit dataset."""

    output_path: str
    run_manifest_path: InputName | str
    analysis_output_path: InputName | str


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


def _expected_wandb_run_name(experiment_name: str, run_name: str) -> str:
    return _truncate_wandb_name(f"{experiment_name}/{run_name}")


def _resolve_wandb_run_for_manifest_row(
    run_rows: list[dict[str, Any]],
    *,
    experiment_name: str,
    run_name: str,
) -> dict[str, Any] | None:
    expected_name = _expected_wandb_run_name(experiment_name, run_name)
    exact_matches = [row for row in run_rows if str(row.get("wandb_run_name") or "") == expected_name]
    if exact_matches:
        finished = [row for row in exact_matches if row.get("status") == "finished"]
        return finished[0] if finished else exact_matches[0]

    marker = f"/{run_name}"
    suffix_matches = [row for row in run_rows if marker in str(row.get("wandb_run_name") or "")]
    if len(suffix_matches) == 1:
        finished = [row for row in suffix_matches if row.get("status") == "finished"]
        return finished[0] if finished else suffix_matches[0]
    return None


def _load_checkpoint_eval_metrics(*, experiment_name: str, run_name: str) -> tuple[str | None, dict[str, float]]:
    checkpoint_root: str | None = None
    try:
        checkpoint_root = resolve_unique_checkpoint_root(source_experiment=experiment_name, run_name=run_name)
    except ValueError:
        return None, {}

    metrics_path = os.path.join(checkpoint_root, CHECKPOINT_EVAL_METRICS_PATH)
    try:
        with fsspec.open(metrics_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning("Checkpoint metrics file missing for %s at %s", run_name, metrics_path)
        return checkpoint_root, {}

    if not lines:
        logger.warning("Checkpoint metrics file is empty for %s at %s", run_name, metrics_path)
        return checkpoint_root, {}

    payload = json.loads(lines[-1])
    checkpoint_metrics = {
        key: float(value) for key, value in payload.items() if key.startswith("eval/") and isinstance(value, int | float)
    }
    return checkpoint_root, checkpoint_metrics


def _collect_manifest_results_frame(
    *,
    manifest_payload: dict[str, Any],
    wandb_entity: str,
    wandb_project: str,
    metric_prefixes: tuple[str, ...],
    extra_metrics: tuple[str, ...] = (),
) -> pd.DataFrame:
    from experiments.domain_phase_mix.analysis import query_wandb_runs, query_wandb_runs_by_name_substrings

    experiment_name = str(manifest_payload["experiment_name"])
    run_rows = query_wandb_runs(
        entity=wandb_entity,
        project=wandb_project,
        tags=[experiment_name],
        metrics=list(extra_metrics),
        metric_prefixes=metric_prefixes,
    )
    missing_expected_names = [
        _expected_wandb_run_name(experiment_name, str(manifest_row["run_name"]))
        for manifest_row in manifest_payload["runs"]
        if _resolve_wandb_run_for_manifest_row(
            run_rows,
            experiment_name=experiment_name,
            run_name=str(manifest_row["run_name"]),
        )
        is None
    ]
    if missing_expected_names:
        logger.warning(
            "Tag-based W&B lookup missed %d/%d manifest runs; falling back to display-name lookup for %s",
            len(missing_expected_names),
            len(manifest_payload["runs"]),
            missing_expected_names,
        )
        run_rows.extend(
            query_wandb_runs_by_name_substrings(
                entity=wandb_entity,
                project=wandb_project,
                name_substrings=missing_expected_names,
                metrics=list(extra_metrics),
                metric_prefixes=metric_prefixes,
            )
        )

    collected_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_payload["runs"]:
        run_name = str(manifest_row["run_name"])
        wandb_row = _resolve_wandb_run_for_manifest_row(
            run_rows,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        row: dict[str, Any] = {
            "run_id": int(manifest_row["run_id"]),
            "run_name": run_name,
            "cohort": str(manifest_row["cohort"]),
            "trainer_seed": manifest_row.get("trainer_seed"),
            "data_seed": manifest_row.get("data_seed"),
            "simulated_epoch_subset_seed": manifest_row.get("simulated_epoch_subset_seed"),
            "source_run_name": manifest_row.get("source_run_name"),
            **_phase_weights_to_columns(manifest_row["phase_weights"]),
        }

        for extra_key in PRESERVED_MANIFEST_METADATA_KEYS:
            if extra_key in manifest_row:
                row[extra_key] = manifest_row[extra_key]

        checkpoint_root, checkpoint_eval_metrics = _load_checkpoint_eval_metrics(
            experiment_name=experiment_name,
            run_name=run_name,
        )
        if checkpoint_root is not None:
            row["checkpoint_root"] = checkpoint_root

        if wandb_row is None:
            row.update(
                {
                    "wandb_run_id": None,
                    "wandb_run_name": None,
                    "status": "checkpoint_eval_only" if checkpoint_eval_metrics else "not_found",
                }
            )
            row.update(checkpoint_eval_metrics)
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
            if isinstance(value, int | float) and any(key.startswith(prefix) for prefix in metric_prefixes):
                row[key] = float(value)
        for key, value in checkpoint_eval_metrics.items():
            row.setdefault(key, value)
        collected_rows.append(row)

    return pd.DataFrame(collected_rows).sort_values("run_id").reset_index(drop=True)


def collect_manifest_results(config: CollectManifestResultsConfig) -> None:
    """Collect run summaries from W&B for the runs listed in a manifest."""
    with fsspec.open(str(config.run_manifest_path), "r") as f:
        manifest_payload = json.load(f)
    results_df = _collect_manifest_results_frame(
        manifest_payload=manifest_payload,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        metric_prefixes=config.metric_prefixes,
        extra_metrics=config.extra_metrics,
    )
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        results_df.to_csv(f, index=False)


def _ordered_fit_dataset_columns(df: pd.DataFrame) -> list[str]:
    prefix_columns = [
        "wandb_run_id",
        "run_id",
        "run_name",
        "status",
        "candidate_run_id",
        "candidate_run_name",
        "candidate_source_experiment",
        "trainer_seed",
        "data_seed",
        "simulated_epoch_subset_seed",
        "cohort",
        "source_run_name",
    ]
    ordered = [column for column in prefix_columns if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)
    return ordered


def create_fit_dataset_export(config: FitDatasetExportConfig) -> None:
    """Write a completed long-form fit dataset and completion summary from collected results."""
    with fsspec.open(str(config.run_manifest_path), "r") as f:
        manifest_payload = json.load(f)

    results_path = os.path.join(str(config.analysis_output_path), RESULTS_CSV)
    results_df = pd.read_csv(results_path)
    results_df["status"] = results_df["status"].fillna("not_found")

    completed = results_df[results_df["status"] == "completed"].copy()
    sort_columns = [column for column in ("candidate_run_id", "trainer_seed", "run_id") if column in completed.columns]
    if sort_columns:
        completed = completed.sort_values(sort_columns).reset_index(drop=True)
    completed = completed.loc[:, _ordered_fit_dataset_columns(completed)]

    manifest_runs = pd.DataFrame(manifest_payload["runs"])
    expected_candidate_columns = [
        column
        for column in ("candidate_run_id", "candidate_run_name", "candidate_source_experiment")
        if column in manifest_runs.columns
    ]
    expected_seed_columns = [column for column in ("trainer_seed",) if column in manifest_runs.columns]

    per_candidate_completion_counts: list[dict[str, Any]] = []
    if expected_candidate_columns:
        expected_candidates = (
            manifest_runs[expected_candidate_columns]
            .drop_duplicates()
            .sort_values(
                [column for column in ("candidate_run_id", "candidate_run_name") if column in expected_candidate_columns]
            )
        )
        for candidate_row in expected_candidates.itertuples(index=False):
            candidate = candidate_row._asdict()
            mask = pd.Series(True, index=results_df.index)
            for key, value in candidate.items():
                mask &= results_df[key] == value
            per_candidate_completion_counts.append(
                {
                    **candidate,
                    "n_expected": int(mask.sum()),
                    "n_completed": int((results_df.loc[mask, "status"] == "completed").sum()),
                }
            )

    per_seed_completion_counts: list[dict[str, Any]] = []
    if expected_seed_columns:
        expected_seeds = manifest_runs[expected_seed_columns].drop_duplicates().sort_values(expected_seed_columns)
        for seed_row in expected_seeds.itertuples(index=False):
            seed_values = seed_row._asdict()
            mask = pd.Series(True, index=results_df.index)
            for key, value in seed_values.items():
                mask &= results_df[key] == value
            per_seed_completion_counts.append(
                {
                    **seed_values,
                    "n_expected": int(mask.sum()),
                    "n_completed": int((results_df.loc[mask, "status"] == "completed").sum()),
                }
            )

    summary = {
        "experiment_name": manifest_payload["experiment_name"],
        "total_candidates_expected": (
            len(per_candidate_completion_counts) if per_candidate_completion_counts else len(manifest_payload["runs"])
        ),
        "total_seeds_expected": len(per_seed_completion_counts),
        "total_rows_expected": len(manifest_payload["runs"]),
        "total_rows_completed": len(completed),
        "per_candidate_completion_counts": per_candidate_completion_counts,
        "per_seed_completion_counts": per_seed_completion_counts,
    }

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, FIT_DATASET_CSV), "w") as f:
        completed.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, FIT_DATASET_SUMMARY_JSON), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


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


def _statistic_value(values: np.ndarray, statistic: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan
    if statistic == "variance":
        return float(np.std(values, ddof=1) ** 2) if len(values) > 1 else 0.0
    if statistic == "range":
        return float(np.max(values) - np.min(values))
    raise ValueError(f"Unsupported statistic {statistic!r}")


def _fetch_active_compute_hours(
    *,
    run_id: str,
    wandb_entity: str,
    wandb_project: str,
) -> float:
    return float(sum(_get_step_times_from_wandb(run_id=run_id, entity=wandb_entity, project=wandb_project)) / 3600.0)


def _active_compute_hours_summary(
    run_ids: list[str],
    *,
    wandb_entity: str,
    wandb_project: str,
    max_workers: int = 6,
) -> dict[str, float]:
    if not run_ids:
        return {
            "active_compute_hours_mean": np.nan,
            "active_compute_hours_std": np.nan,
            "active_compute_hours_n": 0,
        }

    durations: list[float] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(run_ids))) as pool:
        future_by_run_id = {
            pool.submit(
                _fetch_active_compute_hours,
                run_id=run_id,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
            ): run_id
            for run_id in run_ids
        }
        for future in as_completed(future_by_run_id):
            durations.append(float(future.result()))

    duration_array = np.asarray(durations, dtype=np.float64)
    return {
        "active_compute_hours_mean": float(duration_array.mean()),
        "active_compute_hours_std": float(duration_array.std(ddof=1)) if len(duration_array) > 1 else 0.0,
        "active_compute_hours_n": int(duration_array.size),
    }


def _bootstrap_ratio_ci(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    *,
    statistic: str,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    numerator_values = np.asarray(numerator_values, dtype=np.float64)
    denominator_values = np.asarray(denominator_values, dtype=np.float64)
    if len(numerator_values) == 0 or len(denominator_values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    num_idx = rng.integers(0, len(numerator_values), size=(n_boot, len(numerator_values)))
    den_idx = rng.integers(0, len(denominator_values), size=(n_boot, len(denominator_values)))
    ratios: list[float] = []
    for boot_idx in range(n_boot):
        numerator_stat = _statistic_value(numerator_values[num_idx[boot_idx]], statistic)
        denominator_stat = _statistic_value(denominator_values[den_idx[boot_idx]], statistic)
        if not np.isfinite(numerator_stat) or not np.isfinite(denominator_stat) or denominator_stat == 0.0:
            continue
        ratios.append(float(numerator_stat / denominator_stat))

    if not ratios:
        return np.nan, np.nan

    low_q = alpha / 2
    high_q = 1.0 - alpha / 2
    return float(np.quantile(ratios, low_q)), float(np.quantile(ratios, high_q))


def _build_compute_scaling_summary_rows(
    *,
    current_runs_df: pd.DataFrame,
    baseline_runs_df: pd.DataFrame,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    metrics = tuple(dict.fromkeys((*primary_metrics, *secondary_metrics)))
    cohort_specs = (
        ("baseline_1x", "regmix60m_1x", baseline_runs_df, "primary_baseline"),
        (
            "regmix60m_6b",
            "regmix60m_6b",
            current_runs_df[current_runs_df["ladder"] == "regmix60m_6b"].copy(),
            "paired_continuation",
        ),
        (
            "olmo3_30m_3b",
            "olmo3_30m_3b",
            current_runs_df[current_runs_df["ladder"] == "olmo3_30m_3b"].copy(),
            "exploratory_cross_family",
        ),
    )

    baseline_metric_values = {
        metric: (
            baseline_runs_df[metric].dropna().to_numpy(dtype=np.float64)
            if metric in baseline_runs_df.columns
            else np.array([])
        )
        for metric in metrics
    }

    rows: list[dict[str, Any]] = []
    for cohort_idx, (cohort_key, cohort_label, cohort_df, role) in enumerate(cohort_specs):
        for metric_idx, metric in enumerate(metrics):
            if metric in cohort_df.columns:
                values = cohort_df[metric].dropna().to_numpy(dtype=np.float64)
            else:
                values = np.array([])
            stats = _metric_summary(values)
            mean_ci, std_ci = _bootstrap_mean_and_std_ci(
                values,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + cohort_idx * 1_000 + metric_idx,
            )
            row: dict[str, Any] = {
                "cohort": cohort_key,
                "cohort_label": cohort_label,
                "cohort_role": role,
                "metric": metric,
                "is_primary_metric": metric in primary_metrics,
                **stats,
                "mean_ci_low": float(mean_ci[0]),
                "mean_ci_high": float(mean_ci[1]),
                "std_ci_low": float(std_ci[0]),
                "std_ci_high": float(std_ci[1]),
                "variance_ratio_vs_baseline_1x": np.nan,
                "variance_ratio_ci_low": np.nan,
                "variance_ratio_ci_high": np.nan,
                "range_ratio_vs_baseline_1x": np.nan,
                "range_ratio_ci_low": np.nan,
                "range_ratio_ci_high": np.nan,
            }
            if cohort_key == "regmix60m_6b":
                baseline_values = baseline_metric_values[metric]
                baseline_variance = _statistic_value(baseline_values, "variance")
                baseline_range = _statistic_value(baseline_values, "range")
                current_variance = _statistic_value(values, "variance")
                current_range = _statistic_value(values, "range")
                if np.isfinite(current_variance) and np.isfinite(baseline_variance) and baseline_variance != 0.0:
                    row["variance_ratio_vs_baseline_1x"] = float(current_variance / baseline_variance)
                    ci_low, ci_high = _bootstrap_ratio_ci(
                        values,
                        baseline_values,
                        statistic="variance",
                        n_boot=bootstrap_samples,
                        seed=bootstrap_seed + 10_000 + metric_idx,
                    )
                    row["variance_ratio_ci_low"] = ci_low
                    row["variance_ratio_ci_high"] = ci_high
                if np.isfinite(current_range) and np.isfinite(baseline_range) and baseline_range != 0.0:
                    row["range_ratio_vs_baseline_1x"] = float(current_range / baseline_range)
                    ci_low, ci_high = _bootstrap_ratio_ci(
                        values,
                        baseline_values,
                        statistic="range",
                        n_boot=bootstrap_samples,
                        seed=bootstrap_seed + 20_000 + metric_idx,
                    )
                    row["range_ratio_ci_low"] = ci_low
                    row["range_ratio_ci_high"] = ci_high
            rows.append(row)
    return rows


def _build_fixed_subset_summary_rows(
    *,
    current_runs_df: pd.DataFrame,
    baseline_runs_df: pd.DataFrame,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    metrics = tuple(dict.fromkeys((*primary_metrics, *secondary_metrics)))
    cohort_specs = (
        ("baseline_1x", "original_seed_study", baseline_runs_df, "primary_baseline"),
        ("fixed_subset_1x", "fixed_subset_1x", current_runs_df, "fixed_subset_variant"),
    )

    baseline_metric_values = {
        metric: (
            baseline_runs_df[metric].dropna().to_numpy(dtype=np.float64)
            if metric in baseline_runs_df.columns
            else np.array([])
        )
        for metric in metrics
    }

    rows: list[dict[str, Any]] = []
    for cohort_idx, (cohort_key, cohort_label, cohort_df, role) in enumerate(cohort_specs):
        for metric_idx, metric in enumerate(metrics):
            if metric in cohort_df.columns:
                values = cohort_df[metric].dropna().to_numpy(dtype=np.float64)
            else:
                values = np.array([])
            stats = _metric_summary(values)
            mean_ci, std_ci = _bootstrap_mean_and_std_ci(
                values,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + cohort_idx * 1_000 + metric_idx,
            )
            row: dict[str, Any] = {
                "cohort": cohort_key,
                "cohort_label": cohort_label,
                "cohort_role": role,
                "metric": metric,
                "is_primary_metric": metric in primary_metrics,
                **stats,
                "mean_ci_low": float(mean_ci[0]),
                "mean_ci_high": float(mean_ci[1]),
                "std_ci_low": float(std_ci[0]),
                "std_ci_high": float(std_ci[1]),
                "variance_ratio_vs_baseline_1x": np.nan,
                "variance_ratio_ci_low": np.nan,
                "variance_ratio_ci_high": np.nan,
                "range_ratio_vs_baseline_1x": np.nan,
                "range_ratio_ci_low": np.nan,
                "range_ratio_ci_high": np.nan,
            }
            if cohort_key == "fixed_subset_1x":
                baseline_values = baseline_metric_values[metric]
                baseline_variance = _statistic_value(baseline_values, "variance")
                baseline_range = _statistic_value(baseline_values, "range")
                current_variance = _statistic_value(values, "variance")
                current_range = _statistic_value(values, "range")
                if np.isfinite(current_variance) and np.isfinite(baseline_variance) and baseline_variance != 0.0:
                    row["variance_ratio_vs_baseline_1x"] = float(current_variance / baseline_variance)
                    ci_low, ci_high = _bootstrap_ratio_ci(
                        values,
                        baseline_values,
                        statistic="variance",
                        n_boot=bootstrap_samples,
                        seed=bootstrap_seed + 10_000 + metric_idx,
                    )
                    row["variance_ratio_ci_low"] = ci_low
                    row["variance_ratio_ci_high"] = ci_high
                if np.isfinite(current_range) and np.isfinite(baseline_range) and baseline_range != 0.0:
                    row["range_ratio_vs_baseline_1x"] = float(current_range / baseline_range)
                    ci_low, ci_high = _bootstrap_ratio_ci(
                        values,
                        baseline_values,
                        statistic="range",
                        n_boot=bootstrap_samples,
                        seed=bootstrap_seed + 20_000 + metric_idx,
                    )
                    row["range_ratio_ci_low"] = ci_low
                    row["range_ratio_ci_high"] = ci_high
            rows.append(row)
    return rows


def _pairwise_absolute_differences(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.array([], dtype=np.float64)
    pairwise = np.abs(values[:, None] - values[None, :])
    tri_upper = np.triu_indices(len(values), k=1)
    return pairwise[tri_upper]


def _build_panel_vs_noise_summary_rows(
    *,
    panel_runs_df: pd.DataFrame,
    baseline_runs_df: pd.DataFrame,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    metrics = tuple(dict.fromkeys((*primary_metrics, *secondary_metrics)))
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel_values = (
            panel_runs_df[metric].dropna().to_numpy(dtype=np.float64)
            if metric in panel_runs_df.columns
            else np.array([])
        )
        baseline_values = (
            baseline_runs_df[metric].dropna().to_numpy(dtype=np.float64)
            if metric in baseline_runs_df.columns
            else np.array([])
        )
        panel_stats = _metric_summary(panel_values)
        baseline_stats = _metric_summary(baseline_values)
        pairwise_diffs = _pairwise_absolute_differences(panel_values)
        baseline_std = baseline_stats["std"]
        baseline_range = baseline_stats["range"]
        rows.append(
            {
                "metric": metric,
                "is_primary_metric": metric in primary_metrics,
                "panel_n": panel_stats["n"],
                "panel_mean": panel_stats["mean"],
                "panel_std": panel_stats["std"],
                "panel_min": panel_stats["min"],
                "panel_max": panel_stats["max"],
                "panel_range": panel_stats["range"],
                "panel_iqr": panel_stats["iqr"],
                "baseline_n": baseline_stats["n"],
                "baseline_std": baseline_std,
                "baseline_range": baseline_range,
                "panel_std_ratio_vs_noise": (
                    float(panel_stats["std"] / baseline_std)
                    if np.isfinite(panel_stats["std"]) and np.isfinite(baseline_std) and baseline_std != 0.0
                    else np.nan
                ),
                "panel_range_ratio_vs_noise": (
                    float(panel_stats["range"] / baseline_range)
                    if np.isfinite(panel_stats["range"]) and np.isfinite(baseline_range) and baseline_range != 0.0
                    else np.nan
                ),
                "pairwise_n": len(pairwise_diffs),
                "pairwise_abs_diff_mean": float(np.mean(pairwise_diffs)) if len(pairwise_diffs) else np.nan,
                "pairwise_abs_diff_median": float(np.median(pairwise_diffs)) if len(pairwise_diffs) else np.nan,
                "pairwise_abs_diff_min": float(np.min(pairwise_diffs)) if len(pairwise_diffs) else np.nan,
                "pairwise_abs_diff_max": float(np.max(pairwise_diffs)) if len(pairwise_diffs) else np.nan,
                "frac_pairwise_gt_1x_noise_std": (
                    float(np.mean(pairwise_diffs > baseline_std))
                    if len(pairwise_diffs) and np.isfinite(baseline_std)
                    else np.nan
                ),
                "frac_pairwise_gt_2x_noise_std": (
                    float(np.mean(pairwise_diffs > (2.0 * baseline_std)))
                    if len(pairwise_diffs) and np.isfinite(baseline_std)
                    else np.nan
                ),
            }
        )
    return rows


def _build_model_size_noise_summary_rows(
    *,
    current_runs_df: pd.DataFrame,
    fixed_subset_baseline_df: pd.DataFrame,
    compute_baseline_df: pd.DataFrame,
    runtime_by_cohort: dict[str, dict[str, float]],
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    metrics = tuple(dict.fromkeys((*primary_metrics, *secondary_metrics)))
    cohort_specs = (
        ("fixed_subset_1x", "60M / 1.2B Fixed Subset", fixed_subset_baseline_df, "baseline_fixed_subset_1x"),
        ("regmix60m_6b", "60M / 6B", compute_baseline_df, "baseline_regmix60m_6b"),
        ("regmix300m_6b", "300M / 6B", current_runs_df, "study_300m_6b"),
    )
    baseline_metric_values = {
        metric: {
            "fixed_subset_1x": (
                fixed_subset_baseline_df[metric].dropna().to_numpy(dtype=np.float64)
                if metric in fixed_subset_baseline_df.columns
                else np.array([])
            ),
            "regmix60m_6b": (
                compute_baseline_df[metric].dropna().to_numpy(dtype=np.float64)
                if metric in compute_baseline_df.columns
                else np.array([])
            ),
        }
        for metric in metrics
    }

    rows: list[dict[str, Any]] = []
    for cohort_idx, (cohort_key, cohort_label, cohort_df, role) in enumerate(cohort_specs):
        runtime_stats = runtime_by_cohort.get(
            cohort_key,
            {
                "active_compute_hours_mean": np.nan,
                "active_compute_hours_std": np.nan,
                "active_compute_hours_n": 0,
            },
        )
        for metric_idx, metric in enumerate(metrics):
            values = (
                cohort_df[metric].dropna().to_numpy(dtype=np.float64) if metric in cohort_df.columns else np.array([])
            )
            stats = _metric_summary(values)
            mean_ci, std_ci = _bootstrap_mean_and_std_ci(
                values,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + cohort_idx * 1_000 + metric_idx,
            )
            row: dict[str, Any] = {
                "cohort": cohort_key,
                "cohort_label": cohort_label,
                "cohort_role": role,
                "metric": metric,
                "is_primary_metric": metric in primary_metrics,
                **stats,
                "mean_ci_low": float(mean_ci[0]),
                "mean_ci_high": float(mean_ci[1]),
                "std_ci_low": float(std_ci[0]),
                "std_ci_high": float(std_ci[1]),
                **runtime_stats,
                "variance_ratio_vs_fixed_subset_1x": np.nan,
                "variance_ratio_vs_fixed_subset_1x_ci_low": np.nan,
                "variance_ratio_vs_fixed_subset_1x_ci_high": np.nan,
                "range_ratio_vs_fixed_subset_1x": np.nan,
                "range_ratio_vs_fixed_subset_1x_ci_low": np.nan,
                "range_ratio_vs_fixed_subset_1x_ci_high": np.nan,
                "variance_ratio_vs_regmix60m_6b": np.nan,
                "variance_ratio_vs_regmix60m_6b_ci_low": np.nan,
                "variance_ratio_vs_regmix60m_6b_ci_high": np.nan,
                "range_ratio_vs_regmix60m_6b": np.nan,
                "range_ratio_vs_regmix60m_6b_ci_low": np.nan,
                "range_ratio_vs_regmix60m_6b_ci_high": np.nan,
            }
            if cohort_key == "regmix300m_6b":
                for baseline_key in ("fixed_subset_1x", "regmix60m_6b"):
                    baseline_values = baseline_metric_values[metric][baseline_key]
                    baseline_variance = _statistic_value(baseline_values, "variance")
                    baseline_range = _statistic_value(baseline_values, "range")
                    current_variance = _statistic_value(values, "variance")
                    current_range = _statistic_value(values, "range")
                    if np.isfinite(current_variance) and np.isfinite(baseline_variance) and baseline_variance != 0.0:
                        row[f"variance_ratio_vs_{baseline_key}"] = float(current_variance / baseline_variance)
                        ci_low, ci_high = _bootstrap_ratio_ci(
                            values,
                            baseline_values,
                            statistic="variance",
                            n_boot=bootstrap_samples,
                            seed=bootstrap_seed
                            + 10_000
                            + metric_idx
                            + (0 if baseline_key == "fixed_subset_1x" else 1_000),
                        )
                        row[f"variance_ratio_vs_{baseline_key}_ci_low"] = ci_low
                        row[f"variance_ratio_vs_{baseline_key}_ci_high"] = ci_high
                    if np.isfinite(current_range) and np.isfinite(baseline_range) and baseline_range != 0.0:
                        row[f"range_ratio_vs_{baseline_key}"] = float(current_range / baseline_range)
                        ci_low, ci_high = _bootstrap_ratio_ci(
                            values,
                            baseline_values,
                            statistic="range",
                            n_boot=bootstrap_samples,
                            seed=bootstrap_seed
                            + 20_000
                            + metric_idx
                            + (0 if baseline_key == "fixed_subset_1x" else 1_000),
                        )
                        row[f"range_ratio_vs_{baseline_key}_ci_low"] = ci_low
                        row[f"range_ratio_vs_{baseline_key}_ci_high"] = ci_high
            rows.append(row)
    return rows


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


def _save_compute_scaling_plot(summary_df: pd.DataFrame, output_path: str, primary_metrics: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    plot_df = summary_df[summary_df["metric"].isin(primary_metrics)].copy()
    if plot_df.empty:
        return

    metric_labels = [metric.removeprefix("lm_eval/mmlu_5shot/") for metric in primary_metrics]
    metric_to_x = {metric: idx for idx, metric in enumerate(primary_metrics)}
    cohort_order = ["baseline_1x", "regmix60m_6b", "olmo3_30m_3b"]
    cohort_labels = {
        "baseline_1x": "60M / 1.2B",
        "regmix60m_6b": "60M / 6B",
        "olmo3_30m_3b": "30M / 3B",
    }
    colors = {
        cohort: cm.get_cmap("RdYlGn_r")(position)
        for cohort, position in zip(cohort_order, (0.15, 0.5, 0.85), strict=True)
    }
    offsets = {
        "baseline_1x": -0.22,
        "regmix60m_6b": 0.0,
        "olmo3_30m_3b": 0.22,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for cohort in cohort_order:
        cohort_df = plot_df[plot_df["cohort"] == cohort]
        x = [metric_to_x[metric] + offsets[cohort] for metric in cohort_df["metric"]]
        y = cohort_df["std"].to_numpy(dtype=np.float64)
        yerr = np.vstack(
            [
                y - cohort_df["std_ci_low"].to_numpy(dtype=np.float64),
                cohort_df["std_ci_high"].to_numpy(dtype=np.float64) - y,
            ]
        )
        axes[0].errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=colors[cohort],
            capsize=4,
            label=cohort_labels[cohort],
        )
    axes[0].set_xticks(range(len(primary_metrics)), metric_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Seed-jitter std")
    axes[0].set_title("MMLU noise by compute ladder")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    ratio_df = plot_df[plot_df["cohort"] == "regmix60m_6b"].copy()
    ratio_x = range(len(primary_metrics))
    ratio_y = ratio_df["variance_ratio_vs_baseline_1x"].to_numpy(dtype=np.float64)
    ratio_yerr = np.vstack(
        [
            ratio_y - ratio_df["variance_ratio_ci_low"].to_numpy(dtype=np.float64),
            ratio_df["variance_ratio_ci_high"].to_numpy(dtype=np.float64) - ratio_y,
        ]
    )
    axes[1].axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[1].errorbar(
        ratio_x,
        ratio_y,
        yerr=ratio_yerr,
        fmt="o",
        color=colors["regmix60m_6b"],
        capsize=4,
    )
    axes[1].set_xticks(range(len(primary_metrics)), metric_labels, rotation=15, ha="right")
    axes[1].set_ylabel("Variance ratio vs 60M / 1.2B")
    axes[1].set_title("60M 6B vs 1.2B variance ratio")
    axes[1].grid(axis="y", alpha=0.3)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with fsspec.open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def _save_fixed_subset_plot(summary_df: pd.DataFrame, output_path: str, primary_metrics: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    plot_df = summary_df[summary_df["metric"].isin(primary_metrics)].copy()
    if plot_df.empty:
        return

    metric_labels = [metric.removeprefix("lm_eval/mmlu_5shot/") for metric in primary_metrics]
    metric_to_x = {metric: idx for idx, metric in enumerate(primary_metrics)}
    cohort_order = ["baseline_1x", "fixed_subset_1x"]
    cohort_labels = {
        "baseline_1x": "Original 60M / 1.2B",
        "fixed_subset_1x": "Fixed subset 60M / 1.2B",
    }
    colors = {
        cohort: cm.get_cmap("RdYlGn_r")(position) for cohort, position in zip(cohort_order, (0.2, 0.8), strict=True)
    }
    offsets = {
        "baseline_1x": -0.12,
        "fixed_subset_1x": 0.12,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for cohort in cohort_order:
        cohort_df = plot_df[plot_df["cohort"] == cohort]
        x = [metric_to_x[metric] + offsets[cohort] for metric in cohort_df["metric"]]
        y = cohort_df["std"].to_numpy(dtype=np.float64)
        yerr = np.vstack(
            [
                y - cohort_df["std_ci_low"].to_numpy(dtype=np.float64),
                cohort_df["std_ci_high"].to_numpy(dtype=np.float64) - y,
            ]
        )
        axes[0].errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=colors[cohort],
            capsize=4,
            label=cohort_labels[cohort],
        )
    axes[0].set_xticks(range(len(primary_metrics)), metric_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Seed-jitter std")
    axes[0].set_title("MMLU noise: original vs fixed subset")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    ratio_df = plot_df[plot_df["cohort"] == "fixed_subset_1x"].copy()
    ratio_x = range(len(primary_metrics))
    ratio_y = ratio_df["variance_ratio_vs_baseline_1x"].to_numpy(dtype=np.float64)
    ratio_yerr = np.vstack(
        [
            ratio_y - ratio_df["variance_ratio_ci_low"].to_numpy(dtype=np.float64),
            ratio_df["variance_ratio_ci_high"].to_numpy(dtype=np.float64) - ratio_y,
        ]
    )
    axes[1].axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[1].errorbar(
        ratio_x,
        ratio_y,
        yerr=ratio_yerr,
        fmt="o",
        color=colors["fixed_subset_1x"],
        capsize=4,
    )
    axes[1].set_xticks(range(len(primary_metrics)), metric_labels, rotation=15, ha="right")
    axes[1].set_ylabel("Variance ratio vs original")
    axes[1].set_title("Fixed-subset variance ratio")
    axes[1].grid(axis="y", alpha=0.3)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with fsspec.open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def _save_panel_vs_noise_plot(summary_df: pd.DataFrame, output_path: str, primary_metrics: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    plot_df = summary_df[summary_df["metric"].isin(primary_metrics)].copy()
    if plot_df.empty:
        return

    metric_labels = [metric.removeprefix("lm_eval/mmlu_5shot/") for metric in primary_metrics]
    cmap = cm.get_cmap("RdYlGn_r")
    colors = {
        "baseline": cmap(0.25),
        "panel": cmap(0.8),
        "gt1": cmap(0.62),
        "gt2": cmap(0.9),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(primary_metrics))
    width = 0.32

    baseline_std = [plot_df.loc[plot_df["metric"] == metric, "baseline_std"].iloc[0] for metric in primary_metrics]
    panel_std = [plot_df.loc[plot_df["metric"] == metric, "panel_std"].iloc[0] for metric in primary_metrics]
    axes[0].bar(x - width / 2, baseline_std, width=width, color=colors["baseline"], label="Fixed-subset noise std")
    axes[0].bar(x + width / 2, panel_std, width=width, color=colors["panel"], label="10-run panel std")
    axes[0].set_xticks(x, metric_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Std")
    axes[0].set_title("Panel spread vs fixed-subset noise")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    frac_gt1 = [
        plot_df.loc[plot_df["metric"] == metric, "frac_pairwise_gt_1x_noise_std"].iloc[0] for metric in primary_metrics
    ]
    frac_gt2 = [
        plot_df.loc[plot_df["metric"] == metric, "frac_pairwise_gt_2x_noise_std"].iloc[0] for metric in primary_metrics
    ]
    axes[1].bar(x - width / 2, frac_gt1, width=width, color=colors["gt1"], label=">|1x noise std|")
    axes[1].bar(x + width / 2, frac_gt2, width=width, color=colors["gt2"], label=">|2x noise std|")
    axes[1].set_xticks(x, metric_labels, rotation=15, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Fraction of pairwise diffs")
    axes[1].set_title("How often mix differences beat the noise floor")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=False)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with fsspec.open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def _save_model_size_noise_plot(summary_df: pd.DataFrame, output_path: str, primary_metrics: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = summary_df[summary_df["metric"].isin(primary_metrics)].copy()
    if plot_df.empty:
        return

    metric_labels = [metric.removeprefix("lm_eval/mmlu_5shot/") for metric in primary_metrics]
    metric_to_x = {metric: idx for idx, metric in enumerate(primary_metrics)}
    cohort_order = ["fixed_subset_1x", "regmix60m_6b", "regmix300m_6b"]
    cohort_labels = {
        "fixed_subset_1x": "60M / 1.2B Fixed Subset",
        "regmix60m_6b": "60M / 6B",
        "regmix300m_6b": "300M / 6B",
    }
    cmap = plt.colormaps["RdYlGn_r"]
    colors = {cohort: cmap(position) for cohort, position in zip(cohort_order, (0.2, 0.5, 0.82), strict=True)}
    offsets = {
        "fixed_subset_1x": -0.22,
        "regmix60m_6b": 0.0,
        "regmix300m_6b": 0.22,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for cohort in cohort_order:
        cohort_df = plot_df[plot_df["cohort"] == cohort]
        x = [metric_to_x[metric] + offsets[cohort] for metric in cohort_df["metric"]]
        y = cohort_df["std"].to_numpy(dtype=np.float64)
        yerr = np.vstack(
            [
                y - cohort_df["std_ci_low"].to_numpy(dtype=np.float64),
                cohort_df["std_ci_high"].to_numpy(dtype=np.float64) - y,
            ]
        )
        axes[0].errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=colors[cohort],
            capsize=4,
            label=cohort_labels[cohort],
        )
    axes[0].set_xticks(range(len(primary_metrics)), metric_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Seed-jitter std")
    axes[0].set_title("MMLU noise by model/compute setting")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    runtime_rows = plot_df.drop_duplicates("cohort").set_index("cohort")
    runtime_x = np.arange(len(cohort_order))
    runtime_y = np.array(
        [runtime_rows.loc[cohort, "active_compute_hours_mean"] for cohort in cohort_order],
        dtype=np.float64,
    )
    runtime_yerr = np.array(
        [runtime_rows.loc[cohort, "active_compute_hours_std"] for cohort in cohort_order],
        dtype=np.float64,
    )
    axes[1].bar(
        runtime_x,
        runtime_y,
        yerr=runtime_yerr,
        color=[colors[cohort] for cohort in cohort_order],
        capsize=4,
    )
    axes[1].set_xticks(runtime_x, [cohort_labels[cohort] for cohort in cohort_order], rotation=15, ha="right")
    axes[1].set_ylabel("Active compute time (hours)")
    axes[1].set_title("Mean active runtime per run")
    axes[1].grid(axis="y", alpha=0.3)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with fsspec.open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def create_compute_scaling_noise_report(config: ComputeScalingNoiseReportConfig) -> None:
    """Compare seed-jitter noise across higher-compute ladders against the 1x reference study."""
    with fsspec.open(str(config.run_manifest_path), "r") as f:
        manifest_payload = json.load(f)
    results_path = os.path.join(str(config.analysis_output_path), RESULTS_CSV)
    merged = pd.read_csv(results_path)
    merged["status"] = merged["status"].fillna("not_found")

    completed_current = merged[(merged["status"] == "completed") & merged["wandb_run_id"].notna()].copy()
    current_traj = _collect_trajectory_rows(
        completed_current,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        objective_metric=config.primary_metrics[0],
    )
    if not current_traj.empty:
        trajectory_meta = completed_current[
            ["run_id", "ladder", "model_family", "experiment_budget", "num_train_steps"]
        ].drop_duplicates("run_id")
        current_traj = current_traj.merge(trajectory_meta, on="run_id", how="left")

    baseline_manifest_payload = json.loads(config.baseline_manifest_json)
    baseline_results = _collect_manifest_results_frame(
        manifest_payload=baseline_manifest_payload,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        metric_prefixes=("eval/", "lm_eval/"),
        extra_metrics=tuple(dict.fromkeys((*config.primary_metrics, *config.secondary_metrics))),
    )
    baseline_completed = baseline_results[
        (baseline_results["status"] == "completed") & baseline_results["wandb_run_id"].notna()
    ].copy()

    summary_rows = _build_compute_scaling_summary_rows(
        current_runs_df=completed_current,
        baseline_runs_df=baseline_completed,
        primary_metrics=config.primary_metrics,
        secondary_metrics=config.secondary_metrics,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    summary_df = pd.DataFrame(summary_rows)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        merged.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, TRAJECTORY_RAW_PARQUET), "wb") as f:
        current_traj.to_parquet(f, index=False)
    with fsspec.open(os.path.join(config.output_path, COMPUTE_SCALING_NOISE_SUMMARY_CSV), "w") as f:
        summary_df.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, COMPUTE_SCALING_NOISE_SUMMARY_JSON), "w") as f:
        json.dump(
            {
                "primary_metrics": list(config.primary_metrics),
                "secondary_metrics": list(config.secondary_metrics),
                "baseline_experiment": baseline_manifest_payload["experiment_name"],
                "rows": summary_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    _save_compute_scaling_plot(
        summary_df,
        os.path.join(config.output_path, MMLU_NOISE_VS_COMPUTE_PNG),
        config.primary_metrics,
    )


def create_fixed_subset_noise_report(config: FixedSubsetNoiseReportConfig) -> None:
    """Compare fixed-subset seed jitter against the original run_00097 seed study."""
    analysis_path = str(config.analysis_output_path)
    seed_runs_path = os.path.join(analysis_path, SEED_RUNS_CSV)
    control_report_path = os.path.join(analysis_path, DETERMINISM_CONTROL_REPORT_JSON)

    current_seed_runs = pd.read_csv(seed_runs_path)
    current_seed_runs = current_seed_runs[
        (current_seed_runs["status"] == "completed") & current_seed_runs["wandb_run_id"].notna()
    ].copy()

    baseline_manifest_payload = json.loads(config.baseline_manifest_json)
    baseline_results = _collect_manifest_results_frame(
        manifest_payload=baseline_manifest_payload,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        metric_prefixes=("eval/", "lm_eval/"),
        extra_metrics=tuple(dict.fromkeys((*config.primary_metrics, *config.secondary_metrics))),
    )
    baseline_completed = baseline_results[
        (baseline_results["status"] == "completed") & baseline_results["wandb_run_id"].notna()
    ].copy()

    summary_rows = _build_fixed_subset_summary_rows(
        current_runs_df=current_seed_runs,
        baseline_runs_df=baseline_completed,
        primary_metrics=config.primary_metrics,
        secondary_metrics=config.secondary_metrics,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    summary_df = pd.DataFrame(summary_rows)

    control_report: dict[str, Any] | None = None
    control_fs, _, _ = fsspec.get_fs_token_paths(control_report_path)
    if control_fs.exists(control_report_path):
        with fsspec.open(control_report_path, "r") as f:
            control_report = json.load(f)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, FIXED_SUBSET_NOISE_SUMMARY_CSV), "w") as f:
        summary_df.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, FIXED_SUBSET_NOISE_SUMMARY_JSON), "w") as f:
        json.dump(
            {
                "primary_metrics": list(config.primary_metrics),
                "secondary_metrics": list(config.secondary_metrics),
                "baseline_experiment": baseline_manifest_payload["experiment_name"],
                "current_seed_runs_path": seed_runs_path,
                "control_report": control_report,
                "rows": summary_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    _save_fixed_subset_plot(
        summary_df,
        os.path.join(config.output_path, MMLU_NOISE_FIXED_SUBSET_VS_ORIGINAL_PNG),
        config.primary_metrics,
    )


def create_panel_vs_noise_report(config: PanelVsNoiseReportConfig) -> None:
    """Compare a fixed-subset observed-run panel against the fixed-subset noise baseline."""
    with fsspec.open(str(config.run_manifest_path), "r") as f:
        manifest_payload = json.load(f)
    results_path = os.path.join(str(config.analysis_output_path), RESULTS_CSV)
    merged = pd.read_csv(results_path)
    merged["status"] = merged["status"].fillna("not_found")
    panel_completed = merged[(merged["status"] == "completed") & merged["wandb_run_id"].notna()].copy()

    baseline_manifest_payload = json.loads(config.baseline_manifest_json)
    baseline_results = _collect_manifest_results_frame(
        manifest_payload=baseline_manifest_payload,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        metric_prefixes=("eval/", "lm_eval/"),
        extra_metrics=tuple(dict.fromkeys((*config.primary_metrics, *config.secondary_metrics))),
    )
    baseline_completed = baseline_results[
        (baseline_results["status"] == "completed") & baseline_results["wandb_run_id"].notna()
    ].copy()

    summary_rows = _build_panel_vs_noise_summary_rows(
        panel_runs_df=panel_completed,
        baseline_runs_df=baseline_completed,
        primary_metrics=config.primary_metrics,
        secondary_metrics=config.secondary_metrics,
    )
    summary_df = pd.DataFrame(summary_rows)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        merged.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, PANEL_VS_NOISE_SUMMARY_CSV), "w") as f:
        summary_df.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, PANEL_VS_NOISE_SUMMARY_JSON), "w") as f:
        json.dump(
            {
                "primary_metrics": list(config.primary_metrics),
                "secondary_metrics": list(config.secondary_metrics),
                "baseline_experiment": baseline_manifest_payload["experiment_name"],
                "rows": summary_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    _save_panel_vs_noise_plot(
        summary_df,
        os.path.join(config.output_path, MMLU_PANEL_VS_NOISE_PNG),
        config.primary_metrics,
    )


def create_model_size_noise_report(config: ModelSizeNoiseReportConfig) -> None:
    """Compare a larger-model fixed-subset study against the 60M fixed-subset baselines."""
    with fsspec.open(str(config.run_manifest_path), "r") as f:
        manifest_payload = json.load(f)
    results_path = os.path.join(str(config.analysis_output_path), RESULTS_CSV)
    merged = pd.read_csv(results_path)
    merged["status"] = merged["status"].fillna("not_found")

    current_completed = merged[(merged["status"] == "completed") & merged["wandb_run_id"].notna()].copy()
    current_traj = _collect_trajectory_rows(
        current_completed,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        objective_metric=config.primary_metrics[0],
    )

    fixed_subset_manifest_payload = json.loads(config.fixed_subset_baseline_manifest_json)
    fixed_subset_results = _collect_manifest_results_frame(
        manifest_payload=fixed_subset_manifest_payload,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        metric_prefixes=("eval/", "lm_eval/"),
        extra_metrics=tuple(dict.fromkeys((*config.primary_metrics, *config.secondary_metrics))),
    )
    fixed_subset_completed = fixed_subset_results[
        (fixed_subset_results["status"] == "completed") & fixed_subset_results["wandb_run_id"].notna()
    ].copy()

    compute_manifest_payload = json.loads(config.compute_baseline_manifest_json)
    compute_results = _collect_manifest_results_frame(
        manifest_payload=compute_manifest_payload,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        metric_prefixes=("eval/", "lm_eval/"),
        extra_metrics=tuple(dict.fromkeys((*config.primary_metrics, *config.secondary_metrics))),
    )
    compute_completed = compute_results[
        (compute_results["status"] == "completed") & compute_results["wandb_run_id"].notna()
    ].copy()

    runtime_by_cohort = {
        "fixed_subset_1x": _active_compute_hours_summary(
            fixed_subset_completed["wandb_run_id"].astype(str).tolist(),
            wandb_entity=config.wandb_entity,
            wandb_project=config.wandb_project,
        ),
        "regmix60m_6b": _active_compute_hours_summary(
            compute_completed["wandb_run_id"].astype(str).tolist(),
            wandb_entity=config.wandb_entity,
            wandb_project=config.wandb_project,
        ),
        "regmix300m_6b": _active_compute_hours_summary(
            current_completed["wandb_run_id"].astype(str).tolist(),
            wandb_entity=config.wandb_entity,
            wandb_project=config.wandb_project,
        ),
    }

    summary_rows = _build_model_size_noise_summary_rows(
        current_runs_df=current_completed,
        fixed_subset_baseline_df=fixed_subset_completed,
        compute_baseline_df=compute_completed,
        runtime_by_cohort=runtime_by_cohort,
        primary_metrics=config.primary_metrics,
        secondary_metrics=config.secondary_metrics,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    summary_df = pd.DataFrame(summary_rows)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        merged.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, TRAJECTORY_RAW_PARQUET), "wb") as f:
        current_traj.to_parquet(f, index=False)
    with fsspec.open(os.path.join(config.output_path, NOISE_SUMMARY_CSV), "w") as f:
        summary_df.to_csv(f, index=False)
    with fsspec.open(os.path.join(config.output_path, NOISE_SUMMARY_JSON), "w") as f:
        json.dump(
            {
                "primary_metrics": list(config.primary_metrics),
                "secondary_metrics": list(config.secondary_metrics),
                "fixed_subset_baseline_experiment": fixed_subset_manifest_payload["experiment_name"],
                "compute_baseline_experiment": compute_manifest_payload["experiment_name"],
                "rows": summary_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    _save_model_size_noise_plot(
        summary_df,
        os.path.join(config.output_path, MMLU_NOISE_VS_RUNTIME_300M_6B_PNG),
        config.primary_metrics,
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


def create_compute_scaling_noise_report_step(
    *,
    name_prefix: str,
    run_manifest_step: ExecutorStep,
    analysis_step: ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
    baseline_manifest_json: str,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...] = (),
) -> ExecutorStep:
    """Create an ExecutorStep for the run_00097 compute-scaling noise report."""
    return ExecutorStep(
        name=f"{name_prefix}/compute_scaling_noise_report",
        description="Compare MMLU seed-jitter noise across higher-compute run_00097 ladders",
        fn=create_compute_scaling_noise_report,
        config=ComputeScalingNoiseReportConfig(
            output_path=this_output_path(),
            run_manifest_path=output_path_of(run_manifest_step, RUN_MANIFEST_FILE),
            analysis_output_path=output_path_of(analysis_step),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            baseline_manifest_json=baseline_manifest_json,
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
        ),
    )


def create_fixed_subset_noise_report_step(
    *,
    name_prefix: str,
    analysis_step: ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
    baseline_manifest_json: str,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...] = (),
) -> ExecutorStep:
    """Create an ExecutorStep for the fixed-subset vs original seed-study noise report."""
    return ExecutorStep(
        name=f"{name_prefix}/fixed_subset_noise_report",
        description="Compare fixed-subset MMLU seed jitter against the original run_00097 study",
        fn=create_fixed_subset_noise_report,
        config=FixedSubsetNoiseReportConfig(
            output_path=this_output_path(),
            analysis_output_path=output_path_of(analysis_step),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            baseline_manifest_json=baseline_manifest_json,
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
        ),
    )


def create_panel_vs_noise_report_step(
    *,
    name_prefix: str,
    run_manifest_step: ExecutorStep,
    analysis_step: ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
    baseline_manifest_json: str,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...] = (),
) -> ExecutorStep:
    """Create an ExecutorStep for comparing an observed fixed-subset panel against the fixed-subset noise floor."""
    return ExecutorStep(
        name=f"{name_prefix}/panel_vs_noise_report",
        description="Compare observed-mix panel spread against fixed-subset MMLU noise",
        fn=create_panel_vs_noise_report,
        config=PanelVsNoiseReportConfig(
            output_path=this_output_path(),
            run_manifest_path=output_path_of(run_manifest_step, RUN_MANIFEST_FILE),
            analysis_output_path=output_path_of(analysis_step),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            baseline_manifest_json=baseline_manifest_json,
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
        ),
    )


def create_model_size_noise_report_step(
    *,
    name_prefix: str,
    run_manifest_step: ExecutorStep,
    analysis_step: ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
    fixed_subset_baseline_manifest_json: str,
    compute_baseline_manifest_json: str,
    primary_metrics: tuple[str, ...],
    secondary_metrics: tuple[str, ...] = (),
) -> ExecutorStep:
    """Create an ExecutorStep for model-size noise comparisons against 60M baselines."""
    return ExecutorStep(
        name=f"{name_prefix}/noise_report",
        description="Compare 300M fixed-subset MMLU noise against 60M fixed-subset baselines",
        fn=create_model_size_noise_report,
        config=ModelSizeNoiseReportConfig(
            output_path=this_output_path(),
            run_manifest_path=output_path_of(run_manifest_step, RUN_MANIFEST_FILE),
            analysis_output_path=output_path_of(analysis_step),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            fixed_subset_baseline_manifest_json=fixed_subset_baseline_manifest_json,
            compute_baseline_manifest_json=compute_baseline_manifest_json,
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
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


def create_fit_dataset_export_step(
    *,
    name_prefix: str,
    run_manifest_step: ExecutorStep,
    analysis_step: ExecutorStep,
) -> ExecutorStep:
    """Create an ExecutorStep that exports a completed long-form fit dataset CSV."""
    return ExecutorStep(
        name=f"{name_prefix}/fit_dataset_export",
        description="Export a completed long-form fit dataset for surrogate fitting",
        fn=create_fit_dataset_export,
        config=FitDatasetExportConfig(
            output_path=this_output_path(),
            run_manifest_path=output_path_of(run_manifest_step, RUN_MANIFEST_FILE),
            analysis_output_path=output_path_of(analysis_step),
        ),
    )
