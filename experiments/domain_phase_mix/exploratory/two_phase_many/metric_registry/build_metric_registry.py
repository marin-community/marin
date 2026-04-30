# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Build the two-phase-many metric registry.

The registry treats metric values as facts with explicit provenance. Generated
wide tables are views over the long fact table, not the source of truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

import fsspec
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import summarize_eval_signal_to_noise as snr
from experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline import (
    build_run_spec as build_stratified_run_spec,
)
from experiments.domain_phase_mix.qsplit240_replay import build_qsplit240_replay_run_specs
from experiments.domain_phase_mix.scaling_study_recipes import (
    ScalingStudyCell,
    ScalingStudyPath,
    build_strong_tier_cells,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_original_qsplit240_with_core_baselines

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent

RUNS_CSV = SCRIPT_DIR / "runs.csv"
EVAL_ARTIFACTS_CSV = SCRIPT_DIR / "eval_artifacts.csv"
METRICS_ALL_SOURCES_CSV = SCRIPT_DIR / "metrics_all_sources_long.csv.gz"
METRICS_LONG_CSV = SCRIPT_DIR / "metrics_long.csv.gz"
METRICS_WIDE_CSV = SCRIPT_DIR / "metrics_wide.csv"
METRIC_CONFLICTS_CSV = SCRIPT_DIR / "metric_conflicts.csv.gz"
COVERAGE_CSV = SCRIPT_DIR / "coverage.csv"
BACKFILLS_CSV = SCRIPT_DIR / "backfills.csv"
MANUAL_BACKFILLS_CSV = SCRIPT_DIR / "manual_backfills.csv"
SUMMARY_JSON = SCRIPT_DIR / "summary.json"
DEFAULT_CHECKPOINT_BUCKET = "marin-us-east5"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
STRATIFIED_RUN_ID = 3
STRATIFIED_RUN_NAME = "baseline_stratified"
QSPLIT240_300M_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
STRATIFIED_300M_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b"
STRICT_300M_SUCCESS_CSV = TWO_PHASE_MANY_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"
RUN_REGISTRY_LOGICAL_RUNS_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"
SWARM_REFERENCE_60M_COHORT = "swarm_reference_60m"
STRONG_TIER_FAMILIES = {
    "strong_tier_qsplit_representative12",
    "strong_tier_stratified",
    "strong_tier_qsplit_baselines3_holdout",
    "strong_tier_stratified_holdout",
}
SINGLE_PHASE_EXPOSURE_AVERAGE_FAMILY = "single_phase_exposure_average_60m_1p2b"
SINGLE_PHASE_GRP_NO_L2_FAMILY = "single_phase_grp_no_l2_60m_1p2b"
SINGLE_PHASE_FAMILIES = {
    SINGLE_PHASE_EXPOSURE_AVERAGE_FAMILY,
    SINGLE_PHASE_GRP_NO_L2_FAMILY,
}
STRONG_TIER_COHORT_BY_PATH = {
    ScalingStudyPath.QSPLIT_REPRESENTATIVE12.value: "representative12",
    ScalingStudyPath.STRATIFIED.value: "stratified",
    ScalingStudyPath.QSPLIT_BASELINES3_HOLDOUT.value: "holdout_baselines3",
    ScalingStudyPath.STRATIFIED_HOLDOUT.value: "holdout_stratified",
}
LM_EVAL_RESULTS_GLOB = "lm_eval_artifacts/lm_eval_harness_results*.json"
LM_EVAL_METRIC_SUFFIX = ",none"

LOCAL_SOURCE_CSVS = (
    (
        "local_two_phase_many",
        TWO_PHASE_MANY_DIR / "two_phase_many.csv",
        "60m_1p2b",
        "signal",
        50,
        "local_wandb_collect",
    ),
    (
        "local_two_phase_many_with_mmlu_sl_verb",
        TWO_PHASE_MANY_DIR / "two_phase_many_with_mmlu_sl_verb.csv",
        "60m_1p2b",
        "signal",
        60,
        "local_wandb_collect",
    ),
    (
        "local_two_phase_many_all_60m_1p2b",
        TWO_PHASE_MANY_DIR / "two_phase_many_all_60m_1p2b.csv",
        "60m_1p2b",
        "signal",
        70,
        "local_wandb_collect",
    ),
    (
        "local_qsplit240_300m_6b_partial_results",
        TWO_PHASE_MANY_DIR / "qsplit240_300m_6b_partial_results.csv",
        "300m_6b",
        "original_swarm_300m",
        55,
        "local_wandb_collect",
    ),
    (
        "local_two_phase_many_all_60m_1p2b_swarm_reference",
        TWO_PHASE_MANY_DIR / "two_phase_many_all_60m_1p2b.csv",
        "60m_1p2b",
        SWARM_REFERENCE_60M_COHORT,
        75,
        "local_wandb_collect",
    ),
)
LOCAL_COLLECTED_EVAL_CSVS = (
    (
        "local_300m_gsm8k_humaneval_completion",
        SCRIPT_DIR / "300m_gsm8k_humaneval_completion" / "300m_gsm8k_humaneval_eval_results.csv",
        "300m_6b",
        "signal",
        120,
        "local_collected_downstream_eval",
    ),
)

METRIC_PREFIXES = ("eval/", "lm_eval/")
AMBIGUOUS_METRIC_PREFIXES = ("lm_eval/averages/",)
WEIGHT_PREFIXES = ("phase_0_", "phase_1_")
KNOWN_ID_COLUMNS = (
    "registry_run_key",
    "scale",
    "cohort",
    "source_cohort",
    "study_path",
    "study_panel",
    "study_cell_status",
    "run_id",
    "run_name",
    "source_run_name",
    "source_experiment",
    "source_name_prefix",
    "wandb_run_id",
    "wandb_run_name",
    "checkpoint_root",
    "status",
    "experiment_budget",
    "target_budget",
    "target_budget_multiplier",
    "num_train_steps",
    "model_family",
    "trainer_seed",
    "data_seed",
    "candidate_source_experiment",
    "candidate_run_id",
    "candidate_run_name",
    "source_run_id",
    "source_two_phase_experiment",
    "single_phase_strategy",
    "priority_rank",
    "priority_tier",
    "phase_tv",
    "source_60m_bpb",
    "source_60m_rank",
    "source_100m_bpb",
    "source_100m_rank",
    "rank_shift",
    "has_objective_metric_value",
    "has_checkpoint_root",
    "has_checkpoint_backed_objective",
    "analysis_attempt_root",
    "analysis_executor_status",
    "max_checkpoint_step",
    "target_final_checkpoint_step",
    "reached_target_step",
    "is_perplexity_ready",
)
RUN_FIRST_VALUE_COLUMNS = (
    "run_id",
    "run_name",
    "scale",
    "cohort",
    "source_cohort",
    "study_path",
    "study_panel",
    "study_cell_status",
    "source_run_name",
    "source_experiment",
    "source_name_prefix",
    "wandb_run_id",
    "wandb_run_name",
    "checkpoint_root",
    "status",
    "experiment_budget",
    "target_budget",
    "target_budget_multiplier",
    "num_train_steps",
    "model_family",
    "trainer_seed",
    "data_seed",
    "candidate_source_experiment",
    "candidate_run_id",
    "candidate_run_name",
    "source_run_id",
    "source_two_phase_experiment",
    "single_phase_strategy",
    "priority_rank",
    "priority_tier",
    "phase_tv",
    "source_60m_bpb",
    "source_60m_rank",
    "source_100m_bpb",
    "source_100m_rank",
    "rank_shift",
    "has_objective_metric_value",
    "has_checkpoint_root",
    "has_checkpoint_backed_objective",
    "analysis_attempt_root",
    "analysis_executor_status",
    "max_checkpoint_step",
    "target_final_checkpoint_step",
    "reached_target_step",
    "is_perplexity_ready",
)
FEWSHOT_TASK_RE = re.compile(r"^(?P<task>.+)_(?P<num_fewshot>[0-9]+)shot$")
CONFLICT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SourceFrame:
    """One metric source loaded into a frame."""

    source_name: str
    source_uri: str
    source_kind: str
    scale: str
    default_cohort: str
    source_priority: int
    frame: pd.DataFrame
    default_source_experiment: str | None = None
    default_status: str | None = None


def canonicalize_metric_key(metric_key: str) -> dict[str, Any]:
    """Return canonical metric metadata for a raw metric column."""
    if metric_key.startswith("lm_eval/"):
        rest = metric_key.removeprefix("lm_eval/")
        task_key, metric_name = rest.rsplit("/", 1)
        match = FEWSHOT_TASK_RE.match(task_key)
        if match is None:
            task = task_key
            num_fewshot = pd.NA
            canonical = metric_key
        else:
            task = match.group("task")
            num_fewshot = int(match.group("num_fewshot"))
            canonical = f"lm_eval/{task}_{num_fewshot}shot/{metric_name}"
        return {
            "suite": "lm_eval",
            "task": task,
            "num_fewshot": num_fewshot,
            "metric": metric_name,
            "canonical_metric_key": canonical,
            "higher_is_better": _higher_is_better(metric_name),
        }

    if metric_key.startswith("eval/"):
        rest = metric_key.removeprefix("eval/")
        parts = rest.split("/")
        metric_name = parts[-1]
        task = "/".join(parts[:-1]) if len(parts) > 1 else "global"
        return {
            "suite": "eval",
            "task": task,
            "num_fewshot": pd.NA,
            "metric": metric_name,
            "canonical_metric_key": metric_key,
            "higher_is_better": _higher_is_better(metric_name),
        }

    raise ValueError(f"Unsupported metric key: {metric_key}")


def _higher_is_better(metric_name: str) -> bool:
    lower = metric_name.lower()
    if any(token in lower for token in ("bpb", "loss", "perplexity")):
        return False
    if any(token in lower for token in ("acc", "logprob", "prob", "f1", "exact_match")):
        return True
    return True


def _read_local_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_strict_300m_success_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    qsplit_run_ids = {run.run_name: run.run_id for run in load_original_qsplit240_with_core_baselines()}
    qsplit_run_ids[STRATIFIED_RUN_NAME] = STRATIFIED_RUN_ID
    out = frame.rename(columns={"bpb_300m_6b": OBJECTIVE_METRIC, "status_300m_6b": "status"}).copy()
    out["status"] = out["status"].map({"SUCCESS": "completed"}).fillna(out["status"])
    out["run_id"] = out["run_name"].map(qsplit_run_ids)
    out["source_experiment"] = QSPLIT240_300M_SOURCE_EXPERIMENT
    out.loc[out["run_name"].eq(STRATIFIED_RUN_NAME), "source_experiment"] = STRATIFIED_300M_SOURCE_EXPERIMENT
    out["cohort"] = "signal"
    out = _hydrate_checkpoint_eval_metrics(out)
    return out


def _read_checkpoint_eval_metrics(checkpoint_root: str) -> dict[str, float]:
    metrics_path = checkpoint_root.rstrip("/") + "/checkpoints/eval_metrics.jsonl"
    with fsspec.open(metrics_path, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{metrics_path} is empty")

    payload = json.loads(lines[-1])
    metrics = {
        key: float(value)
        for key, value in payload.items()
        if key.startswith(METRIC_PREFIXES) and isinstance(value, int | float)
    }
    tracker_metrics_path = checkpoint_root.rstrip("/") + "/tracker_metrics.jsonl"
    try:
        with fsspec.open(tracker_metrics_path, "rt") as f:
            tracker_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        tracker_lines = []
    if tracker_lines:
        tracker_payload = json.loads(tracker_lines[-1])
        summary = tracker_payload.get("summary", {})
        if isinstance(summary, dict):
            for key, value in summary.items():
                if key.startswith(METRIC_PREFIXES) and isinstance(value, int | float):
                    metrics.setdefault(key, float(value))
    for key, value in _read_lm_eval_harness_metrics(checkpoint_root).items():
        metrics.setdefault(key, value)
    return metrics


def _lm_eval_artifact_step(path: str) -> int:
    match = re.search(r"lm_eval_harness_results\.(\d+)\.json$", path)
    if match is None:
        return -1
    return int(match.group(1))


def _read_lm_eval_harness_metrics(checkpoint_root: str) -> dict[str, float]:
    pattern = checkpoint_root.rstrip("/") + f"/{LM_EVAL_RESULTS_GLOB}"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    matches = [match if str(match).startswith("gs://") else f"gs://{match}" for match in fs.glob(pattern)]
    if not matches:
        return {}

    best_path = max(matches, key=lambda path: (_lm_eval_artifact_step(path), path))
    with fsspec.open(best_path, "rt") as f:
        payload = json.load(f)

    results = payload.get("results", {})
    if not isinstance(results, dict):
        return {}

    metrics: dict[str, float] = {}
    for task_key, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_key, value in task_metrics.items():
            if not metric_key.endswith(LM_EVAL_METRIC_SUFFIX):
                continue
            metric_name = metric_key.removesuffix(LM_EVAL_METRIC_SUFFIX)
            if metric_name.endswith("_stderr"):
                continue
            if not isinstance(value, int | float):
                continue
            metrics[f"lm_eval/{task_key}/{metric_name}"] = float(value)
    return metrics


def _hydrate_checkpoint_eval_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if "checkpoint_root" not in frame.columns:
        return frame

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        out = row.to_dict()
        checkpoint_root = row.get("checkpoint_root")
        if isinstance(checkpoint_root, str) and checkpoint_root.strip():
            out.update(_read_checkpoint_eval_metrics(checkpoint_root))
        rows.append(out)
    return pd.DataFrame.from_records(rows)


def _read_standard_noise_frame() -> pd.DataFrame:
    path = snr.RUN00097_STANDARD_BACKFILL_RESULTS_CSV
    if path.exists():
        return pd.read_csv(path)
    return snr._read_gcs_csv(snr.RUN00097_STANDARD_RESULTS_URI)


def _source_frames(*, include_gcs: bool) -> list[SourceFrame]:
    sources: list[SourceFrame] = []
    for source_name, path, scale, default_cohort, priority, source_kind in LOCAL_SOURCE_CSVS:
        if not path.exists():
            continue
        sources.append(
            SourceFrame(
                source_name=source_name,
                source_uri=str(path),
                source_kind=source_kind,
                scale=scale,
                default_cohort=default_cohort,
                source_priority=priority,
                frame=_read_local_csv(path),
                default_source_experiment=(
                    "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
                    if source_name == "local_qsplit240_300m_6b_partial_results"
                    else None
                ),
            )
        )

    for source_name, path, scale, default_cohort, priority, source_kind in LOCAL_COLLECTED_EVAL_CSVS:
        if not path.exists():
            continue
        sources.append(
            SourceFrame(
                source_name=source_name,
                source_uri=str(path),
                source_kind=source_kind,
                scale=scale,
                default_cohort=default_cohort,
                source_priority=priority,
                frame=_read_local_csv(path),
                default_status="completed",
            )
        )

    if STRICT_300M_SUCCESS_CSV.exists():
        sources.append(
            SourceFrame(
                source_name="gcs_qsplit240_300m_6b_strict_success",
                source_uri=str(STRICT_300M_SUCCESS_CSV),
                source_kind="gcs_checkpoint_eval_metrics",
                scale="300m_6b",
                default_cohort="signal",
                source_priority=90,
                frame=_read_strict_300m_success_csv(STRICT_300M_SUCCESS_CSV),
                default_status="completed",
            )
        )

    sources.extend(_strong_tier_source_frames())
    sources.extend(_single_phase_exposure_average_source_frames())

    if not include_gcs:
        return sources

    sources.extend(
        [
            SourceFrame(
                source_name="gcs_qsplit240_olmo_base_easy_overlap",
                source_uri=snr.QSPLIT240_OVERLAP_RESULTS_GLOB,
                source_kind="gcs_collect_results",
                scale="60m_1p2b",
                default_cohort="signal",
                source_priority=100,
                frame=snr._load_overlap_signal_frame(),
                default_status="completed",
            ),
            SourceFrame(
                source_name="gcs_run00097_olmo_base_easy_overlap_seed_noise",
                source_uri=snr.RUN00097_OVERLAP_RESULTS_URI,
                source_kind="gcs_collect_results",
                scale="60m_1p2b",
                default_cohort="seed_sweep",
                source_priority=100,
                frame=snr._read_gcs_csv(snr.RUN00097_OVERLAP_RESULTS_URI),
                default_source_experiment="pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study",
                default_status="completed",
            ),
            SourceFrame(
                source_name="gcs_qsplit240_mmlu_sl_verb",
                source_uri=snr.QSPLIT240_SL_VERB_RESULTS_URI,
                source_kind="gcs_collect_results",
                scale="60m_1p2b",
                default_cohort="signal",
                source_priority=100,
                frame=snr._read_gcs_csv(snr.QSPLIT240_SL_VERB_RESULTS_URI),
                default_status="completed",
            ),
            SourceFrame(
                source_name="gcs_run00097_mmlu_sl_verb_seed_noise",
                source_uri=snr.RUN00097_SL_VERB_RESULTS_URI,
                source_kind="gcs_collect_results",
                scale="60m_1p2b",
                default_cohort="seed_sweep",
                source_priority=100,
                frame=snr._read_gcs_csv(snr.RUN00097_SL_VERB_RESULTS_URI),
                default_source_experiment="pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study",
                default_status="completed",
            ),
            SourceFrame(
                source_name="standard_run00097_seed_noise",
                source_uri=str(
                    snr.RUN00097_STANDARD_BACKFILL_RESULTS_CSV
                    if snr.RUN00097_STANDARD_BACKFILL_RESULTS_CSV.exists()
                    else snr.RUN00097_STANDARD_RESULTS_URI
                ),
                source_kind="collect_results",
                scale="60m_1p2b",
                default_cohort="seed_sweep",
                source_priority=95,
                frame=_read_standard_noise_frame(),
                default_source_experiment="pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study",
                default_status="completed",
            ),
        ]
    )
    return sources


def _strong_tier_family(cell: ScalingStudyCell) -> str:
    if cell.path == ScalingStudyPath.QSPLIT_REPRESENTATIVE12:
        return "strong_tier_qsplit_representative12"
    if cell.path == ScalingStudyPath.STRATIFIED:
        return "strong_tier_stratified"
    if cell.path == ScalingStudyPath.QSPLIT_BASELINES3_HOLDOUT:
        return "strong_tier_qsplit_baselines3_holdout"
    if cell.path == ScalingStudyPath.STRATIFIED_HOLDOUT:
        return "strong_tier_stratified_holdout"
    raise ValueError(f"Unsupported strong-tier cell path: {cell.path!r}")


def _strong_tier_run_specs(cell: ScalingStudyCell) -> list[dict[str, Any]]:
    if cell.path in {ScalingStudyPath.QSPLIT_REPRESENTATIVE12, ScalingStudyPath.QSPLIT_BASELINES3_HOLDOUT}:
        return [
            spec.__dict__
            for spec in build_qsplit240_replay_run_specs(
                cohort=cell.cohort,
                model_family=cell.model_family,
                experiment_budget=cell.experiment_budget,
                target_budget=cell.target_budget,
                target_budget_multiplier=cell.target_budget_multiplier,
                num_train_steps=cell.num_train_steps,
                panel=cell.panel or "",
            )
        ]
    return [
        build_stratified_run_spec(
            scale=cell.scale,
            experiment_budget=cell.experiment_budget,
            target_budget=cell.target_budget,
            target_budget_multiplier=cell.target_budget_multiplier,
            cohort=cell.cohort,
        ).__dict__
    ]


def _flatten_phase_weights(phase_weights: dict[str, dict[str, float]]) -> dict[str, float]:
    rows: dict[str, float] = {}
    for phase_name, domain_weights in phase_weights.items():
        for domain_name, weight in domain_weights.items():
            rows[f"{phase_name}_{domain_name}"] = float(weight)
    return rows


def _strong_tier_expected_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in build_strong_tier_cells():
        family = _strong_tier_family(cell)
        cohort = STRONG_TIER_COHORT_BY_PATH[cell.path.value]
        for run_spec in _strong_tier_run_specs(cell):
            rows.append(
                {
                    "registry_id": f"{family}:{cell.name_prefix}:{run_spec['run_name']}",
                    "family": family,
                    "scale": cell.scale.value,
                    "cohort": cohort,
                    "source_cohort": cell.cohort,
                    "study_path": cell.path.value,
                    "study_panel": cell.panel,
                    "study_cell_status": cell.status.value,
                    "source_experiment": cell.name_prefix,
                    "source_name_prefix": cell.source_name_prefix,
                    "run_id": int(run_spec["run_id"]),
                    "run_name": str(run_spec["run_name"]),
                    "candidate_run_id": run_spec.get("candidate_run_id"),
                    "candidate_run_name": run_spec.get("candidate_run_name"),
                    "candidate_source_experiment": run_spec.get("candidate_source_experiment"),
                    "model_family": cell.model_family,
                    "experiment_budget": cell.experiment_budget,
                    "target_budget": cell.target_budget,
                    "target_budget_multiplier": cell.target_budget_multiplier,
                    "num_train_steps": cell.num_train_steps,
                    **_flatten_phase_weights(run_spec["phase_weights"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _strong_tier_source_frames() -> list[SourceFrame]:
    if not RUN_REGISTRY_LOGICAL_RUNS_CSV.exists():
        return []

    logical_runs = pd.read_csv(RUN_REGISTRY_LOGICAL_RUNS_CSV, low_memory=False)
    logical_runs = logical_runs.loc[logical_runs["family"].isin(STRONG_TIER_FAMILIES)].copy()
    if logical_runs.empty:
        return []

    expected = _strong_tier_expected_frame()
    merge_columns = [
        "registry_id",
        "family",
        "source_experiment",
        "run_id",
        "run_name",
        "logical_status",
        "checkpoint_root",
        "wandb_run_id",
        "source_status",
        "has_objective_metric_value",
        "has_checkpoint_root",
        "has_checkpoint_backed_objective",
        "analysis_attempt_root",
        "analysis_executor_status",
        "max_checkpoint_step",
        "target_final_checkpoint_step",
        "reached_target_step",
        "is_perplexity_ready",
    ]
    merged = expected.merge(
        logical_runs[merge_columns],
        on=["registry_id", "family", "run_id", "run_name"],
        how="left",
        suffixes=("", "_logical"),
    )
    merged["source_experiment"] = merged["source_experiment_logical"].combine_first(merged["source_experiment"])
    merged["status"] = merged["logical_status"].fillna("planned")
    merged = merged.drop(columns=["source_experiment_logical", "logical_status"])
    merged = _hydrate_checkpoint_eval_metrics(merged)

    sources: list[SourceFrame] = []
    for (scale, cohort), group in merged.groupby(["scale", "cohort"], sort=True):
        sources.append(
            SourceFrame(
                source_name=f"run_registry_{cohort}_{scale}",
                source_uri=str(RUN_REGISTRY_LOGICAL_RUNS_CSV),
                source_kind="run_registry_checkpoint_metrics",
                scale=str(scale),
                default_cohort=str(cohort),
                source_priority=95,
                frame=group.reset_index(drop=True),
            )
        )
    return sources


def _single_phase_exposure_average_source_frames() -> list[SourceFrame]:
    if not RUN_REGISTRY_LOGICAL_RUNS_CSV.exists():
        return []

    logical_runs = pd.read_csv(RUN_REGISTRY_LOGICAL_RUNS_CSV, low_memory=False)
    frame = logical_runs.loc[logical_runs["family"].isin(SINGLE_PHASE_FAMILIES)].copy()
    if frame.empty:
        return []

    frame["status"] = frame["logical_status"].fillna("planned")
    frame = _hydrate_checkpoint_eval_metrics(frame)
    sources: list[SourceFrame] = []
    for family, group in frame.groupby("family", sort=True):
        sources.append(
            SourceFrame(
                source_name=f"run_registry_{family}",
                source_uri=str(RUN_REGISTRY_LOGICAL_RUNS_CSV),
                source_kind="run_registry_checkpoint_metrics",
                scale="60m_1p2b",
                default_cohort=str(group["cohort"].dropna().iloc[0]) if group["cohort"].notna().any() else family,
                source_priority=95,
                frame=group.reset_index(drop=True),
            )
        )
    return sources


def _wandb_run_id_from_checkpoint_root(checkpoint_root: object) -> str | None:
    if not isinstance(checkpoint_root, str) or checkpoint_root.strip() == "":
        return None
    return checkpoint_root.rstrip("/").rsplit("/", 1)[-1]


def _checkpoint_root_from_source(row: pd.Series) -> str | None:
    source_experiment = row.get("source_experiment")
    wandb_run_id = row.get("wandb_run_id")
    if not isinstance(source_experiment, str) or source_experiment.strip() == "":
        return None
    if not isinstance(wandb_run_id, str) or wandb_run_id.strip() == "":
        return None
    return f"gs://{DEFAULT_CHECKPOINT_BUCKET}/checkpoints/{source_experiment.strip('/')}/{wandb_run_id}"


def _normalized_source_frame(source: SourceFrame) -> pd.DataFrame:
    frame = source.frame.copy()
    if source.default_source_experiment is not None:
        if "source_experiment" not in frame.columns:
            frame["source_experiment"] = source.default_source_experiment
        else:
            frame["source_experiment"] = frame["source_experiment"].fillna(source.default_source_experiment)
    if source.default_status is not None:
        if "status" not in frame.columns:
            frame["status"] = source.default_status
        else:
            frame["status"] = frame["status"].fillna(source.default_status)
    if "checkpoint_root" in frame.columns:
        derived_wandb_run_id = frame["checkpoint_root"].map(_wandb_run_id_from_checkpoint_root)
        if "wandb_run_id" not in frame.columns:
            frame["wandb_run_id"] = derived_wandb_run_id
        else:
            frame["wandb_run_id"] = frame["wandb_run_id"].fillna(derived_wandb_run_id)
    if "source_experiment" in frame.columns and "wandb_run_id" in frame.columns:
        derived_checkpoint_root = frame.apply(_checkpoint_root_from_source, axis=1)
        if "checkpoint_root" not in frame.columns:
            frame["checkpoint_root"] = derived_checkpoint_root
        else:
            frame["checkpoint_root"] = frame["checkpoint_root"].fillna(derived_checkpoint_root)
    return frame


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(METRIC_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[column])
        and not column.startswith(AMBIGUOUS_METRIC_PREFIXES)
    ]


def _weight_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(WEIGHT_PREFIXES)]


def _source_cohort_series(source: SourceFrame) -> pd.Series:
    frame = _normalized_source_frame(source)
    if "cohort" not in frame.columns:
        return pd.Series([source.default_cohort] * len(frame), index=frame.index)
    return frame["cohort"].fillna(source.default_cohort).astype(str)


def _normalized_cohort(source: SourceFrame) -> pd.Series:
    source_cohort = _source_cohort_series(source)
    normalized = source_cohort.copy()
    normalized.loc[normalized.str.startswith("original_swarm")] = "signal"
    return normalized


def _run_key(scale: str, cohort: pd.Series, source_experiment: pd.Series, run_name: pd.Series) -> pd.Series:
    normalized_source = source_experiment.fillna("<missing_source_experiment>").astype(str)
    return scale + ":" + cohort.astype(str) + ":" + normalized_source + ":" + run_name.astype(str)


def _artifact_row(source: SourceFrame) -> dict[str, Any]:
    frame = _normalized_source_frame(source)
    metric_columns = _metric_columns(frame)
    return {
        "source_name": source.source_name,
        "source_uri": source.source_uri,
        "source_kind": source.source_kind,
        "scale": source.scale,
        "default_cohort": source.default_cohort,
        "source_priority": source.source_priority,
        "row_count": len(frame),
        "metric_column_count": len(metric_columns),
        "non_null_metric_value_count": int(frame[metric_columns].notna().sum().sum()) if metric_columns else 0,
        "weight_column_count": len(_weight_columns(frame)),
    }


def _runs_from_source(source: SourceFrame) -> pd.DataFrame:
    source_frame = _normalized_source_frame(source)
    if "run_name" not in source_frame.columns:
        raise ValueError(f"{source.source_name} is missing run_name")
    source_cohort = _source_cohort_series(source)
    cohort = _normalized_cohort(source)
    rows = source_frame.copy()
    rows["scale"] = source.scale
    rows["cohort"] = cohort
    rows["source_cohort"] = source_cohort
    if "source_experiment" not in rows.columns:
        rows["source_experiment"] = pd.NA
    rows["registry_run_key"] = _run_key(source.scale, cohort, rows["source_experiment"], rows["run_name"])
    rows["source_name"] = source.source_name
    rows["source_priority"] = source.source_priority

    keep_columns = [
        column
        for column in (*KNOWN_ID_COLUMNS, "source_name", "source_priority", *_weight_columns(rows))
        if column in rows.columns
    ]
    return rows[keep_columns].copy()


def _metrics_from_source(source: SourceFrame) -> pd.DataFrame:
    source_frame = _normalized_source_frame(source)
    metric_columns = _metric_columns(source_frame)
    if not metric_columns:
        return pd.DataFrame()
    source_cohort = _source_cohort_series(source)
    cohort = _normalized_cohort(source)
    frame = source_frame.copy()
    frame["scale"] = source.scale
    frame["cohort"] = cohort
    frame["source_cohort"] = source_cohort
    if "source_experiment" not in frame.columns:
        frame["source_experiment"] = pd.NA
    frame["registry_run_key"] = _run_key(source.scale, cohort, frame["source_experiment"], frame["run_name"])

    id_columns = [column for column in KNOWN_ID_COLUMNS if column in frame.columns]
    melted = frame.melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="original_metric_key",
        value_name="value",
    ).dropna(subset=["value"])
    if melted.empty:
        return melted

    metadata = pd.DataFrame.from_records(
        [canonicalize_metric_key(metric_key) for metric_key in melted["original_metric_key"]]
    )
    metadata.index = melted.index
    out = pd.concat([melted, metadata], axis=1)
    out["source_name"] = source.source_name
    out["source_uri"] = source.source_uri
    out["source_kind"] = source.source_kind
    out["source_priority"] = source.source_priority
    out["value"] = out["value"].astype(float)
    return out


def _manual_backfill_metrics() -> pd.DataFrame:
    if not MANUAL_BACKFILLS_CSV.exists():
        return pd.DataFrame()
    frame = pd.read_csv(MANUAL_BACKFILLS_CSV)
    if frame.empty:
        return pd.DataFrame()
    required = {"run_name", "scale", "cohort", "canonical_metric_key", "value", "reason"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{MANUAL_BACKFILLS_CSV} missing required columns: {missing}")
    rows = frame.copy()
    if "source_experiment" not in rows.columns:
        rows["source_experiment"] = pd.NA
    rows["registry_run_key"] = _run_key(
        rows["scale"].astype(str),
        rows["cohort"].astype(str),
        rows["source_experiment"],
        rows["run_name"],
    )
    rows["original_metric_key"] = rows["canonical_metric_key"]
    metadata = pd.DataFrame.from_records(
        [canonicalize_metric_key(metric_key) for metric_key in rows["canonical_metric_key"]]
    )
    metadata.index = rows.index
    out = pd.concat([rows, metadata.drop(columns=["canonical_metric_key"])], axis=1)
    out["source_name"] = "manual_backfills"
    out["source_uri"] = str(MANUAL_BACKFILLS_CSV)
    out["source_kind"] = "manual_backfill"
    out["source_priority"] = 1000
    out["value"] = out["value"].astype(float)
    return out


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _canonical_runs(source_runs: pd.DataFrame) -> pd.DataFrame:
    if source_runs.empty:
        return source_runs
    sorted_runs = source_runs.sort_values(["registry_run_key", "source_priority"], ascending=[True, False])
    weight_columns = sorted(column for column in sorted_runs.columns if column.startswith(WEIGHT_PREFIXES))
    aggregate_columns = [
        column
        for column in (*RUN_FIRST_VALUE_COLUMNS, *weight_columns)
        if column in sorted_runs.columns and column != "registry_run_key"
    ]
    rows: list[dict[str, Any]] = []
    for registry_run_key, group in sorted_runs.groupby("registry_run_key", sort=True):
        row = {"registry_run_key": registry_run_key}
        for column in aggregate_columns:
            row[column] = _first_non_null(group[column])
        row["source_names"] = ",".join(sorted(set(group["source_name"].astype(str))))
        row["max_source_priority"] = int(group["source_priority"].max())
        rows.append(row)
    runs = pd.DataFrame.from_records(rows)
    qsplit_core_names = {run.run_name for run in load_original_qsplit240_with_core_baselines()}
    runs["is_qsplit240_core"] = runs["run_name"].isin(qsplit_core_names)
    runs["is_baseline_olmix"] = runs["run_name"].eq("baseline_olmix_loglinear")
    runs["is_baseline_stratified"] = runs["run_name"].eq(STRATIFIED_RUN_NAME)
    runs["is_fit_swarm_60m_default"] = (
        runs["scale"].eq("60m_1p2b")
        & runs["cohort"].eq("signal")
        & (runs["is_qsplit240_core"] | runs["is_baseline_olmix"] | runs["is_baseline_stratified"])
    )
    return runs


def _metric_conflicts(all_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (registry_run_key, canonical_metric_key), group in all_metrics.groupby(
        ["registry_run_key", "canonical_metric_key"], sort=True
    ):
        values = group["value"].astype(float)
        if float(values.max() - values.min()) <= CONFLICT_TOLERANCE:
            continue
        rows.append(
            {
                "registry_run_key": registry_run_key,
                "canonical_metric_key": canonical_metric_key,
                "value_min": float(values.min()),
                "value_max": float(values.max()),
                "value_range": float(values.max() - values.min()),
                "source_names": ",".join(sorted(set(group["source_name"].astype(str)))),
                "source_uris": " ".join(sorted(set(group["source_uri"].astype(str)))),
                "value_count": len(group),
            }
        )
    return pd.DataFrame.from_records(rows)


def _canonical_metrics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_metrics.empty:
        return all_metrics
    sorted_metrics = all_metrics.sort_values(
        ["registry_run_key", "canonical_metric_key", "source_priority", "source_name"],
        ascending=[True, True, False, True],
    )
    return sorted_metrics.drop_duplicates(
        subset=["registry_run_key", "canonical_metric_key"],
        keep="first",
    ).reset_index(drop=True)


def _coverage(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    grouped = metrics.groupby(
        ["scale", "cohort", "suite", "task", "num_fewshot", "metric", "canonical_metric_key"],
        dropna=False,
        sort=True,
    )
    return grouped.agg(
        run_count=("registry_run_key", "nunique"),
        source_names=("source_name", lambda values: ",".join(sorted(set(values.astype(str))))),
    ).reset_index()


def _wide_metrics(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return runs
    wide_values = metrics.pivot_table(
        index="registry_run_key",
        columns="canonical_metric_key",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_values.columns.name = None
    return runs.merge(wide_values, on="registry_run_key", how="left")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_registry(*, include_gcs: bool, fail_on_conflicts: bool) -> dict[str, Any]:
    sources = _source_frames(include_gcs=include_gcs)
    artifact_rows = [_artifact_row(source) for source in sources]
    source_runs = pd.concat([_runs_from_source(source) for source in sources], ignore_index=True, sort=False)
    source_metric_frames = [_metrics_from_source(source) for source in sources]
    manual_backfills = _manual_backfill_metrics()
    if not manual_backfills.empty:
        source_metric_frames.append(manual_backfills)
    all_metrics = pd.concat(source_metric_frames, ignore_index=True, sort=False)

    runs = _canonical_runs(source_runs)
    conflicts = _metric_conflicts(all_metrics)
    if fail_on_conflicts and not conflicts.empty:
        raise ValueError(f"Found {len(conflicts)} conflicting metric facts. See {METRIC_CONFLICTS_CSV}")
    metrics = _canonical_metrics(all_metrics)
    coverage = _coverage(metrics)
    wide = _wide_metrics(runs, metrics)
    backfills = all_metrics.loc[all_metrics["source_kind"] == "manual_backfill"].copy()
    eval_artifacts = pd.DataFrame.from_records(artifact_rows)

    _write_csv(runs, RUNS_CSV)
    _write_csv(eval_artifacts, EVAL_ARTIFACTS_CSV)
    _write_csv(all_metrics, METRICS_ALL_SOURCES_CSV)
    _write_csv(metrics, METRICS_LONG_CSV)
    _write_csv(wide, METRICS_WIDE_CSV)
    _write_csv(conflicts, METRIC_CONFLICTS_CSV)
    _write_csv(coverage, COVERAGE_CSV)
    _write_csv(backfills, BACKFILLS_CSV)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "include_gcs": include_gcs,
        "source_count": len(sources),
        "run_count": len(runs),
        "metric_fact_count_all_sources": len(all_metrics),
        "metric_fact_count_canonical": len(metrics),
        "canonical_metric_count": int(metrics["canonical_metric_key"].nunique()) if not metrics.empty else 0,
        "conflict_count": len(conflicts),
        "manual_backfill_count": len(backfills),
        "outputs": {
            "runs": str(RUNS_CSV),
            "eval_artifacts": str(EVAL_ARTIFACTS_CSV),
            "metrics_all_sources_long": str(METRICS_ALL_SOURCES_CSV),
            "metrics_long": str(METRICS_LONG_CSV),
            "metrics_wide": str(METRICS_WIDE_CSV),
            "metric_conflicts": str(METRIC_CONFLICTS_CSV),
            "coverage": str(COVERAGE_CSV),
            "backfills": str(BACKFILLS_CSV),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gcs",
        action="store_true",
        help="Only use local CSVs. This skips the olmo-base/easy-overlap and seed-noise GCS collectors.",
    )
    parser.add_argument(
        "--fail-on-conflicts",
        action="store_true",
        help="Fail if the same run/metric has multiple distinct values across sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = _build_registry(include_gcs=not args.no_gcs, fail_on_conflicts=args.fail_on_conflicts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
