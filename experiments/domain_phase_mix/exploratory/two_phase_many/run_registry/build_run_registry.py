# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Build a canonical two-phase-many run registry.

This registry is intentionally split into:
- logical runs: one row per conceptual run we care about analyzing or babysitting
- attempts: one row per discovered checkpoint-backed attempt
- live watchlist: one row per currently tracked parent job

The goal is provenance first. Derived analysis tables should read from this
directory instead of re-deriving run identity ad hoc.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import fsspec
import pandas as pd

from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b import (
    NAME as PENALTY_300M_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b import (
    build_run_specs as build_penalty_300m_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    NAME as QSPLIT240_300M_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    build_run_specs as build_qsplit240_300m_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_fixed_subset_study import (
    NAME as RUN00097_300M_FIXED_SUBSET_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_fixed_subset_study import (
    build_run_specs as build_run00097_300m_fixed_subset_run_specs,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import STRATIFIED_RUN_ID, STRATIFIED_RUN_NAME
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    genericfamily_penalty_raw_optimum_summaries,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_power_family_penalty_no_l2_raw_subset_optima import (
    GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_power_family_penalty_no_l2_raw_subset_optima import (
    genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_subset_optima import (
    OLMIX_LOGLINEAR_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    olmix_loglinear_subset_optima_summaries,
)
from experiments.domain_phase_mix.two_phase_many_regmix_raw_subset_optima import (
    REGMIX_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    regmix_raw_subset_optima_summaries,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent
OUTPUT_LOGICAL_RUNS_CSV = SCRIPT_DIR / "logical_runs.csv"
OUTPUT_ATTEMPTS_CSV = SCRIPT_DIR / "run_attempts.csv"
OUTPUT_LIVE_WATCHLIST_CSV = SCRIPT_DIR / "live_watchlist.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "summary.json"
README_PATH = SCRIPT_DIR / "README.md"
SCRATCH_DIR = Path(__file__).resolve().parents[5] / "scratch"

INPUT_60M_ALL_RUNS_CSV = TWO_PHASE_MANY_DIR / "two_phase_many_all_60m_1p2b.csv"
INPUT_300M_COMPLETED_CSV = TWO_PHASE_MANY_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"

RUN00097_300M_FIXED_SUBSET_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_300m_6b_fixed_subset/collect_results-605e6a/results.csv"
)

CHECKPOINT_REGIONS = ("us-east5", "us-central1")
DEFAULT_QSPLIT300M_SHARD_COUNT = 8
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
RUN_NAME_PATTERN = re.compile(r"run_\d{5}$")
DEFAULT_LIVE_STATUS_TIMEOUT = 10


@dataclass(frozen=True)
class FamilyMetadata:
    """Static metadata for one logical run family."""

    family: str
    scale: str
    launcher_module: str | None
    resubmit_scope: str
    objective_metric: str
    resubmit_supported: bool


@dataclass(frozen=True)
class LiveJobSpec:
    """One actively tracked parent job."""

    job_id: str
    label: str
    family: str
    launcher_module: str | None
    resubmit_hint: str


@dataclass(frozen=True)
class RecoveryJobSpec:
    """One active single-run recovery parent."""

    run_name: str
    job_id: str
    submission_workspace: str
    resubmit_command: str
    latest_status: str
    latest_note: str


FAMILY_METADATA = {
    "two_phase_many_60m_export": FamilyMetadata(
        family="two_phase_many_60m_export",
        scale="60m_1p2b",
        launcher_module=None,
        resubmit_scope="manual",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=False,
    ),
    "qsplit240_300m_6b": FamilyMetadata(
        family="qsplit240_300m_6b",
        scale="300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b",
        resubmit_scope="shard",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "stratified_300m_6b": FamilyMetadata(
        family="stratified_300m_6b",
        scale="300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline",
        resubmit_scope="family",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "run00097_300m_6b_fixed_subset": FamilyMetadata(
        family="run00097_300m_6b_fixed_subset",
        scale="300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_fixed_subset_study",
        resubmit_scope="family",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "grp_penalty_raw_optima_300m_6b": FamilyMetadata(
        family="grp_penalty_raw_optima_300m_6b",
        scale="300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b",
        resubmit_scope="run",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "grp_power_family_penalty_no_l2_subset_optima": FamilyMetadata(
        family="grp_power_family_penalty_no_l2_subset_optima",
        scale="60m_1p2b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_genericfamily_power_family_penalty_no_l2_raw_subset_optima",
        resubmit_scope="run",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "olmix_loglinear_subset_optima": FamilyMetadata(
        family="olmix_loglinear_subset_optima",
        scale="60m_1p2b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_olmix_loglinear_subset_optima",
        resubmit_scope="run",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "regmix_raw_subset_optima": FamilyMetadata(
        family="regmix_raw_subset_optima",
        scale="60m_1p2b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_regmix_raw_subset_optima",
        resubmit_scope="run",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "qsplit240_300m_6b_parity_rerun": FamilyMetadata(
        family="qsplit240_300m_6b_parity_rerun",
        scale="300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b_parity_rerun",
        resubmit_scope="family",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
    "run00097_300m_6b_parity_rerun": FamilyMetadata(
        family="run00097_300m_6b_parity_rerun",
        scale="300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_parity_rerun",
        resubmit_scope="family",
        objective_metric=OBJECTIVE_METRIC,
        resubmit_supported=True,
    ),
}

TRACKED_LIVE_JOBS = (
    LiveJobSpec(
        job_id="/calvinxu/dm-genericfamily-penalty-raw-optima-300m-6b-20260414-174514",
        label="penalty-300m",
        family="grp_penalty_raw_optima_300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b",
        resubmit_hint=(
            "--variants power_family_penalty "
            "--tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a --max-concurrent 1"
        ),
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-qsplit240-300m-6b-olmix-20260414-174533",
        label="olmix-300m",
        family="qsplit240_300m_6b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b",
        resubmit_hint=(
            "--panel baselines3 --shard-count 3 --shard-index 2 --max-concurrent 1 "
            "--tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
        ),
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-olmix-loglinear-subset-optima-20260414-174548",
        label="olmix-subset",
        family="olmix_loglinear_subset_optima",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_olmix_loglinear_subset_optima",
        resubmit_hint=(
            "--subset-sizes all --max-concurrent 4 " "--tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
        ),
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-genericfamily-power-family-penalty-no-l2-raw-subset-optima-20260414-174605",
        label="no-l2-subset",
        family="grp_power_family_penalty_no_l2_subset_optima",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_genericfamily_power_family_penalty_no_l2_raw_subset_optima",
        resubmit_hint=(
            "--subset-sizes all --max-concurrent 4 " "--tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
        ),
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-qsplit240-300m-6b-parity-rerun-20260414-174621",
        label="parity-qsplit",
        family="qsplit240_300m_6b_parity_rerun",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b_parity_rerun",
        resubmit_hint="family rerun; current launcher resolves completed checkpoints dynamically",
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-run00097-300m-6b-parity-rerun-20260414-174637",
        label="parity-r97",
        family="run00097_300m_6b_parity_rerun",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_parity_rerun",
        resubmit_hint="family rerun; current launcher resolves completed checkpoints dynamically",
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-qsplit240-520m-chinchilla-pilot-20260414-232456",
        label="520m-pilot",
        family="pilot_520m",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_qsplit240_520m_chinchilla_pilot",
        resubmit_hint="--max-concurrent 2",
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-qsplit240-1-2b-chinchilla-pilot-20260414-225110",
        label="1.2b-pilot",
        family="pilot_1_2b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_qsplit240_1_2b_chinchilla_pilot",
        resubmit_hint="--max-concurrent 3",
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-stratified-520m-10p4b-20260414-232528",
        label="strat-520m",
        family="stratified_520m",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline",
        resubmit_hint="--scale 520m_10p4b",
    ),
    LiveJobSpec(
        job_id="/calvinxu/dm-stratified-1-2b-24b-20260414-225142",
        label="strat-1.2b",
        family="stratified_1_2b",
        launcher_module="experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline",
        resubmit_hint="--scale 1_2b_24b",
    ),
)


def _latest_recovery_state_path() -> Path | None:
    candidates = sorted(SCRATCH_DIR.glob("*_monitoring_state.json"))
    if not candidates:
        return None
    return candidates[-1]


def _load_qsplit300m_recoveries() -> dict[str, RecoveryJobSpec]:
    path = _latest_recovery_state_path()
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    recoveries = payload.get("recoveries", [])
    result: dict[str, RecoveryJobSpec] = {}
    for item in recoveries:
        run_name = str(item["run_name"])
        result[run_name] = RecoveryJobSpec(
            run_name=run_name,
            job_id=str(item["job_id"]),
            submission_workspace=str(item["submission_workspace"]),
            resubmit_command=str(item["resubmit_command"]),
            latest_status=str(item.get("latest_status", "planned")),
            latest_note=str(item.get("latest_note", "")),
        )
    return result


def _wandb_run_id_from_checkpoint_root(checkpoint_root: str) -> str:
    return checkpoint_root.rstrip("/").rsplit("/", 1)[-1]


def _extract_region_from_gcs_path(path: str) -> str | None:
    if "marin-us-east5" in path:
        return "us-east5"
    if "marin-us-central1" in path:
        return "us-central1"
    return None


def _read_optional_jsonl_last_record(path: str) -> dict[str, Any] | None:
    try:
        with fsspec.open(path, "r") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except FileNotFoundError:
        return None
    if not lines:
        return None
    return json.loads(lines[-1])


def _status_updated(fs: fsspec.AbstractFileSystem, path: str) -> str | None:
    try:
        info = fs.info(path)
    except FileNotFoundError:
        return None
    updated = info.get("updated")
    return None if updated is None else str(updated)


def _normalize_logical_status(status: str | None) -> str:
    if status is None:
        return "planned"
    normalized = str(status).strip().lower()
    if normalized in {"success", "succeeded", "completed", "complete"}:
        return "completed"
    if normalized in {"running", "run"}:
        return "running"
    if normalized in {"pending"}:
        return "pending"
    if normalized in {"failed", "failure"}:
        return "failed"
    if normalized in {"planned", "missing"}:
        return normalized
    return normalized


def _scan_checkpoint_attempts(
    *,
    family: str,
    source_experiment: str,
    run_name: str,
    run_id: int | None,
    objective_metric: str,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for region in CHECKPOINT_REGIONS:
        pattern = f"gs://marin-{region}/checkpoints/{source_experiment}/{run_name}-*/.executor_status"
        fs, _, _ = fsspec.get_fs_token_paths(pattern)
        for match in sorted(fs.glob(pattern)):
            status_path = match if str(match).startswith("gs://") else f"gs://{match}"
            checkpoint_root = status_path.removesuffix("/.executor_status")
            with fsspec.open(status_path, "r") as handle:
                executor_status = handle.read().strip()
            metrics_payload = _read_optional_jsonl_last_record(f"{checkpoint_root}/checkpoints/eval_metrics.jsonl")
            attempts.append(
                {
                    "registry_id": f"{family}:{run_name}",
                    "family": family,
                    "source_experiment": source_experiment,
                    "run_name": run_name,
                    "run_id": run_id,
                    "attempt_root": checkpoint_root,
                    "checkpoint_root": checkpoint_root,
                    "wandb_run_id": _wandb_run_id_from_checkpoint_root(checkpoint_root),
                    "region": _extract_region_from_gcs_path(checkpoint_root),
                    "executor_status": executor_status,
                    "status_updated": _status_updated(fs, status_path),
                    "has_eval_metrics": metrics_payload is not None,
                    "objective_metric": objective_metric,
                    "objective_metric_value": (
                        float(metrics_payload[objective_metric])
                        if metrics_payload is not None and isinstance(metrics_payload.get(objective_metric), int | float)
                        else None
                    ),
                }
            )
    return attempts


def _attempt_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    priority = {
        "SUCCESS": 4,
        "RUNNING": 3,
        "PENDING": 2,
        "FAILED": 1,
    }.get(str(row["executor_status"]), 0)
    updated = "" if row.get("status_updated") is None else str(row["status_updated"])
    return (priority, updated, str(row["attempt_root"]))


def _canonical_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attempts:
        return None
    return max(attempts, key=_attempt_sort_key)


def _qsplit300m_shard_index(*, run_name: str, shard_count: int = DEFAULT_QSPLIT300M_SHARD_COUNT) -> int:
    run_specs = build_qsplit240_300m_run_specs(panel="all")
    run_names = [spec.run_name for spec in run_specs]
    try:
        run_position = run_names.index(run_name)
    except ValueError as exc:
        raise ValueError(f"Unknown qsplit240 300M run name: {run_name}") from exc
    shard_size = math.ceil(len(run_specs) / shard_count)
    return run_position // shard_size


def _load_60m_export_rows() -> pd.DataFrame:
    frame = pd.read_csv(
        INPUT_60M_ALL_RUNS_CSV,
        usecols=["source_experiment", "run_id", "run_name", "wandb_run_id", "status"],
    ).rename(columns={"status": "source_status"})
    frame["registry_id"] = "two_phase_many_60m_export:" + frame["run_name"].astype(str)
    frame["family"] = "two_phase_many_60m_export"
    frame["scale"] = FAMILY_METADATA["two_phase_many_60m_export"].scale
    frame["checkpoint_root"] = pd.NA
    frame["objective_metric"] = OBJECTIVE_METRIC
    frame["objective_metric_value"] = pd.NA
    frame["canonical_attempt_root"] = pd.NA
    frame["attempt_count"] = 0
    frame["successful_attempt_count"] = 0
    frame["launcher_module"] = None
    frame["resubmit_supported"] = False
    frame["resubmit_scope"] = "manual"
    frame["resubmit_selector"] = pd.NA
    frame["resubmit_hint"] = "historical 60M export; no canonical launcher recipe wired yet"
    frame["logical_status"] = frame["source_status"].map(_normalize_logical_status)
    return frame


def _load_qsplit300m_rows() -> pd.DataFrame:
    completed = pd.read_csv(INPUT_300M_COMPLETED_CSV)
    completed_by_run = completed.set_index("run_name").to_dict(orient="index")
    recoveries_by_run = _load_qsplit300m_recoveries()
    rows: list[dict[str, Any]] = []
    for spec in build_qsplit240_300m_run_specs(panel="all"):
        completed_row = completed_by_run.get(spec.run_name)
        recovery = recoveries_by_run.get(spec.run_name)
        shard_index = _qsplit300m_shard_index(run_name=spec.run_name)
        checkpoint_root = None if completed_row is None else str(completed_row["checkpoint_root"])
        rows.append(
            {
                "registry_id": f"qsplit240_300m_6b:{spec.run_name}",
                "family": "qsplit240_300m_6b",
                "scale": FAMILY_METADATA["qsplit240_300m_6b"].scale,
                "source_experiment": QSPLIT240_300M_SOURCE_EXPERIMENT,
                "run_id": spec.run_id,
                "run_name": spec.run_name,
                "wandb_run_id": None if checkpoint_root is None else _wandb_run_id_from_checkpoint_root(checkpoint_root),
                "checkpoint_root": checkpoint_root,
                "objective_metric": OBJECTIVE_METRIC,
                "objective_metric_value": None if completed_row is None else float(completed_row["bpb_300m_6b"]),
                "canonical_attempt_root": checkpoint_root,
                "attempt_count": 1 if completed_row is not None else 0,
                "successful_attempt_count": 1 if completed_row is not None else 0,
                "launcher_module": FAMILY_METADATA["qsplit240_300m_6b"].launcher_module,
                "resubmit_supported": True,
                "resubmit_scope": "shard",
                "resubmit_selector": f"shard-count={DEFAULT_QSPLIT300M_SHARD_COUNT},shard-index={shard_index}",
                "resubmit_hint": (
                    "--panel all "
                    f"--shard-count {DEFAULT_QSPLIT300M_SHARD_COUNT} "
                    f"--shard-index {shard_index} "
                    "--max-concurrent 2 --tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
                ),
                "logical_status": (
                    "completed"
                    if completed_row is not None
                    else _normalize_logical_status(recovery.latest_status if recovery is not None else "missing")
                ),
                "source_status": None,
                "active_recovery_job_id": None if recovery is None else recovery.job_id,
                "active_recovery_note": None if recovery is None else recovery.latest_note,
                "active_recovery_workspace": None if recovery is None else recovery.submission_workspace,
            }
        )
    return pd.DataFrame(rows)


def _load_stratified300m_row() -> pd.DataFrame:
    completed = pd.read_csv(INPUT_300M_COMPLETED_CSV)
    row = completed.loc[completed["run_name"] == STRATIFIED_RUN_NAME]
    if row.empty:
        checkpoint_root = None
        metric = None
        logical_status = "missing"
        attempt_count = 0
        successful_attempt_count = 0
    else:
        checkpoint_root = str(row.iloc[0]["checkpoint_root"])
        metric = float(row.iloc[0]["bpb_300m_6b"])
        logical_status = "completed"
        attempt_count = 1
        successful_attempt_count = 1
    return pd.DataFrame(
        [
            {
                "registry_id": f"stratified_300m_6b:{STRATIFIED_RUN_NAME}",
                "family": "stratified_300m_6b",
                "scale": FAMILY_METADATA["stratified_300m_6b"].scale,
                "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b",
                "run_id": STRATIFIED_RUN_ID,
                "run_name": STRATIFIED_RUN_NAME,
                "wandb_run_id": None if checkpoint_root is None else _wandb_run_id_from_checkpoint_root(checkpoint_root),
                "checkpoint_root": checkpoint_root,
                "objective_metric": OBJECTIVE_METRIC,
                "objective_metric_value": metric,
                "canonical_attempt_root": checkpoint_root,
                "attempt_count": attempt_count,
                "successful_attempt_count": successful_attempt_count,
                "launcher_module": FAMILY_METADATA["stratified_300m_6b"].launcher_module,
                "resubmit_supported": True,
                "resubmit_scope": "family",
                "resubmit_selector": "--scale 300m_6b",
                "resubmit_hint": "--scale 300m_6b",
                "logical_status": logical_status,
                "source_status": None,
            }
        ]
    )


def _read_gcs_csv(uri: str) -> pd.DataFrame:
    with fsspec.open(uri, "r") as handle:
        return pd.read_csv(handle)


def _load_run00097_fixed_subset_rows() -> pd.DataFrame:
    run_specs = build_run00097_300m_fixed_subset_run_specs()
    try:
        results = _read_gcs_csv(RUN00097_300M_FIXED_SUBSET_RESULTS_URI)
    except FileNotFoundError:
        results = pd.DataFrame()
    results_by_name = (
        {}
        if results.empty
        else results.drop_duplicates(subset=["run_name"], keep="last").set_index("run_name").to_dict(orient="index")
    )
    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        result_row = results_by_name.get(spec.run_name)
        rows.append(
            {
                "registry_id": f"run00097_300m_6b_fixed_subset:{spec.run_name}",
                "family": "run00097_300m_6b_fixed_subset",
                "scale": FAMILY_METADATA["run00097_300m_6b_fixed_subset"].scale,
                "source_experiment": RUN00097_300M_FIXED_SUBSET_SOURCE_EXPERIMENT,
                "run_id": spec.run_id,
                "run_name": spec.run_name,
                "wandb_run_id": None if result_row is None else result_row.get("wandb_run_id"),
                "checkpoint_root": None if result_row is None else result_row.get("checkpoint_root"),
                "objective_metric": OBJECTIVE_METRIC,
                "objective_metric_value": None if result_row is None else result_row.get(OBJECTIVE_METRIC),
                "canonical_attempt_root": None if result_row is None else result_row.get("checkpoint_root"),
                "attempt_count": 1 if result_row is not None else 0,
                "successful_attempt_count": (
                    1 if result_row is not None and result_row.get("status") == "completed" else 0
                ),
                "launcher_module": FAMILY_METADATA["run00097_300m_6b_fixed_subset"].launcher_module,
                "resubmit_supported": True,
                "resubmit_scope": "family",
                "resubmit_selector": None,
                "resubmit_hint": "--tpu-type v5p-8",
                "logical_status": (
                    "completed" if result_row is not None and result_row.get("status") == "completed" else "missing"
                ),
                "source_status": None if result_row is None else result_row.get("status"),
            }
        )
    return pd.DataFrame(rows)


def _family_rows_from_attempt_scan(
    *,
    family: str,
    source_experiment: str,
    summaries: list[dict[str, Any]],
    resubmit_hint_fn: callable,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    logical_rows: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    metadata = FAMILY_METADATA[family]
    for summary in summaries:
        attempts = _scan_checkpoint_attempts(
            family=family,
            source_experiment=source_experiment,
            run_name=str(summary["run_name"]),
            run_id=int(summary["run_id"]),
            objective_metric=metadata.objective_metric,
        )
        all_attempts.extend(attempts)
        canonical = _canonical_attempt(attempts)
        logical_rows.append(
            {
                "registry_id": f"{family}:{summary['run_name']}",
                "family": family,
                "scale": metadata.scale,
                "source_experiment": source_experiment,
                "run_id": int(summary["run_id"]),
                "run_name": str(summary["run_name"]),
                "wandb_run_id": None if canonical is None else canonical["wandb_run_id"],
                "checkpoint_root": None if canonical is None else canonical["checkpoint_root"],
                "objective_metric": metadata.objective_metric,
                "objective_metric_value": None if canonical is None else canonical["objective_metric_value"],
                "canonical_attempt_root": None if canonical is None else canonical["attempt_root"],
                "attempt_count": len(attempts),
                "successful_attempt_count": sum(1 for attempt in attempts if attempt["executor_status"] == "SUCCESS"),
                "launcher_module": metadata.launcher_module,
                "resubmit_supported": metadata.resubmit_supported,
                "resubmit_scope": metadata.resubmit_scope,
                "resubmit_selector": resubmit_hint_fn(summary)[0],
                "resubmit_hint": resubmit_hint_fn(summary)[1],
                "logical_status": (
                    "planned" if canonical is None else _normalize_logical_status(str(canonical["executor_status"]))
                ),
                "source_status": None if canonical is None else canonical["executor_status"],
            }
        )
    return pd.DataFrame(logical_rows), all_attempts


def _penalty300m_rows() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source_variant_by_run_name = {
        summary.run_name: summary.variant_name for summary in genericfamily_penalty_raw_optimum_summaries(None)
    }
    summaries = [
        {
            "run_name": spec.run_name,
            "run_id": spec.run_id,
            "variant_name": source_variant_by_run_name[spec.candidate_run_name],
        }
        for spec in build_penalty_300m_run_specs()
    ]
    return _family_rows_from_attempt_scan(
        family="grp_penalty_raw_optima_300m_6b",
        source_experiment=PENALTY_300M_SOURCE_EXPERIMENT,
        summaries=summaries,
        resubmit_hint_fn=lambda summary: (
            f"--variants {summary['variant_name']}",
            (
                f"--variants {summary['variant_name']} "
                "--tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a --max-concurrent 1"
            ),
        ),
    )


def _no_l2_subset_rows() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summaries = [summary.__dict__ for summary in genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries()]
    return _family_rows_from_attempt_scan(
        family="grp_power_family_penalty_no_l2_subset_optima",
        source_experiment=GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
        summaries=summaries,
        resubmit_hint_fn=lambda summary: (
            f"--subset-sizes {summary['subset_size']}",
            (
                f"--subset-sizes {summary['subset_size']} "
                "--max-concurrent 1 --tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
            ),
        ),
    )


def _olmix_subset_rows() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summaries = [summary.__dict__ for summary in olmix_loglinear_subset_optima_summaries()]
    return _family_rows_from_attempt_scan(
        family="olmix_loglinear_subset_optima",
        source_experiment=OLMIX_LOGLINEAR_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
        summaries=summaries,
        resubmit_hint_fn=lambda summary: (
            f"--subset-sizes {summary['subset_size']}",
            (
                f"--subset-sizes {summary['subset_size']} "
                "--max-concurrent 1 --tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
            ),
        ),
    )


def _regmix_subset_rows() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summaries = [summary.__dict__ for summary in regmix_raw_subset_optima_summaries()]
    return _family_rows_from_attempt_scan(
        family="regmix_raw_subset_optima",
        source_experiment=REGMIX_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
        summaries=summaries,
        resubmit_hint_fn=lambda summary: (
            f"--subset-sizes {summary['subset_size']}",
            (
                f"--subset-sizes {summary['subset_size']} "
                "--max-concurrent 1 --tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a"
            ),
        ),
    )


def _live_job_status(job_id: str, *, timeout: int) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "iris",
        "--config",
        "lib/iris/examples/marin.yaml",
        "job",
        "summary",
        job_id,
        "--json",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[5],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "state": "timeout",
            "failure_count": None,
            "preemption_count": None,
            "error": f"iris job summary timed out after {timeout}s",
        }
    if result.returncode != 0:
        return {
            "state": "unavailable",
            "failure_count": None,
            "preemption_count": None,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    payload = json.loads(result.stdout)
    return {
        "state": payload.get("state"),
        "failure_count": payload.get("failure_count"),
        "preemption_count": payload.get("preemption_count"),
        "error": payload.get("error") or "",
    }


def _live_watchlist_frame(*, include_live_status: bool, live_status_timeout: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in TRACKED_LIVE_JOBS:
        status = (
            {"state": "skipped", "failure_count": None, "preemption_count": None, "error": ""}
            if not include_live_status
            else _live_job_status(spec.job_id, timeout=live_status_timeout)
        )
        rows.append(
            {
                "job_id": spec.job_id,
                "label": spec.label,
                "family": spec.family,
                "launcher_module": spec.launcher_module,
                "resubmit_hint": spec.resubmit_hint,
                "job_state": status["state"],
                "failure_count": status["failure_count"],
                "preemption_count": status["preemption_count"],
                "job_error": status["error"],
            }
        )
    for recovery in _load_qsplit300m_recoveries().values():
        status = (
            {"state": "skipped", "failure_count": None, "preemption_count": None, "error": ""}
            if not include_live_status
            else _live_job_status(recovery.job_id, timeout=live_status_timeout)
        )
        rows.append(
            {
                "job_id": recovery.job_id,
                "label": f"qsplit300m-recovery-{recovery.run_name}",
                "family": "qsplit240_300m_6b",
                "launcher_module": "scratch/20260414-2147_qsplit240_300m_resume_recovery.py",
                "resubmit_hint": recovery.resubmit_command,
                "job_state": status["state"],
                "failure_count": status["failure_count"],
                "preemption_count": status["preemption_count"],
                "job_error": status["error"] or recovery.latest_note,
            }
        )
    return pd.DataFrame(rows)


def _write_readme() -> None:
    README_PATH.write_text(
        """# Two-Phase-Many Run Registry

This directory is the canonical provenance layer for two-phase-many experiments.

Files:
- `logical_runs.csv`: one row per conceptual run
- `run_attempts.csv`: one row per discovered checkpoint-backed attempt
- `live_watchlist.csv`: current parent jobs we are actively babysitting
- `summary.json`: aggregate counts for quick handoff checks

Design:
- `logical_runs.csv` is the table analysis code should join against first
- `run_attempts.csv` preserves retries, failures, and superseded attempts
- W&B is treated as a convenient mirror, not the sole source of truth
- checkpoint-backed artifacts and executor status remain authoritative

Backfill policy:
- missing metrics should be backfilled from authoritative artifacts with an idempotent script
- if metrics are pushed to W&B later, record them as backfilled rather than pretending they were original

Operational notes:
- `live_watchlist.csv` is best-effort and slower because each Iris status query establishes a controller tunnel
- for a fast deterministic refresh, use `--no-include-live-status`
"""
    )


def build_registry(
    *, include_live_status: bool, live_status_timeout: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logical_frames = [
        _load_60m_export_rows(),
        _load_qsplit300m_rows(),
        _load_stratified300m_row(),
        _load_run00097_fixed_subset_rows(),
    ]
    attempts: list[dict[str, Any]] = []
    for builder in (_penalty300m_rows, _no_l2_subset_rows, _olmix_subset_rows, _regmix_subset_rows):
        logical_frame, family_attempts = builder()
        logical_frames.append(logical_frame)
        attempts.extend(family_attempts)

    logical_runs = pd.concat(logical_frames, ignore_index=True, sort=False)
    logical_runs = logical_runs.sort_values(["family", "run_name"]).reset_index(drop=True)
    run_attempts = (
        pd.DataFrame(attempts)
        .sort_values(["family", "run_name", "status_updated", "attempt_root"])
        .reset_index(drop=True)
        if attempts
        else pd.DataFrame()
    )
    live_watchlist = _live_watchlist_frame(
        include_live_status=include_live_status,
        live_status_timeout=live_status_timeout,
    )
    return logical_runs, run_attempts, live_watchlist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-live-status",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("CI") is None,
        help="Query live Iris parent-job status for the watchlist.",
    )
    parser.add_argument(
        "--live-status-timeout",
        type=int,
        default=DEFAULT_LIVE_STATUS_TIMEOUT,
        help="Per-job timeout in seconds for live Iris watchlist enrichment.",
    )
    return parser.parse_args()


def _summary(logical_runs: pd.DataFrame, run_attempts: pd.DataFrame, live_watchlist: pd.DataFrame) -> dict[str, Any]:
    return {
        "logical_run_count": len(logical_runs),
        "attempt_count": len(run_attempts),
        "live_watch_count": len(live_watchlist),
        "logical_runs_by_family": {
            str(key): int(value) for key, value in logical_runs.groupby("family").size().sort_index().items()
        },
        "logical_runs_by_status": {
            str(key): int(value) for key, value in logical_runs.groupby("logical_status").size().sort_index().items()
        },
        "live_jobs_by_state": {
            str(key): int(value) for key, value in live_watchlist.groupby("job_state").size().sort_index().items()
        },
        "qsplit240_300m_incomplete_runs": sorted(
            logical_runs.loc[
                (logical_runs["family"] == "qsplit240_300m_6b") & (logical_runs["logical_status"] != "completed"),
                "run_name",
            ]
            .astype(str)
            .tolist()
        ),
        "qsplit240_300m_active_recovery_runs": sorted(
            logical_runs.loc[
                (logical_runs["family"] == "qsplit240_300m_6b") & (logical_runs["active_recovery_job_id"].notna()),
                "run_name",
            ]
            .astype(str)
            .tolist()
        ),
        "qsplit240_300m_unrecovered_missing_runs": sorted(
            logical_runs.loc[
                (logical_runs["family"] == "qsplit240_300m_6b")
                & (logical_runs["logical_status"] != "completed")
                & (logical_runs["active_recovery_job_id"].isna()),
                "run_name",
            ]
            .astype(str)
            .tolist()
        ),
    }


def main() -> None:
    args = _parse_args()
    logical_runs, run_attempts, live_watchlist = build_registry(
        include_live_status=args.include_live_status,
        live_status_timeout=args.live_status_timeout,
    )
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    logical_runs.to_csv(OUTPUT_LOGICAL_RUNS_CSV, index=False)
    run_attempts.to_csv(OUTPUT_ATTEMPTS_CSV, index=False)
    live_watchlist.to_csv(OUTPUT_LIVE_WATCHLIST_CSV, index=False)
    OUTPUT_SUMMARY_JSON.write_text(
        json.dumps(_summary(logical_runs, run_attempts, live_watchlist), indent=2, sort_keys=True)
    )
    _write_readme()
    print(f"Wrote {OUTPUT_LOGICAL_RUNS_CSV}")
    print(f"Wrote {OUTPUT_ATTEMPTS_CSV}")
    print(f"Wrote {OUTPUT_LIVE_WATCHLIST_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")
    print(f"Wrote {README_PATH}")


if __name__ == "__main__":
    main()
