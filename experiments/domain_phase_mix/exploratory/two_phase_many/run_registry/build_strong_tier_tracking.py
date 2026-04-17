# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "torch"]
# ///
"""Build a focused tracking snapshot for the active strong-tier scaling study."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline import (
    build_run_spec as build_stratified_run_spec,
)
from experiments.domain_phase_mix.qsplit240_replay import build_qsplit240_replay_run_specs
from experiments.domain_phase_mix.scaling_study_recipes import (
    ScalingStudyCell,
    ScalingStudyPath,
    build_strong_tier_cells,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_LOGICAL_RUNS_CSV = SCRIPT_DIR / "strong_tier_logical_runs.csv"
OUTPUT_CHILD_JOBS_CSV = SCRIPT_DIR / "strong_tier_child_jobs.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "strong_tier_summary.json"
STRONG_TIER_PARENT_JOB_ID = "/calvinxu/dm-strong-tier-scaling-study-20260415-235849"
IRIS_CONFIG = "lib/iris/examples/marin.yaml"


def _logical_family(cell: ScalingStudyCell) -> str:
    if cell.path == ScalingStudyPath.QSPLIT_REPRESENTATIVE12:
        return "strong_tier_qsplit_representative12"
    if cell.path == ScalingStudyPath.STRATIFIED:
        return "strong_tier_stratified"
    if cell.path == ScalingStudyPath.QSPLIT_BASELINES3_HOLDOUT:
        return "strong_tier_qsplit_baselines3_holdout"
    if cell.path == ScalingStudyPath.STRATIFIED_HOLDOUT:
        return "strong_tier_stratified_holdout"
    raise ValueError(f"Unsupported strong-tier cell path: {cell.path!r}")


def _qsplit_specs(cell: ScalingStudyCell) -> list[dict[str, Any]]:
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


def _stratified_spec(cell: ScalingStudyCell) -> dict[str, Any]:
    return build_stratified_run_spec(
        scale=cell.scale,
        experiment_budget=cell.experiment_budget,
        target_budget=cell.target_budget,
        target_budget_multiplier=cell.target_budget_multiplier,
        cohort=cell.cohort,
    ).__dict__


def build_logical_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in build_strong_tier_cells():
        family = _logical_family(cell)
        run_specs = (
            _qsplit_specs(cell)
            if cell.path in {ScalingStudyPath.QSPLIT_REPRESENTATIVE12, ScalingStudyPath.QSPLIT_BASELINES3_HOLDOUT}
            else [_stratified_spec(cell)]
        )
        for run_spec in run_specs:
            rows.append(
                {
                    "registry_id": f"{family}:{cell.name_prefix}:{run_spec['run_name']}",
                    "family": family,
                    "scale": cell.scale.value,
                    "study_parent_job_id": STRONG_TIER_PARENT_JOB_ID,
                    "study_cell_status": cell.status.value,
                    "study_path": cell.path.value,
                    "study_cohort": cell.cohort,
                    "study_panel": cell.panel,
                    "source_experiment": cell.name_prefix,
                    "source_name_prefix": cell.source_name_prefix,
                    "run_id": int(run_spec["run_id"]),
                    "run_name": str(run_spec["run_name"]),
                    "candidate_run_name": run_spec.get("candidate_run_name"),
                    "candidate_source_experiment": run_spec.get("candidate_source_experiment"),
                    "model_family": cell.model_family,
                    "experiment_budget": cell.experiment_budget,
                    "target_budget": cell.target_budget,
                    "target_budget_multiplier": cell.target_budget_multiplier,
                    "num_train_steps": cell.num_train_steps,
                    "batch_size": cell.batch_size,
                    "seq_len": cell.seq_len,
                    "tpu_type": cell.tpu_type,
                    "tpu_regions": ",".join(cell.tpu_regions),
                    "tpu_zone": cell.tpu_zone,
                    "is_new_submission": cell.status.value == "new",
                    "is_reused_submission": cell.status.value == "reused",
                    "is_holdout_only": cell.status.value == "holdout_only",
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["scale", "study_path", "source_experiment", "run_name"]).reset_index(drop=True)


def build_child_jobs() -> pd.DataFrame:
    command = [
        "uv",
        "run",
        "iris",
        "--config",
        IRIS_CONFIG,
        "job",
        "list",
        "--json",
        "--prefix",
        STRONG_TIER_PARENT_JOB_ID,
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[5],
        capture_output=True,
        text=True,
        check=True,
    )
    jobs = json.loads(result.stdout)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if job["job_id"] == STRONG_TIER_PARENT_JOB_ID:
            continue
        variant = job.get("resources", {}).get("device", {}).get("tpu", {}).get("variant")
        count = job.get("resources", {}).get("device", {}).get("tpu", {}).get("count")
        rows.append(
            {
                "parent_job_id": STRONG_TIER_PARENT_JOB_ID,
                "job_id": job["job_id"],
                "state": job["state"],
                "submitted_at_epoch_ms": job.get("submitted_at", {}).get("epoch_ms"),
                "task_count": job.get("task_count"),
                "completed_count": job.get("completed_count"),
                "failure_count": job.get("failure_count"),
                "preemption_count": job.get("preemption_count"),
                "pending_reason": job.get("pending_reason", ""),
                "tpu_variant": variant,
                "tpu_count": count,
                "run_stem": str(job["job_id"]).rsplit("/", 1)[-1].split("-", 1)[0],
            }
        )
    return pd.DataFrame(rows).sort_values(["state", "job_id"]).reset_index(drop=True)


def build_summary(logical_runs: pd.DataFrame, child_jobs: pd.DataFrame) -> dict[str, Any]:
    return {
        "parent_job_id": STRONG_TIER_PARENT_JOB_ID,
        "logical_run_count": len(logical_runs),
        "new_run_count": int(logical_runs["is_new_submission"].sum()),
        "reused_run_count": int(logical_runs["is_reused_submission"].sum()),
        "holdout_run_count": int(logical_runs["is_holdout_only"].sum()),
        "logical_runs_by_scale": {
            str(key): int(value) for key, value in logical_runs.groupby("scale").size().sort_index().items()
        },
        "new_runs_by_scale": {
            str(key): int(value)
            for key, value in (
                logical_runs.loc[logical_runs["is_new_submission"]].groupby("scale").size().sort_index().items()
            )
        },
        "logical_runs_by_family": {
            str(key): int(value) for key, value in logical_runs.groupby("family").size().sort_index().items()
        },
        "child_job_count": len(child_jobs),
        "child_jobs_by_state": {
            str(key): int(value) for key, value in child_jobs.groupby("state").size().sort_index().items()
        },
        "child_jobs_by_tpu_variant": {
            str(key): int(value) for key, value in child_jobs.groupby("tpu_variant").size().sort_index().items()
        },
    }


def main() -> None:
    logical_runs = build_logical_runs()
    child_jobs = build_child_jobs()
    summary = build_summary(logical_runs, child_jobs)
    logical_runs.to_csv(OUTPUT_LOGICAL_RUNS_CSV, index=False)
    child_jobs.to_csv(OUTPUT_CHILD_JOBS_CSV, index=False)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_LOGICAL_RUNS_CSV}")
    print(f"Wrote {OUTPUT_CHILD_JOBS_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
