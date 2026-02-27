# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge, dedupe, and export steps for next-gen mixture loops."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fsspec
import pandas as pd

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

from experiments.domain_phase_mix.nextgen.collect import (
    IMPORTED_RUNS_FILE,
    IMPORTED_TRAJ_FILE,
    NEW_RUNS_FILE,
    NEW_TRAJ_FILE,
)
from experiments.domain_phase_mix.nextgen.contracts import LoopState, RunRecord
from experiments.domain_phase_mix.nextgen.state_store import STATE_FILE, load_loop_state
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights, stable_hash, write_json

logger = logging.getLogger(__name__)

MERGED_RUNS_JSON = "merged_runs.json"
MERGED_TRAJ_PARQUET = "merged_trajectories.parquet"
RESULTS_CSV = "results.csv"
TRAJ_CSV = "trajectories.csv"
RUNS_PARQUET = "runs.parquet"
TRAJ_PARQUET = "trajectories.parquet"


@dataclass(frozen=True)
class MergeDatasetConfig:
    """Executor config for merging run and trajectory data."""

    output_path: str
    loop_name: str
    objective_metric: str
    state_output_path: InputName | str
    imported_output_path: InputName | str
    new_output_path: InputName | str


@dataclass(frozen=True)
class ExportDatasetConfig:
    """Executor config for exporting merged artifacts."""

    output_path: str
    merged_output_path: InputName | str


def _decode_run_record(row: dict) -> RunRecord:
    return RunRecord(
        wandb_run_id=row.get("wandb_run_id"),
        source_experiment=row["source_experiment"],
        local_run_id=row.get("local_run_id"),
        run_name=row.get("run_name"),
        phase_weights=normalize_phase_weights(row.get("phase_weights", {})),
        status=row.get("status", "unknown"),
        metrics={str(k): float(v) for k, v in row.get("metrics", {}).items()},
    )


def _run_identity_key(run: RunRecord) -> str:
    if run.wandb_run_id:
        return f"wandb:{run.wandb_run_id}"
    return stable_hash(
        {
            "source_experiment": run.source_experiment,
            "run_name": run.run_name,
            "phase_weights": run.phase_weights,
        },
        prefix="fallback",
    )


def _merge_runs(prior: list[RunRecord], imported: list[RunRecord], new_runs: list[RunRecord]) -> list[RunRecord]:
    merged: dict[str, RunRecord] = {}

    for run in [*prior, *imported, *new_runs]:
        key = _run_identity_key(run)
        prev = merged.get(key)
        if prev is None:
            merged[key] = run
            continue

        # Prefer rows with completed status and richer metrics, then merge
        # metadata so local_run_id/run_name/phase_weights are preserved.
        score_prev = (1 if prev.status == "completed" else 0, len(prev.metrics))
        score_new = (1 if run.status == "completed" else 0, len(run.metrics))
        winner = run if score_new >= score_prev else prev
        loser = prev if winner is run else run

        merged[key] = RunRecord(
            wandb_run_id=winner.wandb_run_id or loser.wandb_run_id,
            source_experiment=winner.source_experiment,
            local_run_id=winner.local_run_id if winner.local_run_id is not None else loser.local_run_id,
            run_name=winner.run_name or loser.run_name,
            phase_weights=winner.phase_weights or loser.phase_weights,
            status=winner.status,
            metrics={**loser.metrics, **winner.metrics},
        )

    return sorted(
        merged.values(),
        key=lambda r: (
            r.local_run_id is None,
            r.local_run_id if r.local_run_id is not None else 10**18,
            r.run_name or "",
        ),
    )


def _load_run_records(path: str) -> list[RunRecord]:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return []
    with fsspec.open(path, "r") as f:
        payload = json.load(f)
    return [_decode_run_record(row) for row in payload]


def _load_trajectories(path: str) -> pd.DataFrame:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return pd.DataFrame(columns=_trajectory_columns())
    with fsspec.open(path, "rb") as f:
        return pd.read_parquet(f)


def _trajectory_columns() -> list[str]:
    return [
        "wandb_run_id",
        "source_experiment",
        "local_run_id",
        "run_name",
        "step",
        "total_tokens",
        "metric_key",
        "metric_value",
    ]


def _runs_to_dataframe(runs: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "wandb_run_id": run.wandb_run_id,
            "source_experiment": run.source_experiment,
            "run_id": run.local_run_id,
            "run_name": run.run_name,
            "status": run.status,
        }

        for phase_name, domain_weights in run.phase_weights.items():
            for domain_name, weight in domain_weights.items():
                row[f"{phase_name}_{domain_name}"] = float(weight)

        row.update({key: float(value) for key, value in run.metrics.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def merge_dataset(config: MergeDatasetConfig) -> None:
    """Merge prior state, imported data, and new run data with dedupe rules."""
    state_path = os.path.join(str(config.state_output_path), STATE_FILE)
    prior_state = load_loop_state(state_path, loop_name=config.loop_name, objective_metric=config.objective_metric)

    imported_runs = _load_run_records(os.path.join(str(config.imported_output_path), IMPORTED_RUNS_FILE))
    new_runs = _load_run_records(os.path.join(str(config.new_output_path), NEW_RUNS_FILE))

    merged_runs = _merge_runs(prior=prior_state.runs, imported=imported_runs, new_runs=new_runs)

    prior_traj = _load_trajectories(os.path.join(config.output_path, MERGED_TRAJ_PARQUET))
    imported_traj = _load_trajectories(os.path.join(str(config.imported_output_path), IMPORTED_TRAJ_FILE))
    new_traj = _load_trajectories(os.path.join(str(config.new_output_path), NEW_TRAJ_FILE))

    non_empty_frames = [frame for frame in [prior_traj, imported_traj, new_traj] if not frame.empty]
    merged_traj = (
        pd.concat(non_empty_frames, ignore_index=True)
        if non_empty_frames
        else pd.DataFrame(columns=_trajectory_columns())
    )
    if non_empty_frames:
        merged_traj = merged_traj.drop_duplicates(
            subset=["wandb_run_id", "source_experiment", "step", "metric_key"],
            keep="last",
        )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    merged_runs_json = os.path.join(config.output_path, MERGED_RUNS_JSON)
    with fsspec.open(merged_runs_json, "w") as f:
        json.dump([dataclasses.asdict(run) for run in merged_runs], f, indent=2, sort_keys=True)

    merged_traj_path = os.path.join(config.output_path, MERGED_TRAJ_PARQUET)
    with fsspec.open(merged_traj_path, "wb") as f:
        merged_traj.to_parquet(f, index=False)

    # Persist an updated state snapshot for downstream stages.
    max_local_run_id = max((run.local_run_id for run in merged_runs if run.local_run_id is not None), default=-1)
    updated_state = LoopState(
        loop_name=prior_state.loop_name,
        objective_metric=prior_state.objective_metric,
        next_local_run_id=max_local_run_id + 1,
        runs=merged_runs,
        validated_candidates=prior_state.validated_candidates,
    )
    write_json(os.path.join(config.output_path, "state_snapshot.json"), dataclasses.asdict(updated_state))



def export_dataset(config: ExportDatasetConfig) -> None:
    """Export merged run and trajectory data to CSV and parquet."""
    merged_dir = str(config.merged_output_path)
    runs = _load_run_records(os.path.join(merged_dir, MERGED_RUNS_JSON))
    run_df = _runs_to_dataframe(runs)

    traj_df = _load_trajectories(os.path.join(merged_dir, MERGED_TRAJ_PARQUET))

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "w") as f:
        run_df.to_csv(f, index=False)

    with fsspec.open(os.path.join(config.output_path, TRAJ_CSV), "w") as f:
        traj_df.to_csv(f, index=False)

    with fsspec.open(os.path.join(config.output_path, RUNS_PARQUET), "wb") as f:
        run_df.to_parquet(f, index=False)

    with fsspec.open(os.path.join(config.output_path, TRAJ_PARQUET), "wb") as f:
        traj_df.to_parquet(f, index=False)



def create_merge_step(
    *,
    loop_name: str,
    objective_metric: str,
    state_step: ExecutorStep,
    imported_step: ExecutorStep,
    new_step: ExecutorStep,
    output_override_path: str,
) -> ExecutorStep:
    """Create merge step."""
    return ExecutorStep(
        name=f"{loop_name}/merge/dataset",
        description="Merge prior state, imported trajectories, and newly collected runs",
        fn=merge_dataset,
        config=MergeDatasetConfig(
            output_path=this_output_path(),
            loop_name=loop_name,
            objective_metric=objective_metric,
            state_output_path=output_path_of(state_step),
            imported_output_path=output_path_of(imported_step),
            new_output_path=output_path_of(new_step),
        ),
        override_output_path=output_override_path,
    )


def create_export_step(*, loop_name: str, merge_step: ExecutorStep, output_override_path: str) -> ExecutorStep:
    """Create export step writing CSV/parquet artifacts."""
    return ExecutorStep(
        name=f"{loop_name}/export/csv",
        description="Export merged dataset to results.csv and trajectories.csv",
        fn=export_dataset,
        config=ExportDatasetConfig(
            output_path=this_output_path(),
            merged_output_path=output_path_of(merge_step),
        ),
        override_output_path=output_override_path,
    )
