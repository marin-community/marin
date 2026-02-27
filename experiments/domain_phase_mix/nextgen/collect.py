# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collection steps for legacy imports and newly launched runs."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass

import fsspec
import pandas as pd

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

from experiments.domain_phase_mix.analysis import query_wandb_runs
from experiments.domain_phase_mix.nextgen.contracts import PlannedRun, RunRecord
from experiments.domain_phase_mix.nextgen.import_sources import (
    LegacyDomainPhaseImportSource,
    source_from_dict,
)

logger = logging.getLogger(__name__)

IMPORTED_RUNS_FILE = "imported_runs.json"
IMPORTED_TRAJ_FILE = "imported_trajectories.parquet"
NEW_RUNS_FILE = "new_runs.json"
NEW_TRAJ_FILE = "new_trajectories.parquet"


@dataclass(frozen=True)
class CollectImportedConfig:
    """Executor config for importing legacy runs/trajectories."""

    output_path: str
    objective_metric: str
    import_sources_json: str


@dataclass(frozen=True)
class CollectNewRunDataConfig:
    """Executor config for collecting new run trajectories."""

    output_path: str
    loop_name: str
    objective_metric: str
    wandb_entity: str
    wandb_project: str
    planned_runs_json: str
    depends_on: tuple[InputName, ...] = ()


def collect_imported_data(config: CollectImportedConfig) -> None:
    """Collect run metadata and trajectories from configured import sources."""
    sources_payload = json.loads(config.import_sources_json)
    sources = [source_from_dict(item) for item in sources_payload]

    run_records: list[RunRecord] = []
    traj_frames: list[pd.DataFrame] = []

    for source in sources:
        records = source.collect_runs()
        if records:
            run_records.extend(records)

        traj = source.collect_trajectories(config.objective_metric)
        if not traj.empty:
            traj_frames.append(traj)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    run_target = os.path.join(config.output_path, IMPORTED_RUNS_FILE)
    with fsspec.open(run_target, "w") as f:
        json.dump([dataclasses.asdict(record) for record in run_records], f, indent=2, sort_keys=True)

    if traj_frames:
        imported_traj = pd.concat(traj_frames, ignore_index=True)
    else:
        imported_traj = pd.DataFrame(
            columns=[
                "wandb_run_id",
                "source_experiment",
                "local_run_id",
                "run_name",
                "step",
                "total_tokens",
                "metric_key",
                "metric_value",
            ]
        )

    traj_target = os.path.join(config.output_path, IMPORTED_TRAJ_FILE)
    with fsspec.open(traj_target, "wb") as f:
        imported_traj.to_parquet(f, index=False)


def _resolve_wandb_run_for_planned(run_rows: list[dict], planned: PlannedRun) -> dict | None:
    marker = f"/{planned.run_name}"
    matches = [row for row in run_rows if marker in (row.get("wandb_run_name") or "")]
    if not matches:
        return None

    # Prefer completed run if multiple are present.
    finished = [row for row in matches if row.get("status") == "finished"]
    if finished:
        return finished[0]
    return matches[0]


def _scan_trajectory(wandb_run_id: str, entity: str, project: str, objective_metric: str) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{wandb_run_id}")
    keys = ["_step", objective_metric, "throughput/total_tokens"]

    rows: list[dict] = []
    for entry in run.scan_history(keys=keys):
        metric_value = entry.get(objective_metric)
        step = entry.get("_step")
        if metric_value is None or step is None:
            continue
        rows.append(
            {
                "step": int(step),
                "total_tokens": float(entry["throughput/total_tokens"])
                if entry.get("throughput/total_tokens") is not None
                else None,
                "metric_key": objective_metric,
                "metric_value": float(metric_value),
            }
        )

    return pd.DataFrame(rows)


def collect_new_run_data(config: CollectNewRunDataConfig) -> None:
    """Collect run-level and trajectory data for newly planned runs."""
    planned_payload = json.loads(config.planned_runs_json)
    planned_runs = [PlannedRun(**row) for row in planned_payload]

    if not planned_runs:
        fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
        fs.makedirs(config.output_path, exist_ok=True)
        with fsspec.open(os.path.join(config.output_path, NEW_RUNS_FILE), "w") as f:
            json.dump([], f)
        with fsspec.open(os.path.join(config.output_path, NEW_TRAJ_FILE), "wb") as f:
            pd.DataFrame().to_parquet(f, index=False)
        return

    run_rows = query_wandb_runs(
        entity=config.wandb_entity,
        project=config.wandb_project,
        tags=[config.loop_name],
        metrics=[config.objective_metric],
    )

    records: list[RunRecord] = []
    trajectories: list[pd.DataFrame] = []

    for planned in planned_runs:
        wb_row = _resolve_wandb_run_for_planned(run_rows, planned)
        if wb_row is None:
            records.append(
                RunRecord(
                    wandb_run_id=None,
                    source_experiment=config.loop_name,
                    local_run_id=planned.local_run_id,
                    run_name=planned.run_name,
                    phase_weights=planned.phase_weights,
                    status="not_found",
                )
            )
            continue

        wandb_run_id = wb_row.get("wandb_run_id")
        status = wb_row.get("status", "unknown")

        metrics = {
            key: float(value)
            for key, value in wb_row.items()
            if isinstance(value, int | float) and key.startswith(("eval/", "lm_eval/"))
        }

        records.append(
            RunRecord(
                wandb_run_id=wandb_run_id,
                source_experiment=config.loop_name,
                local_run_id=planned.local_run_id,
                run_name=planned.run_name,
                phase_weights=planned.phase_weights,
                status="completed" if status == "finished" else status,
                metrics=metrics,
            )
        )

        if wandb_run_id is None:
            continue

        try:
            traj = _scan_trajectory(
                wandb_run_id=wandb_run_id,
                entity=config.wandb_entity,
                project=config.wandb_project,
                objective_metric=config.objective_metric,
            )
        except Exception:
            logger.exception("Failed trajectory scan for run %s", wandb_run_id)
            continue

        if traj.empty:
            continue

        traj["wandb_run_id"] = wandb_run_id
        traj["source_experiment"] = config.loop_name
        traj["local_run_id"] = planned.local_run_id
        traj["run_name"] = planned.run_name
        trajectories.append(traj)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fsspec.open(os.path.join(config.output_path, NEW_RUNS_FILE), "w") as f:
        json.dump([dataclasses.asdict(row) for row in records], f, indent=2, sort_keys=True)

    traj_df = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame(
        columns=[
            "wandb_run_id",
            "source_experiment",
            "local_run_id",
            "run_name",
            "step",
            "total_tokens",
            "metric_key",
            "metric_value",
        ]
    )
    with fsspec.open(os.path.join(config.output_path, NEW_TRAJ_FILE), "wb") as f:
        traj_df.to_parquet(f, index=False)


def serialize_import_sources(sources: tuple) -> str:
    """Serialize import source objects into JSON for executor configs."""
    return json.dumps([source_to_dict(source) for source in sources], sort_keys=True)


def source_to_dict(source) -> dict:
    if not isinstance(source, LegacyDomainPhaseImportSource):
        raise TypeError(f"Unsupported import source object: {type(source)}")
    return {
        "type": "legacy_domain_phase",
        "source_experiment": source.source_experiment,
        "wandb_entity": source.wandb_entity,
        "wandb_project": source.wandb_project,
        "wandb_tags": list(source.wandb_tags),
        "weight_configs_path": source.weight_configs_path,
        "metric_prefixes": list(source.metric_prefixes),
    }


def create_collect_imported_step(
    loop_name: str,
    objective_metric: str,
    sources: tuple,
    output_override_path: str,
) -> ExecutorStep:
    """Create import-legacy collection step."""
    source_json = serialize_import_sources(tuple(sources))
    return ExecutorStep(
        name=f"{loop_name}/collect/import_legacy",
        description="Collect run metadata and trajectories from legacy sources",
        fn=collect_imported_data,
        config=CollectImportedConfig(
            output_path=this_output_path(),
            objective_metric=objective_metric,
            import_sources_json=source_json,
        ),
        override_output_path=output_override_path,
    )


def create_collect_new_step(
    *,
    loop_name: str,
    objective_metric: str,
    planned_runs: list[PlannedRun],
    wandb_entity: str,
    wandb_project: str,
    depends_on: list[ExecutorStep],
    output_override_path: str,
) -> ExecutorStep:
    """Create new-run collection step."""
    return ExecutorStep(
        name=f"{loop_name}/collect/new_run_trajectories",
        description="Collect trajectories and summaries for newly launched runs",
        fn=collect_new_run_data,
        config=CollectNewRunDataConfig(
            output_path=this_output_path(),
            loop_name=loop_name,
            objective_metric=objective_metric,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            planned_runs_json=json.dumps([dataclasses.asdict(p) for p in planned_runs], sort_keys=True),
            depends_on=tuple(output_path_of(step) for step in depends_on),
        ),
        override_output_path=output_override_path,
    )
