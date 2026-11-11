# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt receipts and numeric metrics shared by RL writers and readers.

Metrics are keyed by attempt, step and name. A resumed history contains only its
recorded checkpoint ancestors, bounded by each continuation's starting step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import prefix_join

from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.schema import arrow_schema
from finestore.store import DataStore, DataTable

RESOLVED_LAUNCH_FILENAME = "resolved-launch.yaml"
TERMINAL_RECEIPT_FILENAME = "terminal.json"
EXPORTS_SUBDIRECTORY = "exports"
RUN_RECORDS_SUBDIRECTORY = "run-records"
METRICS_TABLE = "metrics"
ATTEMPTS_TABLE = "attempts"
ENGINE_CONFIG_KEYS = (
    "backend",
    "model_dtype",
    "vllm_attention_backend",
    "enable_prefix_caching",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enforce_eager",
    "engine_init_kwargs",
)


class LaunchState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExportedPolicy(BaseModel):
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


class LaunchResult(BaseModel):
    state: LaunchState
    run_id: str | None = None
    attempt_id: str | None = None
    iris_job_id: str | None = None
    iris_job_state: str | None = None
    provenance: dict[str, Any] | None = None
    runtime_profile: str | None = None
    model: ExportedPolicy | None = None
    failure: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class LaunchReceipt(BaseModel):
    config: dict[str, Any]
    result: LaunchResult


class CheckpointReference(BaseModel):
    export_root: str
    attempt_id: str
    step: int


class AttemptRecord(BaseModel):
    """One execution's native receipt, startup facts and checkpoint ancestry."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    attempt_id: str
    started_at: str | None = None
    vllm_version: str | None = None
    metrics_started_at: str | None = None
    resumed_step: int | None = None
    parent: CheckpointReference | None = None
    launch: LaunchReceipt | None = None

    @property
    def native(self) -> LaunchReceipt:
        """Return the native receipt of an Iris-launched attempt."""
        if self.launch is None:
            raise ValueError(f"Attempt {self.attempt_id} has no native launch receipt")
        return self.launch


class MetricRow(BaseModel):
    attempt_id: str
    step: int
    name: str
    value: float


class AttemptRow(BaseModel):
    attempt_id: str
    record_json: str


ROW_MODELS: dict[str, type[BaseModel]] = {METRICS_TABLE: MetricRow, ATTEMPTS_TABLE: AttemptRow}
PRIMARY_KEYS = {METRICS_TABLE: ("attempt_id", "step", "name"), ATTEMPTS_TABLE: ("attempt_id",)}


@dataclass(frozen=True)
class RunTables:
    metrics: DataTable
    attempts: DataTable


@dataclass(frozen=True)
class AttemptHistory:
    export_root: str
    record: AttemptRecord
    through_step: int | None


def run_records_root(export_root: str) -> str:
    return prefix_join(export_root, RUN_RECORDS_SUBDIRECTORY)


def register_run_tables(store: DataStore) -> RunTables:
    """Register metrics and attempts with their shared schemas and primary keys."""
    tables = {
        name: store.table(
            name,
            primary_key=PRIMARY_KEYS[name],
            schema=arrow_schema(model),
            on_conflict=OnConflict.SUPERSEDE,
        )
        for name, model in ROW_MODELS.items()
    }
    return RunTables(tables[METRICS_TABLE], tables[ATTEMPTS_TABLE])


def write_attempt(table: DataTable, record: AttemptRecord) -> None:
    """Append an attempt's complete record to its registered attempts table."""
    table.extend([AttemptRow(attempt_id=record.attempt_id, record_json=record.model_dump_json()).model_dump()])


def _attempt(view: ReadView, attempt_id: str) -> AttemptRecord | None:
    row = view.point(ATTEMPTS_TABLE, attempt_id=attempt_id)
    return AttemptRecord.model_validate_json(row["record_json"]) if row is not None else None


def read_attempt(export_root: str, attempt_id: str) -> AttemptRecord | None:
    """Read one attempt, including its native receipt and startup facts."""
    return _attempt(ReadView(run_records_root(export_root)), attempt_id)


def read_attempts(export_root: str) -> dict[str, AttemptRecord]:
    """Read the stored attempts from one export root's committed snapshot."""
    view = ReadView(run_records_root(export_root))
    return {
        row["attempt_id"]: AttemptRecord.model_validate_json(row["record_json"])
        for row in view.iter_rows(ATTEMPTS_TABLE)
    }


def _lineage(export_root: str, attempt_id: str, views: dict[str, ReadView]) -> tuple[AttemptHistory, ...]:
    history = []
    seen = set()
    through_step = None
    while True:
        export_root = export_root.rstrip("/")
        identity = (export_root, attempt_id)
        if identity in seen:
            raise ValueError(f"Checkpoint ancestry contains a cycle at {identity}")
        seen.add(identity)
        if export_root not in views:
            views[export_root] = ReadView(run_records_root(export_root))
        record = _attempt(views[export_root], attempt_id)
        if record is None:
            raise FileNotFoundError(f"Checkpoint ancestry has no attempt {attempt_id} at {export_root}")
        history.append(AttemptHistory(export_root, record, through_step))
        if record.resumed_step is None or record.resumed_step == 0:
            return tuple(history)
        if record.parent is None:
            raise ValueError(f"Attempt {attempt_id} has no recorded checkpoint parent")
        bound = min(record.resumed_step, record.parent.step)
        through_step = bound if through_step is None else min(through_step, bound)
        export_root, attempt_id = record.parent.export_root, record.parent.attempt_id


def attempt_lineage(export_root: str, attempt_id: str) -> tuple[AttemptHistory, ...]:
    """Return the attempt and its bounded ancestors, stopping at a fresh stage."""
    return _lineage(export_root, attempt_id, {})


def read_attempt_metrics(export_root: str, attempt_id: str) -> dict[int, dict[str, Any]]:
    """Read one attempt's metrics with checkpoint ancestors capped at resume bounds."""
    views: dict[str, ReadView] = {}
    history = _lineage(export_root, attempt_id, views)
    metrics: dict[int, dict[str, Any]] = {}
    for entry in reversed(history):
        for row in views[entry.export_root].iter_rows(
            METRICS_TABLE, where=[("attempt_id", "==", entry.record.attempt_id)]
        ):
            step = row["step"]
            if entry.through_step is not None and step > entry.through_step:
                continue
            values = metrics.setdefault(step, {"step": step, "attempt_id": entry.record.attempt_id})
            values[row["name"]] = row["value"]
    return dict(sorted(metrics.items()))


def attempt_tracking(record: AttemptRecord) -> dict[str, Any]:
    """Describe the native attempt's startup, engine configuration and W&B run."""
    if record.launch is None or record.started_at is None:
        return {}
    config = record.launch.config
    generator = config["skyrl"]["generator"]
    engine = {key: generator.get(key) for key in ENGINE_CONFIG_KEYS}
    engine.update(
        batch_invariant=config["skyrl"]["trainer"]["algorithm"]["batch_invariant"],
        vllm_env={key: value for key, value in config["runtime"]["task_env"].items() if key.startswith("VLLM_")},
        vllm_version=record.vllm_version,
    )
    entity = config["iris"]["wandb_entity"]
    trainer = config["skyrl"]["trainer"]
    wandb_url = (
        f"https://wandb.ai/{entity}/{trainer['project_name']}/runs/{record.attempt_id}"
        if entity and "wandb" in trainer["logger"]
        else None
    )
    return {
        "run_id": record.run_id,
        "attempt_id": record.attempt_id,
        "iris_job_id": record.native.result.iris_job_id,
        "coordinator_iris_job_id": config["run"].get("coordinator_iris_job_id"),
        "started_at": record.started_at,
        "engine": engine,
        "wandb_url": wandb_url,
    }
