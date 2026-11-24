# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt receipts, numeric metrics and task evaluations shared by RL writers and readers.

Metrics are keyed by attempt, step and name. A resumed history contains only its
recorded checkpoint ancestors, bounded by each continuation's starting step.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from rigging.filesystem.storage_path import prefix_join

from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.rl.rl_progress import RunIndexIdentity
from finestore.rl.strata import TaskStrata
from finestore.schema import arrow_schema
from finestore.store import DataStore, DataTable

RUN_RECORDS_SUBDIRECTORY = "run-records"
METRICS_TABLE = "metrics"
ATTEMPTS_TABLE = "attempts"
EVALUATIONS_TABLE = "evaluations"
METRIC_ATTEMPT_IDS = "metric_attempt_ids"
METRIC_METADATA_KEYS = frozenset(("step", "attempt_id", METRIC_ATTEMPT_IDS))


class LaunchState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (LaunchState.SUCCEEDED, LaunchState.FAILED)


class ExportedPolicy(BaseModel):
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str | None
    tokenizer_revision: str | None
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
    """One native attempt with its receipt, startup facts and checkpoint parent."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    attempt_id: str
    started_at: str | None = None
    iris_job_id: str | None = None
    vllm_version: str | None = None
    metrics_started_at: str | None = None
    evaluation_storage_version: Literal[1] | None = None
    resumed_step: int | None = None
    parent: CheckpointReference | None = None
    config: dict[str, Any] | None = None
    result: LaunchResult | None = None
    issue: int | None = None
    index: RunIndexIdentity | None = None
    elapsed_excluding_eval_seconds: Annotated[FiniteFloat, Field(ge=0)] | None = None

    @property
    def index_run_id(self) -> str:
        """Return the cross-system identity when this attempt has one."""
        return self.index.run_id if self.index is not None else self.run_id

    @property
    def native(self) -> LaunchReceipt:
        """Return the native receipt of an Iris-launched attempt."""
        if self.config is None:
            raise ValueError(f"Attempt {self.attempt_id} has no native launch receipt")
        result = self.result or LaunchResult(
            state=LaunchState.SUBMITTED if self.started_at is not None else LaunchState.PREPARED,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            iris_job_id=self.iris_job_id,
            started_at=self.started_at,
        )
        return LaunchReceipt(config=self.config, result=result)


class MetricRow(BaseModel):
    attempt_id: str
    step: int
    name: str
    value: float


class AttemptRow(BaseModel):
    attempt_id: str
    field: str
    value_json: str


class EvaluationAnswerRow(BaseModel):
    ordinal: int
    native_uid: str
    optimization_reward: float
    verifier_score: float | None
    raw_verifier_score: float | None
    score_min: float | None
    score_max: float | None
    verification_status: str | None
    passed: bool
    outcome: str
    error_type: str | None
    error_treatment: str | None
    response_token_count: int | None = None


class EvaluationTaskRow(BaseModel):
    attempt_id: str
    step: int
    evaluation: str
    panel_name: str | None
    panel_version: str | None
    task_id: str
    data_source: str | None
    strata: TaskStrata
    task_digest: str | None = None
    score: float | None
    outcome: str
    error_type: str | None
    answers: list[EvaluationAnswerRow]
    model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def read_legacy_agentic(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("agentic"), bool):
            strata = value.get("strata", {})
            if isinstance(strata, dict) and strata.get("agentic") is None:
                return {**value, "strata": {**strata, "agentic": value["agentic"]}}
        return value


_TABLES: dict[str, tuple[type[BaseModel], tuple[str, ...]]] = {
    METRICS_TABLE: (MetricRow, ("attempt_id", "step", "name")),
    ATTEMPTS_TABLE: (AttemptRow, ("attempt_id", "field")),
    EVALUATIONS_TABLE: (EvaluationTaskRow, ("attempt_id", "step", "evaluation", "task_id")),
}


@dataclass(frozen=True)
class LineageEntry:
    records_root: str
    record: AttemptRecord
    through_step: int | None


def run_records_root(export_root: str) -> str:
    return prefix_join(export_root, RUN_RECORDS_SUBDIRECTORY)


def register_run_tables(store: DataStore) -> dict[str, DataTable]:
    """Register the run tables with their shared schemas and primary keys."""
    return {
        name: store.table(name, primary_key=key, schema=arrow_schema(model), on_conflict=OnConflict.SUPERSEDE)
        for name, (model, key) in _TABLES.items()
    }


def update_attempt(store: DataStore, record: AttemptRecord) -> None:
    """Commit a fresh writer-owned update; omitted fields stay stored, explicit null clears a field."""
    values = record.model_dump(mode="json", exclude_unset=True)
    values.pop("attempt_id")
    rows = [
        AttemptRow(attempt_id=record.attempt_id, field=name, value_json=json.dumps(value)).model_dump()
        for name, value in values.items()
    ]
    register_run_tables(store)[ATTEMPTS_TABLE].extend(rows)
    store.flush()


def write_attempt(export_root: str, record: AttemptRecord) -> None:
    """Commit one attempt update to its export root."""
    with DataStore.open(run_records_root(export_root)) as store:
        update_attempt(store, record)


def _records(rows: Iterable[dict[str, Any]]) -> dict[str, AttemptRecord]:
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        records.setdefault(row["attempt_id"], {})[row["field"]] = json.loads(row["value_json"])
    for values in records.values():
        if values.get("issue") == "opted_out":
            values["issue"] = None
    return {attempt_id: AttemptRecord(attempt_id=attempt_id, **values) for attempt_id, values in records.items()}


def _attempt(view: ReadView, attempt_id: str) -> AttemptRecord | None:
    return _records(view.iter_rows(ATTEMPTS_TABLE, where=[("attempt_id", "==", attempt_id)])).get(attempt_id)


def read_attempt(export_root: str, attempt_id: str) -> AttemptRecord | None:
    """Read one attempt, including its native receipt and startup facts."""
    return _attempt(ReadView(run_records_root(export_root)), attempt_id)


def read_attempts(export_root: str) -> dict[str, AttemptRecord]:
    """Read the stored attempts from one export root's committed snapshot."""
    view = ReadView(run_records_root(export_root))
    return _records(view.iter_rows(ATTEMPTS_TABLE))


def _lineage(export_root: str, attempt_id: str, views: dict[str, ReadView]) -> tuple[LineageEntry, ...]:
    history = []
    seen = set()
    through_step = None
    while True:
        records_root = run_records_root(export_root)
        identity = (records_root, attempt_id)
        if identity in seen:
            raise ValueError(f"Checkpoint ancestry contains a cycle at {identity}")
        seen.add(identity)
        if records_root not in views:
            views[records_root] = ReadView(records_root)
        record = _attempt(views[records_root], attempt_id)
        if record is None:
            return tuple(history)
        history.append(LineageEntry(records_root, record, through_step))
        if record.resumed_step is None or record.resumed_step == 0:
            return tuple(history)
        if record.parent is None:
            return tuple(history)
        bound = record.parent.step
        through_step = bound if through_step is None else min(through_step, bound)
        export_root, attempt_id = record.parent.export_root, record.parent.attempt_id


def attempt_lineage(export_root: str, attempt_id: str) -> tuple[LineageEntry, ...]:
    """Return newest-first checkpoint ancestry with inclusive through_step bounds."""
    return _lineage(export_root, attempt_id, {})


def read_attempt_metrics(export_root: str, attempt_id: str) -> dict[int, dict[str, Any]]:
    """Join bounded history by step, retaining each metric's source attempt.

    The step's attempt_id names its latest contributor; metric_attempt_ids
    identifies the writer of each numeric value in a joined step.
    """
    views: dict[str, ReadView] = {}
    history = _lineage(export_root, attempt_id, views)
    metrics: dict[int, dict[str, Any]] = {}
    for entry in reversed(history):
        for row in views[entry.records_root].iter_rows(
            METRICS_TABLE, where=[("attempt_id", "==", entry.record.attempt_id)]
        ):
            step = row["step"]
            if entry.through_step is not None and step > entry.through_step:
                continue
            values = metrics.setdefault(step, {"step": step, METRIC_ATTEMPT_IDS: {}})
            values["attempt_id"] = entry.record.attempt_id
            values[METRIC_ATTEMPT_IDS][row["name"]] = entry.record.attempt_id
            values[row["name"]] = row["value"]
    return dict(sorted(metrics.items()))


def write_evaluations(export_root: str, rows: Sequence[EvaluationTaskRow]) -> None:
    """Commit one evaluation's task answers in the run's canonical records store."""
    with DataStore.open(run_records_root(export_root)) as store:
        register_run_tables(store)[EVALUATIONS_TABLE].extend(row.model_dump(mode="json") for row in rows)
        store.flush()


def read_evaluations(
    export_root: str, *, attempt_id: str, evaluation: str, step: int | None = None
) -> list[EvaluationTaskRow]:
    """Read one attempt and evaluation without combining retries or other profiles."""
    view = ReadView(run_records_root(export_root))
    where = [("attempt_id", "==", attempt_id), ("evaluation", "==", evaluation)]
    if step is not None:
        where.append(("step", "==", step))
    return [EvaluationTaskRow.model_validate(row) for row in view.iter_rows(EVALUATIONS_TABLE, where=where)]
