# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Finelog copies of the controller's job, job config and attempt rows.

The controller DB deletes terminal jobs after a few days. A write path that
changes a job or an attempt marks its id here. Just before the transaction
commits, the marked rows are read back in batches, so each row is the entity's
final state in that transaction; after commit they are written to ``iris.job``
/ ``iris.job_config`` / ``iris.task_attempt``, the same post-commit path
``iris.task_event`` uses. A rolled-back transaction writes nothing.

Only work this controller executes is written: rows for jobs whose ``cluster`` is
``LOCAL_CLUSTER``. A job handed off to a federation peer, and the mirror rows of
a peer's jobs and attempts, are written by the peer that runs them, so every job
has one writer and one key.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import batched
from typing import Protocol

from finelog.client.log_client import Table
from rigging.timing import Timestamp
from sqlalchemy import select, tuple_
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.stats.tables import IrisJob, IrisJobConfig, IrisTaskAttempt, stats_timestamp
from iris.cluster.types import LOCAL_CLUSTER, JobName
from iris.rpc import job_pb2

_MEMO_KEY = "controller_rows"
# Keeps each IN list under SQLite's 999 bound parameters (an attempt key is two).
_ID_CHUNK_SIZE = 400


@dataclass(frozen=True)
class ControllerRowTables:
    """The Finelog tables the controller copies its rows to, and this cluster's name."""

    cluster: str
    job_table: Table
    job_config_table: Table
    task_attempt_table: Table

    def __post_init__(self) -> None:
        if not self.cluster:
            raise ValueError("ControllerRowTables needs a non-empty cluster name (the cluster config name)")


class RowCursor(Protocol):
    """The write-transaction surface this module needs (``db.Tx``)."""

    controller_rows: ControllerRowTables | None
    memo: dict[str, object]

    def execute(self, stmt, params=None) -> CursorResult: ...

    def register(self, hook: Callable[[], None]) -> None: ...

    def register_pre_commit(self, hook: Callable[[], None]) -> None: ...


@dataclass
class _Marked:
    """Entities changed in one transaction, and their rows once snapshotted."""

    jobs: dict[JobName, None] = field(default_factory=dict)
    submitted: dict[JobName, None] = field(default_factory=dict)
    attempts: dict[tuple[JobName, int], None] = field(default_factory=dict)
    rows: list[IrisJob] = field(default_factory=list)
    config_rows: list[IrisJobConfig] = field(default_factory=list)
    attempt_rows: list[IrisTaskAttempt] = field(default_factory=list)


def _marked(cur: RowCursor) -> _Marked | None:
    tables = cur.controller_rows
    if tables is None:
        return None
    marked = cur.memo.get(_MEMO_KEY)
    if marked is None:
        marked = _Marked()
        cur.memo[_MEMO_KEY] = marked
        cur.register_pre_commit(lambda: _snapshot(cur, tables, marked))
        cur.register(lambda: _write(tables, marked))
    assert isinstance(marked, _Marked)
    return marked


def record_job_changes(cur: RowCursor, job_ids: Iterable[JobName]) -> None:
    """Mark ``job_ids`` as changed in this transaction."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.jobs.update(dict.fromkeys(job_ids))


def record_job_submit(cur: RowCursor, job_id: JobName) -> None:
    """Mark a just-submitted job; its ``job_config`` row is written too.

    Submit is the one place a job's request is written, so ``iris.job_config``
    gets exactly one row per job.
    """
    marked = _marked(cur)
    if marked is None:
        return
    marked.jobs[job_id] = None
    marked.submitted[job_id] = None


def record_attempt_changes(cur: RowCursor, attempts: Iterable[tuple[JobName, int]]) -> None:
    """Mark ``(task_id, attempt_id)`` pairs as changed in this transaction."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.attempts.update(dict.fromkeys(attempts))


def _write(tables: ControllerRowTables, marked: _Marked) -> None:
    if marked.rows:
        tables.job_table.write(marked.rows)
    if marked.config_rows:
        tables.job_config_table.write(marked.config_rows)
    if marked.attempt_rows:
        tables.task_attempt_table.write(marked.attempt_rows)


def _dt(value: Timestamp | None) -> datetime | None:
    return value.as_naive_utc() if value is not None else None


def _snapshot(cur: RowCursor, tables: ControllerRowTables, marked: _Marked) -> None:
    ts = stats_timestamp()
    j, c, a, t = jobs_table.c, job_config_table.c, task_attempts_table.c, tasks_table.c
    for chunk in batched(marked.jobs, _ID_CHUNK_SIZE):
        for r in cur.execute(select(jobs_table).where(j.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)):
            marked.rows.append(
                IrisJob(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    parent_job_id=r.parent_job_id.to_wire() if r.parent_job_id is not None else None,
                    depth=r.depth,
                    name=r.name,
                    user_id=r.user_id,
                    submitting_user=r.submitting_user,
                    state=job_pb2.JobState.Name(r.state),
                    submitted_at=r.submitted_at_ms.as_naive_utc(),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    exit_code=r.exit_code,
                    error=r.error,
                    num_tasks=r.num_tasks,
                    ts=ts,
                )
            )
    for chunk in batched(marked.submitted, _ID_CHUNK_SIZE):
        query = (
            select(job_config_table, j.root_job_id, j.root_submitted_at_ms)
            .join(jobs_table, j.job_id == c.job_id)
            .where(c.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            marked.config_rows.append(
                IrisJobConfig(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    res_cpu_millicores=r.res_cpu_millicores,
                    res_memory_bytes=r.res_memory_bytes,
                    res_disk_bytes=r.res_disk_bytes,
                    res_device_json=r.res_device_json,
                    constraints_json=r.constraints_json,
                    has_coscheduling=bool(r.has_coscheduling),
                    priority_band=job_pb2.PriorityBand.Name(r.priority_band),
                    preemption_policy=job_pb2.JobPreemptionPolicy.Name(r.preemption_policy),
                    max_retries_failure=r.max_retries_failure,
                    max_retries_preemption=r.max_retries_preemption,
                    timeout_ms=r.timeout_ms,
                    task_image=r.task_image,
                    ts=ts,
                )
            )
    for chunk in batched(marked.attempts, _ID_CHUNK_SIZE):
        query = (
            select(task_attempts_table, t.job_id, j.root_job_id, j.root_submitted_at_ms)
            .join(tasks_table, t.task_id == a.task_id)
            .join(jobs_table, j.job_id == t.job_id)
            .where(tuple_(a.task_id, a.attempt_id).in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            marked.attempt_rows.append(
                IrisTaskAttempt(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    task_id=r.task_id.to_wire(),
                    attempt_id=r.attempt_id,
                    attempt_uid=r.attempt_uid,
                    state=job_pb2.TaskState.Name(r.state),
                    created_at=r.created_at_ms.as_naive_utc(),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    exit_code=r.exit_code,
                    error=r.error,
                    terminal_reason=r.terminal_reason,
                    worker_id=str(r.worker_id) if r.worker_id is not None else None,
                    node_name=r.node_name,
                    pod_name=r.pod_name,
                    pod_uid=r.pod_uid,
                    ts=ts,
                )
            )
