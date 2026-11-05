# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Finelog copies of the controller's job, job config and attempt rows.

The controller DB deletes terminal jobs after a few days. Each write transaction
that changes a job or an attempt records the changed ids here; the full row is
read back inside the transaction, and one snapshot per changed entity is written
to ``iris.job`` / ``iris.job_config`` / ``iris.task_attempt`` after commit, the
same post-commit path ``iris.task_event`` uses. A rolled-back transaction writes
nothing.

Only work this controller executes is written: rows for jobs whose ``cluster`` is
``LOCAL_CLUSTER``. A job handed off to a federation peer, and the mirror rows of
a peer's jobs and attempts, are written by the peer that runs them, so every job
has one writer and one key.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import batched
from typing import Protocol

from finelog.client.log_client import Table
from rigging.timing import Timestamp
from sqlalchemy import and_, or_, select
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.stats.tables import JobConfigRow, JobRow, TaskAttemptRow, stats_timestamp
from iris.cluster.types import LOCAL_CLUSTER, JobName
from iris.rpc import job_pb2

_MEMO_KEY = "controller_rows"
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


@dataclass
class _PendingRows:
    """Latest snapshot per entity within one transaction, flushed after commit."""

    jobs: dict[JobName, JobRow] = field(default_factory=dict)
    job_configs: dict[JobName, JobConfigRow] = field(default_factory=dict)
    attempts: dict[tuple[JobName, int], TaskAttemptRow] = field(default_factory=dict)


def _pending(cur: RowCursor, tables: ControllerRowTables) -> _PendingRows:
    pending = cur.memo.get(_MEMO_KEY)
    if pending is None:
        pending = _PendingRows()
        cur.memo[_MEMO_KEY] = pending
        cur.register(lambda: _write(tables, pending))
    assert isinstance(pending, _PendingRows)
    return pending


def _write(tables: ControllerRowTables, pending: _PendingRows) -> None:
    if pending.jobs:
        tables.job_table.write(list(pending.jobs.values()))
    if pending.job_configs:
        tables.job_config_table.write(list(pending.job_configs.values()))
    if pending.attempts:
        tables.task_attempt_table.write(list(pending.attempts.values()))


def _dt(value: Timestamp | None):
    return value.as_naive_utc() if value is not None else None


def _job_state_name(state: int) -> str:
    return job_pb2.JobState.Name(state)


def _task_state_name(state: int) -> str:
    return job_pb2.TaskState.Name(state)


def record_job_changes(cur: RowCursor, job_ids: Iterable[JobName]) -> None:
    """Snapshot the current ``jobs`` rows of ``job_ids`` for writing after commit."""
    tables = cur.controller_rows
    if tables is None:
        return
    ids = list(dict.fromkeys(job_ids))
    if not ids:
        return
    pending = _pending(cur, tables)
    ts = stats_timestamp()
    j = jobs_table.c
    for chunk in batched(ids, _ID_CHUNK_SIZE):
        for r in cur.execute(select(jobs_table).where(j.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)):
            pending.jobs[r.job_id] = JobRow(
                cluster=tables.cluster,
                root_job_id=r.root_job_id,
                root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                job_id=r.job_id.to_wire(),
                parent_job_id=r.parent_job_id.to_wire() if r.parent_job_id is not None else None,
                depth=r.depth,
                name=r.name,
                user_id=r.user_id,
                submitting_user=r.submitting_user,
                state=_job_state_name(r.state),
                submitted_at=r.submitted_at_ms.as_naive_utc(),
                started_at=_dt(r.started_at_ms),
                finished_at=_dt(r.finished_at_ms),
                exit_code=r.exit_code,
                error=r.error,
                num_tasks=r.num_tasks,
                ts=ts,
            )


def record_job_submit(cur: RowCursor, job_id: JobName) -> None:
    """Snapshot a just-submitted job's ``jobs`` and ``job_config`` rows.

    Submit is the one place a job's request is written, so ``iris.job_config``
    gets exactly one row per job.
    """
    record_job_changes(cur, [job_id])
    tables = cur.controller_rows
    if tables is None:
        return
    pending = _pending(cur, tables)
    ts = stats_timestamp()
    c, j = job_config_table.c, jobs_table.c
    query = (
        select(job_config_table, j.root_job_id, j.root_submitted_at_ms)
        .join(jobs_table, j.job_id == c.job_id)
        .where(c.job_id == job_id, j.cluster == LOCAL_CLUSTER)
    )
    for r in cur.execute(query):
        pending.job_configs[r.job_id] = JobConfigRow(
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
            priority_band=r.priority_band,
            preemption_policy=r.preemption_policy,
            max_retries_failure=r.max_retries_failure,
            max_retries_preemption=r.max_retries_preemption,
            timeout_ms=r.timeout_ms,
            task_image=r.task_image,
            ts=ts,
        )


def record_attempt_changes(cur: RowCursor, attempts: Iterable[tuple[JobName, int]]) -> None:
    """Snapshot the current ``task_attempts`` rows of ``(task_id, attempt_id)`` pairs."""
    tables = cur.controller_rows
    if tables is None:
        return
    keys = list(dict.fromkeys(attempts))
    if not keys:
        return
    pending = _pending(cur, tables)
    ts = stats_timestamp()
    a, t, j = task_attempts_table.c, tasks_table.c, jobs_table.c
    for chunk in batched(keys, _ID_CHUNK_SIZE):
        query = (
            select(task_attempts_table, t.job_id, j.root_job_id, j.root_submitted_at_ms)
            .join(tasks_table, t.task_id == a.task_id)
            .join(jobs_table, j.job_id == t.job_id)
            .where(
                j.cluster == LOCAL_CLUSTER,
                or_(*(and_(a.task_id == task_id, a.attempt_id == attempt_id) for task_id, attempt_id in chunk)),
            )
        )
        for r in cur.execute(query):
            pending.attempts[(r.task_id, r.attempt_id)] = TaskAttemptRow(
                cluster=tables.cluster,
                root_job_id=r.root_job_id,
                root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                job_id=r.job_id.to_wire(),
                task_id=r.task_id.to_wire(),
                attempt_id=r.attempt_id,
                attempt_uid=r.attempt_uid,
                state=_task_state_name(r.state),
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
