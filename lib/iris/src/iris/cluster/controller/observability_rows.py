# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Finelog copies of the controller's job, job config, task and attempt rows.

This is observability data, off the scheduling critical path: the controller
never reads these copies, and a failure here only loses rows (see the end of
this docstring).

The controller DB deletes terminal jobs after a few days. A write path that
changes a job, task or attempt marks its id here. Just before the transaction
commits, the marked rows are read back in batches, so each row is the entity's
final state in that transaction; after commit they are written to the
``iris.controller_db__*`` namespaces, the same post-commit path ``iris.task_event``
uses. A rolled-back transaction writes nothing.

Only work this controller executes is written: rows for jobs whose ``cluster`` is
``LOCAL_CLUSTER``. A job handed off to a federation peer, and the mirror rows of
a peer's jobs and attempts, are written by the peer that runs them, so every job
has one writer and one key.

Each row's ``cluster`` column is the writing controller's cluster name. A
regional Finelog forwards to the hub the rows named for its own
``forwarding.cluster``, which each cluster config sets to the cluster's name.

A failed snapshot or write is logged and drops that transaction's rows; it never
rolls back the controller's own writes or stops its later post-commit hooks.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import batched
from typing import NamedTuple, Protocol

from finelog.client.log_client import Table as FinelogTable
from rigging.provenance import LAUNCH_PROVENANCE_ENV, Provenance
from rigging.timing import RateLimiter, Timestamp
from sqlalchemy import select, tuple_
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.constraints import get_device_variant
from iris.cluster.controller.codec import proto_from_json
from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.write_lock_cost import CostWindow, RowTransactionCost
from iris.cluster.redaction import redact_string
from iris.cluster.stats.tables import IrisJob, IrisJobConfig, IrisTask, IrisTaskAttempt, stats_timestamp
from iris.cluster.types import LOCAL_CLUSTER, JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

_MEMO_KEY = "observability_rows"
# Keeps each IN list under SQLite's 999 bound parameters (an attempt key is two).
_ID_CHUNK_SIZE = 400
# A submitter that sends a malformed MARIN_PROVENANCE usually sends it on every job.
_MALFORMED_PROVENANCE_LOG_INTERVAL = 60.0
_malformed_provenance_log = RateLimiter(interval_seconds=_MALFORMED_PROVENANCE_LOG_INTERVAL)
# Rows this slow to snapshot and enqueue hold every other writer up noticeably.
SLOW_ROWS_SECONDS = 0.1
_SLOW_ROWS_LOG_INTERVAL = 60.0
_slow_rows_log = RateLimiter(interval_seconds=_SLOW_ROWS_LOG_INTERVAL)


@dataclass(frozen=True)
class ObservabilityRowTables:
    """The Finelog tables the controller copies its rows to, not the controller DB's own tables.

    ``cluster`` is this cluster's name, written to every row's ``cluster`` column.
    """

    cluster: str
    job_table: FinelogTable
    job_config_table: FinelogTable
    task_table: FinelogTable
    task_attempt_table: FinelogTable


class RowCursor(Protocol):
    """The write-transaction surface this module needs (``db.Tx``)."""

    observability_rows: ObservabilityRowTables | None
    lock_cost: CostWindow | None
    memo: dict[str, object]

    def execute(self, stmt, params=None) -> CursorResult: ...

    def register(self, hook: Callable[[], None]) -> None: ...

    def register_pre_commit(self, hook: Callable[[], None]) -> None: ...


@dataclass
class _Marked:
    """Entities changed in one transaction, and their rows once snapshotted."""

    jobs: dict[JobName, None] = field(default_factory=dict)
    submitted: dict[JobName, None] = field(default_factory=dict)
    tasks: dict[JobName, None] = field(default_factory=dict)
    attempts: dict[tuple[JobName, int], None] = field(default_factory=dict)
    # Attempts' workers as they were before ``remove_worker`` cleared them.
    removed_workers: dict[tuple[JobName, int], WorkerId] = field(default_factory=dict)
    job_rows: list[IrisJob] = field(default_factory=list)
    config_rows: list[IrisJobConfig] = field(default_factory=list)
    task_rows: list[IrisTask] = field(default_factory=list)
    attempt_rows: list[IrisTaskAttempt] = field(default_factory=list)
    # Time spent reading the rows before commit, under the write lock.
    snapshot_seconds: float = 0.0


def _marked(cur: RowCursor) -> _Marked | None:
    tables = cur.observability_rows
    if tables is None:
        return None
    marked = cur.memo.get(_MEMO_KEY)
    if marked is None:
        marked = _Marked()
        cur.memo[_MEMO_KEY] = marked
        lock_cost = cur.lock_cost
        assert lock_cost is not None, "observability rows need a write transaction"
        cur.register_pre_commit(lambda: _snapshot_or_drop(cur, tables, marked, lock_cost))
        cur.register(lambda: _write(tables, marked, lock_cost))
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

    Submit is the one place a job's request is written, so ``iris.controller_db__job_config``
    gets exactly one row per job.
    """
    marked = _marked(cur)
    if marked is None:
        return
    marked.jobs[job_id] = None
    marked.submitted[job_id] = None


def record_task_changes(cur: RowCursor, task_ids: Iterable[JobName]) -> None:
    """Mark ``task_ids`` whose state, attempt, worker, error or exit code changed."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.tasks.update(dict.fromkeys(task_ids))


def record_attempt_changes(cur: RowCursor, attempts: Iterable[tuple[JobName, int]]) -> None:
    """Mark ``(task_id, attempt_id)`` pairs as changed in this transaction."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.attempts.update(dict.fromkeys(attempts))


def record_worker_removal(cur: RowCursor, worker_id: WorkerId) -> None:
    """Remember which attempts ran on ``worker_id`` before ``remove_worker`` clears them.

    A failed worker's attempts finish in the same transaction that removes it, so
    without this their final rows would lose ``worker_id``.
    """
    marked = _marked(cur)
    if marked is None:
        return
    a = task_attempts_table.c
    for r in cur.execute(select(a.task_id, a.attempt_id).where(a.worker_id == worker_id)):
        marked.removed_workers[(r.task_id, r.attempt_id)] = worker_id
    t = tasks_table.c
    marked.tasks.update(dict.fromkeys(cur.execute(select(t.task_id).where(t.current_worker_id == worker_id)).scalars()))


def _write(tables: ObservabilityRowTables, marked: _Marked, lock_cost: CostWindow) -> None:
    """Enqueue the snapshotted rows; a table whose write fails is logged and its rows dropped.

    This runs after commit but still under the write lock, so its time counts with
    the snapshot's as the rows' cost: the transaction's row counts and both times go
    to ``lock_cost`` as one record, and a slow total is logged.
    """
    started = time.monotonic()
    for table, rows in (
        (tables.job_table, marked.job_rows),
        (tables.job_config_table, marked.config_rows),
        (tables.task_table, marked.task_rows),
        (tables.task_attempt_table, marked.attempt_rows),
    ):
        if not rows:
            continue
        try:
            table.write(rows)
        # This runs among the post-commit hooks: raising would reach the caller of a
        # committed transaction and skip the hooks registered after it (cache invalidations).
        except Exception:
            logger.exception("observability rows: write failed, dropping %d %s rows", len(rows), type(rows[0]).__name__)
    cost = RowTransactionCost(
        job_rows=len(marked.job_rows),
        job_config_rows=len(marked.config_rows),
        task_rows=len(marked.task_rows),
        attempt_rows=len(marked.attempt_rows),
        snapshot_seconds=marked.snapshot_seconds,
        enqueue_seconds=time.monotonic() - started,
    )
    rows = cost.job_rows + cost.job_config_rows + cost.task_rows + cost.attempt_rows
    if rows == 0:
        # Marked ids that turned out to need no row (e.g. a job outside this cluster).
        return
    lock_cost.add_row_transaction(cost)
    seconds = cost.snapshot_seconds + cost.enqueue_seconds
    if seconds >= SLOW_ROWS_SECONDS and _slow_rows_log.should_run():
        logger.warning(
            "observability rows: %d rows took %.0f ms under the write lock "
            "(at most one such warning per %.0f s; set finelog.controller_rows: false to turn the rows off)",
            rows,
            seconds * 1000,
            _SLOW_ROWS_LOG_INTERVAL,
        )


class _Accelerator(NamedTuple):
    """``iris.controller_db__job_config``'s flattened device columns, counted as the scheduler counts them."""

    kind: str | None
    type: str | None
    per_task: int


def _accelerator(device_json: str | None) -> _Accelerator:
    device = proto_from_json(device_json, job_pb2.DeviceConfig) if device_json else job_pb2.DeviceConfig()
    if device.HasField("gpu"):
        return _Accelerator("gpu", get_device_variant(device), get_gpu_count(device))
    if device.HasField("tpu"):
        return _Accelerator("tpu", get_device_variant(device), get_tpu_count(device))
    return _Accelerator(None, get_device_variant(device), 0)


def _launch_provenance(job_id: str, environment_json: str) -> Provenance | None:
    """The job's ``MARIN_PROVENANCE``, or ``None`` when it is unset or malformed.

    A malformed value drops only the ``provenance_*`` columns, never the job's rows.
    The warning names the job and the error, never the value, which is user text.
    """
    raw = proto_from_json(environment_json, job_pb2.EnvironmentConfig).env_vars.get(LAUNCH_PROVENANCE_ENV)
    if raw is None:
        return None
    try:
        return Provenance.from_json(raw)
    except ValueError as error:
        if _malformed_provenance_log.should_run():
            logger.warning(
                "observability rows: job %s has a malformed %s (%s); its provenance columns are NULL "
                "(at most one such warning per %.0f s)",
                job_id,
                LAUNCH_PROVENANCE_ENV,
                type(error).__name__,
                _MALFORMED_PROVENANCE_LOG_INTERVAL,
            )
        return None


def _free_text(value: str | None) -> str | None:
    """Error and status text stay readable, with token-like strings redacted."""
    return redact_string(value) if value is not None else None


def _dt(value: Timestamp | None) -> datetime | None:
    return value.as_naive_utc() if value is not None else None


def _snapshot_or_drop(cur: RowCursor, tables: ObservabilityRowTables, marked: _Marked, lock_cost: CostWindow) -> None:
    """Snapshot the marked rows; on failure, log and drop them so the transaction still commits.

    These rows are a telemetry copy: a bug here must not block scheduling. The
    snapshot runs under the write lock, so its time is recorded (see
    :mod:`iris.cluster.controller.write_lock_cost`).
    """
    started = time.monotonic()
    try:
        _snapshot(cur, tables, marked)
    except Exception:
        logger.exception(
            "observability rows: snapshot failed, dropping %d job, %d attempt rows (jobs %s, attempts %s)",
            len(marked.jobs),
            len(marked.attempts),
            [j.to_wire() for j in list(marked.jobs)[:5]],
            [(t.to_wire(), a) for t, a in list(marked.attempts)[:5]],
        )
        marked.job_rows.clear()
        marked.config_rows.clear()
        marked.task_rows.clear()
        marked.attempt_rows.clear()
    marked.snapshot_seconds = time.monotonic() - started


def _snapshot(cur: RowCursor, tables: ObservabilityRowTables, marked: _Marked) -> None:
    ts = stats_timestamp()
    _snapshot_jobs(cur, tables, marked, ts)
    _snapshot_configs(cur, tables, marked, ts)
    _snapshot_tasks(cur, tables, marked, ts)
    _snapshot_attempts(cur, tables, marked, ts)


def _snapshot_jobs(cur: RowCursor, tables: ObservabilityRowTables, marked: _Marked, ts: datetime) -> None:
    j = jobs_table.c
    for chunk in batched(marked.jobs, _ID_CHUNK_SIZE):
        for r in cur.execute(select(jobs_table).where(j.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)):
            marked.job_rows.append(
                IrisJob(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.submitted_at_ms.epoch_ms(),
                    parent_job_id=r.parent_job_id.to_wire() if r.parent_job_id is not None else None,
                    depth=r.depth,
                    name=r.name,
                    user_id=r.user_id,
                    submitting_user=r.submitting_user,
                    state=job_pb2.JobState.Name(r.state),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    exit_code=r.exit_code,
                    error=_free_text(r.error),
                    num_tasks=r.num_tasks,
                    scheduling_deadline_at=(
                        _dt(Timestamp.from_ms(r.scheduling_deadline_epoch_ms))
                        if r.scheduling_deadline_epoch_ms is not None
                        else None
                    ),
                    backend_id=r.backend_id,
                    placement=r.cluster,
                    ts=ts,
                )
            )


def _snapshot_configs(cur: RowCursor, tables: ObservabilityRowTables, marked: _Marked, ts: datetime) -> None:
    """Config rows for the jobs submitted in this transaction."""
    j, c = jobs_table.c, job_config_table.c
    for chunk in batched(marked.submitted, _ID_CHUNK_SIZE):
        query = (
            select(job_config_table, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms)
            .join(jobs_table, j.job_id == c.job_id)
            .where(c.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            accelerator = _accelerator(r.res_device_json)
            provenance = _launch_provenance(r.job_id.to_wire(), r.environment_json)
            marked.config_rows.append(
                IrisJobConfig(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.submitted_at_ms.epoch_ms(),
                    res_cpu_millicores=r.res_cpu_millicores,
                    res_memory_bytes=r.res_memory_bytes,
                    res_disk_bytes=r.res_disk_bytes,
                    res_device_json=r.res_device_json,
                    accelerator_kind=accelerator.kind,
                    accelerator_type=accelerator.type,
                    accelerators_per_task=accelerator.per_task,
                    constraints_json=r.constraints_json,
                    has_coscheduling=bool(r.has_coscheduling),
                    coscheduling_group_by=r.coscheduling_group_by,
                    coscheduling_group_sizes=json.dumps(list(r.coscheduling_group_sizes)),
                    priority_band=job_pb2.PriorityBand.Name(r.priority_band),
                    preemption_policy=job_pb2.JobPreemptionPolicy.Name(r.preemption_policy),
                    existing_job_policy=job_pb2.ExistingJobPolicy.Name(r.existing_job_policy),
                    max_retries_failure=r.max_retries_failure,
                    max_retries_preemption=r.max_retries_preemption,
                    max_task_failures=r.max_task_failures,
                    timeout_ms=r.timeout_ms,
                    scheduling_timeout_ms=r.scheduling_timeout_ms,
                    termination_grace_ms=r.termination_grace_ms,
                    task_image=r.task_image,
                    fail_if_exists=bool(r.fail_if_exists),
                    container_profile=job_pb2.ContainerProfile.Name(r.container_profile),
                    egress_policy=job_pb2.EgressPolicy.Name(r.egress_policy),
                    provenance_base_commit=provenance.base_commit if provenance else None,
                    provenance_tree_hash=provenance.tree_hash if provenance else None,
                    provenance_dirty=provenance.dirty if provenance else None,
                    provenance_branch=_free_text(provenance.branch) if provenance else None,
                    ts=ts,
                )
            )


def _snapshot_tasks(cur: RowCursor, tables: ObservabilityRowTables, marked: _Marked, ts: datetime) -> None:
    j, t = jobs_table.c, tasks_table.c
    for chunk in batched(marked.tasks, _ID_CHUNK_SIZE):
        query = (
            select(tasks_table, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms.label("job_submitted_at_ms"))
            .join(jobs_table, j.job_id == t.job_id)
            .where(t.task_id.in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            marked.task_rows.append(
                IrisTask(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.job_submitted_at_ms.epoch_ms(),
                    task_id=r.task_id.to_wire(),
                    task_index=r.task_index,
                    state=job_pb2.TaskState.Name(r.state),
                    error=_free_text(r.error),
                    exit_code=r.exit_code,
                    task_submitted_at=r.submitted_at_ms.as_naive_utc(),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    max_retries_failure=r.max_retries_failure,
                    max_retries_preemption=r.max_retries_preemption,
                    current_attempt_id=r.current_attempt_id,
                    priority_root_submitted_ms=r.priority_root_submitted_ms,
                    priority_band=job_pb2.PriorityBand.Name(r.priority_band),
                    container_id=r.container_id,
                    status_message=_free_text(r.status_message),
                    current_worker_id=str(r.current_worker_id) if r.current_worker_id is not None else None,
                    backend_id=r.backend_id,
                    placement=r.cluster,
                    ts=ts,
                )
            )


def _snapshot_attempts(cur: RowCursor, tables: ObservabilityRowTables, marked: _Marked, ts: datetime) -> None:
    j, a, t = jobs_table.c, task_attempts_table.c, tasks_table.c
    for chunk in batched(marked.attempts, _ID_CHUNK_SIZE):
        query = (
            select(task_attempts_table, t.job_id, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms)
            .join(tasks_table, t.task_id == a.task_id)
            .join(jobs_table, j.job_id == t.job_id)
            .where(tuple_(a.task_id, a.attempt_id).in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            worker_id = r.worker_id if r.worker_id is not None else marked.removed_workers.get((r.task_id, r.attempt_id))
            marked.attempt_rows.append(
                IrisTaskAttempt(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.submitted_at_ms.epoch_ms(),
                    task_id=r.task_id.to_wire(),
                    attempt_id=r.attempt_id,
                    attempt_uid=r.attempt_uid,
                    state=job_pb2.TaskState.Name(r.state),
                    created_at=r.created_at_ms.as_naive_utc(),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    exit_code=r.exit_code,
                    error=_free_text(r.error),
                    terminal_reason=r.terminal_reason,
                    worker_id=str(worker_id) if worker_id is not None else None,
                    node_name=r.node_name,
                    pod_name=r.pod_name,
                    pod_uid=r.pod_uid,
                    backend_id=r.backend_id,
                    topology_json=json.dumps(r.topology_json, sort_keys=True),
                    ts=ts,
                )
            )
