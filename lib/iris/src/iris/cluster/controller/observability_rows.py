# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Finelog copies of the controller's job, job config, task and attempt rows.

This is observability data, off the scheduling critical path: the controller
never reads these copies, and a failure here only loses rows (see the end of
this docstring).

The controller DB deletes terminal jobs after a few days. SQLite triggers on the
write connection record which jobs, configs, tasks and attempts a transaction
inserts or changes, in TEMP tables private to that connection. Just before the
transaction commits, the recorded rows are read back, so each row is the
entity's final state in that transaction; after commit they are written to the
``iris.controller_db__*`` namespaces, the same post-commit path ``iris.task_event``
uses. A rolled-back transaction rolls back its recorded keys too and writes
nothing. Deletes (pruning, worker removal cascades) record nothing.

What counts as a change is stated once, in the trigger conditions:

* a job: any column of ``jobs``;
* a task: ``state``, ``current_attempt_id``, ``current_worker_id``, ``error`` or
  ``exit_code`` (not the pending reason or the container id);
* an attempt: any column of ``task_attempts``, except that clearing ``worker_id``
  when a worker is removed is not a change; the row keeps the old worker;
* a job config: its insert, at submit, so each job has exactly one config row.

TEMP triggers fire only for writes on the connection that created them. Every
controller write to these tables goes through ``ControllerDB.transaction()``,
whose single pooled write connection gets the triggers the first time it is used
with rows attached, including a fresh connection after a reopen or checkpoint
restore.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple, Protocol

from finelog.client.log_client import Table as FinelogTable
from rigging.provenance import LAUNCH_PROVENANCE_ENV, Provenance
from rigging.timing import RateLimiter, Timestamp
from sqlalchemy import Column, Integer, MetaData, Table, delete, exists, or_, select, text, tuple_
from sqlalchemy.engine import Connection
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.constraints import get_device_variant
from iris.cluster.controller.codec import proto_from_json
from iris.cluster.controller.schema import (
    JobNameType,
    WorkerIdType,
    job_config_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
)
from iris.cluster.controller.write_lock_cost import CostWindow, RowTransactionCost
from iris.cluster.redaction import redact_string
from iris.cluster.stats.tables import IrisJob, IrisJobConfig, IrisTask, IrisTaskAttempt, stats_timestamp
from iris.cluster.types import LOCAL_CLUSTER, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Task columns whose change writes a task row (see the module docstring).
TASK_CHANGE_COLUMNS = ("state", "current_attempt_id", "current_worker_id", "error", "exit_code")
# Marks a DBAPI connection whose TEMP tables and triggers exist.
_INSTALLED_KEY = "observability_row_triggers"
# A submitter that sends a malformed MARIN_PROVENANCE usually sends it on every job.
_MALFORMED_PROVENANCE_LOG_INTERVAL = 60.0
_malformed_provenance_log = RateLimiter(interval_seconds=_MALFORMED_PROVENANCE_LOG_INTERVAL)
# Rows this slow to snapshot and enqueue hold every other writer up noticeably.
SLOW_ROWS_SECONDS = 0.1
_SLOW_ROWS_LOG_INTERVAL = 60.0
_slow_rows_log = RateLimiter(interval_seconds=_SLOW_ROWS_LOG_INTERVAL)

# The TEMP tables the triggers fill. They live on the write connection only and are
# emptied before every commit, so they never reach the DB file or a checkpoint.
_changed = MetaData(schema="temp")
changed_jobs = Table("observability_changed_jobs", _changed, Column("job_id", JobNameType, primary_key=True))
changed_configs = Table("observability_changed_configs", _changed, Column("job_id", JobNameType, primary_key=True))
changed_tasks = Table("observability_changed_tasks", _changed, Column("task_id", JobNameType, primary_key=True))
changed_attempts = Table(
    "observability_changed_attempts",
    _changed,
    Column("task_id", JobNameType, primary_key=True),
    Column("attempt_id", Integer, primary_key=True),
)
# Attempts' workers as they were before worker removal cleared them.
removed_workers = Table(
    "observability_removed_workers",
    _changed,
    Column("task_id", JobNameType, primary_key=True),
    Column("attempt_id", Integer, primary_key=True),
    Column("worker_id", WorkerIdType, nullable=False),
)
_CHANGED_TABLES = (changed_jobs, changed_configs, changed_tasks, changed_attempts, removed_workers)
# The snapshot queries select with ``key IN (SELECT ... FROM temp table)`` rather than a
# join: SQLite has no statistics for the TEMP tables, and with a join it may scan the
# 100k-row ``task_attempts`` table and probe the TEMP table per row (~50 ms a
# transaction). The IN form always starts from the few changed keys.


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


class CaptureCursor(Protocol):
    """The write-transaction surface capture needs (``db.Tx``)."""

    lock_cost: CostWindow | None

    def execute(self, stmt, params=None) -> CursorResult: ...

    def register(self, hook: Callable[[], None]) -> None: ...


@dataclass
class _Rows:
    """One transaction's snapshotted rows."""

    jobs: list[IrisJob] = field(default_factory=list)
    configs: list[IrisJobConfig] = field(default_factory=list)
    tasks: list[IrisTask] = field(default_factory=list)
    attempts: list[IrisTaskAttempt] = field(default_factory=list)


def _differs(columns: list[str]) -> str:
    return " OR ".join(f'OLD."{c}" IS NOT NEW."{c}"' for c in columns)


def _trigger(name: str, event: str, table: str, insert: str, when: str | None = None) -> str:
    when_clause = f" WHEN {when}" if when else ""
    return (
        f"CREATE TEMP TRIGGER IF NOT EXISTS observability_{name} AFTER {event} ON main.{table}{when_clause} "
        f"BEGIN INSERT OR IGNORE INTO {insert}; END"
    )


def _trigger_ddl() -> list[str]:
    job_columns = [c.name for c in jobs_table.columns]
    attempt_columns = [c.name for c in task_attempts_table.columns if c.name != "worker_id"]
    task_columns = ", ".join(f'"{c}"' for c in TASK_CHANGE_COLUMNS)
    job = f"{changed_jobs.name} VALUES (NEW.job_id)"
    config = f"{changed_configs.name} VALUES (NEW.job_id)"
    task = f"{changed_tasks.name} VALUES (NEW.task_id)"
    attempt = f"{changed_attempts.name} VALUES (NEW.task_id, NEW.attempt_id)"
    return [
        _trigger("job_insert", "INSERT", "jobs", job),
        _trigger("job_update", "UPDATE", "jobs", job, _differs(job_columns)),
        _trigger("config_insert", "INSERT", "job_config", config),
        _trigger("task_insert", "INSERT", "tasks", task),
        _trigger("task_update", f"UPDATE OF {task_columns}", "tasks", task, _differs(list(TASK_CHANGE_COLUMNS))),
        _trigger("attempt_insert", "INSERT", "task_attempts", attempt),
        _trigger(
            "attempt_update",
            "UPDATE",
            "task_attempts",
            attempt,
            f"{_differs(attempt_columns)} OR (OLD.worker_id IS NOT NEW.worker_id AND NEW.worker_id IS NOT NULL)",
        ),
        _trigger(
            "attempt_worker_removed",
            "UPDATE OF worker_id",
            "task_attempts",
            f"{removed_workers.name} VALUES (NEW.task_id, NEW.attempt_id, OLD.worker_id)",
            "OLD.worker_id IS NOT NULL AND NEW.worker_id IS NULL",
        ),
    ]


def install_capture(conn: Connection) -> None:
    """Create the TEMP tables and triggers on ``conn`` once per DBAPI connection.

    Runs outside a transaction (the write engine autocommits between transactions),
    so a later rollback cannot undo it. ``conn.info`` belongs to the DBAPI
    connection, so a reconnect after a reopen or restore installs again.
    """
    if conn.info.get(_INSTALLED_KEY):
        return
    _changed.create_all(conn)
    for statement in _trigger_ddl():
        conn.execute(text(statement))
    conn.info[_INSTALLED_KEY] = True


def capture_changes(cur: CaptureCursor, tables: ObservabilityRowTables) -> None:
    """Snapshot this transaction's changed rows, empty the TEMP tables, and write the rows after commit.

    Called just before ``COMMIT``, under the write lock. A transaction with rows gets
    one cost record (see :mod:`iris.cluster.controller.write_lock_cost`) whose snapshot
    time includes the check for changed keys. That check also runs on transactions
    with no changes, which have no record, so its time there counts only as lock time,
    as do the triggers' own inserts, which run inside the controller's statements.
    """
    lock_cost = cur.lock_cost
    assert lock_cost is not None, "observability rows need a write transaction"
    started = time.monotonic()
    if not cur.execute(select(_any_changed())).scalar():
        return
    rows = _Rows()
    try:
        rows = _snapshot(cur, tables)
    except Exception:
        logger.exception("observability rows: snapshot failed, dropping this transaction's rows")
        rows = _Rows()
    finally:
        for table in _CHANGED_TABLES:
            cur.execute(delete(table))
    snapshot_seconds = time.monotonic() - started
    cur.register(lambda: _write(tables, rows, snapshot_seconds, lock_cost))


def _any_changed():
    return or_(*(exists().select_from(table) for table in _CHANGED_TABLES))


def _write(tables: ObservabilityRowTables, rows: _Rows, snapshot_seconds: float, lock_cost: CostWindow) -> None:
    """Enqueue the snapshotted rows; a table whose write fails is logged and its rows dropped.

    This runs after commit but still under the write lock, so its time counts with
    the snapshot's as the rows' cost: the transaction's row counts and both times go
    to ``lock_cost`` as one record, and a slow total is logged.
    """
    started = time.monotonic()
    for table, batch in (
        (tables.job_table, rows.jobs),
        (tables.job_config_table, rows.configs),
        (tables.task_table, rows.tasks),
        (tables.task_attempt_table, rows.attempts),
    ):
        if not batch:
            continue
        try:
            table.write(batch)
        # This runs among the post-commit hooks: raising would reach the caller of a
        # committed transaction and skip the hooks registered after it (cache invalidations).
        except Exception:
            logger.exception(
                "observability rows: write failed, dropping %d %s rows", len(batch), type(batch[0]).__name__
            )
    cost = RowTransactionCost(
        job_rows=len(rows.jobs),
        job_config_rows=len(rows.configs),
        task_rows=len(rows.tasks),
        attempt_rows=len(rows.attempts),
        snapshot_seconds=snapshot_seconds,
        enqueue_seconds=time.monotonic() - started,
    )
    count = cost.job_rows + cost.job_config_rows + cost.task_rows + cost.attempt_rows
    if count == 0:
        # Changed keys that turned out to need no row: only jobs outside this cluster,
        # or a snapshot that failed and was dropped.
        return
    lock_cost.add_row_transaction(cost)
    seconds = cost.snapshot_seconds + cost.enqueue_seconds
    if seconds >= SLOW_ROWS_SECONDS and _slow_rows_log.should_run():
        logger.warning(
            "observability rows: %d rows took %.0f ms under the write lock "
            "(at most one such warning per %.0f s; set finelog.controller_rows: false to turn the rows off)",
            count,
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


def _snapshot(cur: CaptureCursor, tables: ObservabilityRowTables) -> _Rows:
    ts = stats_timestamp()
    return _Rows(
        jobs=_snapshot_jobs(cur, tables, ts),
        configs=_snapshot_configs(cur, tables, ts),
        tasks=_snapshot_tasks(cur, tables, ts),
        attempts=_snapshot_attempts(cur, tables, ts),
    )


def _snapshot_jobs(cur: CaptureCursor, tables: ObservabilityRowTables, ts: datetime) -> list[IrisJob]:
    j = jobs_table.c
    query = select(jobs_table).where(j.job_id.in_(select(changed_jobs.c.job_id)), j.cluster == LOCAL_CLUSTER)
    return [
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
        for r in cur.execute(query)
    ]


def _snapshot_configs(cur: CaptureCursor, tables: ObservabilityRowTables, ts: datetime) -> list[IrisJobConfig]:
    j, c = jobs_table.c, job_config_table.c
    query = (
        select(job_config_table, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms)
        .join(jobs_table, j.job_id == c.job_id)
        .where(c.job_id.in_(select(changed_configs.c.job_id)), j.cluster == LOCAL_CLUSTER)
    )
    rows = []
    for r in cur.execute(query):
        accelerator = _accelerator(r.res_device_json)
        provenance = _launch_provenance(r.job_id.to_wire(), r.environment_json)
        rows.append(
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
    return rows


def _snapshot_tasks(cur: CaptureCursor, tables: ObservabilityRowTables, ts: datetime) -> list[IrisTask]:
    j, t = jobs_table.c, tasks_table.c
    query = (
        select(tasks_table, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms.label("job_submitted_at_ms"))
        .join(jobs_table, j.job_id == t.job_id)
        .where(t.task_id.in_(select(changed_tasks.c.task_id)), j.cluster == LOCAL_CLUSTER)
    )
    return [
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
        for r in cur.execute(query)
    ]


def _snapshot_attempts(cur: CaptureCursor, tables: ObservabilityRowTables, ts: datetime) -> list[IrisTaskAttempt]:
    j, a, t, ch, rw = jobs_table.c, task_attempts_table.c, tasks_table.c, changed_attempts.c, removed_workers.c
    query = (
        select(task_attempts_table, rw.worker_id.label("removed_worker_id"), t.job_id, j.root_job_id)
        .add_columns(j.root_submitted_at_ms, j.submitted_at_ms)
        .join(tasks_table, t.task_id == a.task_id)
        .join(jobs_table, j.job_id == t.job_id)
        .outerjoin(removed_workers, (rw.task_id == a.task_id) & (rw.attempt_id == a.attempt_id))
        .where(tuple_(a.task_id, a.attempt_id).in_(select(ch.task_id, ch.attempt_id)), j.cluster == LOCAL_CLUSTER)
    )
    rows = []
    for r in cur.execute(query):
        worker_id = r.worker_id if r.worker_id is not None else r.removed_worker_id
        rows.append(
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
    return rows
