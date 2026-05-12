# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-tick read helpers (SA Core expression language).

All queries use ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on schema_v2 columns decode values on read.

Return shapes:

* ``jobs_with_reservations`` — ``list[Row]`` (job_id, reservation_json)
* ``resource_usage_by_worker`` — ``dict[WorkerId, WorkerResourceUsage]``
* ``reconcile_rows_for_workers`` — ``list[ReconcileRow]``
* ``running_tasks_by_worker`` — ``dict[WorkerId, set[JobName]]``
* ``timed_out_executing_tasks`` — ``list[TimedOutTask]``

Performance notes:

* ``resource_usage_by_worker`` still uses a two-step approach (fetch
  reservation-holder ids, then filter in Python) to avoid a JOIN-driven
  full-table scan — see inline comment.
* ``reconcile_rows_for_workers`` filters workers in Python to keep the
  partial index ``idx_task_attempts_live_workerbound`` in play.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp
from sqlalchemy import bindparam, select

from iris.cluster.controller.codec import device_counts_from_json
from iris.cluster.controller.db import Tx
from iris.cluster.controller.rows import ReconcileRow, WorkerResourceUsage
from iris.cluster.controller.schema import (
    job_config_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Reservation-holding jobs
# ---------------------------------------------------------------------------

# Slim 2-column projection for the per-tick reservation-claim recomputation.
# Filters on ``jobs.has_reservation = 1`` (partial index
# ``idx_jobs_has_reservation``) and joins ``job_config`` solely to pull
# ``reservation_json``. The expanding ``:states`` bind lets SA reuse the
# compiled statement across different state-tuple lengths.
JOBS_WITH_RESERVATIONS_QUERY = (
    select(jobs_table.c.job_id, job_config_table.c.reservation_json)
    .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
    .where(
        jobs_table.c.state.in_(bindparam("states", expanding=True)),
        jobs_table.c.has_reservation == True,  # noqa: E712 — SQLAlchemy requires == not `is`
    )
)


def jobs_with_reservations(tx: Tx, states: tuple[int, ...]) -> list:
    """Return ``(job_id, reservation_json)`` for reservation-holding jobs in ``states``.

    Returns list[Row]. TypeDecorators decode job_id to JobName.
    """
    return tx.execute(JOBS_WITH_RESERVATIONS_QUERY, {"states": list(states)}).all()


# ---------------------------------------------------------------------------
# Per-worker resource usage
# ---------------------------------------------------------------------------

# Two-step query mirroring ``TaskAttemptStore.resource_usage_by_worker``:
# the inline ``JOIN jobs ON is_reservation_holder = 0`` is intentionally
# avoided because it drives SQLite from the ``jobs`` table (full scan ~24k
# rows on production) and pushes the read from ~3 ms to ~380 ms. The small
# set of reservation-holder job ids is fetched once and filtered in Python.
HOLDER_JOBS_QUERY = select(jobs_table.c.job_id).where(jobs_table.c.is_reservation_holder == True)  # noqa: E712

RESOURCE_USAGE_QUERY = (
    select(
        task_attempts_table.c.worker_id,
        tasks_table.c.job_id,
        job_config_table.c.res_cpu_millicores,
        job_config_table.c.res_memory_bytes,
        job_config_table.c.res_device_json,
    )
    .select_from(
        task_attempts_table.join(tasks_table, tasks_table.c.task_id == task_attempts_table.c.task_id).join(
            job_config_table, job_config_table.c.job_id == tasks_table.c.job_id
        )
    )
    .where(
        task_attempts_table.c.worker_id.is_not(None),
        task_attempts_table.c.finished_at_ms.is_(None),
    )
)


def resource_usage_by_worker(tx: Tx) -> dict[WorkerId, WorkerResourceUsage]:
    """Aggregate resources held by unfinished worker-bound attempts.

    Returns dict[WorkerId, WorkerResourceUsage]. Reservation-holder job rows
    are excluded (filtered in Python, not SQL — see module note).
    """
    holder_rows = tx.execute(HOLDER_JOBS_QUERY).all()
    # job_id column uses JobNameType so row.job_id is already a JobName.
    holder_jobs: set[JobName] = {row.job_id for row in holder_rows}
    rows = tx.execute(RESOURCE_USAGE_QUERY).all()

    cpu: dict[WorkerId, int] = {}
    mem: dict[WorkerId, int] = {}
    gpu: dict[WorkerId, int] = {}
    tpu: dict[WorkerId, int] = {}
    for row in rows:
        if row.job_id in holder_jobs:
            continue
        # WorkerIdType decodes worker_id to WorkerId already.
        wid: WorkerId = row.worker_id
        cpu[wid] = cpu.get(wid, 0) + int(row.res_cpu_millicores)
        mem[wid] = mem.get(wid, 0) + int(row.res_memory_bytes)
        counts = device_counts_from_json(row.res_device_json)
        gpu[wid] = gpu.get(wid, 0) + counts.gpu
        tpu[wid] = tpu.get(wid, 0) + counts.tpu
    return {
        wid: WorkerResourceUsage(
            cpu_millicores=cpu.get(wid, 0),
            memory_bytes=mem.get(wid, 0),
            gpu_count=gpu.get(wid, 0),
            tpu_count=tpu.get(wid, 0),
        )
        for wid in cpu.keys() | mem.keys() | gpu.keys() | tpu.keys()
    }


# ---------------------------------------------------------------------------
# Per-worker reconcile rows
# ---------------------------------------------------------------------------

_RECONCILE_TASK_STATES = (
    int(job_pb2.TASK_STATE_ASSIGNED),
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)

# ASSIGNED/BUILDING/RUNNING filter is static; worker_ids are filtered in Python
# to keep the partial index ``idx_task_attempts_live_workerbound`` in play —
# a long IN list on worker_id degrades to a scan.
RECONCILE_ROWS_QUERY = (
    select(
        task_attempts_table.c.worker_id,
        tasks_table.c.task_id,
        task_attempts_table.c.attempt_id,
        tasks_table.c.state.label("task_state"),
        task_attempts_table.c.state.label("attempt_state"),
        tasks_table.c.job_id,
    )
    .select_from(
        task_attempts_table.join(
            tasks_table,
            (tasks_table.c.task_id == task_attempts_table.c.task_id)
            & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
        )
    )
    .where(
        task_attempts_table.c.worker_id.is_not(None),
        task_attempts_table.c.finished_at_ms.is_(None),
        tasks_table.c.state.in_(bindparam("states", expanding=True)),
    )
)


def reconcile_rows_for_workers(tx: Tx, worker_ids: Sequence[WorkerId]) -> list[ReconcileRow]:
    """Snapshot current attempts for ``worker_ids``.

    Returns list[ReconcileRow]. Workers not in ``worker_ids`` are filtered
    in Python so the partial index ``idx_task_attempts_live_workerbound``
    remains active rather than falling back to a scan on a long IN list.
    """
    if not worker_ids:
        return []
    target_ids: set[WorkerId] = set(worker_ids)
    rows = tx.execute(RECONCILE_ROWS_QUERY, {"states": list(_RECONCILE_TASK_STATES)}).all()
    return [
        ReconcileRow(
            worker_id=row.worker_id,
            task_id=row.task_id,
            attempt_id=int(row.attempt_id),
            task_state=int(row.task_state),
            attempt_state=int(row.attempt_state),
            job_id=row.job_id,
        )
        for row in rows
        if row.worker_id in target_ids
    ]


# ---------------------------------------------------------------------------
# Shared read helpers
# ---------------------------------------------------------------------------

_ACTIVE_TASK_STATES = (
    int(job_pb2.TASK_STATE_ASSIGNED),
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)

RUNNING_TASKS_QUERY = select(tasks_table.c.current_worker_id.label("worker_id"), tasks_table.c.task_id).where(
    tasks_table.c.current_worker_id.in_(bindparam("worker_ids", expanding=True)),
    tasks_table.c.state.in_(bindparam("states", expanding=True)),
)


def running_tasks_by_worker(tx: Tx, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker.

    Returns dict[WorkerId, set[JobName]]. TypeDecorators decode worker_id to
    WorkerId and task_id to JobName.
    """
    if not worker_ids:
        return {}
    rows = tx.execute(
        RUNNING_TASKS_QUERY,
        {"worker_ids": list(worker_ids), "states": list(_ACTIVE_TASK_STATES)},
    ).all()
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
    return running


_EXECUTING_TASK_STATES = (
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)

TIMED_OUT_QUERY = (
    select(
        tasks_table.c.task_id,
        tasks_table.c.current_worker_id,
        task_attempts_table.c.started_at_ms,
        job_config_table.c.timeout_ms,
    )
    .select_from(
        tasks_table.join(job_config_table, job_config_table.c.job_id == tasks_table.c.job_id).join(
            task_attempts_table,
            (task_attempts_table.c.task_id == tasks_table.c.task_id)
            & (task_attempts_table.c.attempt_id == tasks_table.c.current_attempt_id),
        )
    )
    .where(
        tasks_table.c.state.in_(bindparam("states", expanding=True)),
        job_config_table.c.timeout_ms.is_not(None),
        job_config_table.c.timeout_ms > 0,
        task_attempts_table.c.started_at_ms.is_not(None),
    )
)


# Local dataclass for timed-out tasks — kept here so the SA Core read does
# not import the legacy ``db`` module just for this shape.
@dataclass(frozen=True, slots=True)
class TimedOutTask:
    """A running task that has exceeded its execution timeout."""

    task_id: JobName
    worker_id: WorkerId | None


def timed_out_executing_tasks(tx: Tx, now: Timestamp) -> list[TimedOutTask]:
    """Return executing tasks whose current attempt has exceeded the job's execution timeout.

    Returns list[TimedOutTask]. Comparison is done in Python against ``now``
    to avoid converting now -> raw int in the SQL bind.
    """
    now_ms = now.epoch_ms()
    rows = tx.execute(TIMED_OUT_QUERY, {"states": list(_EXECUTING_TASK_STATES)}).all()
    result: list[TimedOutTask] = []
    for row in rows:
        # started_at_ms is decoded by TimestampMsType to a Timestamp.
        attempt_started_at_ms = row.started_at_ms.epoch_ms()
        timeout_ms = int(row.timeout_ms)
        if attempt_started_at_ms + timeout_ms <= now_ms:
            result.append(TimedOutTask(task_id=row.task_id, worker_id=row.current_worker_id))
    return result
