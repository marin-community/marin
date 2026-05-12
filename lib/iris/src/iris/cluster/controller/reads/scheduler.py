# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-tick read helpers.

Named SA Core ``text`` constants and helper functions for the hot
per-tick queries driven by the scheduler loop. Stage 5 of the
SQLAlchemy Core migration introduced this module with the
``_jobs_with_reservations`` port (the perf canary). Stage 9 adds the
heavier reads: ``resource_usage_by_worker``,
``reconcile_rows_for_workers``, plus the shared
``running_tasks_by_worker`` / ``timed_out_executing_tasks`` helpers
that used to live in :mod:`db`.

The hot-path queries use ``text(...)`` with bound parameters rather
than ``select(...)`` Core ORM expressions. The two forms produce
identical SQL but ``text()`` avoids ~370 µs/call of per-call statement
compilation overhead that ``select()`` reintroduces even after caching.
``bindparams(expanding=True)`` lets one compiled statement service
``IN (?, ?, ...)`` calls with varying list lengths.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp
from sqlalchemy import bindparam, text

from iris.cluster.controller.codec import device_counts_from_json
from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import JobReservationRow
from iris.cluster.controller.stores import ReconcileRow, WorkerResourceUsage
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Reservation-holding jobs (Stage 5 canary)
# ---------------------------------------------------------------------------

# Slim 2-column projection for the per-tick reservation-claim recomputation.
# Filters on ``jobs.has_reservation = 1`` (partial index
# ``idx_jobs_has_reservation``) and joins ``job_config`` solely to pull
# ``reservation_json``. The expanding ``:states`` bind lets SA reuse the
# compiled statement across different state-tuple lengths.
_JOBS_WITH_RESERVATIONS_SQL = text(
    "SELECT j.job_id AS job_id, jc.reservation_json AS reservation_json "
    "FROM jobs j JOIN job_config jc ON j.job_id = jc.job_id "
    "WHERE j.state IN :states AND j.has_reservation = 1"
).bindparams(bindparam("states", expanding=True))


def jobs_with_reservations(tx: Tx, states: tuple[int, ...]) -> list[JobReservationRow]:
    """Fetch ``(job_id, reservation_json)`` for reservation-holding jobs in ``states``."""
    rows = tx.execute(_JOBS_WITH_RESERVATIONS_SQL, {"states": list(states)}).all()
    return [
        JobReservationRow(job_id=JobName.from_wire(row.job_id), reservation_json=row.reservation_json) for row in rows
    ]


# ---------------------------------------------------------------------------
# Per-worker resource usage (Stage 9 hot path)
# ---------------------------------------------------------------------------

# Two-step query mirroring ``TaskAttemptStore.resource_usage_by_worker``:
# the inline ``JOIN jobs ON is_reservation_holder = 0`` is intentionally
# avoided because it drives SQLite from the ``jobs`` table (full scan ~24k
# rows on production) and pushes the read from ~3 ms to ~380 ms. The small
# set of reservation-holder job ids is fetched once and filtered in Python.
_HOLDER_JOBS_SQL = text("SELECT job_id FROM jobs WHERE is_reservation_holder = 1")
_RESOURCE_USAGE_SQL = text(
    "SELECT ta.worker_id AS worker_id, t.job_id AS job_id, "
    "jc.res_cpu_millicores AS res_cpu_millicores, "
    "jc.res_memory_bytes AS res_memory_bytes, "
    "jc.res_device_json AS res_device_json "
    "FROM task_attempts ta "
    "JOIN tasks t ON t.task_id = ta.task_id "
    "JOIN job_config jc ON jc.job_id = t.job_id "
    "WHERE ta.worker_id IS NOT NULL AND ta.finished_at_ms IS NULL"
)


def resource_usage_by_worker(tx: Tx) -> dict[WorkerId, WorkerResourceUsage]:
    """Aggregate resources held by unfinished worker-bound attempts.

    See :meth:`stores.TaskAttemptStore.resource_usage_by_worker` for the
    full rationale; this is a verbatim SA Core port. The partial index
    ``idx_task_attempts_live_workerbound`` (migration 0045) drives the
    second query; reservation-holder rows are skipped in Python.
    """
    holder_rows = tx.execute(_HOLDER_JOBS_SQL).all()
    holder_jobs = {str(row.job_id) for row in holder_rows}
    rows = tx.execute(_RESOURCE_USAGE_SQL).all()

    cpu: dict[WorkerId, int] = {}
    mem: dict[WorkerId, int] = {}
    gpu: dict[WorkerId, int] = {}
    tpu: dict[WorkerId, int] = {}
    for row in rows:
        if str(row.job_id) in holder_jobs:
            continue
        wid = WorkerId(str(row.worker_id))
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
# Per-worker reconcile rows (Stage 9 hot path)
# ---------------------------------------------------------------------------

# ASSIGNED/BUILDING/RUNNING tuple is hoisted as a static bind because the
# scheduler tick always passes the same three states (matching the legacy
# ``WHERE t.state IN (?, ?, ?)`` literal). Keeping ``worker_ids`` as a
# Python-side filter is intentional — see the legacy method's docstring for
# the index regression that prompted that decision.
_RECONCILE_ROWS_SQL = text(
    "SELECT ta.worker_id AS worker_id, t.task_id AS task_id, ta.attempt_id AS attempt_id, "
    "t.state AS task_state, ta.state AS attempt_state, t.job_id AS job_id "
    "FROM task_attempts ta "
    "JOIN tasks t "
    "  ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
    "WHERE ta.worker_id IS NOT NULL AND ta.finished_at_ms IS NULL "
    "  AND t.state IN :states"
).bindparams(bindparam("states", expanding=True))


_RECONCILE_TASK_STATES = (
    int(job_pb2.TASK_STATE_ASSIGNED),
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)


def reconcile_rows_for_workers(tx: Tx, worker_ids: Sequence[WorkerId]) -> list[ReconcileRow]:
    """Snapshot current attempts for ``worker_ids``.

    Verbatim SA Core port of
    :meth:`stores.TaskAttemptStore.reconcile_rows_for_workers`. Filters
    out workers not in ``worker_ids`` in Python so the SQL keeps using
    the partial index ``idx_task_attempts_live_workerbound`` rather than
    falling back to a scan on a long ``IN`` list.
    """
    if not worker_ids:
        return []
    wire_ids = {str(w) for w in worker_ids}
    rows = tx.execute(_RECONCILE_ROWS_SQL, {"states": list(_RECONCILE_TASK_STATES)}).all()
    return [
        ReconcileRow(
            worker_id=WorkerId(str(row.worker_id)),
            task_id=JobName.from_wire(str(row.task_id)),
            attempt_id=int(row.attempt_id),
            task_state=int(row.task_state),
            attempt_state=int(row.attempt_state),
            job_id=JobName.from_wire(str(row.job_id)),
        )
        for row in rows
        if str(row.worker_id) in wire_ids
    ]


# ---------------------------------------------------------------------------
# Shared read helpers (Stage 9 — copied alongside the originals in db.py).
# ---------------------------------------------------------------------------

# Matches ``ACTIVE_TASK_STATES`` from ``db.py``. Hoisted as a tuple so the
# expanding ``:states`` bind compiles once.
_ACTIVE_TASK_STATES = (
    int(job_pb2.TASK_STATE_ASSIGNED),
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)


_RUNNING_TASKS_SQL = text(
    "SELECT t.current_worker_id AS worker_id, t.task_id AS task_id "
    "FROM tasks t "
    "WHERE t.current_worker_id IN :worker_ids AND t.state IN :states"
).bindparams(
    bindparam("worker_ids", expanding=True),
    bindparam("states", expanding=True),
)


def running_tasks_by_worker(tx: Tx, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker.

    SA Core port of :func:`db.running_tasks_by_worker` — same behaviour,
    takes a ``Tx`` rather than a ``ControllerDB`` so callers reuse an
    existing read snapshot.
    """
    if not worker_ids:
        return {}
    wires = [str(wid) for wid in worker_ids]
    rows = tx.execute(
        _RUNNING_TASKS_SQL,
        {"worker_ids": wires, "states": list(_ACTIVE_TASK_STATES)},
    ).all()
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[WorkerId(str(row.worker_id))].add(JobName.from_wire(str(row.task_id)))
    return running


# ``EXECUTING_TASK_STATES`` literal hoisted for the expanding bind.
_EXECUTING_TASK_STATES = (
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)


_TIMED_OUT_SQL = text(
    "SELECT t.task_id AS task_id, t.current_worker_id AS worker_id, "
    "ta.started_at_ms AS attempt_started_at_ms, jc.timeout_ms AS timeout_ms "
    "FROM tasks t "
    "JOIN job_config jc ON jc.job_id = t.job_id "
    "JOIN task_attempts ta ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id "
    "WHERE t.state IN :states "
    "AND jc.timeout_ms IS NOT NULL AND jc.timeout_ms > 0 "
    "AND ta.started_at_ms IS NOT NULL"
).bindparams(bindparam("states", expanding=True))


# Local copy of ``db.TimedOutTask`` — kept here so the SA Core read does
# not depend on the legacy module. The dataclass shape is identical so
# call sites can swap implementations transparently in Stage 13.
@dataclass(frozen=True, slots=True)
class TimedOutTask:
    """A running task that has exceeded its execution timeout."""

    task_id: JobName
    worker_id: WorkerId | None


def timed_out_executing_tasks(tx: Tx, now: Timestamp) -> list[TimedOutTask]:
    """Find executing tasks whose current attempt exceeded the job's execution timeout.

    SA Core port of :func:`db.timed_out_executing_tasks`.
    """
    now_ms = now.epoch_ms()
    rows = tx.execute(_TIMED_OUT_SQL, {"states": list(_EXECUTING_TASK_STATES)}).all()
    result: list[TimedOutTask] = []
    for row in rows:
        attempt_started_at_ms = int(row.attempt_started_at_ms)
        timeout_ms = int(row.timeout_ms)
        if attempt_started_at_ms + timeout_ms <= now_ms:
            worker_id = WorkerId(str(row.worker_id)) if row.worker_id is not None else None
            result.append(TimedOutTask(task_id=JobName.from_wire(str(row.task_id)), worker_id=worker_id))
    return result
