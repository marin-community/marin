# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-tick read helpers.

Return shapes:

* ``resource_usage_by_worker`` — ``dict[WorkerId, WorkerResourceUsage]``
* ``running_tasks_by_worker`` — ``dict[WorkerId, set[JobName]]``

Performance notes:

* ``resource_usage_by_worker`` uses a two-step approach (fetch
  reservation-holder ids, then filter in Python) to avoid a JOIN-driven
  full-table scan — see inline comment.
"""

from sqlalchemy import select

from iris.cluster.controller.codec import device_counts_from_json
from iris.cluster.controller.db import Tx
from iris.cluster.controller.rows import WorkerResourceUsage
from iris.cluster.controller.schema import (
    job_config_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Per-worker resource usage
# ---------------------------------------------------------------------------


def resource_usage_by_worker(tx: Tx) -> dict[WorkerId, WorkerResourceUsage]:
    """Aggregate resources held by unfinished worker-bound attempts.

    Reservation-holder job rows are excluded (filtered in Python, not SQL).
    Two-step approach: the inline ``JOIN jobs ON is_reservation_holder = 0``
    is intentionally avoided because it drives SQLite from the ``jobs`` table
    (full scan ~24k rows on production) and pushes the read from ~3 ms to
    ~380 ms. The small set of reservation-holder job ids is fetched once and
    filtered in Python.
    """
    holder_rows = tx.execute(
        select(jobs_table.c.job_id).where(jobs_table.c.is_reservation_holder == True)  # noqa: E712
    ).all()
    holder_jobs: set[JobName] = {row.job_id for row in holder_rows}
    rows = tx.execute(
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
    ).all()

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
# Shared read helpers
# ---------------------------------------------------------------------------

_ACTIVE_TASK_STATES = (
    int(job_pb2.TASK_STATE_ASSIGNED),
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)


def running_tasks_by_worker(tx: Tx, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker."""
    if not worker_ids:
        return {}
    rows = tx.execute(
        select(tasks_table.c.current_worker_id.label("worker_id"), tasks_table.c.task_id).where(
            tasks_table.c.current_worker_id.in_(list(worker_ids)),
            tasks_table.c.state.in_(list(_ACTIVE_TASK_STATES)),
        ),
    ).all()
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
    return running
