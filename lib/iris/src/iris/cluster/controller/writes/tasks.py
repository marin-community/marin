# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write helpers for the ``tasks`` table."""

from sqlalchemy import func, insert, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import task_attempts_table, tasks_table
from iris.cluster.controller.writes import writes_to
from iris.cluster.controller.writes.task_attempts import insert_attempt
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2


@writes_to(tasks_table)
def insert_task(
    tx: Tx,
    *,
    task_id: JobName,
    job_id: JobName,
    task_index: int,
    state: int,
    submitted_at_ms: int,
    max_retries_failure: int,
    max_retries_preemption: int,
    priority_neg_depth: int,
    priority_root_submitted_ms: int,
    priority_insertion: int,
    priority_band: int,
) -> None:
    """Insert one row into ``tasks``."""
    tx.execute(
        insert(tasks_table).values(
            task_id=task_id,
            job_id=job_id,
            task_index=task_index,
            state=state,
            error=None,
            exit_code=None,
            submitted_at_ms=submitted_at_ms,
            started_at_ms=None,
            finished_at_ms=None,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            failure_count=0,
            preemption_count=0,
            current_attempt_id=-1,
            priority_neg_depth=priority_neg_depth,
            priority_root_submitted_ms=priority_root_submitted_ms,
            priority_insertion=priority_insertion,
            priority_band=priority_band,
        )
    )


@writes_to(tasks_table, task_attempts_table)
def assign_task(
    tx: Tx,
    task_id: JobName,
    worker_id: WorkerId | None,
    worker_address: str | None,
    attempt_id: int,
    now_ms: int,
    priority_band: int | None = None,
) -> None:
    """Insert a fresh ``task_attempts`` row and move the task to ASSIGNED.

    A single transaction creates the attempt row and stamps the task with
    the worker / attempt fields. ``priority_band`` is stamped at assign
    time so the preemption pass treats a running task's band as fixed;
    ``None`` leaves the column untouched.
    """
    insert_attempt(
        tx,
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=worker_id,
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=now_ms,
    )
    values: dict = {
        "state": job_pb2.TASK_STATE_ASSIGNED,
        "current_attempt_id": attempt_id,
        "started_at_ms": func.coalesce(tasks_table.c.started_at_ms, now_ms),
    }
    if worker_id is not None:
        values["current_worker_id"] = worker_id
        values["current_worker_address"] = worker_address
    if priority_band is not None:
        values["priority_band"] = priority_band
    tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(**values))


@writes_to(tasks_table)
def apply_state_update(
    tx: Tx,
    *,
    task_id: JobName,
    state: int,
    error: str | None,
    exit_code: int | None,
    started_at_ms: int | None,
    finished_at_ms: int | None,
    failure_count: int,
    preemption_count: int,
    active_states: set[int],
) -> None:
    """Apply a computed task state update.

    Active target states preserve ``current_worker_id`` /
    ``current_worker_address``; non-active states clear them so the row
    is consistent with terminal-transition writes.
    """
    values: dict = {
        "state": state,
        "error": func.coalesce(error, tasks_table.c.error),
        "exit_code": func.coalesce(exit_code, tasks_table.c.exit_code),
        "started_at_ms": func.coalesce(tasks_table.c.started_at_ms, started_at_ms),
        "finished_at_ms": finished_at_ms,
        "failure_count": failure_count,
        "preemption_count": preemption_count,
    }
    if state not in active_states:
        values["current_worker_id"] = None
        values["current_worker_address"] = None
    tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(**values))
