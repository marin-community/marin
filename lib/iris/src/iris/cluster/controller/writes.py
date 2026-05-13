# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write-side helpers: module-level functions decorated with ``@writes_to``.

The :func:`writes_to` decorator records the table set on the function as
``fn.writes_to`` / ``fn.cascades_into`` and appends the function to
:data:`REGISTERED_WRITE_FUNCTIONS`. The startup check in
``projections/__init__.py`` walks that registry and the ``PROJECTIONS``
list to verify no Projection-owned table is written outside its Projection.

Areas covered (previously split across writes/<entity>.py):
  jobs           — jobs, job_config, meta sequence
  task_attempts  — task_attempts
  tasks          — tasks (insert, assign, state update)
  workers        — workers, worker_attributes
  budgets        — users, user_budgets (previously db.py methods)
"""

from collections.abc import Callable

from rigging.timing import Timestamp
from sqlalchemy import Table, delete, func, insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.controller.db import ACTIVE_TASK_STATES, Tx
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import (
    USER_ROLE_DEFAULT,
    job_config_table,
    jobs_table,
    meta_table,
    task_attempts_table,
    tasks_table,
    user_budgets_table,
    users_table,
    workers_table,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

REGISTERED_WRITE_FUNCTIONS: list[Callable] = []


def writes_to(
    *tables: Table,
    cascades_into: tuple[Table, ...] = (),
) -> Callable:
    """Mark a write function with the tables it mutates.

    Pure metadata. The startup-time owned-table check in
    ``projections/__init__.py`` reads ``fn.writes_to`` and
    ``fn.cascades_into`` to verify no Projection-owned table is written
    outside its Projection.

    ``cascades_into`` lists tables mutated via FK ``ON DELETE CASCADE``
    by writes to ``tables``; the check treats them identically to direct
    writes.
    """

    def deco(fn: Callable) -> Callable:
        fn.writes_to = tables  # type: ignore[attr-defined]
        fn.cascades_into = cascades_into  # type: ignore[attr-defined]
        REGISTERED_WRITE_FUNCTIONS.append(fn)
        return fn

    return deco


# ---------------------------------------------------------------------------
# Meta sequence (shared by jobs and priority insertion)
# ---------------------------------------------------------------------------


@writes_to(meta_table)
def meta_sequence_bump(tx: Tx, key: str) -> int:
    """Bump the named sequence in ``meta`` and return the new value.

    If the key is absent it is inserted with value 1. Callers reserving N
    task slots use ``base + i`` for ``i in range(N)``.
    """
    row = tx.execute(select(meta_table.c.value).where(meta_table.c.key == key)).fetchone()
    if row is None:
        tx.execute(insert(meta_table).values(key=key, value=1))
        return 1
    value = int(row[0]) + 1
    tx.execute(update(meta_table).where(meta_table.c.key == key).values(value=value))
    return value


# ---------------------------------------------------------------------------
# Job writes (previously writes/jobs.py)
# ---------------------------------------------------------------------------

_PRIORITY_INSERTION_KEY = "task_priority_insertion"


@writes_to(jobs_table)
def insert_job(
    tx: Tx,
    *,
    job_id: JobName,
    user_id: str,
    parent_job_id: JobName | None,
    root_job_id: str,
    depth: int,
    state: int,
    submitted_at_ms: int,
    root_submitted_at_ms: int,
    started_at_ms: int | None,
    finished_at_ms: int | None,
    scheduling_deadline_epoch_ms: int | None,
    error: str | None,
    exit_code: int | None,
    num_tasks: int,
    is_reservation_holder: bool,
    name: str,
    has_reservation: bool,
) -> None:
    """Insert one row into ``jobs``.

    TypeDecorators handle JobName → wire string and bool → 0/1 automatically.
    """
    tx.execute(
        insert(jobs_table).values(
            job_id=job_id,
            user_id=user_id,
            parent_job_id=parent_job_id,
            root_job_id=root_job_id,
            depth=depth,
            state=state,
            submitted_at_ms=submitted_at_ms,
            root_submitted_at_ms=root_submitted_at_ms,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            scheduling_deadline_epoch_ms=scheduling_deadline_epoch_ms,
            error=error,
            exit_code=exit_code,
            num_tasks=num_tasks,
            is_reservation_holder=is_reservation_holder,
            name=name,
            has_reservation=has_reservation,
        )
    )


@writes_to(job_config_table)
def insert_job_config(
    tx: Tx,
    *,
    job_id: JobName,
    name: str,
    has_reservation: bool,
    res_cpu_millicores: int,
    res_memory_bytes: int,
    res_disk_bytes: int,
    res_device_json: str | None,
    constraints_json: str,
    has_coscheduling: bool,
    coscheduling_group_by: str,
    scheduling_timeout_ms: int | None,
    max_task_failures: int,
    entrypoint_json: str,
    environment_json: str,
    bundle_id: str,
    ports_json: list,
    max_retries_failure: int,
    max_retries_preemption: int,
    timeout_ms: int | None,
    preemption_policy: int,
    existing_job_policy: int,
    priority_band: int,
    task_image: str,
    submit_argv_json: list | None = None,
    reservation_json: str | None = None,
    fail_if_exists: bool = False,
) -> None:
    """Insert one row into ``job_config``."""
    tx.execute(
        insert(job_config_table).values(
            job_id=job_id,
            name=name,
            has_reservation=has_reservation,
            res_cpu_millicores=res_cpu_millicores,
            res_memory_bytes=res_memory_bytes,
            res_disk_bytes=res_disk_bytes,
            res_device_json=res_device_json,
            constraints_json=constraints_json,
            has_coscheduling=has_coscheduling,
            coscheduling_group_by=coscheduling_group_by,
            scheduling_timeout_ms=scheduling_timeout_ms,
            max_task_failures=max_task_failures,
            entrypoint_json=entrypoint_json,
            environment_json=environment_json,
            bundle_id=bundle_id,
            ports_json=ports_json,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            timeout_ms=timeout_ms,
            preemption_policy=preemption_policy,
            existing_job_policy=existing_job_policy,
            priority_band=priority_band,
            task_image=task_image,
            submit_argv_json=submit_argv_json if submit_argv_json is not None else [],
            reservation_json=reservation_json,
            fail_if_exists=fail_if_exists,
        )
    )


@writes_to(jobs_table)
def delete_job(tx: Tx, job_id: JobName) -> None:
    """Delete a job row. ``ON DELETE CASCADE`` handles tasks, attempts, endpoints."""
    tx.execute(delete(jobs_table).where(jobs_table.c.job_id == job_id))


@writes_to(meta_table)
def reserve_priority_insertion_base(tx: Tx) -> int:
    """Bump the ``task_priority_insertion`` sequence and return the new value.

    Callers reserving N task slots use ``base + i`` for ``i in range(N)``.
    Delegates to :func:`meta_sequence_bump`.
    """
    return meta_sequence_bump(tx, _PRIORITY_INSERTION_KEY)


# ---------------------------------------------------------------------------
# Task-attempt writes (previously writes/task_attempts.py)
# ---------------------------------------------------------------------------


@writes_to(task_attempts_table)
def insert_attempt(
    tx: Tx,
    *,
    task_id: JobName,
    attempt_id: int,
    worker_id: WorkerId | None,
    state: int,
    created_at_ms: int,
) -> None:
    """Insert one row into ``task_attempts``."""
    tx.execute(
        insert(task_attempts_table).values(
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            state=state,
            created_at_ms=created_at_ms,
        )
    )


@writes_to(task_attempts_table)
def apply_attempt_update(
    tx: Tx,
    *,
    task_id: JobName,
    attempt_id: int,
    state: int,
    started_at_ms: int | None,
    finished_at_ms: int | None,
    exit_code: int | None,
    error: str | None,
) -> None:
    """Apply a worker/direct-provider attempt update."""
    tx.execute(
        update(task_attempts_table)
        .where(
            task_attempts_table.c.task_id == task_id,
            task_attempts_table.c.attempt_id == attempt_id,
        )
        .values(
            state=state,
            started_at_ms=func.coalesce(task_attempts_table.c.started_at_ms, started_at_ms),
            finished_at_ms=func.coalesce(task_attempts_table.c.finished_at_ms, finished_at_ms),
            exit_code=func.coalesce(exit_code, task_attempts_table.c.exit_code),
            error=func.coalesce(error, task_attempts_table.c.error),
        )
    )


# ---------------------------------------------------------------------------
# Task writes (previously writes/tasks.py)
# ---------------------------------------------------------------------------


@writes_to(tasks_table)
def bulk_insert_tasks(tx: Tx, task_rows: list[dict]) -> None:
    """Insert multiple rows into ``tasks`` in a single executemany call.

    Each dict in ``task_rows`` must contain all columns required by
    :func:`insert_task`. Use :func:`task_row` to build the dicts.
    """
    if not task_rows:
        return
    tx.execute(insert(tasks_table), task_rows)


def task_row(
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
) -> dict:
    """Build a parameter dict for :func:`bulk_insert_tasks`."""
    return {
        "task_id": task_id,
        "job_id": job_id,
        "task_index": task_index,
        "state": state,
        "error": None,
        "exit_code": None,
        "submitted_at_ms": submitted_at_ms,
        "started_at_ms": None,
        "finished_at_ms": None,
        "max_retries_failure": max_retries_failure,
        "max_retries_preemption": max_retries_preemption,
        "failure_count": 0,
        "preemption_count": 0,
        "current_attempt_id": -1,
        "priority_neg_depth": priority_neg_depth,
        "priority_root_submitted_ms": priority_root_submitted_ms,
        "priority_insertion": priority_insertion,
        "priority_band": priority_band,
    }


@writes_to(tasks_table, task_attempts_table)
def assign_to_worker(
    tx: Tx,
    task_id: JobName,
    worker_id: WorkerId,
    worker_address: str,
    attempt_id: int,
    now_ms: int,
    priority_band: int | None = None,
) -> None:
    """Insert a fresh ``task_attempts`` row and assign the task to a worker.

    Stamps ``current_worker_id`` and ``current_worker_address`` on the task
    row. ``priority_band`` is stamped when provided so the preemption pass
    treats a running task's band as fixed; ``None`` leaves the column untouched.
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
        "current_worker_id": worker_id,
        "current_worker_address": worker_address,
    }
    if priority_band is not None:
        values["priority_band"] = priority_band
    tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(**values))


@writes_to(tasks_table, task_attempts_table)
def promote_to_direct_provider(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    now_ms: int,
) -> None:
    """Insert a fresh ``task_attempts`` row and promote the task for direct-provider dispatch.

    No worker is assigned; ``current_worker_id`` is left NULL so the
    direct-provider path can track and dispatch the task via K8s.
    """
    insert_attempt(
        tx,
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=None,
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=now_ms,
    )
    tx.execute(
        update(tasks_table)
        .where(tasks_table.c.task_id == task_id)
        .values(
            state=job_pb2.TASK_STATE_ASSIGNED,
            current_attempt_id=attempt_id,
            started_at_ms=func.coalesce(tasks_table.c.started_at_ms, now_ms),
        )
    )


@writes_to(tasks_table)
def apply_task_state_update(
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
    if state not in ACTIVE_TASK_STATES:
        values["current_worker_id"] = None
        values["current_worker_address"] = None
    tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(**values))


# ---------------------------------------------------------------------------
# Worker writes (previously writes/workers.py)
# ---------------------------------------------------------------------------


@writes_to(workers_table, cascades_into=(task_attempts_table,))
def remove_worker(
    tx: Tx,
    worker_id: WorkerId,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
) -> None:
    """Delete a worker row and clear back-references on attempts / tasks.

    ``cascades_into`` records the FK fanout to ``task_attempts``.
    The cascade into ``worker_attributes`` is Projection-owned and therefore
    not declared on the decorator; instead this function calls
    :meth:`WorkerAttrsProjection.invalidate_for_worker` inline so the
    cache update commits under the same write lock as the SQL.

    The pre-emptive ``UPDATE`` on ``task_attempts`` / ``tasks`` sets
    ``current_worker_*`` to NULL before the delete so the row history
    is observable to readers in the same write transaction.
    """
    tx.execute(update(task_attempts_table).where(task_attempts_table.c.worker_id == worker_id).values(worker_id=None))
    tx.execute(update(tasks_table).where(tasks_table.c.current_worker_id == worker_id).values(current_worker_id=None))
    tx.execute(delete(workers_table).where(workers_table.c.worker_id == worker_id))
    worker_attrs.invalidate_for_worker(tx, worker_id)
    tx.register(lambda: health.forget(worker_id))


# ---------------------------------------------------------------------------
# User / budget writes (previously ControllerDB methods)
# ---------------------------------------------------------------------------


@writes_to(users_table)
def ensure_user(tx: Tx, user_id: str, now: Timestamp, role: str = USER_ROLE_DEFAULT) -> None:
    """Create user if not exists. Does not update role for existing users."""
    stmt = sqlite_insert(users_table).values(
        user_id=user_id,
        created_at_ms=now,
        role=role,
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=["user_id"])
    tx.execute(stmt)


@writes_to(users_table)
def set_user_role(tx: Tx, user_id: str, role: str) -> None:
    """Update the role for an existing user."""
    tx.execute(update(users_table).where(users_table.c.user_id == user_id).values(role=role))


@writes_to(user_budgets_table)
def set_user_budget(tx: Tx, user_id: str, budget_limit: int, max_band: int, now: Timestamp) -> None:
    """Insert or update a user's budget configuration."""
    stmt = sqlite_insert(user_budgets_table).values(
        user_id=user_id,
        budget_limit=budget_limit,
        max_band=max_band,
        updated_at_ms=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id"],
        set_={"budget_limit": budget_limit, "max_band": max_band, "updated_at_ms": now},
    )
    tx.execute(stmt)
