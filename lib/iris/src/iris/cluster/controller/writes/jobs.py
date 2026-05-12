# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``jobs``, ``job_config``, ``job_workdir_files``,
``users`` and the ``meta`` sequence counter.

Stage M2 of the SA Core migration: replaces raw ``text("INSERT INTO ...")``
strings with SA Core expression-language constructs. TypeDecorators on the
table columns handle all bind-side conversions (``JobName.to_wire()``,
``Timestamp.epoch_ms()``, ``bool`` → 0/1) automatically.
"""

from collections.abc import Iterable, Mapping, Sequence

from sqlalchemy import case, delete, func, insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import (
    job_config_table,
    job_workdir_files_table,
    jobs_table,
    meta_table,
    users_table,
)
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import TERMINAL_JOB_STATES, JobName
from iris.rpc import job_pb2

_PRIORITY_INSERTION_KEY = "task_priority_insertion"

# Job states that warrant recording an error message on finished_at_ms.
_ERROR_STATES: frozenset[int] = frozenset(
    [
        job_pb2.JOB_STATE_FAILED,
        job_pb2.JOB_STATE_KILLED,
        job_pb2.JOB_STATE_UNSCHEDULABLE,
        job_pb2.JOB_STATE_WORKER_FAILED,
    ]
)


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
    ports_json: str,
    max_retries_failure: int,
    max_retries_preemption: int,
    timeout_ms: int | None,
    preemption_policy: int,
    existing_job_policy: int,
    priority_band: int,
    task_image: str,
    submit_argv_json: str = "[]",
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
            submit_argv_json=submit_argv_json,
            reservation_json=reservation_json,
            fail_if_exists=fail_if_exists,
        )
    )


@writes_to(jobs_table)
def delete_job(tx: Tx, job_id: JobName) -> None:
    """Delete a job row. ``ON DELETE CASCADE`` handles tasks, attempts, endpoints."""
    tx.execute(delete(jobs_table).where(jobs_table.c.job_id == job_id))


@writes_to(job_workdir_files_table)
def insert_workdir_files(tx: Tx, job_id: JobName, files: Mapping[str, bytes]) -> None:
    """Insert each ``{filename: data}`` pair as a row in ``job_workdir_files``."""
    if not files:
        return
    tx.execute(
        insert(job_workdir_files_table),
        [{"job_id": job_id, "filename": name, "data": data} for name, data in files.items()],
    )


@writes_to(jobs_table)
def update_state_if_not_terminal(
    tx: Tx,
    job_id: JobName,
    new_state: int,
    error: str | None,
    finished_at_ms: int | None,
) -> None:
    """Set a new state on a single job, skipping rows already in a terminal state."""
    tx.execute(
        update(jobs_table)
        .where(
            jobs_table.c.job_id == job_id,
            jobs_table.c.state.not_in(TERMINAL_JOB_STATES),
        )
        .values(
            state=new_state,
            error=error,
            finished_at_ms=func.coalesce(jobs_table.c.finished_at_ms, finished_at_ms),
        )
    )


@writes_to(jobs_table)
def bulk_update_state(
    tx: Tx,
    job_ids: Sequence[JobName],
    new_state: int,
    error: str | None,
    finished_at_ms: int | None,
    guard_states: Iterable[int],
) -> None:
    """Set state on many jobs; rows in any of ``guard_states`` are skipped."""
    if not job_ids:
        return
    guard = list(guard_states)
    tx.execute(
        update(jobs_table)
        .where(
            jobs_table.c.job_id.in_(job_ids),
            jobs_table.c.state.not_in(guard),
        )
        .values(
            state=new_state,
            error=error,
            finished_at_ms=func.coalesce(jobs_table.c.finished_at_ms, finished_at_ms),
        )
    )


@writes_to(jobs_table)
def mark_running_if_pending(tx: Tx, job_id: JobName, now_ms: int) -> None:
    """Advance PENDING → RUNNING and set ``started_at_ms`` if not already populated."""
    tx.execute(
        update(jobs_table)
        .where(jobs_table.c.job_id == job_id)
        .values(
            state=case(
                (jobs_table.c.state == job_pb2.JOB_STATE_PENDING, job_pb2.JOB_STATE_RUNNING),
                else_=jobs_table.c.state,
            ),
            started_at_ms=func.coalesce(jobs_table.c.started_at_ms, now_ms),
        )
    )


@writes_to(jobs_table)
def apply_recomputed_state(
    tx: Tx,
    job_id: JobName,
    new_state: int,
    now_ms: int,
    error: str | None,
) -> None:
    """Write the result of ``_recompute_job_state`` back to the row.

    Sets ``started_at_ms`` (if moving to RUNNING), ``finished_at_ms``
    (if moving to a terminal state), and ``error`` (if the terminal
    reason warrants one). The caller has already decided ``new_state``
    differs from the current state.

    Uses SQL CASE/WHEN for the three conditional columns so a single
    UPDATE round-trip handles every branch.
    """
    tx.execute(
        update(jobs_table)
        .where(jobs_table.c.job_id == job_id)
        .values(
            state=new_state,
            started_at_ms=(
                func.coalesce(jobs_table.c.started_at_ms, now_ms)
                if new_state == job_pb2.JOB_STATE_RUNNING
                else jobs_table.c.started_at_ms
            ),
            finished_at_ms=now_ms if new_state in TERMINAL_JOB_STATES else jobs_table.c.finished_at_ms,
            error=error if new_state in _ERROR_STATES else jobs_table.c.error,
        )
    )


@writes_to(meta_table)
def reserve_priority_insertion_base(tx: Tx) -> int:
    """Bump the ``task_priority_insertion`` sequence and return the new value.

    Callers reserving N task slots use ``base + i`` for ``i in range(N)``.
    Mirrors :meth:`ControllerDB.next_sequence` against the ``meta`` table.
    """
    row = tx.execute(select(meta_table.c.value).where(meta_table.c.key == _PRIORITY_INSERTION_KEY)).fetchone()
    if row is None:
        tx.execute(insert(meta_table).values(key=_PRIORITY_INSERTION_KEY, value=1))
        return 1
    value = int(row[0]) + 1
    tx.execute(update(meta_table).where(meta_table.c.key == _PRIORITY_INSERTION_KEY).values(value=value))
    return value


@writes_to(users_table)
def ensure_user(tx: Tx, user_id: str, now_ms: int) -> None:
    """Idempotently create a ``users`` row at submission time."""
    tx.execute(
        sqlite_insert(users_table)
        .values(user_id=user_id, created_at_ms=now_ms)
        .on_conflict_do_nothing(index_elements=["user_id"])
    )
