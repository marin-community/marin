# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write helpers for ``jobs``, ``job_config``, and the ``meta`` sequence counter.

TypeDecorators on the table columns handle all bind-side conversions
(``JobName.to_wire()``, ``Timestamp.epoch_ms()``, ``bool`` → 0/1) automatically.
"""

from sqlalchemy import delete, insert, select, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import (
    job_config_table,
    jobs_table,
    meta_table,
)
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import JobName

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
    Mirrors :meth:`ControllerDB.next_sequence` against the ``meta`` table.
    """
    row = tx.execute(select(meta_table.c.value).where(meta_table.c.key == _PRIORITY_INSERTION_KEY)).fetchone()
    if row is None:
        tx.execute(insert(meta_table).values(key=_PRIORITY_INSERTION_KEY, value=1))
        return 1
    value = int(row[0]) + 1
    tx.execute(update(meta_table).where(meta_table.c.key == _PRIORITY_INSERTION_KEY).values(value=value))
    return value
