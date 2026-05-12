# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write helpers for ``task_attempts``."""

from collections.abc import Sequence

from sqlalchemy import func, insert, select, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import task_attempts_table, tasks_table
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import JobName, WorkerId


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
def mark_finished(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    state: int,
    finished_at_ms: int,
    error: str | None,
) -> None:
    """Stamp ``finished_at_ms`` and final state on an attempt."""
    tx.execute(
        update(task_attempts_table)
        .where(
            task_attempts_table.c.task_id == task_id,
            task_attempts_table.c.attempt_id == attempt_id,
        )
        .values(
            state=state,
            finished_at_ms=func.coalesce(task_attempts_table.c.finished_at_ms, finished_at_ms),
            error=error,
        )
    )


@writes_to(task_attempts_table)
def apply_attempt_state(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    state: int,
    error: str | None,
) -> None:
    """Update an attempt's reporting state without stamping ``finished_at_ms``.

    Producing transitions (cancel, preempt, timeout, gang cascade) call
    this to record the controller's intent while the worker still holds
    the container. The heartbeat path stamps ``finished_at_ms`` via
    :func:`mark_finished` once the worker confirms termination.
    """
    tx.execute(
        update(task_attempts_table)
        .where(
            task_attempts_table.c.task_id == task_id,
            task_attempts_table.c.attempt_id == attempt_id,
        )
        .values(
            state=state,
            error=func.coalesce(error, task_attempts_table.c.error),
        )
    )


@writes_to(task_attempts_table)
def apply_update(
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


@writes_to(task_attempts_table)
def bulk_apply_attempt_state(
    tx: Tx,
    job_ids: Sequence[JobName],
    state: int,
    error: str,
    active_states: set[int],
) -> None:
    """Update reporting state on every active attempt under ``job_ids``."""
    if not job_ids:
        return
    tx.execute(
        update(task_attempts_table)
        .where(
            task_attempts_table.c.task_id.in_(select(tasks_table.c.task_id).where(tasks_table.c.job_id.in_(job_ids))),
            task_attempts_table.c.state.in_(active_states),
        )
        .values(
            state=state,
            error=func.coalesce(task_attempts_table.c.error, error),
        )
    )
