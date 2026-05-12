# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write helpers for ``task_attempts``."""

from sqlalchemy import func, insert, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import task_attempts_table
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
