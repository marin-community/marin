# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-attempt read helpers.

Return shapes:

* ``list_for_task`` — ``list[Row]`` ordered by attempt_id ASC
* ``bulk_get_for_updates`` — ``dict[(JobName, int), Row]``

``bulk_get_for_updates`` chunks at 450 (task_id, attempt_id) pairs to keep
the per-statement parameter count under SQLite's 999-parameter limit (2 binds per pair).
"""

from collections.abc import Sequence

from sqlalchemy import bindparam, select, tuple_

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.types import JobName

# ---------------------------------------------------------------------------
# Shared column list for all attempt reads
# ---------------------------------------------------------------------------

_ATTEMPT_COLS = (
    task_attempts_table.c.task_id,
    task_attempts_table.c.attempt_id,
    task_attempts_table.c.worker_id,
    task_attempts_table.c.state,
    task_attempts_table.c.created_at_ms,
    task_attempts_table.c.started_at_ms,
    task_attempts_table.c.finished_at_ms,
    task_attempts_table.c.exit_code,
    task_attempts_table.c.error,
)

# ---------------------------------------------------------------------------
# Single-attempt lookup
# ---------------------------------------------------------------------------


def list_for_task(tx: Tx, task_id: JobName) -> list:
    """Return every attempt Row for ``task_id`` ordered by attempt_id ascending."""
    return tx.execute(
        select(*_ATTEMPT_COLS)
        .where(task_attempts_table.c.task_id == bindparam("task_id"))
        .order_by(task_attempts_table.c.attempt_id.asc()),
        {"task_id": task_id},
    ).all()


# ---------------------------------------------------------------------------
# Bulk attempt lookup for the heartbeat update path
# ---------------------------------------------------------------------------

_BULK_GET_CHUNK_SIZE = 450


def bulk_get_for_updates(
    tx: Tx,
    keys: Sequence[tuple[JobName, int]],
) -> dict[tuple[JobName, int], object]:
    """Return ``{(task_id, attempt_id): Row}`` for the requested keys.

    Drives lookups through the ``task_attempts`` PK. Missing keys are silently
    absent. Chunks at 450 keys per statement to keep the bound parameter list
    under SQLite's 999-parameter limit (2 binds per pair).
    """
    if not keys:
        return {}
    unique: list[tuple[JobName, int]] = list({k: None for k in keys}.keys())
    result: dict[tuple[JobName, int], object] = {}
    pair_cols = tuple_(task_attempts_table.c.task_id, task_attempts_table.c.attempt_id)
    for chunk_start in range(0, len(unique), _BULK_GET_CHUNK_SIZE):
        chunk = unique[chunk_start : chunk_start + _BULK_GET_CHUNK_SIZE]
        stmt = select(*_ATTEMPT_COLS).where(pair_cols.in_(chunk))
        rows = tx.execute(stmt).all()
        for row in rows:
            result[(row.task_id, row.attempt_id)] = row
    return result
