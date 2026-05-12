# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-attempt read helpers (SA Core expression language).

All queries use ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on the schema_v2 columns decode values on read so
callers receive ``JobName``, ``Timestamp``, and ``WorkerId`` directly.

Return shapes:

* ``get`` — SA ``Row`` or ``None``
* ``get_state`` — ``int | None``
* ``get_worker_id`` — ``WorkerId | None``
* ``list_for_task`` — ``list[Row]`` ordered by attempt_id ASC
* ``bulk_get_for_updates`` — ``dict[(JobName, int), Row]``

Bulk chunking: ``bulk_get_for_updates`` chunks at 450 (task_id, attempt_id)
pairs so the per-statement parameter count stays under SQLite's 999-parameter
default (2 binds per pair). SA's ``tuple_(...).in_(...)`` handles the PK
comparison natively.
"""

from collections.abc import Sequence

from sqlalchemy import bindparam, select, tuple_

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import task_attempts_table
from iris.cluster.types import JobName, WorkerId

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

GET_ATTEMPT_QUERY = select(*_ATTEMPT_COLS).where(
    task_attempts_table.c.task_id == bindparam("task_id"),
    task_attempts_table.c.attempt_id == bindparam("attempt_id"),
)

LIST_FOR_TASK_QUERY = (
    select(*_ATTEMPT_COLS)
    .where(task_attempts_table.c.task_id == bindparam("task_id"))
    .order_by(task_attempts_table.c.attempt_id.asc())
)

GET_STATE_QUERY = select(task_attempts_table.c.state).where(
    task_attempts_table.c.task_id == bindparam("task_id"),
    task_attempts_table.c.attempt_id == bindparam("attempt_id"),
)

GET_WORKER_ID_QUERY = select(task_attempts_table.c.worker_id).where(
    task_attempts_table.c.task_id == bindparam("task_id"),
    task_attempts_table.c.attempt_id == bindparam("attempt_id"),
)


def get(tx: Tx, task_id: JobName, attempt_id: int):
    """Return SA Row for ``(task_id, attempt_id)`` or None.

    Row fields (TypeDecorator-decoded): task_id (JobName), worker_id
    (WorkerId|None), created_at_ms (Timestamp), started_at_ms (Timestamp|None),
    finished_at_ms (Timestamp|None). Remaining fields are plain int/str/None.
    """
    return tx.execute(GET_ATTEMPT_QUERY, {"task_id": task_id, "attempt_id": attempt_id}).first()


def get_state(tx: Tx, task_id: JobName, attempt_id: int) -> int | None:
    """Return the attempt's ``state`` integer, or None if absent."""
    row = tx.execute(GET_STATE_QUERY, {"task_id": task_id, "attempt_id": attempt_id}).first()
    return int(row.state) if row is not None else None


def get_worker_id(tx: Tx, task_id: JobName, attempt_id: int) -> WorkerId | None:
    """Return the attempt's bound :class:`WorkerId`, or None.

    WorkerIdType decodes to WorkerId on read; returns None if unbound or absent.
    """
    row = tx.execute(GET_WORKER_ID_QUERY, {"task_id": task_id, "attempt_id": attempt_id}).first()
    if row is None or row.worker_id is None:
        return None
    return row.worker_id


def list_for_task(tx: Tx, task_id: JobName) -> list:
    """Return every attempt Row for ``task_id`` ordered by attempt_id ascending.

    Returns list[Row]. Each Row has TypeDecorator-decoded fields.
    """
    return tx.execute(LIST_FOR_TASK_QUERY, {"task_id": task_id}).all()


# ---------------------------------------------------------------------------
# Bulk attempt lookup for the heartbeat update path
# ---------------------------------------------------------------------------

_BULK_GET_CHUNK_SIZE = 450


def bulk_get_for_updates(
    tx: Tx,
    keys: Sequence[tuple[JobName, int]],
) -> dict[tuple[JobName, int], object]:
    """Return ``{(task_id, attempt_id): Row}`` for the requested keys.

    Drives lookups through the ``task_attempts`` PK via
    ``(task_id, attempt_id) IN ((…), …)``. Missing keys are silently absent.
    Chunks at 450 keys per statement to keep the bound parameter list under
    SQLite's 999-parameter cap (2 binds per pair).

    Returns dict[(JobName, int), Row].
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
