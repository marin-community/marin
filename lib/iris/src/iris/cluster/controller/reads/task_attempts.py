# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-attempt read helpers (SA Core port).

Named ``text(...)`` SQL constants and helpers for the non-hot-path read
methods on :class:`iris.cluster.controller.stores.TaskAttemptStore`. The
heavy hot-path readers (``resource_usage_by_worker`` and
``reconcile_rows_for_workers``) live in :mod:`reads.scheduler` since
they drive the scheduler tick directly.

Notes:

* ``bulk_get_for_updates`` preserves the legacy 450-tuple chunk size
  (2 placeholders per row, well under SQLite's 999-parameter default
  limit). SA's expanding bindparam emits
  ``IN ((:k1_1, :k1_2), (:k2_1, :k2_2), …)`` under the hood; we still
  chunk explicitly so per-statement compile is bounded.
* :class:`AttemptRow` decoding mirrors the legacy
  ``ATTEMPT_PROJECTION.decode`` byte-for-byte: nullable timestamps via
  ``_nullable(decode_timestamp_ms)``, ``WorkerId(str(...))`` for the
  worker reference, etc.
"""

from collections.abc import Sequence

from rigging.timing import Timestamp
from sqlalchemy import select, text, tuple_

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import (
    AttemptRow,
    _nullable,
    decode_timestamp_ms,
)
from iris.cluster.controller.schema_v2 import task_attempts_table
from iris.cluster.types import JobName, WorkerId

# ---------------------------------------------------------------------------
# Single-attempt lookup
# ---------------------------------------------------------------------------

_ATTEMPT_COLS = (
    "ta.task_id AS task_id, ta.attempt_id AS attempt_id, ta.worker_id AS worker_id, "
    "ta.state AS state, ta.created_at_ms AS created_at_ms, "
    "ta.started_at_ms AS started_at_ms, ta.finished_at_ms AS finished_at_ms, "
    "ta.exit_code AS exit_code, ta.error AS error"
)


_GET_ATTEMPT_SQL = text(
    f"SELECT {_ATTEMPT_COLS} FROM task_attempts ta " "WHERE ta.task_id = :tid AND ta.attempt_id = :aid"
)


_LIST_FOR_TASK_SQL = text(
    f"SELECT {_ATTEMPT_COLS} FROM task_attempts ta " "WHERE ta.task_id = :tid ORDER BY ta.attempt_id ASC"
)


_NULL_INT = _nullable(int)
_NULL_STR = _nullable(str)


def _decode_job_name(value):
    if isinstance(value, JobName):
        return value
    return JobName.from_wire(str(value))


def _decode_worker_id_nullable(value):
    # ``WorkerId`` is a ``NewType`` over ``str`` so isinstance() is not
    # applicable — wrap unconditionally except for the None case.
    if value is None:
        return None
    return WorkerId(str(value))


def _decode_timestamp(value):
    if isinstance(value, Timestamp):
        return value
    return decode_timestamp_ms(value)


def _decode_timestamp_nullable(value):
    if value is None:
        return None
    return _decode_timestamp(value)


def _row_to_attempt(row) -> AttemptRow:
    # The bulk path uses SA ``select(...)`` against ``task_attempts_table``
    # so values come back already decoded via the TypeDecorators
    # (``JobName``, ``WorkerId``, ``Timestamp``). The single-row text()
    # paths return raw scalars. The decoder helpers below handle both.
    return AttemptRow(
        task_id=_decode_job_name(row.task_id),
        attempt_id=int(row.attempt_id),
        worker_id=_decode_worker_id_nullable(row.worker_id),
        state=int(row.state),
        created_at=_decode_timestamp(row.created_at_ms),
        started_at=_decode_timestamp_nullable(row.started_at_ms),
        finished_at=_decode_timestamp_nullable(row.finished_at_ms),
        exit_code=_NULL_INT(row.exit_code),
        error=_NULL_STR(row.error),
    )


def get(tx: Tx, task_id: JobName, attempt_id: int) -> AttemptRow | None:
    """Return the :class:`AttemptRow` for ``(task_id, attempt_id)``, or None."""
    row = tx.execute(_GET_ATTEMPT_SQL, {"tid": task_id.to_wire(), "aid": attempt_id}).first()
    if row is None:
        return None
    return _row_to_attempt(row)


_GET_STATE_SQL = text("SELECT state FROM task_attempts WHERE task_id = :tid AND attempt_id = :aid")


def get_state(tx: Tx, task_id: JobName, attempt_id: int) -> int | None:
    """Return the attempt's ``state`` integer, or None if absent."""
    row = tx.execute(_GET_STATE_SQL, {"tid": task_id.to_wire(), "aid": attempt_id}).first()
    return int(row.state) if row is not None else None


_GET_WORKER_ID_SQL = text("SELECT worker_id FROM task_attempts WHERE task_id = :tid AND attempt_id = :aid")


def get_worker_id(tx: Tx, task_id: JobName, attempt_id: int) -> WorkerId | None:
    """Return the attempt's bound :class:`WorkerId`, or None."""
    row = tx.execute(_GET_WORKER_ID_SQL, {"tid": task_id.to_wire(), "aid": attempt_id}).first()
    if row is None or row.worker_id is None:
        return None
    return WorkerId(str(row.worker_id))


def list_for_task(tx: Tx, task_id: JobName) -> list[AttemptRow]:
    """Return every attempt for ``task_id`` ordered by attempt_id ascending."""
    rows = tx.execute(_LIST_FOR_TASK_SQL, {"tid": task_id.to_wire()}).all()
    return [_row_to_attempt(row) for row in rows]


# ---------------------------------------------------------------------------
# Bulk attempt lookup for the heartbeat update path
# ---------------------------------------------------------------------------

# Legacy ``bulk_get_for_updates`` chunked at 450 (task_id, attempt_id) pairs
# so the per-statement param count stayed under SQLite's 999 default. SA's
# tuple expanding bind handles arbitrarily-long lists by re-rendering
# ``IN ((?, ?), ...)``, but we still chunk so compile cost stays bounded.
_BULK_GET_CHUNK_SIZE = 450


def bulk_get_for_updates(
    tx: Tx,
    keys: Sequence[tuple[JobName, int]],
) -> dict[tuple[JobName, int], AttemptRow]:
    """Return ``{(task_id, attempt_id): AttemptRow}`` for the requested keys.

    Drives lookups through the ``task_attempts`` PK
    (``(task_id, attempt_id) IN ((…), …)``). Missing keys are silently
    absent. Chunks at 450 keys per statement to keep the bound parameter
    list under SQLite's 999-parameter cap (2 binds per pair).
    """
    if not keys:
        return {}
    # Deduplicate so the IN list never carries the same (task, attempt) twice.
    unique: list[tuple[JobName, int]] = list({k: None for k in keys}.keys())
    result: dict[tuple[JobName, int], AttemptRow] = {}
    pair_cols = tuple_(task_attempts_table.c.task_id, task_attempts_table.c.attempt_id)
    for chunk_start in range(0, len(unique), _BULK_GET_CHUNK_SIZE):
        chunk = unique[chunk_start : chunk_start + _BULK_GET_CHUNK_SIZE]
        # Pass JobName/int through the SA TypeDecorator (JobNameType
        # converts via ``to_wire``); supplying wire strings here bypasses
        # the decorator and raises an AttributeError.
        pairs = [(tid, aid) for tid, aid in chunk]
        stmt = select(
            task_attempts_table.c.task_id.label("task_id"),
            task_attempts_table.c.attempt_id.label("attempt_id"),
            task_attempts_table.c.worker_id.label("worker_id"),
            task_attempts_table.c.state.label("state"),
            task_attempts_table.c.created_at_ms.label("created_at_ms"),
            task_attempts_table.c.started_at_ms.label("started_at_ms"),
            task_attempts_table.c.finished_at_ms.label("finished_at_ms"),
            task_attempts_table.c.exit_code.label("exit_code"),
            task_attempts_table.c.error.label("error"),
        ).where(pair_cols.in_(pairs))
        rows = tx.execute(stmt).all()
        for row in rows:
            attempt = _row_to_attempt(row)
            result[(attempt.task_id, attempt.attempt_id)] = attempt
    return result
