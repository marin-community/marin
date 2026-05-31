# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the controller-minted ``attempt_uid`` routing key to ``task_attempts``.

``attempt_uid`` is a 16 hex-char identifier the controller mints per attempt
and routes worker observations by. The final schema (``schema.py``) declares it
``NOT NULL`` with a ``UNIQUE`` index. SQLite cannot add a ``NOT NULL`` column to
a populated table in place, so this migration:

1. Adds the column as nullable ``TEXT``.
2. Backfills every existing row with ``secrets.token_hex(8)``.
3. Rebuilds the table to promote the column to ``NOT NULL``.
4. Creates the unique index ``idx_task_attempts_uid``.

Every step is idempotent: a crash mid-migration is re-run from scratch on the
next controller startup (the runner records the migration only after
``migrate`` returns).
"""

import secrets

BACKFILL_CHUNK = 1000


def _columns(raw_conn, table: str) -> list[tuple]:
    return raw_conn.execute(f"PRAGMA table_info({table})").fetchall()


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in _columns(raw_conn, table))


def _column_is_notnull(raw_conn, table: str, column: str) -> bool:
    for row in _columns(raw_conn, table):
        if row[1] == column:
            return bool(row[3])
    return False


def _add_nullable_column(raw_conn) -> None:
    if _has_column(raw_conn, "task_attempts", "attempt_uid"):
        return
    raw_conn.execute("ALTER TABLE task_attempts ADD COLUMN attempt_uid TEXT")


def _backfill(raw_conn) -> None:
    """Fill every NULL ``attempt_uid`` with a fresh per-row token, in chunks.

    Plain SQLite ``UPDATE`` has no ``LIMIT`` clause unless compiled with
    ``SQLITE_ENABLE_UPDATE_DELETE_LIMIT``, which we cannot assume. Instead we
    select a bounded set of rowids and update exactly those.
    """
    while True:
        rowids = [
            row[0]
            for row in raw_conn.execute(
                "SELECT rowid FROM task_attempts WHERE attempt_uid IS NULL LIMIT ?",
                (BACKFILL_CHUNK,),
            ).fetchall()
        ]
        if not rowids:
            return
        raw_conn.executemany(
            "UPDATE task_attempts SET attempt_uid = ? WHERE rowid = ?",
            [(secrets.token_hex(8), rowid) for rowid in rowids],
        )


def _rebuild_not_null(raw_conn) -> None:
    """Rebuild ``task_attempts`` with ``attempt_uid`` promoted to ``NOT NULL``.

    The rebuilt table mirrors ``schema.py``'s ``task_attempts_table`` exactly:
    same columns, the ``(task_id, attempt_id)`` primary key, the FKs to
    ``tasks`` and ``workers``, and the two non-unique indexes. The unique
    ``idx_task_attempts_uid`` index is created separately by ``migrate``.
    """
    if _column_is_notnull(raw_conn, "task_attempts", "attempt_uid"):
        return

    # Foreign keys must be off during a table rebuild so the rename does not
    # trip the FKs that reference task_attempts. SQLite ignores PRAGMA
    # foreign_keys changes inside a transaction, so toggle it outside one.
    raw_conn.commit()
    raw_conn.execute("PRAGMA foreign_keys=OFF")
    try:
        raw_conn.execute("BEGIN IMMEDIATE")
        raw_conn.execute(
            """
            CREATE TABLE task_attempts_new (
                task_id VARCHAR NOT NULL,
                attempt_id INTEGER NOT NULL,
                worker_id VARCHAR,
                state INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                started_at_ms INTEGER,
                finished_at_ms INTEGER,
                exit_code INTEGER,
                error VARCHAR,
                attempt_uid VARCHAR NOT NULL,
                PRIMARY KEY (task_id, attempt_id),
                FOREIGN KEY(task_id) REFERENCES tasks (task_id) ON DELETE CASCADE,
                FOREIGN KEY(worker_id) REFERENCES workers (worker_id) ON DELETE SET NULL
            )
            """
        )
        raw_conn.execute(
            """
            INSERT INTO task_attempts_new (
                task_id, attempt_id, worker_id, state, created_at_ms,
                started_at_ms, finished_at_ms, exit_code, error, attempt_uid
            )
            SELECT
                task_id, attempt_id, worker_id, state, created_at_ms,
                started_at_ms, finished_at_ms, exit_code, error, attempt_uid
            FROM task_attempts
            """
        )
        raw_conn.execute("DROP TABLE task_attempts")
        raw_conn.execute("ALTER TABLE task_attempts_new RENAME TO task_attempts")
        raw_conn.execute("CREATE INDEX idx_task_attempts_worker_task ON task_attempts (worker_id, task_id, attempt_id)")
        raw_conn.execute(
            "CREATE INDEX idx_task_attempts_live_workerbound ON task_attempts (worker_id) "
            "WHERE worker_id IS NOT NULL AND finished_at_ms IS NULL"
        )
        raw_conn.commit()
    except Exception:
        raw_conn.execute("ROLLBACK")
        raise
    finally:
        raw_conn.execute("PRAGMA foreign_keys=ON")


def migrate(raw_conn) -> None:
    _add_nullable_column(raw_conn)
    _backfill(raw_conn)
    _rebuild_not_null(raw_conn)
    raw_conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_task_attempts_uid ON task_attempts (attempt_uid)")
