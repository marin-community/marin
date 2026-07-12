# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``backend_id`` to jobs/tasks/task_attempts and the ``backends`` table.

Multi-backend clusters pin every job, task, and attempt to one backend via a
``backend_id``. Existing single-backend rows are stamped with the implicit
``DEFAULT_BACKEND_ID``, and one ``backends`` row for it is inserted (status
ACTIVE). The same literal is used by the runtime config synthesis
(``iris.cluster.config.resolve_backends``), so upgraded rows match what the
controller resolves at startup.

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""

from iris.cluster.types import DEFAULT_BACKEND_ID, BackendStatus

_BACKEND_ID_TABLES = ("jobs", "tasks", "task_attempts")


def _columns(raw_conn, table: str) -> list[tuple]:
    return raw_conn.execute(f"PRAGMA table_info({table})").fetchall()


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in _columns(raw_conn, table))


def _add_backend_id_columns(raw_conn) -> None:
    for table in _BACKEND_ID_TABLES:
        if not _has_column(raw_conn, table, "backend_id"):
            raw_conn.execute(f"ALTER TABLE {table} ADD COLUMN backend_id VARCHAR NOT NULL DEFAULT ''")


def _create_backends_table(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backends (
            backend_id VARCHAR NOT NULL PRIMARY KEY,
            kind VARCHAR NOT NULL DEFAULT '',
            status INTEGER NOT NULL DEFAULT 0,
            attributes_json VARCHAR NOT NULL DEFAULT '{}',
            allow_policy_json VARCHAR NOT NULL DEFAULT '{}',
            last_seen_ms INTEGER
        )
        """
    )


def _backfill_backend_id(raw_conn) -> None:
    """Stamp every pre-migration row (default '') with the implicit backend id."""
    for table in _BACKEND_ID_TABLES:
        raw_conn.execute(f"UPDATE {table} SET backend_id = ? WHERE backend_id = ''", (DEFAULT_BACKEND_ID,))


def _insert_default_backend(raw_conn) -> None:
    raw_conn.execute(
        "INSERT OR IGNORE INTO backends (backend_id, kind, status, attributes_json, allow_policy_json) "
        "VALUES (?, '', ?, '{}', '{}')",
        (DEFAULT_BACKEND_ID, int(BackendStatus.ACTIVE)),
    )


def migrate(raw_conn) -> None:
    _add_backend_id_columns(raw_conn)
    _create_backends_table(raw_conn)
    _backfill_backend_id(raw_conn)
    _insert_default_backend(raw_conn)
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_backend_state ON tasks (backend_id, state)")
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_task_attempts_backend ON task_attempts (backend_id)")
