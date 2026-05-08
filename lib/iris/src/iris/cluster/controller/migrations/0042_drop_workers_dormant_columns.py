# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the transient-liveness columns on ``workers`` (now in-memory only).

Removes ``last_heartbeat_ms``, ``healthy``, ``active``, and
``consecutive_failures`` along with ``idx_workers_healthy_active`` and the
``trg_task_attempt_active_worker`` trigger that referenced them.
"""

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


_COLUMNS_TO_DROP = (
    "last_heartbeat_ms",
    "healthy",
    "active",
    "consecutive_failures",
)


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_workers_healthy_active")
    conn.execute("DROP TRIGGER IF EXISTS trg_task_attempt_active_worker")
    for col in _COLUMNS_TO_DROP:
        if _has_column(conn, "workers", col):
            conn.execute(f"ALTER TABLE workers DROP COLUMN {col}")
