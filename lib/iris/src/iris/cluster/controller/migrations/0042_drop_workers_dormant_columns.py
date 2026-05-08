# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the dormant transient-liveness columns on ``workers``.

Liveness signals (heartbeat/health/failure counters) now live exclusively in
the in-memory ``WorkerHealthTracker``. This migration removes those four
columns along with the ``idx_workers_healthy_active`` index and the
``trg_task_attempt_active_worker`` trigger that referenced them.

The ``committed_*`` columns are intentionally retained: they record durable
scheduling state owned by the scheduler under a write transaction and must
survive a controller restart.

SQLite >= 3.35 supports ``ALTER TABLE ... DROP COLUMN`` directly.
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
