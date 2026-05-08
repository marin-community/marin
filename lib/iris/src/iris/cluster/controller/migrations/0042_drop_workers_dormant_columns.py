# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the dormant liveness / committed-resource columns on ``workers``.

The runtime source of truth for these signals is the in-memory
``WorkerHealthTracker`` / ``WorkerCommitTracker``. This migration removes the
unused columns along with the ``idx_workers_healthy_active`` index and the
``trg_task_attempt_active_worker`` trigger that referenced them.

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
    "committed_cpu_millicores",
    "committed_mem_bytes",
    "committed_gpu",
    "committed_tpu",
)


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_workers_healthy_active")
    conn.execute("DROP TRIGGER IF EXISTS trg_task_attempt_active_worker")
    for col in _COLUMNS_TO_DROP:
        if _has_column(conn, "workers", col):
            conn.execute(f"ALTER TABLE workers DROP COLUMN {col}")
