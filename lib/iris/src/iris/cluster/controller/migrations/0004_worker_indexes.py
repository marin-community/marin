# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(conn: sqlite3.Connection) -> None:
    # ``healthy`` / ``active`` are dropped from the workers table by 0042; on a
    # fresh DB the columns are absent at this point so the index is a no-op.
    if _has_column(conn, "workers", "healthy") and _has_column(conn, "workers", "active"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_healthy_active ON workers(healthy, active)")
