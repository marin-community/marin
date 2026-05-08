# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(conn: sqlite3.Connection) -> None:
    # Originally this migration also rewrote the `trg_txn_log_retention`
    # trigger; those statements were removed once migration 0037 dropped the
    # `txn_log` / `txn_actions` tables entirely. On DBs that already ran the
    # old form the trigger survives until 0037 executes; 0037 is idempotent
    # (`DROP TRIGGER IF EXISTS`) so no fixup is needed here.
    #
    # ``healthy`` / ``active`` were workers columns when this migration was
    # authored. They are dropped in 0042; on a fresh DB the columns are absent
    # at this point so the index is a no-op.
    if _has_column(conn, "workers", "healthy") and _has_column(conn, "workers", "active"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_healthy_active ON workers(healthy, active)")
