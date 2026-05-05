# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # Originally this migration also rewrote the `trg_txn_log_retention`
    # trigger; those statements were removed once migration 0037 dropped the
    # `txn_log` / `txn_actions` tables entirely. On DBs that already ran the
    # old form the trigger survives until 0037 executes; 0037 is idempotent
    # (`DROP TRIGGER IF EXISTS`) so no fixup is needed here.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_healthy_active ON workers(healthy, active)")
