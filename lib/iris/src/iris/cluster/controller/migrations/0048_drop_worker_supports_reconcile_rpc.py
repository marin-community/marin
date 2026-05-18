# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Drop supports_reconcile_rpc column from workers table.

    Capability negotiation was removed in favour of the global
    IRIS_RECONCILE_RPC_ENABLED flag (see migration 0047 which added this
    column).  SQLite 3.35+ supports DROP COLUMN directly.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
    if "supports_reconcile_rpc" in columns:
        conn.execute("ALTER TABLE workers DROP COLUMN supports_reconcile_rpc")
