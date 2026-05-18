# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Add supports_reconcile_rpc column to workers table."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
    if "supports_reconcile_rpc" not in columns:
        conn.execute("ALTER TABLE workers ADD COLUMN supports_reconcile_rpc BOOLEAN NOT NULL DEFAULT FALSE")
