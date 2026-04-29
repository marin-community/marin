# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the `txn_log` and `txn_actions` audit tables.

The controller no longer records mutating transitions to SQLite; every
state change now emits a semi-structured `logger.info` line via
``transitions.log_event`` that is captured by the Iris log server and
queried through the normal log-store DuckDB interface.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER IF EXISTS trg_txn_log_retention")
    conn.execute("DROP INDEX IF EXISTS idx_txn_actions_txn")
    conn.execute("DROP TABLE IF EXISTS txn_actions")
    conn.execute("DROP TABLE IF EXISTS txn_log")
