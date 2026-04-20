# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the dead `logs` SQLite table.

The `logs` table was schema-only — no code path ever wrote to it. Controller
log data has always lived in the RAM buffer + Parquet segments managed by
`iris.cluster.log_store.duckdb_store.DuckDBLogStore`. SQL consumers that hit
`logs` saw silent zero-row results, which is worse than a missing-table error.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_logs_key")
    conn.execute("DROP TABLE IF EXISTS logs")
