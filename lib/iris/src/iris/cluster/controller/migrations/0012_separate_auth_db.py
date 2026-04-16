# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create auth tables in the attached auth database and migrate data from main.

By the time this migration runs, the auth DB is already ATTACHed as 'auth'
in ControllerDB.__init__. We create the tables in the auth schema, copy any
existing data from the main tables (created by migrations 0004/0006), and
drop the main copies.
"""

import sqlite3

AUTH_TABLES = ("api_keys", "controller_secrets")


def _table_exists(conn: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS auth.api_keys (
            key_id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL UNIQUE,
            key_prefix TEXT NOT NULL,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            last_used_at_ms INTEGER,
            expires_at_ms INTEGER,
            revoked_at_ms INTEGER
        );
        CREATE INDEX IF NOT EXISTS auth.idx_api_keys_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS auth.idx_api_keys_user ON api_keys(user_id);
        CREATE TABLE IF NOT EXISTS auth.controller_secrets (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
    """
    )

    for table in AUTH_TABLES:
        if _table_exists(conn, table, "main"):
            conn.execute(f"INSERT OR IGNORE INTO auth.{table} SELECT * FROM main.{table}")
            conn.execute(f"DROP TABLE main.{table}")
    conn.commit()
