# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
CREATE TABLE IF NOT EXISTS controller_secrets (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
"""
    )
