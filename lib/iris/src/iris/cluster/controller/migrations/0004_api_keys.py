# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    last_used_at_ms INTEGER,
    expires_at_ms INTEGER,
    revoked_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

ALTER TABLE users ADD COLUMN display_name TEXT;
ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
    CHECK (role IN ('admin', 'user', 'worker'));
"""
    )
