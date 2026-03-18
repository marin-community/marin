# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
DROP TABLE IF EXISTS scaling_groups_new;

CREATE TABLE scaling_groups_new (
    name                    TEXT PRIMARY KEY,
    consecutive_failures    INTEGER NOT NULL DEFAULT 0,
    backoff_until_ms        INTEGER NOT NULL DEFAULT 0,
    last_scale_up_ms        INTEGER NOT NULL DEFAULT 0,
    last_scale_down_ms      INTEGER NOT NULL DEFAULT 0,
    quota_exceeded_until_ms INTEGER NOT NULL DEFAULT 0,
    quota_reason            TEXT    NOT NULL DEFAULT '',
    updated_at_ms           INTEGER NOT NULL DEFAULT 0
);
INSERT INTO scaling_groups_new (name, updated_at_ms)
    SELECT name, updated_at_ms FROM scaling_groups;
DROP TABLE scaling_groups;
ALTER TABLE scaling_groups_new RENAME TO scaling_groups;

CREATE TABLE IF NOT EXISTS slices (
    slice_id       TEXT PRIMARY KEY,
    scale_group    TEXT NOT NULL,
    lifecycle      TEXT NOT NULL,
    worker_ids     TEXT NOT NULL DEFAULT '[]',
    created_at_ms  INTEGER NOT NULL DEFAULT 0,
    last_active_ms INTEGER NOT NULL DEFAULT 0,
    error_message  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_slices_scale_group ON slices(scale_group);
"""
    )
