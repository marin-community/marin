# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
CREATE TABLE IF NOT EXISTS slice_transitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name      TEXT    NOT NULL,
    slice_id        TEXT    NOT NULL,
    timestamp_ms    INTEGER NOT NULL,
    event           TEXT    NOT NULL,
    from_state      TEXT    NOT NULL,
    to_state        TEXT    NOT NULL,
    context_json    TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_slice_transitions_group ON slice_transitions(group_name, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_slice_transitions_slice ON slice_transitions(slice_id, timestamp_ms);
"""
    )
