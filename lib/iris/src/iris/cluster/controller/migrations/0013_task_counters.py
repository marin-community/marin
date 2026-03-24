# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "counters_json" not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN counters_json TEXT")
