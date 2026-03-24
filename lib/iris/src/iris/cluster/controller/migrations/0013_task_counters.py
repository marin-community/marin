# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE tasks ADD COLUMN counters_json TEXT")
