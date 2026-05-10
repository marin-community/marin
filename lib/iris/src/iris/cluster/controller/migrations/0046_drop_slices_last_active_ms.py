# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop ``slices.last_active_ms`` (now tracked in-memory as ``quiet_since``).

The autoscaler used to persist a continuously-mutating "last active" stamp,
but the periodic-flush logic was buggy (the elapsed clock was reset every
tick, so the row froze at the first post-restart write). The replacement
tracks active→idle transitions in memory only — there is no need to persist
it, since recovery starts the dwell-time clock fresh on the next tick anyway.
"""

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(conn: sqlite3.Connection) -> None:
    if _has_column(conn, "slices", "last_active_ms"):
        conn.execute("ALTER TABLE slices DROP COLUMN last_active_ms")
