# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the legacy ``profiles.task_profiles`` table; ``db.apply_migrations`` unlinks the file."""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS profiles.trg_task_profiles_cap;
        DROP INDEX IF EXISTS profiles.idx_task_profiles_task_kind;
        DROP TABLE IF EXISTS profiles.task_profiles;
        """
    )
