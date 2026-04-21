# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create task_profiles table in the attached profiles database and migrate data from main.

By the time this migration runs, the profiles DB is already ATTACHed as 'profiles'
in ControllerDB.__init__. We create the table in the profiles schema, copy any
existing data from the main table, and drop the main copy.
"""

import sqlite3


def _table_exists(conn: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS profiles.task_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            profile_data BLOB NOT NULL,
            captured_at_ms INTEGER NOT NULL,
            profile_kind TEXT NOT NULL DEFAULT 'cpu'
        );
        CREATE INDEX IF NOT EXISTS profiles.idx_task_profiles_task_kind
            ON task_profiles(task_id, profile_kind, id DESC);
    """
    )

    # SQLite prohibits qualified table names inside trigger bodies;
    # the trigger is scoped to the profiles schema by its qualified name.
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS profiles.trg_task_profiles_cap
        AFTER INSERT ON task_profiles
        BEGIN
          DELETE FROM task_profiles
           WHERE task_id = NEW.task_id
             AND profile_kind = NEW.profile_kind
             AND id NOT IN (
               SELECT id FROM task_profiles
                WHERE task_id = NEW.task_id
                  AND profile_kind = NEW.profile_kind
                ORDER BY id DESC
                LIMIT 10
             );
        END
    """
    )

    if _table_exists(conn, "task_profiles", "main"):
        conn.execute("INSERT OR IGNORE INTO profiles.task_profiles SELECT * FROM main.task_profiles")
        conn.execute("DROP TRIGGER IF EXISTS trg_task_profiles_cap")
        conn.execute("DROP INDEX IF EXISTS idx_task_profiles_task_kind")
        conn.execute("DROP TABLE main.task_profiles")
    conn.commit()
