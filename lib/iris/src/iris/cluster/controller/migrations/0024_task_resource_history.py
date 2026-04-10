import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_resource_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            attempt_id INTEGER NOT NULL,
            snapshot_proto BLOB NOT NULL,
            timestamp_ms INTEGER NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_resource_history_task_attempt"
        " ON task_resource_history(task_id, attempt_id, id DESC)"
    )
