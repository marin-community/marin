# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record when an Attempt's external runtime is confirmed absent."""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(task_attempts)").fetchall()}
    if "runtime_released_at_ms" not in columns:
        raw_conn.execute("ALTER TABLE task_attempts ADD COLUMN runtime_released_at_ms INTEGER")

    # Existing finished Attempts predate the distinct runtime-release signal.
    # Preserve their historical terminal-is-released interpretation rather than
    # replaying stop requests for the entire retained Attempt history.
    raw_conn.execute(
        """
        UPDATE task_attempts
        SET runtime_released_at_ms = finished_at_ms
        WHERE runtime_released_at_ms IS NULL AND finished_at_ms IS NOT NULL
        """
    )
    raw_conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_task_attempts_runtime_unreleased
        ON task_attempts (attempt_uid)
        WHERE runtime_released_at_ms IS NULL
        """
    )
