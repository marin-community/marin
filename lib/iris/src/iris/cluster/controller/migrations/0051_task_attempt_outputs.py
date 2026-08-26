# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add per-attempt temporary output metadata."""


def migrate(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_attempt_outputs (
            task_id VARCHAR NOT NULL,
            attempt_id INTEGER NOT NULL,
            archive_json VARCHAR NOT NULL,
            PRIMARY KEY (task_id, attempt_id),
            FOREIGN KEY (task_id, attempt_id)
                REFERENCES task_attempts(task_id, attempt_id) ON DELETE CASCADE
        )
        """
    )
