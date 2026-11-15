# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record each attempt's worker region so it outlives the worker row."""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(task_attempts)").fetchall()}
    if "region" not in columns:
        raw_conn.execute("ALTER TABLE task_attempts ADD COLUMN region TEXT")
