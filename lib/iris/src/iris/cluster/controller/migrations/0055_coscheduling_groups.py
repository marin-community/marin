# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist requested locality groups without changing existing jobs."""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(job_config)").fetchall()}
    if "coscheduling_group_sizes" not in columns:
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN coscheduling_group_sizes TEXT NOT NULL DEFAULT '[]'")
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(task_attempts)").fetchall()}
    if "topology_json" not in columns:
        raw_conn.execute("ALTER TABLE task_attempts ADD COLUMN topology_json TEXT NOT NULL DEFAULT '{}'")
