# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store the optional application health policy for each job."""


def _has_column(raw_conn, table: str, column: str) -> bool:
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "job_config", "health_check_json"):
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN health_check_json VARCHAR")
