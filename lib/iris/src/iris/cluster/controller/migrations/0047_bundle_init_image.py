# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist the per-job bundle staging image."""


def _has_column(raw_conn, table: str, column: str) -> bool:
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "job_config", "bundle_init_image"):
        return
    raw_conn.execute("ALTER TABLE job_config ADD COLUMN bundle_init_image TEXT NOT NULL DEFAULT ''")
