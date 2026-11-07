# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain delegated job authority and a durable attempt retirement outbox."""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "shared_grant_id" not in columns:
        raw_conn.execute("ALTER TABLE jobs ADD COLUMN shared_grant_id TEXT NOT NULL DEFAULT ''")
    # No cascading foreign keys: deletion must not erase an undelivered retirement.
    raw_conn.execute(
        "CREATE TABLE IF NOT EXISTS shared_attempt_grants ("
        "attempt_uid TEXT PRIMARY KEY, job_id TEXT NOT NULL, task_id TEXT NOT NULL, "
        "grant_id TEXT NOT NULL, resource TEXT NOT NULL)"
    )
