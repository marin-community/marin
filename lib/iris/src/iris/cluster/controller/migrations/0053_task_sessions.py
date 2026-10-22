# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist task-session generations, credential blocks and administrative audit events."""


def migrate(raw_conn) -> None:
    raw_conn.execute(
        "CREATE TABLE IF NOT EXISTS task_sessions "
        "(attempt_uid TEXT PRIMARY KEY REFERENCES task_attempts(attempt_uid) ON DELETE CASCADE, "
        "generation INTEGER NOT NULL DEFAULT 0, finalizing_at_ms INTEGER)"
    )
    raw_conn.execute(
        "CREATE TABLE IF NOT EXISTS task_credential_blocks "
        "(scope TEXT NOT NULL, target TEXT NOT NULL, PRIMARY KEY (scope, target))"
    )
    raw_conn.execute(
        "CREATE TABLE IF NOT EXISTS task_credential_audit "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, at_ms INTEGER NOT NULL, actor TEXT NOT NULL, "
        "action TEXT NOT NULL, scope TEXT NOT NULL, target TEXT NOT NULL)"
    )
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_task_credential_audit_at ON task_credential_audit(at_ms)")
