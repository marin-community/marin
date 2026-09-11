# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``status_message`` to ``tasks``.

The backend's current human-readable reason a task is not yet running — the
Kubernetes pod's container-waiting reason or the Kueue admission verdict for a
SchedulingGated pod. It is served as ``TaskStatus.status_message`` and mirrored
across federation so a stuck-BUILDING task explains itself on the dashboard, both
locally and on a hub that federated the job out.

Nullable: pre-migration rows and running/quiet tasks carry NULL (rendered as no
message). Idempotent: the column add no-ops when it already exists (a fresh DB has
it from the baseline schema).
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "tasks", "status_message"):
        raw_conn.execute("ALTER TABLE tasks ADD COLUMN status_message VARCHAR")
