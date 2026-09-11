# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile DBs that recorded the pre-revision 0034/0035 federation shape.

0034/0035 were revised in place after they had already been applied to a live
controller: the federation coordinate was renamed ``child_cluster`` -> ``cluster``
(default ``''`` -> ``'local'``) and the ``federated_jobs`` / ``federation_sync_state``
bookkeeping columns ``remote_job_id`` / ``last_sync_ms`` / ``terminal_error`` /
``last_full_resync_ms`` were dropped. A DB that recorded the old stems keeps the
old shape forever, because a recorded migration is never re-run. This forward
migration brings that shape to the current schema.

Federation has never been enabled, so every row's coordinate is local: the added
``cluster`` column defaults to ``'local'``, already the correct value for every
existing row.

Idempotent: a fresh DB (or one already on the current schema) has ``cluster`` and
none of the dropped columns, so every step no-ops.
"""

# federated_jobs / federation_sync_state columns the revised design removed.
_STALE_COLUMNS = (
    ("federated_jobs", "remote_job_id"),
    ("federated_jobs", "last_sync_ms"),
    ("federated_jobs", "terminal_error"),
    ("federation_sync_state", "last_full_resync_ms"),
)


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    for table, column in _STALE_COLUMNS:
        if _has_column(raw_conn, table, column):
            raw_conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")

    # Rename the federation coordinate child_cluster -> cluster on jobs/tasks.
    # The tasks "local" partial indexes filter on child_cluster and would block
    # the column drop, so drop them first; they are recreated on cluster below.
    if _has_column(raw_conn, "jobs", "child_cluster") or _has_column(raw_conn, "tasks", "child_cluster"):
        raw_conn.execute("DROP INDEX IF EXISTS idx_tasks_pending_local")
        raw_conn.execute("DROP INDEX IF EXISTS idx_tasks_state_local")
        for table in ("jobs", "tasks"):
            if not _has_column(raw_conn, table, "child_cluster"):
                continue
            if not _has_column(raw_conn, table, "cluster"):
                # Clause order matches the baseline's SQLAlchemy-emitted DDL so the
                # stored CREATE text is byte-identical to a fresh DB's.
                raw_conn.execute(f"ALTER TABLE {table} ADD COLUMN cluster VARCHAR DEFAULT 'local' NOT NULL")
            raw_conn.execute(f"ALTER TABLE {table} DROP COLUMN child_cluster")

    # Ensure the local-scheduler partial indexes exist on the cluster column.
    # No-op on a fresh DB where the baseline schema already created them.
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_pending_local ON tasks "
        "(state, priority_band, priority_neg_depth, priority_root_submitted_ms, "
        "submitted_at_ms, priority_insertion) WHERE cluster = 'local'"
    )
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state_local ON tasks (state) WHERE cluster = 'local'")
