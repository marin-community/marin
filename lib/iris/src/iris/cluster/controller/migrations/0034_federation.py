# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the federation ``cluster`` coordinate and its sidecar tables.

Federation hands whole jobs off to a peer cluster. ``jobs``/``tasks`` carry a
``cluster`` coordinate that is always set: ``'local'`` (owned by this controller)
or ``'<peer>'`` (handed off to that peer, with ``backend_id = ''``). Handed-off
rows must be structurally invisible to the local scheduler fold, which reads the
``local_tasks`` selectable (``cluster = 'local'``) backed by the partial indexes
created here. The ``federated_jobs``/``federation_sync_state``/``federated_tasks``
tables hold federation-only join metadata; job/task state lives in the main rows.

Existing rows are local, so the ``'local'`` default the ADD COLUMN stamps is
already the correct value — no backfill pass is needed.

Idempotent: re-run from scratch if the controller crashes mid-migration. On a
fresh DB the columns/tables/indexes already exist from the baseline schema, so
every add/create no-ops.
"""

_CLUSTER_TABLES = ("jobs", "tasks")


def _columns(raw_conn, table: str) -> list[tuple]:
    return raw_conn.execute(f"PRAGMA table_info({table})").fetchall()


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in _columns(raw_conn, table))


def _add_cluster_columns(raw_conn) -> None:
    for table in _CLUSTER_TABLES:
        if not _has_column(raw_conn, table, "cluster"):
            raw_conn.execute(f"ALTER TABLE {table} ADD COLUMN cluster VARCHAR NOT NULL DEFAULT 'local'")


def _create_sidecar_tables(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federated_jobs (
            job_id VARCHAR NOT NULL PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
            peer_id VARCHAR NOT NULL,
            owner_principal VARCHAR NOT NULL,
            handoff_state INTEGER NOT NULL,
            cancel_intent_version INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federation_sync_state (
            peer_id VARCHAR NOT NULL PRIMARY KEY,
            cursor VARCHAR NOT NULL DEFAULT ''
        )
        """
    )
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federated_tasks (
            task_id VARCHAR NOT NULL PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
            peer_worker_label VARCHAR NOT NULL DEFAULT ''
        )
        """
    )


def _create_partial_indexes(raw_conn) -> None:
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_pending_local ON tasks "
        "(state, priority_band, priority_neg_depth, priority_root_submitted_ms, "
        "submitted_at_ms, priority_insertion) WHERE cluster = 'local'"
    )
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state_local ON tasks (state) WHERE cluster = 'local'")


def migrate(raw_conn) -> None:
    _add_cluster_columns(raw_conn)
    _create_sidecar_tables(raw_conn)
    _create_partial_indexes(raw_conn)
