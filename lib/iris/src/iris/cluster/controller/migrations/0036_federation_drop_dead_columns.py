# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop dead federation bookkeeping columns.

Three columns were declared but never read: ``federated_jobs.last_sync_ms`` and
``federated_jobs.terminal_error`` (the error path writes ``jobs.error``), and
``federation_sync_state.last_full_resync_ms``. SQLite cannot drop these in place
without rewriting the table, so rebuild each via the create/copy/drop/rename
dance. Federation is not yet rolled out, so these tables are empty and the copy
is trivial.

Idempotent: each rebuild is skipped when its column is already absent — a fresh
DB from the current baseline schema, or an already-migrated one.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def _rebuild_federated_jobs(raw_conn) -> None:
    if not _has_column(raw_conn, "federated_jobs", "last_sync_ms"):
        return
    raw_conn.execute(
        """
        CREATE TABLE federated_jobs_new (
            job_id VARCHAR NOT NULL PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
            direction INTEGER NOT NULL,
            peer_id VARCHAR NOT NULL,
            owner_principal VARCHAR NOT NULL DEFAULT '',
            remote_job_id VARCHAR,
            handoff_state INTEGER,
            cancel_intent_version INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    raw_conn.execute(
        """
        INSERT INTO federated_jobs_new (
            job_id, direction, peer_id, owner_principal, remote_job_id, handoff_state, cancel_intent_version
        )
        SELECT job_id, direction, peer_id, owner_principal, remote_job_id, handoff_state, cancel_intent_version
        FROM federated_jobs
        """
    )
    raw_conn.execute("DROP TABLE federated_jobs")
    raw_conn.execute("ALTER TABLE federated_jobs_new RENAME TO federated_jobs")
    raw_conn.execute("CREATE INDEX idx_federated_jobs_direction_peer ON federated_jobs (direction, peer_id)")


def _rebuild_federation_sync_state(raw_conn) -> None:
    if not _has_column(raw_conn, "federation_sync_state", "last_full_resync_ms"):
        return
    raw_conn.execute(
        """
        CREATE TABLE federation_sync_state_new (
            peer_id VARCHAR NOT NULL PRIMARY KEY,
            cursor VARCHAR NOT NULL DEFAULT ''
        )
        """
    )
    raw_conn.execute(
        "INSERT INTO federation_sync_state_new (peer_id, cursor) SELECT peer_id, cursor FROM federation_sync_state"
    )
    raw_conn.execute("DROP TABLE federation_sync_state")
    raw_conn.execute("ALTER TABLE federation_sync_state_new RENAME TO federation_sync_state")


def migrate(raw_conn) -> None:
    _rebuild_federated_jobs(raw_conn)
    _rebuild_federation_sync_state(raw_conn)
