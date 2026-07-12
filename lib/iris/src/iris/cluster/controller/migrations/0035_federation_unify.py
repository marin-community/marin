# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unify federation bookkeeping into one ``federated_jobs`` table plus a changelog.

``federated_jobs`` now records both directions of a handoff: a SENT row (this
cluster is the parent) tracks the handoff lifecycle to ``peer_id``; a RECEIVED row
(this cluster is the peer) names the requesting cluster in ``peer_id``. A
``direction`` discriminator distinguishes them and the SENT-only ``handoff_state``
becomes nullable. The peer-side ``federation_changelog`` records a change event per
job/task mutation, each row stamped with the ``requester_id`` it belongs to, so
``FederationSync`` reports a requester only its own handoffs without a join — and a
tombstone event survives the job delete (no foreign key to ``jobs``).

These tables stay empty until this controller federates a job, so a controller
that never does is unchanged.

Idempotent: re-run from scratch if the controller crashes mid-migration. On a
fresh DB the tables already exist in their unified shape from the baseline schema,
so the changelog create no-ops and the ``federated_jobs`` rebuild is skipped
(the ``direction`` column is already present).
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def _create_changelog_table(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federation_changelog (
            seq INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            job_id VARCHAR NOT NULL,
            requester_id VARCHAR NOT NULL,
            task_index INTEGER,
            tombstone INTEGER NOT NULL DEFAULT 0,
            written_ms INTEGER NOT NULL
        )
        """
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_federation_changelog_requester ON federation_changelog (requester_id, seq)"
    )


def _rebuild_federated_jobs(raw_conn) -> None:
    """Rebuild the pre-unification ``federated_jobs`` (0034 shape) in place.

    SQLite cannot relax a ``NOT NULL`` constraint in place, so add the
    ``direction`` discriminator and make the SENT-only ``handoff_state`` nullable
    via the create/copy/drop/rename dance. All pre-existing rows are outbound
    handoffs, so they become ``direction = SENT (0)``. Skipped when ``direction``
    already exists (a fresh DB from the baseline schema, or an already-migrated one).
    """
    if _has_column(raw_conn, "federated_jobs", "direction"):
        return
    raw_conn.execute(
        """
        CREATE TABLE federated_jobs_new (
            job_id VARCHAR NOT NULL PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
            direction INTEGER NOT NULL,
            peer_id VARCHAR NOT NULL,
            owner_principal VARCHAR NOT NULL DEFAULT '',
            handoff_state INTEGER,
            cancel_intent_version INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    raw_conn.execute(
        """
        INSERT INTO federated_jobs_new (
            job_id, direction, peer_id, owner_principal, handoff_state,
            cancel_intent_version
        )
        SELECT job_id, 0, peer_id, owner_principal, handoff_state,
               cancel_intent_version
        FROM federated_jobs
        """
    )
    raw_conn.execute("DROP TABLE federated_jobs")
    raw_conn.execute("ALTER TABLE federated_jobs_new RENAME TO federated_jobs")
    raw_conn.execute("CREATE INDEX idx_federated_jobs_direction_peer ON federated_jobs (direction, peer_id)")


def migrate(raw_conn) -> None:
    _create_changelog_table(raw_conn)
    _rebuild_federated_jobs(raw_conn)
