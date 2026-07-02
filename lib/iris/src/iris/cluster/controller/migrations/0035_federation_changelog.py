# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the peer-side federation changelog and received-job ownership tables.

A controller acting as a federation peer records, in the same transaction as
each job/task mutation, a ``federation_changelog`` event and — for a handed-off
job — a ``federation_received_jobs`` ownership row naming the requester. The
``FederationSync`` RPC joins the two so a requester learns only its own handoffs,
and pages through the changelog by a monotonic ``seq`` cursor. ``federation_changelog``
carries no foreign key to ``jobs`` so a tombstone event survives the job delete.
``federation_changelog_floor`` holds the highest seq removed by retention; a
requester whose cursor is at or below it must full-resync.

These tables stay empty until this controller receives a handoff, so a
controller that is never a peer is unchanged.

Idempotent: re-run from scratch if the controller crashes mid-migration. On a
fresh DB the tables already exist from the baseline schema, so every create no-ops.
"""


def _create_changelog_tables(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federation_changelog (
            seq INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            job_id VARCHAR NOT NULL,
            task_index INTEGER,
            tombstone INTEGER NOT NULL DEFAULT 0,
            written_ms INTEGER NOT NULL
        )
        """
    )
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_federation_changelog_job ON federation_changelog (job_id)")
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federation_received_jobs (
            job_id VARCHAR NOT NULL PRIMARY KEY,
            requester_id VARCHAR NOT NULL,
            owner_principal VARCHAR NOT NULL DEFAULT '',
            received_ms INTEGER NOT NULL
        )
        """
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_federation_received_requester ON federation_received_jobs (requester_id)"
    )
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federation_changelog_floor (
            id INTEGER NOT NULL PRIMARY KEY,
            floor INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def migrate(raw_conn) -> None:
    _create_changelog_tables(raw_conn)
