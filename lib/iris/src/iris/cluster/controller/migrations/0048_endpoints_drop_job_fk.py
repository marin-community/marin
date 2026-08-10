# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``endpoints`` foreign keys to ``jobs``/``tasks``.

An endpoint's lifecycle is governed by its lease, not by its owning task row:
reads hide an expired lease, the pruner's ``sweep_expired`` reclaims it, and
``EndpointsProjection.add`` refuses to register or renew once the task row is gone
or terminal. The FKs only ever provided ``ON DELETE CASCADE`` cleanup, which is now
done explicitly at the single ``delete_job`` chokepoint (also keeping the in-memory
endpoint cache in sync, which CASCADE never did). Dropping the FKs decouples an
endpoint row from a live ``jobs``/``tasks`` target and changes no cleanup behavior.
``endpoints`` is referenced by no other table, so the rebuild needs no
``foreign_keys`` toggling.

Idempotent: a no-op once the FKs are gone, so a crash mid-run is safe to retry.
"""


def _has_job_fk(raw_conn) -> bool:
    # PRAGMA foreign_key_list columns: (id, seq, table, from, to, on_update, on_delete, match)
    return any(row[2] in ("jobs", "tasks") for row in raw_conn.execute("PRAGMA foreign_key_list(endpoints)").fetchall())


def migrate(raw_conn) -> None:
    if not _has_job_fk(raw_conn):
        return

    raw_conn.commit()
    raw_conn.execute("BEGIN IMMEDIATE")
    try:
        raw_conn.execute(
            """
            CREATE TABLE endpoints_new (
                endpoint_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                address VARCHAR NOT NULL,
                job_id VARCHAR NOT NULL,
                task_id VARCHAR,
                metadata_json VARCHAR NOT NULL,
                registered_at_ms INTEGER NOT NULL,
                lease_deadline_ms INTEGER,
                access INTEGER,
                peer_id VARCHAR,
                PRIMARY KEY (endpoint_id)
            )
            """
        )
        raw_conn.execute(
            """
            INSERT INTO endpoints_new (
                endpoint_id, name, address, job_id, task_id, metadata_json,
                registered_at_ms, lease_deadline_ms, access, peer_id
            )
            SELECT
                endpoint_id, name, address, job_id, task_id, metadata_json,
                registered_at_ms, lease_deadline_ms, access, peer_id
            FROM endpoints
            """
        )
        raw_conn.execute("DROP TABLE endpoints")
        raw_conn.execute("ALTER TABLE endpoints_new RENAME TO endpoints")
        raw_conn.execute("CREATE INDEX idx_endpoints_name ON endpoints (name)")
        raw_conn.execute("CREATE INDEX idx_endpoints_task ON endpoints (task_id)")
        raw_conn.execute("CREATE INDEX idx_endpoints_job_id ON endpoints (job_id)")
        raw_conn.execute("CREATE INDEX idx_endpoints_peer_id ON endpoints (peer_id)")
        raw_conn.commit()
    except Exception:
        raw_conn.execute("ROLLBACK")
        raise
