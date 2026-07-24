# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``endpoints`` foreign keys to ``jobs``/``tasks``.

A federation parent now mirrors a child's link-access endpoints even for a job it
never received as a handoff (see ``live_local_link_endpoints`` and
``replace_remote_for_peer``). Such an "absorbed" endpoint has no backing
``jobs``/``tasks`` row on the parent — the mint path reads only the endpoint row and
parses the owner from its ``task_id`` string — so the row must be insertable without
an FK target.

The FKs only ever provided ``ON DELETE CASCADE`` cleanup. That is now done
explicitly at the single ``delete_job`` chokepoint (which also keeps the in-memory
endpoint cache in sync, which CASCADE never did), so dropping them changes no
cleanup behavior for local rows. ``endpoints`` is referenced by no other table, so
the rebuild needs no ``foreign_keys`` toggling.

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
