# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``lease_deadline_ms`` to ``endpoints``.

Existing rows are left NULL (the projection treats NULL as never-expiring) so
loading the new schema cannot retroactively expire an endpoint registered more
than a lease ago; each is handed a real deadline on its next re-registration.
Idempotent: safe to re-run after a mid-migration crash.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "endpoints", "lease_deadline_ms"):
        return
    raw_conn.execute("ALTER TABLE endpoints ADD COLUMN lease_deadline_ms INTEGER")
