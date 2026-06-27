# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``lease_deadline_ms`` to ``endpoints``.

Endpoints become a lease: registration grants a deadline and re-registering
renews it; a row past its deadline is hidden from reads and swept by the
pruner, independent of the FK CASCADE that ties it to the owning task row.

Existing rows are left NULL, which the projection treats as never-expiring, so
endpoints already registered keep being served exactly as before until their
registrant next re-registers (and is handed a real lease). Backfilling a
``registered_at``-relative deadline would instead retroactively expire any
endpoint older than the lease the moment this schema loads.

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "endpoints", "lease_deadline_ms"):
        return
    raw_conn.execute("ALTER TABLE endpoints ADD COLUMN lease_deadline_ms INTEGER")
