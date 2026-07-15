# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``handoff_nonce`` to ``federated_jobs``.

Identifies one handoff incarnation: minted per SENT handle, carried on the
delivered request, and stored on the peer's RECEIVED row. Pre-migration rows get
'' on both sides, which compares equal — an in-flight handoff keeps deduping as
an idempotent replay across the migration. Idempotent: safe to re-run after a
mid-migration crash.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "federated_jobs", "handoff_nonce"):
        return
    raw_conn.execute("ALTER TABLE federated_jobs ADD COLUMN handoff_nonce TEXT NOT NULL DEFAULT ''")
