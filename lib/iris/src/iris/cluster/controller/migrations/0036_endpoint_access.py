# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``access`` to ``endpoints``.

Per-endpoint proxy access mode (EndpointAccess int: PRIVATE=0, PUBLIC=1,
BEARER=2). Existing rows are left NULL; the projection reads NULL as PRIVATE, so
loading the new schema keeps every pre-migration endpoint on today's
cluster-identity-required semantics. Idempotent: safe to re-run after a
mid-migration crash.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "endpoints", "access"):
        return
    raw_conn.execute("ALTER TABLE endpoints ADD COLUMN access INTEGER")
