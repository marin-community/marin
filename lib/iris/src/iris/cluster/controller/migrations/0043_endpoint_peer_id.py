# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``peer_id`` to ``endpoints``.

Owning peer cluster id for an endpoint mirrored from a federated child; NULL for a
locally-registered endpoint. The /proxy route forwards a row with ``peer_id`` set to
that peer's controller rather than dialing ``address`` directly. Existing rows stay
NULL (local), so loading the new schema leaves every pre-migration endpoint local.
Idempotent: safe to re-run after a mid-migration crash.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if _has_column(raw_conn, "endpoints", "peer_id"):
        return
    raw_conn.execute("ALTER TABLE endpoints ADD COLUMN peer_id TEXT")
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_endpoints_peer_id ON endpoints (peer_id)")
