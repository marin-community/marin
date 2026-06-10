# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the dead ``api_keys.key_hash`` column.

``key_hash`` was scaffolding for DB-backed API-key verification that was never
wired up: it is written but never read — all token verification happens
in-memory (``StaticTokenVerifier``) or via JWT crypto. It also carried an old
``NOT NULL`` constraint that rejected the worker token (which has no hash),
crashing controller startup on DBs created before the column was meant to go
nullable. Rather than relax the constraint, drop the column outright.

``api_keys`` lives in the attached ``auth`` database (``auth.sqlite3``), so every
statement is schema-qualified. SQLite cannot drop a column referenced by an
index in place across all supported versions, so this rebuilds the table to
match ``schema.py``'s ``auth_api_keys_table`` (no ``key_hash``, no unique
constraint, just ``idx_api_keys_user``). Idempotent: if ``key_hash`` is already
gone it is a no-op, so a crash mid-run is safe to retry.
"""


def _has_key_hash(raw_conn) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == "key_hash" for row in raw_conn.execute("PRAGMA auth.table_info(api_keys)").fetchall())


def migrate(raw_conn) -> None:
    if not _has_key_hash(raw_conn):
        return

    raw_conn.commit()
    raw_conn.execute("BEGIN IMMEDIATE")
    try:
        raw_conn.execute(
            """
            CREATE TABLE auth.api_keys_new (
                key_id VARCHAR NOT NULL,
                key_prefix VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                created_at_ms INTEGER NOT NULL,
                last_used_at_ms INTEGER,
                expires_at_ms INTEGER,
                revoked_at_ms INTEGER,
                PRIMARY KEY (key_id)
            )
            """
        )
        raw_conn.execute(
            """
            INSERT INTO auth.api_keys_new (
                key_id, key_prefix, user_id, name,
                created_at_ms, last_used_at_ms, expires_at_ms, revoked_at_ms
            )
            SELECT
                key_id, key_prefix, user_id, name,
                created_at_ms, last_used_at_ms, expires_at_ms, revoked_at_ms
            FROM auth.api_keys
            """
        )
        raw_conn.execute("DROP TABLE auth.api_keys")
        raw_conn.execute("ALTER TABLE auth.api_keys_new RENAME TO api_keys")
        raw_conn.execute("CREATE INDEX auth.idx_api_keys_user ON api_keys (user_id)")
        raw_conn.commit()
    except Exception:
        raw_conn.execute("ROLLBACK")
        raise
