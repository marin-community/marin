# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``auth.api_keys`` table.

Iris moved to a fully stateless short-TTL EdDSA model: control-plane tokens are
verified by pure crypto plus audience, with no jti revocation set and no DB-backed
key store. ``api_keys`` was the audit/revocation table behind the old symmetric-era
machinery (``create_api_key`` / ``ListApiKeys`` / ``RevokeApiKey`` and the
``load_revocations`` blocklist), all now removed, so the table is dead.

``api_keys`` lives in the attached ``auth`` database (``auth.sqlite3``), so the
statement is schema-qualified; dropping the table also drops its
``idx_api_keys_user`` index. Idempotent: ``DROP TABLE IF EXISTS`` no-ops on a
fresh DB (whose baseline no longer creates the table) or on a re-run, so a crash
mid-run is safe to retry.
"""


def migrate(raw_conn) -> None:
    raw_conn.execute("DROP TABLE IF EXISTS auth.api_keys")
