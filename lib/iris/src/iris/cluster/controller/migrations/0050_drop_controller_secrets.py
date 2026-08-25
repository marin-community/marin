# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``auth.controller_secrets`` table.

The table held the persistent HMAC-SHA256 JWT signing key the controller minted
for itself on first run. Iris now signs control-plane tokens with a per-cluster
Ed25519 key sourced from ``auth.signing_key`` (a secret reference, never the DB),
so the read/create path and its only reader are gone and any surviving row is an
unusable secret from the symmetric era.

``controller_secrets`` lives in the attached ``auth`` database (``auth.sqlite3``),
so the statement is schema-qualified; this leaves that database with no tables.
Idempotent: ``DROP TABLE IF EXISTS`` no-ops on a fresh DB (whose baseline no
longer creates the table) or on a re-run, so a crash mid-run is safe to retry.
"""


def migrate(raw_conn) -> None:
    raw_conn.execute("DROP TABLE IF EXISTS auth.controller_secrets")
