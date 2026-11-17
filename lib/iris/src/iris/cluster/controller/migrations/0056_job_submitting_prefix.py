# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``submitting_prefix`` to ``jobs``.

The job-name prefix the submitting token was scoped to ("" when unscoped),
stamped by the controller from the verified identity at admission. It is read
back when minting the job's task tokens, so a job submitted under a
prefix-scoped token cannot act outside that prefix — the token's own claim is
not enough, because dispatch happens long after the submitting request.

Pre-migration rows predate scoped tokens and keep the empty default (unscoped
behavior, unchanged); every new row is stamped at submit. Idempotent: the
column add no-ops when it already exists (a fresh DB has it from the baseline
schema).
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "jobs", "submitting_prefix"):
        raw_conn.execute("ALTER TABLE jobs ADD COLUMN submitting_prefix VARCHAR NOT NULL DEFAULT ''")
