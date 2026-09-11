# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``submitting_user`` to ``jobs``.

The authenticated principal a job was submitted under, distinct from the friendly
``user_id`` owner: an IAP/JWT email, or ``local_admin`` for a CIDR/loopback
(null-auth) submission. It drives per-cluster federation authorization and rides a
handoff as a signed claim the receiving peer re-checks against its allowlist.

Pre-migration rows predate any authenticated submitter, so they keep the empty
default (rendered as an unknown/legacy submitter); every new row is stamped with a
real value at submit. Idempotent: the column add no-ops when it already exists (a
fresh DB has it from the baseline schema).
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "jobs", "submitting_user"):
        raw_conn.execute("ALTER TABLE jobs ADD COLUMN submitting_user VARCHAR NOT NULL DEFAULT ''")
