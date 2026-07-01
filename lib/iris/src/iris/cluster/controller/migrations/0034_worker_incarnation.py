# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``registered_at_ms`` to ``workers``.

Stamped by ``ops.worker.register`` on every (re)registration, this is the
worker's incarnation marker: a placement authored at schedule time carries the
worker's ``registered_at_ms`` alongside its address, and commit-time validation
drops the placement if the worker has since re-registered (recycled worker_id
or address) even though the row's address is unchanged.

Idempotent: re-run from scratch if the controller crashes mid-migration. On a
fresh DB the column already exists from the baseline schema, so the add no-ops.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "workers", "registered_at_ms"):
        raw_conn.execute("ALTER TABLE workers ADD COLUMN registered_at_ms INTEGER NOT NULL DEFAULT 0")
