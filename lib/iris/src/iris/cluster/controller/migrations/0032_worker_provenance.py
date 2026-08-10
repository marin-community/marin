# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace ``workers.md_git_hash`` with ``md_provenance_json``.

The flat tree hash is folded into a serialized ``Provenance`` (tree hash, base
commit, dirty, branch, builder), so the bare ``md_git_hash`` column is dropped
and replaced by ``md_provenance_json``. Existing rows default to ``'{}'`` and
are refreshed with real provenance on the worker's next re-registration.
Idempotent: safe to re-run after a mid-migration crash.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "workers", "md_provenance_json"):
        raw_conn.execute("ALTER TABLE workers ADD COLUMN md_provenance_json TEXT NOT NULL DEFAULT '{}'")
    if _has_column(raw_conn, "workers", "md_git_hash"):
        raw_conn.execute("ALTER TABLE workers DROP COLUMN md_git_hash")
