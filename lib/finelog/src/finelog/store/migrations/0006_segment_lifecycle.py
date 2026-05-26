# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace ``copied_at_ms`` with an explicit ``location`` enum.

A segment's bytes can live on local disk (``LOCAL``), on the remote
bucket (``REMOTE``), or both (``BOTH``). The catalog row stays put when
eviction flips ``BOTH`` → ``REMOTE``, so the sync loop can tell an
archived segment apart from a compaction-deleted orphan.

Backfill: ``copied_at_ms IS NULL`` becomes ``LOCAL``, ``NOT NULL``
becomes ``BOTH``. NOT NULL is enforced at the application layer because
the ``segments_level_idx`` index blocks ``SET NOT NULL`` DDL.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from finelog.store.migrations._runner import transactional


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    # Drop and recreate the index because DuckDB rejects ``DROP COLUMN`` on
    # an indexed table. The ADD+UPDATE pair runs in one transaction so no
    # query sees a column of NULLs; the surrounding DDLs each auto-commit
    # to sidestep DuckDB's "multiple DDLs + DML on one table in one txn"
    # restriction.
    conn.execute("DROP INDEX IF EXISTS segments_ns_level_minseq")
    with transactional(conn):
        conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS location TEXT")
        conn.execute(
            """
            UPDATE segments
               SET location = CASE WHEN copied_at_ms IS NULL THEN 'LOCAL' ELSE 'BOTH' END
             WHERE location IS NULL
            """
        )
    conn.execute("ALTER TABLE segments DROP COLUMN IF EXISTS copied_at_ms")
    conn.execute("CREATE INDEX IF NOT EXISTS segments_ns_level_minseq ON segments (namespace, level, min_seq)")
