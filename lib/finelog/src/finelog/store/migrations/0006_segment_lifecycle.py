# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace ``copied_at_ms`` with an explicit ``location`` enum.

A segment's bytes can live on local disk, on the remote bucket, or both.
The previous ``copied_at_ms`` column carried that fact implicitly (NULL =
not durable; NOT NULL = remote copy exists), and "row absent from catalog"
was overloaded to mean both "evicted locally" and "deleted by compaction."
The remote-sync loop couldn't distinguish those cases and would delete
durable archives whose only mistake was being a row no longer in the
catalog.

The new model: every segment that is part of the table has a catalog row,
and its ``location`` says where its bytes live:

* ``LOCAL`` — freshly written and not yet uploaded (or a never-uploaded L0).
* ``BOTH`` — on disk and durably on remote.
* ``REMOTE`` — durable archive only; the local file was evicted.

Compaction drops input rows immediately at commit; the merged output is
inserted as ``LOCAL``. The sync loop reconciles remote with catalog:
``LOCAL`` rows are uploaded (or adopted, if the file is already there from
a crash mid-upload), and remote files with no catalog row are ``fs.rm``'d.
Eviction flips ``BOTH`` → ``REMOTE`` instead of dropping the row, so the
durable archive is never mistaken for an orphan.

Backfill: existing rows with ``copied_at_ms IS NULL`` are at ``LOCAL``;
``copied_at_ms IS NOT NULL`` means ``BOTH``. Pre-migration eviction was
the only row-removal path, so every surviving row had a local file.

Non-null is enforced at the application layer rather than via DuckDB's
``SET NOT NULL`` because the existing ``segments_level_idx`` blocks
schema-altering DDL on the table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    del data_dir
    # DuckDB rejects ``DROP COLUMN`` (and ``SET NOT NULL``) on a table
    # referenced by an index, so drop the level index, run the schema
    # changes, then recreate it. NOT NULL is enforced at the application
    # layer (the backfill below populates every existing row, and
    # ``upsert_segment`` always provides ``location``).
    conn.execute("DROP INDEX IF EXISTS segments_ns_level_minseq")
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS location TEXT")
    conn.execute(
        """
        UPDATE segments
           SET location = CASE WHEN copied_at_ms IS NULL THEN 'LOCAL' ELSE 'BOTH' END
         WHERE location IS NULL
        """
    )
    conn.execute("ALTER TABLE segments DROP COLUMN copied_at_ms")
    conn.execute("CREATE INDEX IF NOT EXISTS segments_ns_level_minseq ON segments (namespace, level, min_seq)")
