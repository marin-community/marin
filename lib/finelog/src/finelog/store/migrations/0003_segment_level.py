# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace the lifecycle ``state`` column with a numeric ``level``.

Before this migration, segment lifecycle was encoded in two places:

* ``segments.state`` — a TEXT column containing ``'tmp'`` (post-flush) or
  ``'finalized'`` (post-compaction).
* The on-disk filename — ``tmp_<seq>.parquet`` or ``logs_<seq>.parquet``.

The leveled compaction scheme replaces both with a numeric tier:
``segments.level`` (0 = freshly flushed; promoted to ``level + 1`` when the
planner picks it up) and ``seg_L<n>_<min_seq:019d>.parquet`` filenames.
After this migration there is no ``state`` column and no ``tmp_*`` /
``logs_*`` filename to worry about.

Resumability
------------
The DB-level work (ADD/DROP COLUMN, UPDATE) runs inside the migration
runner's enclosing transaction, so it's atomic. The filesystem rename is
not, but each ``os.rename`` is. We walk every segment row, derive the
target filename from ``(level, min_seq)``, and rename the file:

* If the source still exists and the destination doesn't, ``os.rename``.
* If the destination already exists (a prior crashed pass moved it),
  treat the source as a duplicate, assert size match, ``unlink`` source.
* If neither exists, the parquet is gone — leave the catalog row in
  place; ``DiskLogNamespace`` boot reconciliation will drop it once the
  store opens.

In every case the catalog row's ``path`` is rewritten to the new name.
A re-run after partial filesystem progress converges, because the
``UPDATE segments SET path = ?`` is a no-op against rows already pointing
at the new name.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

_OLD_FILENAME_RE = re.compile(r"^(?P<prefix>tmp_|logs_)(?P<seq>\d+)\.parquet$")


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    # ---- DB schema rewrite -------------------------------------------------
    # DuckDB rejects ``ADD COLUMN ... NOT NULL`` against an existing table,
    # so the column lands nullable and we rely on application code to set
    # ``level`` on every insert going forward. Backfill (only when ``state``
    # is still present, i.e. first-time apply): ``finalized`` rows become
    # L1, ``tmp`` rows become L0. A re-run after a partial crash is a
    # clean no-op for the SQL phase.
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS level INTEGER")
    state_present = bool(
        conn.execute(
            "SELECT 1 FROM information_schema.columns " "WHERE table_name = 'segments' AND column_name = 'state' LIMIT 1"
        ).fetchone()
    )
    if state_present:
        conn.execute("UPDATE segments SET level = CASE WHEN state = 'finalized' THEN 1 ELSE 0 END")
        conn.execute("ALTER TABLE segments DROP COLUMN state")
    conn.execute("CREATE INDEX IF NOT EXISTS segments_ns_level_minseq ON segments (namespace, level, min_seq)")

    if data_dir is None:
        # In-memory store has no parquet files to rename.
        return

    # ---- Filesystem + path rewrite ----------------------------------------
    # Catalog-known files come with an explicit level (from the backfill
    # above). Files on disk that the catalog doesn't yet track (e.g. an
    # older install pre-dating the catalog branch) get their level from
    # the legacy filename prefix: ``tmp_*`` → 0, ``logs_*`` → 1.
    catalog_paths: dict[str, tuple[str, int]] = {}
    rows = conn.execute("SELECT namespace, path, level FROM segments").fetchall()
    for namespace, old_path, level in rows:
        catalog_paths[old_path] = (namespace, level)

    for namespace_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for old in sorted(list(namespace_dir.glob("tmp_*.parquet")) + list(namespace_dir.glob("logs_*.parquet"))):
            old_str = str(old)
            if old_str in catalog_paths:
                namespace, level = catalog_paths[old_str]
            else:
                level = 0 if old.name.startswith("tmp_") else 1
                namespace = None  # not in catalog; rename is purely filesystem
            new = _rewrite_one(old, level)
            if new is None or new == old:
                continue
            if namespace is not None:
                conn.execute(
                    "UPDATE segments SET path = ? WHERE namespace = ? AND path = ?",
                    [str(new), namespace, old_str],
                )


def _rewrite_one(old: Path, level: int) -> Path | None:
    """Rename ``old`` to the leveled scheme. Resumable across crashes.

    Returns the destination path (which may equal ``old`` if the file has
    already been renamed in a previous pass), or ``None`` if the filename
    isn't recognized as either old or new format.
    """
    name = old.name
    match = _OLD_FILENAME_RE.match(name)
    if match is None:
        # Already in seg_L<n>_<seq>.parquet form, or some unrelated file —
        # nothing to do.
        return old if name.startswith("seg_L") else None
    seq = int(match.group("seq"))
    new_name = f"seg_L{level}_{seq:019d}.parquet"
    new = old.with_name(new_name)

    src_exists = old.exists()
    dst_exists = new.exists()

    if dst_exists and src_exists:
        # Prior crashed run moved the file then died before the DB UPDATE
        # committed. Source is the duplicate.
        src_size = old.stat().st_size
        dst_size = new.stat().st_size
        if src_size != dst_size:
            raise RuntimeError(f"migration 0003: size mismatch on resume for {old.name}: src={src_size} dst={dst_size}")
        old.unlink()
        return new
    if src_exists:
        os.rename(old, new)
        return new
    if dst_exists:
        # File already at destination; just rewrite the catalog row.
        return new
    # Neither file present. The boot-time reconciliation will drop the
    # catalog row when the namespace opens and walks the directory.
    logger.warning("migration 0003: parquet missing for both %s and %s", old, new)
    return new
