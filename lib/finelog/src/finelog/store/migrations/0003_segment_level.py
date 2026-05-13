# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace the lifecycle ``state`` column with a numeric ``level``, and
rename ``tmp_*`` / ``logs_*`` parquet files to ``seg_L<n>_<seq>``.

L0 = freshly flushed, promoted to ``level + 1`` when the compaction
planner picks it up. ``state='finalized'`` rows backfill to L1, ``tmp``
rows to L0. The filesystem rename is resumable per-file: if the
destination already exists from a previous crashed pass, the source is
treated as a duplicate (size-checked, then unlinked).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import duckdb

from finelog.store.migrations._runner import transactional

logger = logging.getLogger(__name__)

_OLD_FILENAME_RE = re.compile(r"^(?P<prefix>tmp_|logs_)(?P<seq>\d+)\.parquet$")


def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir: Path | None) -> None:
    # The companion (namespace, level, min_seq) index is created in 0005
    # because DuckDB can't combine DROP COLUMN and CREATE INDEX in one
    # transaction the way we'd need.
    conn.execute("ALTER TABLE segments ADD COLUMN IF NOT EXISTS level INTEGER")
    state_present = bool(
        conn.execute(
            "SELECT 1 FROM information_schema.columns " "WHERE table_name = 'segments' AND column_name = 'state' LIMIT 1"
        ).fetchone()
    )
    if state_present:
        with transactional(conn):
            conn.execute("UPDATE segments SET level = CASE WHEN state = 'finalized' THEN 1 ELSE 0 END")
            conn.execute("ALTER TABLE segments DROP COLUMN state")

    if data_dir is None:
        return

    # Catalog-known files take their level from the backfill above; files
    # on disk that the catalog doesn't track yet get it from the legacy
    # filename prefix (``tmp_*`` → 0, ``logs_*`` → 1).
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
                namespace = None
            new = _rewrite_one(old, level)
            if new is None or new == old:
                continue
            if namespace is not None:
                conn.execute(
                    "UPDATE segments SET path = ? WHERE namespace = ? AND path = ?",
                    [str(new), namespace, old_str],
                )


def _rewrite_one(old: Path, level: int) -> Path | None:
    """Rename ``old`` to the leveled scheme. Returns the destination path
    (or ``old`` if already renamed); ``None`` for unrecognized filenames."""
    name = old.name
    match = _OLD_FILENAME_RE.match(name)
    if match is None:
        return old if name.startswith("seg_L") else None
    seq = int(match.group("seq"))
    new = old.with_name(f"seg_L{level}_{seq:019d}.parquet")

    src_exists = old.exists()
    dst_exists = new.exists()

    if dst_exists and src_exists:
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
        return new
    logger.warning("migration 0003: parquet missing for both %s and %s", old, new)
    return new
