# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog: namespace + segment metadata in a sidecar DuckDB.

Two tables in ``{data_dir}/_finelog_registry.duckdb``:

* ``namespaces`` — one row per registered namespace holding its schema. Every
  ``RegisterTable`` mutates one row here. On finelog startup the table is
  read out and ``LogNamespace`` instances are rehydrated from the rows.

* ``segments`` — one row per parquet segment in the table, regardless
  of whether its bytes currently live on local disk, in the remote
  bucket, or both (``location`` discriminates). Each
  :class:`DiskLogNamespace` writes here in lockstep with its in-memory
  ``_local_segments`` deque (insert on flush finalize, swap atomically on
  compaction, ``location`` flip on upload / eviction). Compaction drops
  input rows immediately; the remote-sync loop reconciles the bucket
  with the catalog and ``fs.rm``s any remote file that has no row.
  Aggregating this table answers every shape-of-the-data query (row
  counts, seq ranges, byte totals, per-namespace segment counts) without
  touching parquet — finelog's catalog, in the Iceberg/ClickHouse-parts
  sense.

The DB is intentionally separate from the per-namespace Parquet directories:
catalog metadata never lives in two places, so a row count or schema change
is one ``UPDATE``, not a smear across every parquet footer in the namespace.

Concurrency: the :class:`Catalog` is not thread-safe by itself. Every caller
in :class:`finelog.store.duckdb_store.DuckDBLogStore` and
:class:`finelog.store.log_namespace.DiskLogNamespace` already holds the
shared ``_insertion_lock`` around mutating calls, which serializes access.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import duckdb

from finelog.store.migrations import apply_migrations, transactional
from finelog.store.schema import Schema, schema_from_json, schema_to_json

logger = logging.getLogger(__name__)

CATALOG_DB_FILENAME = "_finelog_registry.duckdb"


class SegmentLocation(StrEnum):
    """Where a segment's bytes currently live.

    Every catalog row is part of the table; ``location`` says whether the
    bytes are reachable from local disk, the remote bucket, or both. The
    sync loop reconciles remote with catalog; eviction flips ``BOTH`` →
    ``REMOTE`` rather than dropping the row, so a durable archive cannot
    be confused with an orphan compaction input.
    """

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"
    BOTH = "BOTH"


@dataclass(frozen=True)
class SegmentRow:
    """One persisted row in the segments catalog table.

    ``level`` is the segment's tier in the leveled compaction scheme (0 =
    freshly flushed; promoted to ``level + 1`` when the planner picks it
    up). ``min_key_value`` / ``max_key_value`` carry the parquet footer's
    column statistics for the namespace's declared ``Schema.key_column``.
    They are ``None`` for namespaces whose schema has no ``key_column``,
    or for empty segments where no statistics exist.
    """

    namespace: str
    path: str
    level: int
    min_seq: int
    max_seq: int
    row_count: int
    byte_size: int
    created_at_ms: int
    location: SegmentLocation = SegmentLocation.LOCAL
    min_key_value: str | None = None
    max_key_value: str | None = None


@dataclass(frozen=True)
class NamespaceStats:
    """Aggregate counters for one namespace's persisted segments.

    Live (in-RAM) buffer counts are layered on top of these by the namespace
    in :meth:`finelog.store.log_namespace.DiskLogNamespace.stats`.
    """

    row_count: int
    byte_size: int
    min_seq: int
    max_seq: int
    segment_count: int

    @classmethod
    def empty(cls) -> NamespaceStats:
        return cls(row_count=0, byte_size=0, min_seq=0, max_seq=0, segment_count=0)


class Catalog:
    """Sidecar DuckDB holding namespace schemas + segment metadata.

    Not concurrency-safe by itself — callers hold the shared insertion mutex
    around any mutating call.
    """

    def __init__(self, data_dir: Path | None) -> None:
        # ``data_dir is None`` selects an in-memory registry — paired with
        # ``LogNamespace`` instances that hold their segments as Arrow
        # tables instead of parquet files. Used for tests and for any
        # caller that wants a finelog store with no on-disk footprint.
        if data_dir is None:
            self._path: Path | None = None
            self._conn = duckdb.connect(":memory:")
        else:
            self._path = data_dir / CATALOG_DB_FILENAME
            self._conn = duckdb.connect(str(self._path))
        # Schema is owned by ``finelog.store.migrations``; every additive
        # change (new column, new index, derived backfill) lands as a new
        # numbered file in that package. Migrations that need to rename
        # on-disk parquet files receive ``data_dir`` via the runner.
        apply_migrations(self._conn, data_dir=data_dir)

    def close(self) -> None:
        self._conn.close()

    # ----- namespaces table ---------------------------------------------

    def get(self, namespace: str) -> Schema | None:
        row = self._conn.execute("SELECT schema_json FROM namespaces WHERE namespace = ?", [namespace]).fetchone()
        if row is None:
            return None
        return schema_from_json(row[0])

    def list_all(self) -> dict[str, Schema]:
        rows = self._conn.execute("SELECT namespace, schema_json FROM namespaces").fetchall()
        return {name: schema_from_json(payload) for name, payload in rows}

    def delete(self, namespace: str) -> None:
        """Remove the namespace row and any segment rows. Idempotent."""
        self._conn.execute("DELETE FROM segments WHERE namespace = ?", [namespace])
        self._conn.execute("DELETE FROM namespaces WHERE namespace = ?", [namespace])

    def upsert(self, namespace: str, schema: Schema) -> None:
        """Insert or evolve the row for ``namespace``.

        ``last_modified_ms`` is bumped on every call; ``registered_at_ms``
        is set on first insert and preserved on update.
        """
        now_ms = int(time.time() * 1000)
        existing = self._conn.execute(
            "SELECT registered_at_ms FROM namespaces WHERE namespace = ?", [namespace]
        ).fetchone()
        registered_at = existing[0] if existing is not None else now_ms
        self._conn.execute(
            """
            INSERT INTO namespaces (namespace, schema_json, registered_at_ms, last_modified_ms)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (namespace) DO UPDATE
              SET schema_json = excluded.schema_json,
                  last_modified_ms = excluded.last_modified_ms
            """,
            [namespace, schema_to_json(schema), registered_at, now_ms],
        )

    # ----- segments table -----------------------------------------------

    _SEGMENT_COLUMNS = (
        "namespace, path, level, min_seq, max_seq, row_count, byte_size, "
        "created_at_ms, min_key_value, max_key_value, location"
    )

    def list_segments(self, namespace: str, *, min_level: int = 0) -> list[SegmentRow]:
        """Segment rows for ``namespace`` with ``level >= min_level``, ordered by ``min_seq``."""
        rows = self._conn.execute(
            f"""
            SELECT {self._SEGMENT_COLUMNS}
            FROM segments
            WHERE namespace = ? AND level >= ?
            ORDER BY min_seq
            """,
            [namespace, min_level],
        ).fetchall()
        return [self._row_from_tuple(r) for r in rows]

    @staticmethod
    def _row_from_tuple(r: tuple) -> SegmentRow:
        return SegmentRow(
            namespace=r[0],
            path=r[1],
            level=r[2],
            min_seq=r[3],
            max_seq=r[4],
            row_count=r[5],
            byte_size=r[6],
            created_at_ms=r[7],
            min_key_value=r[8],
            max_key_value=r[9],
            location=SegmentLocation(r[10]),
        )

    def upsert_segment(self, segment: SegmentRow) -> None:
        """Insert or replace one segment row (used by flush + reconciliation)."""
        self._conn.execute(
            """
            INSERT INTO segments
                (namespace, path, level, min_seq, max_seq, row_count, byte_size, created_at_ms,
                 min_key_value, max_key_value, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (namespace, path) DO UPDATE SET
                level = excluded.level,
                min_seq = excluded.min_seq,
                max_seq = excluded.max_seq,
                row_count = excluded.row_count,
                byte_size = excluded.byte_size,
                created_at_ms = excluded.created_at_ms,
                min_key_value = excluded.min_key_value,
                max_key_value = excluded.max_key_value,
                location = excluded.location
            """,
            [
                segment.namespace,
                segment.path,
                segment.level,
                segment.min_seq,
                segment.max_seq,
                segment.row_count,
                segment.byte_size,
                segment.created_at_ms,
                segment.min_key_value,
                segment.max_key_value,
                segment.location.value,
            ],
        )

    def replace_segments(
        self,
        namespace: str,
        removed_paths: Sequence[str],
        added: Sequence[SegmentRow],
    ) -> None:
        """Atomically swap ``removed_paths`` for ``added`` rows in one txn.

        Used by compaction, where N inputs at level n collapse into one
        level-(n+1) output. The whole swap must be visible-or-not visible
        to callers of :meth:`list_segments` — never half.
        """
        with transactional(self._conn):
            for path in removed_paths:
                self._conn.execute(
                    "DELETE FROM segments WHERE namespace = ? AND path = ?",
                    [namespace, path],
                )
            for seg in added:
                self.upsert_segment(seg)

    def remove_segment(self, namespace: str, path: str) -> None:
        """Drop one segment row. Idempotent."""
        self._conn.execute("DELETE FROM segments WHERE namespace = ? AND path = ?", [namespace, path])

    def aggregate_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Single-namespace aggregate over the segments table."""
        row = self._conn.execute(
            """
            SELECT
                COALESCE(SUM(row_count), 0),
                COALESCE(SUM(byte_size), 0),
                COALESCE(MIN(min_seq), 0),
                COALESCE(MAX(max_seq), 0),
                COUNT(*)
            FROM segments
            WHERE namespace = ?
            """,
            [namespace],
        ).fetchone()
        if row is None:
            return NamespaceStats.empty()
        return NamespaceStats(
            row_count=int(row[0]),
            byte_size=int(row[1]),
            min_seq=int(row[2]),
            max_seq=int(row[3]),
            segment_count=int(row[4]),
        )

    def set_location(self, namespace: str, path: str, location: SegmentLocation) -> None:
        """Update one segment's ``location`` (after upload completes / eviction)."""
        self._conn.execute(
            "UPDATE segments SET location = ? WHERE namespace = ? AND path = ?",
            [location.value, namespace, path],
        )

    def select_eviction_candidate(self, namespace: str) -> SegmentRow | None:
        """Pick the oldest evictable segment in ``namespace``.

        Eligibility: ``level >= 1`` (L0 is local-only and transient, so
        never evicted) and ``location = 'BOTH'`` (a compaction output
        becomes evictable only once the remote copy is durable).
        Returns the ``SegmentRow`` with the smallest ``min_seq``, or
        ``None`` when no eligible segment exists.
        """
        row = self._conn.execute(
            f"""
            SELECT {self._SEGMENT_COLUMNS}
            FROM segments
            WHERE namespace = ?
              AND level >= 1
              AND location = ?
            ORDER BY min_seq ASC
            LIMIT 1
            """,
            [namespace, SegmentLocation.BOTH.value],
        ).fetchone()
        if row is None:
            return None
        return self._row_from_tuple(row)
