# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog: namespace + segment metadata in a sidecar DuckDB.

Two tables in ``{data_dir}/_finelog_registry.duckdb``:

* ``namespaces`` — one row per registered namespace holding its schema. Every
  ``RegisterTable`` mutates one row here. On finelog startup the table is
  read out and ``LogNamespace`` instances are rehydrated from the rows.

* ``segments`` — one row per locally-tracked parquet file. Each
  :class:`DiskLogNamespace` writes here in lockstep with its in-memory
  ``_local_segments`` deque (insert on flush finalize, swap atomically on
  compaction, delete on eviction). Aggregating this table answers every
  shape-of-the-data query (row counts, seq ranges, byte totals,
  per-namespace segment counts) without touching parquet — finelog's
  catalog, in the Iceberg/ClickHouse-parts sense.

The DB is intentionally separate from the per-namespace Parquet directories:
catalog metadata never lives in two places, so a row count or schema change
is one ``UPDATE``, not a smear across every parquet footer in the namespace.

Concurrency: the :class:`Catalog` is not thread-safe by itself. Every caller
in :class:`finelog.store.duckdb_store.DuckDBLogStore` and
:class:`finelog.store.log_namespace.DiskLogNamespace` already holds the
shared ``_insertion_lock`` around mutating calls, which serializes access.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from finelog.store.migrations import apply_migrations, transactional
from finelog.store.schema import Schema, schema_from_json, schema_to_json

logger = logging.getLogger(__name__)

CATALOG_DB_FILENAME = "_finelog_registry.duckdb"


class SegmentState(enum.StrEnum):
    """Lifecycle state for a parquet segment in the catalog.

    Persisted as ``state`` (TEXT) on each ``segments`` row.
    """

    TMP = "tmp"  # freshly flushed, awaiting compaction
    FINALIZED = "finalized"  # produced by compaction


@dataclass(frozen=True)
class SegmentRow:
    """One persisted row in the segments catalog table."""

    namespace: str
    path: str
    state: SegmentState
    min_seq: int
    max_seq: int
    row_count: int
    byte_size: int
    created_at_ms: int


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
        # numbered file in that package.
        apply_migrations(self._conn)

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

    def list_segments(self, namespace: str) -> list[SegmentRow]:
        rows = self._conn.execute(
            """
            SELECT namespace, path, state, min_seq, max_seq, row_count, byte_size, created_at_ms
            FROM segments
            WHERE namespace = ?
            ORDER BY min_seq
            """,
            [namespace],
        ).fetchall()
        return [
            SegmentRow(
                namespace=ns,
                path=path,
                state=SegmentState(state),
                min_seq=min_seq,
                max_seq=max_seq,
                row_count=row_count,
                byte_size=byte_size,
                created_at_ms=created_at_ms,
            )
            for ns, path, state, min_seq, max_seq, row_count, byte_size, created_at_ms in rows
        ]

    def upsert_segment(self, segment: SegmentRow) -> None:
        """Insert or replace one segment row (used by flush + reconciliation)."""
        self._conn.execute(
            """
            INSERT INTO segments
                (namespace, path, state, min_seq, max_seq, row_count, byte_size, created_at_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (namespace, path) DO UPDATE SET
                state = excluded.state,
                min_seq = excluded.min_seq,
                max_seq = excluded.max_seq,
                row_count = excluded.row_count,
                byte_size = excluded.byte_size,
                created_at_ms = excluded.created_at_ms
            """,
            [
                segment.namespace,
                segment.path,
                segment.state.value,
                segment.min_seq,
                segment.max_seq,
                segment.row_count,
                segment.byte_size,
                segment.created_at_ms,
            ],
        )

    def replace_segments(
        self,
        namespace: str,
        removed_paths: Sequence[str],
        added: Sequence[SegmentRow],
    ) -> None:
        """Atomically swap ``removed_paths`` for ``added`` rows in one txn.

        Used by compaction, where N tmp segments collapse into one finalized
        segment. The whole swap must be visible-or-not visible to callers
        of :meth:`list_segments` — never half.
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

    def reconcile_segments(self, namespace: str, segments: Sequence[SegmentRow]) -> None:
        """Replace this namespace's segment rows wholesale (startup recovery).

        Parquet files on disk are authoritative across crashes; on every
        ``DiskLogNamespace`` boot we discover the on-disk segment set and
        push it through this method so the catalog matches reality.
        """
        with transactional(self._conn):
            self._conn.execute("DELETE FROM segments WHERE namespace = ?", [namespace])
            for seg in segments:
                self.upsert_segment(seg)

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
