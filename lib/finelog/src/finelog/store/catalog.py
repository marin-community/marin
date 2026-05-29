# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog: single source of truth for namespace state, persistent and live.

Three coupled pieces of state, all guarded by one mutex:

* The live ``LogNamespaceProtocol`` registry (``_namespaces``,
  ``_registered_at``) — every name the store routes RPCs to is here.
* The ``_dropping`` reservation set — names whose ``drop_table`` is in
  flight, used to fence off concurrent ``register_table`` calls.
* A sidecar DuckDB at ``{data_dir}/_finelog_registry.duckdb`` with two
  tables — ``namespaces`` (one row per registered namespace + its schema)
  and ``segments`` (one row per parquet segment, with ``location``
  discriminating LOCAL / REMOTE / BOTH). Aggregating the segments table
  answers every shape-of-the-data query (row counts, seq ranges, byte
  totals, per-namespace segment counts) without touching parquet.

Segment rows live in lockstep with each :class:`DiskLogNamespace`'s
in-memory ``_local_segments`` deque (insert on flush finalize, swap
atomically on compaction, ``location`` flip on upload / eviction).
Compaction drops input rows immediately; the remote-sync loop reconciles
the bucket against the catalog and ``fs.rm``s any remote file with no
row.

The sidecar DB is intentionally separate from the per-namespace Parquet
directories: catalog metadata never lives in two places, so a row count
or schema change is one ``UPDATE``, not a smear across every parquet
footer.

Concurrency: ``self._lock`` is an ``RLock``. ``register_or_evolve``
builds and publishes a new namespace under that lock; the namespace's
constructor reads back its boot state via ``Catalog.list_segments``,
which re-acquires the same lock from the same thread. The bg thread on
the new namespace starts inside the constructor but only calls back
into the catalog after ``register_or_evolve`` has released the lock, so
the per-thread reentry is contained.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock

import duckdb

from finelog.store.migrations import apply_migrations, transactional
from finelog.store.policy import StoragePolicy
from finelog.store.schema import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    Schema,
    schema_from_json,
    schema_to_json,
)
from finelog.store.types import LogNamespaceProtocol, NamespaceStats, SegmentLocation, SegmentRow

logger = logging.getLogger(__name__)

CATALOG_DB_FILENAME = "_finelog_registry.duckdb"


class Catalog:
    """Single source of truth for namespace state, persistent and live.

    See module docstring for the full picture. The class owns the
    sidecar DuckDB connection, the live namespace registry, and the
    in-flight drop reservations — all behind a single ``RLock`` so a
    register-or-evolve can build a namespace (whose constructor calls
    back into the catalog) inside one critical section.
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
        # RLock because ``register_or_evolve`` builds a namespace under
        # the lock and the namespace constructor calls back into the
        # catalog (e.g. ``list_segments``) to read its boot state. The
        # callbacks come from the same thread; ``RLock`` lets them
        # nest, foreign threads still serialize.
        self._lock = RLock()

        self._namespaces: dict[str, LogNamespaceProtocol] = {}
        # Monotonic insertion ordinal, used as the eviction tiebreak and
        # to render the ``list_namespaces`` output in registration order.
        self._registered_at: dict[str, int] = {}
        # Names whose drop is mid-flight. ``register_or_evolve`` checks
        # this so a concurrent register can't re-create a namespace that's
        # still having its catalog rows / on-disk directory torn down.
        self._dropping: set[str] = set()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- live namespace registry --------------------------------------

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._namespaces

    def __getitem__(self, name: str) -> LogNamespaceProtocol:
        with self._lock:
            return self._namespaces[name]

    def get_live(self, name: str) -> LogNamespaceProtocol | None:
        with self._lock:
            return self._namespaces.get(name)

    def require_live(self, name: str) -> LogNamespaceProtocol:
        with self._lock:
            ns = self._namespaces.get(name)
        if ns is None:
            raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
        return ns

    def snapshot_live(self) -> list[tuple[str, LogNamespaceProtocol]]:
        """Live namespaces in registration order."""
        with self._lock:
            return sorted(self._namespaces.items(), key=lambda kv: self._registered_at.get(kv[0], 0))

    def live_values(self) -> list[LogNamespaceProtocol]:
        with self._lock:
            return list(self._namespaces.values())

    def is_dropping(self, name: str) -> bool:
        with self._lock:
            return name in self._dropping

    def insert_live(self, name: str, ns: LogNamespaceProtocol) -> None:
        """Publish a freshly-built namespace; used by the rehydrate path."""
        with self._lock:
            self._namespaces[name] = ns
            self._registered_at[name] = len(self._registered_at)

    def register_or_evolve(
        self,
        name: str,
        stored_schema: Schema,
        factory: Callable[[Schema, StoragePolicy], LogNamespaceProtocol],
        *,
        on_existing: Callable[[LogNamespaceProtocol], Schema],
        policy: StoragePolicy = StoragePolicy(),
    ) -> Schema:
        """Atomically register ``name`` or evolve the existing namespace.

        Runs the whole decision-and-publish as one critical section so
        two concurrent calls on the same name cannot both build a
        candidate namespace (each of which would start a bg thread and
        touch on-disk / remote state). The factory and the
        ``_namespaces`` publish are inside one ``self._lock`` acquire;
        ``RLock`` lets the factory's namespace constructor call back
        into other catalog methods on the same thread.

        ``policy`` is the per-namespace retention override. On fresh
        registration it is persisted in ``storage_policies`` and passed
        to the factory. Existing-namespace evolution is delegated to
        ``on_existing``; policy updates on re-register are handled by
        the caller via ``upsert_policy``.

        Returns the effective schema in both branches: ``stored_schema``
        for a fresh registration, or whatever ``on_existing`` resolves
        to for an evolution. Raises ``InvalidNamespaceError`` if a drop
        is in flight.
        """
        with self._lock:
            if name in self._dropping:
                raise InvalidNamespaceError(
                    f"namespace {name!r} is currently being dropped; retry once drop_table completes"
                )
            existing = self._namespaces.get(name)
            if existing is not None:
                return on_existing(existing)

            ns = factory(stored_schema, policy)
            self.upsert(name, stored_schema)
            self.upsert_policy(name, policy)
            self._namespaces[name] = ns
            self._registered_at[name] = len(self._registered_at)
            return stored_schema

    def begin_drop(self, name: str) -> LogNamespaceProtocol:
        """Pop ``name`` from the registry and reserve it in ``_dropping``.

        Atomic so a concurrent ``register_table`` can't slip in between
        the pop and the catalog-row delete that follows. Raises
        ``NamespaceNotFoundError`` if the name isn't live.
        """
        with self._lock:
            ns = self._namespaces.get(name)
            if ns is None:
                raise NamespaceNotFoundError(f"namespace {name!r} is not registered")
            del self._namespaces[name]
            self._registered_at.pop(name, None)
            self._dropping.add(name)
            return ns

    def finish_drop(self, name: str) -> None:
        with self._lock:
            self._dropping.discard(name)

    # ----- namespaces table ---------------------------------------------

    def get(self, namespace: str) -> Schema | None:
        with self._lock:
            row = self._conn.execute("SELECT schema_json FROM namespaces WHERE namespace = ?", [namespace]).fetchone()
        if row is None:
            return None
        return schema_from_json(row[0])

    def list_all(self) -> dict[str, Schema]:
        with self._lock:
            rows = self._conn.execute("SELECT namespace, schema_json FROM namespaces").fetchall()
        return {name: schema_from_json(payload) for name, payload in rows}

    def delete(self, namespace: str) -> None:
        """Remove the namespace row, segment rows, and policy row. Idempotent."""
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE namespace = ?", [namespace])
            self._conn.execute("DELETE FROM storage_policies WHERE namespace = ?", [namespace])
            self._conn.execute("DELETE FROM namespaces WHERE namespace = ?", [namespace])

    # ----- storage_policies table ---------------------------------------

    def get_policy(self, namespace: str) -> StoragePolicy:
        """Return the persisted policy or an empty (inherit-all) policy."""
        with self._lock:
            row = self._conn.execute(
                "SELECT max_segments, max_bytes, max_age_seconds FROM storage_policies WHERE namespace = ?",
                [namespace],
            ).fetchone()
        if row is None:
            return StoragePolicy()
        return StoragePolicy(max_segments=row[0], max_bytes=row[1], max_age_seconds=row[2])

    def upsert_policy(self, namespace: str, policy: StoragePolicy) -> None:
        """Persist ``policy``, or delete the row if every field is ``None``."""
        with self._lock:
            if policy.is_empty():
                self._conn.execute("DELETE FROM storage_policies WHERE namespace = ?", [namespace])
                return
            self._conn.execute(
                """
                INSERT INTO storage_policies (namespace, max_segments, max_bytes, max_age_seconds)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (namespace) DO UPDATE
                  SET max_segments    = excluded.max_segments,
                      max_bytes       = excluded.max_bytes,
                      max_age_seconds = excluded.max_age_seconds
                """,
                [namespace, policy.max_segments, policy.max_bytes, policy.max_age_seconds],
            )

    def upsert(self, namespace: str, schema: Schema) -> None:
        """Insert or evolve the row for ``namespace``.

        ``last_modified_ms`` is bumped on every call; ``registered_at_ms``
        is set on first insert and preserved on update.
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._upsert_segment_locked(segment)

    def _upsert_segment_locked(self, segment: SegmentRow) -> None:
        """Body of ``upsert_segment`` without the lock + transaction context.

        Used by ``replace_segments`` so the swap of removed + added rows
        runs inside one ``transactional(self._conn)`` block under the
        already-held lock. This is *not* a lock-avoidance helper — the
        caller is responsible for holding ``self._lock``.
        """
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
        with self._lock, transactional(self._conn):
            for path in removed_paths:
                self._conn.execute(
                    "DELETE FROM segments WHERE namespace = ? AND path = ?",
                    [namespace, path],
                )
            for seg in added:
                self._upsert_segment_locked(seg)

    def remove_segment(self, namespace: str, path: str) -> None:
        """Drop one segment row. Idempotent."""
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE namespace = ? AND path = ?", [namespace, path])

    def aggregate_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Single-namespace aggregate over the segments table."""
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

    def select_aged_eviction_candidate(self, namespace: str, cutoff_ms: int) -> SegmentRow | None:
        """Pick the oldest-by-``created_at_ms`` evictable segment past ``cutoff_ms``.

        Same eligibility as :meth:`select_eviction_candidate` (``level >= 1``
        and ``location = 'BOTH'``). Ordering by ``created_at_ms`` (not
        ``min_seq``) matters because compaction outputs inherit their
        inputs' ``min_seq`` but get a fresh ``created_at_ms`` — so a
        low-``min_seq`` segment can be the *youngest* one in the
        namespace, and a ``min_seq``-ordered scan would short-circuit
        on it and miss strictly-older siblings at higher ``min_seq``.
        Returns ``None`` if no eligible segment is older than the cutoff.
        """
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT {self._SEGMENT_COLUMNS}
                FROM segments
                WHERE namespace = ?
                  AND level >= 1
                  AND location = ?
                  AND created_at_ms < ?
                ORDER BY created_at_ms ASC
                LIMIT 1
                """,
                [namespace, SegmentLocation.BOTH.value, cutoff_ms],
            ).fetchone()
        if row is None:
            return None
        return self._row_from_tuple(row)
