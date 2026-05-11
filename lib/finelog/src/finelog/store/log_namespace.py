# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace log storage state.

:class:`DiskLogNamespace` is the production path; :class:`MemoryLogNamespace`
backs the in-memory store mode. Both satisfy :class:`LogNamespaceProtocol`.

The two global locks (insertion mutex + query-visibility rwlock) live on the
registry and are passed in at construction. The disk variant additionally
owns a per-namespace flush mutex preventing the test ``_force_flush`` hook
from racing the bg thread on the same segment filename.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock
from typing import NamedTuple, Protocol

import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.timing import RateLimiter

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.store.catalog import (
    Catalog,
    NamespaceStats,
    SegmentLocation,
    SegmentRow,
)
from finelog.store.compactor import (
    CompactionConfig,
    CompactionJob,
    Compactor,
    aggregate_key_bounds,
    compaction_sort_keys,
    parse_seg_filename,
    seg_filename,
)
from finelog.store.rwlock import RWLock
from finelog.store.schema import (
    IMPLICIT_SEQ_COLUMN,
    Column,
    Schema,
    schema_to_arrow,
)
from finelog.types import LogReadResult, parse_attempt_id, str_to_log_level

logger = logging.getLogger(__name__)

# The user-declared schema for the "log" namespace. The registry stamps
# the implicit ``seq`` column on top.
LOG_REGISTERED_SCHEMA = Schema(
    columns=(
        Column(name="key", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        Column(name="source", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        Column(name="data", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        Column(name="epoch_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        Column(name="level", type=stats_pb2.COLUMN_TYPE_INT32, nullable=False),
    ),
    # Per-source tail reads (``WHERE key = $key ORDER BY seq DESC``) dominate;
    # sorting by ``key`` first colocates same-source rows for row-group pruning.
    key_column="key",
)

_ROW_GROUP_SIZE = 16_384

# Append calls slower than this emit a warning so lock contention vs prepare
# work vs critical-section work is visible in production logs.
_SLOW_APPEND_THRESHOLD_MS = 200

# Background loop heartbeat cadence — emits a one-line snapshot of buffer
# and segment state regardless of whether a flush or compaction fires.
_BG_HEARTBEAT_INTERVAL_SEC = 10.0

# Hard ceiling on the per-read parquet working set; safety net for body-LIKE
# queries that cannot be pruned by row-group statistics.
_MAX_PARQUET_BYTES_PER_READ = 2_500 * 1024 * 1024

# 4 MiB amortizes GCS per-write overhead without growing with segment size.
_REMOTE_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024


class SegmentMetadata(NamedTuple):
    """Per-segment metadata derived from filename + parquet footer.

    ``level`` is read out of the filename (``seg_L<n>_*``); the catalog
    holds the runtime source of truth, but a fresh boot reconciles from
    disk so this is what populates the catalog rows on startup.

    ``min_key_value`` / ``max_key_value`` are the parquet column statistics
    for ``key_column``, when one is requested and the column carries
    statistics. ``None`` when the namespace has no ``key_column``, the
    column statistics weren't written, or the segment is empty.
    """

    level: int
    min_seq: int
    max_seq: int
    row_count: int
    min_key_value: object | None
    max_key_value: object | None


_EMPTY_SEGMENT_METADATA = SegmentMetadata(
    level=0, min_seq=0, max_seq=0, row_count=0, min_key_value=None, max_key_value=None
)


def _level_histogram(segments: Iterable[LocalSegment]) -> dict[int, int]:
    """Count segments per level for the bg-loop heartbeat log line."""
    counts: dict[int, int] = {}
    for s in segments:
        counts[s.level] = counts.get(s.level, 0) + 1
    return dict(sorted(counts.items()))


def _level_bytes_summary(segments: Iterable[LocalSegment], level_targets: tuple[int, ...]) -> str:
    """Render ``L<n>=<bytes>/<target>`` per occupied level for the heartbeat.

    ``level_targets[n]`` is the byte threshold for promoting L_n → L_{n+1};
    the terminal level (``len(level_targets)``) has no target and prints
    just the raw byte count. Empty levels are omitted.
    """
    bytes_per_level: dict[int, int] = {}
    for s in segments:
        bytes_per_level[s.level] = bytes_per_level.get(s.level, 0) + s.size_bytes
    parts: list[str] = []
    for level in sorted(bytes_per_level):
        size = bytes_per_level[level]
        if level < len(level_targets):
            parts.append(f"L{level}={size}/{level_targets[level]}")
        else:
            parts.append(f"L{level}={size}")
    return "{" + ", ".join(parts) + "}"


def _key_bounds_from_parquet(metadata: pq.FileMetaData, key_column: str | None) -> tuple[object | None, object | None]:
    """Extract ``(min, max)`` for ``key_column`` across all row groups.

    Returns ``(None, None)`` if no ``key_column`` was requested, the column
    isn't present, or no row group carries statistics. The returned values
    are the parquet-native Python types (int / str / float / bytes /
    datetime). Stringification happens at the catalog boundary so numeric
    keys preserve their ordering through ``aggregate_key_bounds``.
    """
    if not key_column:
        return None, None
    try:
        col_idx = metadata.schema.names.index(key_column)
    except ValueError:
        return None, None
    overall_min: object | None = None
    overall_max: object | None = None
    for rg_idx in range(metadata.num_row_groups):
        stats = metadata.row_group(rg_idx).column(col_idx).statistics
        if stats is None or not stats.has_min_max:
            continue
        rg_min, rg_max = stats.min, stats.max
        if overall_min is None or rg_min < overall_min:
            overall_min = rg_min
        if overall_max is None or rg_max > overall_max:
            overall_max = rg_max
    if overall_min is None:
        return None, None
    return overall_min, overall_max


def _read_segment_metadata(path: Path, key_column: str | None = None) -> SegmentMetadata:
    """Recover (level, seq, row count, key_column bounds) from a segment file.

    Returns ``_EMPTY_SEGMENT_METADATA`` for unparseable filenames or footer-read
    failures — the caller treats that as an empty/discardable segment.
    """
    parsed = parse_seg_filename(path.name)
    if parsed is None:
        return _EMPTY_SEGMENT_METADATA
    level, min_seq = parsed
    try:
        metadata = pq.read_metadata(path)
    except Exception:
        logger.warning("Failed to read parquet metadata for %s; treating as empty", path, exc_info=True)
        return _EMPTY_SEGMENT_METADATA
    num_rows = metadata.num_rows
    if num_rows <= 0:
        return SegmentMetadata(
            level=level, min_seq=min_seq, max_seq=min_seq, row_count=0, min_key_value=None, max_key_value=None
        )
    min_key, max_key = _key_bounds_from_parquet(metadata, key_column)
    return SegmentMetadata(
        level=level,
        min_seq=min_seq,
        max_seq=min_seq + num_rows - 1,
        row_count=num_rows,
        min_key_value=min_key,
        max_key_value=max_key,
    )


def _discover_segments(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("seg_L*_*.parquet"))


def _recover_next_seq(log_dir: Path) -> int:
    next_seq = 1
    for p in _discover_segments(log_dir):
        meta = _read_segment_metadata(p)
        if meta.max_seq + 1 > next_seq:
            next_seq = meta.max_seq + 1
    return next_seq


def _merge_chunks(chunks: list[pa.Table]) -> list[pa.Table]:
    """Log-structured merge: keep each chunk at least 2x the previous one.

    Bounds ``len(chunks)`` logarithmically in total row count.
    """
    if len(chunks) < 2:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if merged[-1].num_rows <= chunk.num_rows:
            merged[-1] = pa.concat_tables([merged[-1], chunk])
        else:
            merged.append(chunk)
    return merged


@dataclass
class _SealedBuffer:
    table: pa.Table
    min_seq: int
    max_seq: int


@dataclass
class LocalSegment:
    path: str
    size_bytes: int
    level: int
    min_seq: int
    max_seq: int
    row_count: int
    # ``created_at_ms`` is stamped once at flush/merge time and preserved
    # across level bumps and catalog reconcile so the planner can age out
    # quiet L0 segments via ``CompactionConfig.max_l0_age_sec``.
    created_at_ms: int = 0
    # Typed key-column bounds (Python int / str / float / bool / bytes
    # depending on schema). Stringified only at the catalog boundary in
    # ``_segment_to_row``; held typed in memory so ``aggregate_key_bounds``
    # can compare numeric keys with native ordering.
    min_key_value: object | None = None
    max_key_value: object | None = None
    # Where the bytes live. The deque only ever holds ``LOCAL`` or ``BOTH``
    # entries; eviction flips to ``REMOTE`` and removes the entry from the
    # deque (the row stays in the catalog as a durable-archive pointer).
    location: SegmentLocation = SegmentLocation.LOCAL


def _stamp_seq_column(batch: pa.RecordBatch, first_seq: int, arrow_schema: pa.Schema) -> pa.Table:
    """Project ``batch`` to ``arrow_schema`` with the implicit seq column filled."""
    seq_array = pa.array(range(first_seq, first_seq + batch.num_rows), type=pa.int64())
    arrays: list[pa.Array] = []
    for field in arrow_schema:
        if field.name == IMPLICIT_SEQ_COLUMN:
            arrays.append(seq_array)
        else:
            arrays.append(batch.column(field.name))
    return pa.Table.from_arrays(arrays, schema=arrow_schema)


def _build_log_table(buffer: list[tuple], arrow_schema: pa.Schema) -> pa.Table:
    """Build an Arrow table from log-namespace ``(seq, key, source, data, epoch_ms, level)`` tuples.

    Used by :class:`MemoryLogNamespace`. The disk path goes through
    :meth:`RamBuffers.append_table` with pre-built columnar arrays.
    """
    if not buffer:
        return arrow_schema.empty_table()
    n = 6
    cols: list[list] = [[] for _ in range(n)]
    for row in buffer:
        for i, val in enumerate(row):
            cols[i].append(val)
    arrays = [
        pa.array(cols[0], type=pa.int64()),
        pa.array(cols[1], type=pa.string()),
        pa.array(cols[2], type=pa.string()),
        pa.array(cols[3], type=pa.string()),
        pa.array(cols[4], type=pa.int64()),
        pa.array(cols[5], type=pa.int32()),
    ]
    return pa.table(arrays, schema=arrow_schema)


class RamBuffers:
    """Owns the in-RAM write state for a single namespace.

    Holds the merged log chunks plus the in-flight ``flushing`` table, the
    seq counter, and a maintained ``_ram_bytes`` tally so callers don't
    rescan ``self._chunks`` on every append. Not thread-safe — the
    enclosing namespace serializes calls under ``_insertion_lock``.

    ``pa.concat_tables`` (used by ``_merge_chunks``) is zero-copy for
    primitive columns and shares string buffers via Arrow's reference
    counting, so total ``nbytes`` is conserved across merges. We can
    therefore maintain ``_ram_bytes`` incrementally rather than scanning.
    """

    def __init__(self, *, arrow_schema: pa.Schema, next_seq: int) -> None:
        self._arrow_schema = arrow_schema
        self._chunks: list[pa.Table] = []
        self._flushing: _SealedBuffer | None = None
        self._next_seq = next_seq
        self._ram_bytes = 0
        self._ram_rows = 0

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def allocate_seq(self, count: int) -> int:
        first = self._next_seq
        self._next_seq += count
        return first

    def append_table(self, table: pa.Table) -> None:
        added = table.nbytes
        self._chunks.append(table)
        self._chunks = _merge_chunks(self._chunks)
        self._ram_bytes += added
        self._ram_rows += table.num_rows

    def ram_bytes(self) -> int:
        flushing_b = self._flushing.table.nbytes if self._flushing is not None else 0
        return self._ram_bytes + flushing_b

    def ram_rows(self) -> int:
        flushing_n = self._flushing.table.num_rows if self._flushing is not None else 0
        return self._ram_rows + flushing_n

    def chunk_count(self) -> int:
        return len(self._chunks)

    def has_chunks(self) -> bool:
        return bool(self._chunks)

    def seal(self) -> _SealedBuffer | None:
        """Move accumulated chunks into a sealed flushing buffer.

        Returns ``None`` if there is nothing to flush. The returned buffer
        is also stored on ``self._flushing`` so queries see in-flight rows.
        """
        if not self._chunks:
            return None
        tables = self._chunks
        self._chunks = []
        self._ram_bytes = 0
        self._ram_rows = 0
        visible_table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        seq_col = visible_table.column(IMPLICIT_SEQ_COLUMN)
        sealed = _SealedBuffer(
            table=visible_table,
            min_seq=pc.min(seq_col).as_py(),
            max_seq=pc.max(seq_col).as_py(),
        )
        self._flushing = sealed
        return sealed

    def commit_flush(self) -> None:
        """Drop the in-flight flushing buffer (parquet write succeeded)."""
        self._flushing = None

    def restore_flush(self) -> None:
        """Push the in-flight buffer back to the head of chunks (write failed)."""
        if self._flushing is None:
            return
        table = self._flushing.table
        self._chunks.insert(0, table)
        self._ram_bytes += table.nbytes
        self._ram_rows += table.num_rows
        self._flushing = None

    def query_snapshot(self) -> list[pa.Table]:
        """Return chunks plus any in-flight flushing table (for read paths)."""
        snap = list(self._chunks)
        if self._flushing is not None:
            snap.append(self._flushing.table)
        return snap


class LogNamespaceProtocol(Protocol):
    name: str
    schema: Schema

    def append_log_batch(self, items: list[tuple[str, list]]) -> None: ...

    def append_record_batch(self, batch: pa.RecordBatch) -> None: ...

    def get_logs(
        self,
        key: str,
        *,
        match_scope: int = logging_pb2.MATCH_SCOPE_EXACT,
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult: ...

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]: ...

    def all_segments_unlocked(self) -> list[LocalSegment]: ...

    def update_schema(self, new_schema: Schema) -> None: ...

    def evict_segment(self, path: str) -> int: ...

    def remove_local_storage(self) -> None: ...

    def close(self) -> None: ...

    def stop_and_join(self) -> None: ...

    def stats(self) -> NamespaceStats: ...

    def ram_bytes(self) -> int: ...

    def chunk_count(self) -> int: ...


class DiskLogNamespace:
    """Disk-backed per-namespace storage.

    Owns the in-memory write buffer, the on-disk Parquet segment registry,
    the flush thread, and the compaction state for a single namespace.
    The ``log`` namespace exposes a key/source/data read API on top of the
    same storage; that path is hardcoded for log columns.
    """

    def __init__(
        self,
        *,
        name: str,
        schema: Schema,
        data_dir: Path,
        remote_log_dir: str,
        segment_target_bytes: int,
        flush_interval_sec: float,
        compaction_config: CompactionConfig,
        insertion_lock: Lock,
        query_visibility_lock: RWLock,
        duckdb_memory_limit: str,
        read_pool: _ReadPoolProtocol,
        catalog: Catalog,
    ) -> None:
        self.name = name
        self.schema = schema
        self._arrow_schema = schema_to_arrow(schema)
        self._data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Empty string disables remote sync.
        self._remote_namespace_dir = f"{remote_log_dir.rstrip('/')}/{name}" if remote_log_dir else ""

        self._segment_target_bytes = segment_target_bytes
        self._compactor = Compactor(compaction_config)

        self._insertion_lock = insertion_lock
        self._query_visibility_lock = query_visibility_lock
        # Prevents the test ``_force_flush`` hook racing the bg thread on
        # the same segment filename (both derive from this namespace's
        # _next_seq).
        self._flush_lock = Lock()

        # Compaction uses a fresh DuckDB connection per merge (see
        # ``_open_compaction_conn``). DuckDB's per-connection spill
        # accountant leaks bytes when a COPY fails; reusing one long-lived
        # connection eventually wedges with a bogus "165 GiB / 16383 PiB
        # used" OOM. A short-lived per-merge conn sidesteps the leak,
        # and the connect cost (~ms) is negligible vs. a multi-second COPY.
        self._compaction_tmp = data_dir / ".duckdb_tmp"
        self._compaction_conn_memory_limit = duckdb_memory_limit
        self._read_pool = read_pool
        self._catalog = catalog

        self._buffers = RamBuffers(
            arrow_schema=self._arrow_schema,
            next_seq=_recover_next_seq(data_dir),
        )
        self._local_segments: deque[LocalSegment] = deque()

        # Reconcile boot state across three sources of truth (in order):
        #   1. Catalog rows — authoritative for ``REMOTE`` segments whose
        #      local file was evicted before the prior shutdown.
        #   2. Local parquet files — authoritative for unflushed catalog
        #      state (genuine boot from disk, no prior catalog).
        #   3. Remote bucket — authoritative for wiped-catalog recovery,
        #      handled below by ``_adopt_remote_segments`` *before* the
        #      bg loop starts. We can't defer this to the first sync tick:
        #      that tick's phase-2 orphan-delete would see an empty
        #      catalog with a populated bucket and ``fs.rm`` everything.
        #      The cost is a network round-trip per parquet file at boot
        #      when the bucket is non-empty.
        #
        # Each catalog row is reattached to its on-disk file when present;
        # ``REMOTE``-only rows stay in the catalog but never enter the
        # deque (queries don't see archived data). Local files without a
        # catalog row are inserted as ``LOCAL`` (sync uploads them next
        # tick).
        key_column = self.schema.key_column or None
        catalog_rows = {row.path: row for row in self._catalog.list_segments(self.name)}
        local_files = {str(p): p for p in _discover_segments(data_dir)}
        seen_paths: set[str] = set()

        # Pass 1: walk catalog rows. Local-present → deque entry; otherwise
        # leave the row in place as the durable-archive pointer.
        for path_str, row in catalog_rows.items():
            seen_paths.add(path_str)
            local_path = local_files.get(path_str)
            if local_path is None:
                if row.location == SegmentLocation.LOCAL:
                    # Local file went missing while the catalog still
                    # claims LOCAL — data is lost. Drop the row so sync
                    # doesn't keep trying to upload a non-existent file.
                    logger.warning(
                        "Catalog row %s missing local file and has no remote copy; dropping",
                        path_str,
                    )
                    self._catalog.remove_segment(self.name, path_str)
                elif row.location == SegmentLocation.BOTH:
                    # Local file lost but bucket copy is durable; collapse
                    # to ``REMOTE`` so queries don't try to read the
                    # missing file.
                    self._catalog.set_location(self.name, path_str, SegmentLocation.REMOTE)
                continue
            meta = _read_segment_metadata(local_path, key_column=key_column)
            location = SegmentLocation.BOTH if row.location == SegmentLocation.REMOTE else row.location
            self._local_segments.append(
                LocalSegment(
                    path=path_str,
                    size_bytes=local_path.stat().st_size,
                    level=meta.level,
                    min_seq=meta.min_seq,
                    max_seq=meta.max_seq,
                    row_count=meta.row_count,
                    created_at_ms=row.created_at_ms,
                    min_key_value=meta.min_key_value,
                    max_key_value=meta.max_key_value,
                    location=location,
                )
            )

        # Pass 2: walk local files. Anything not already accounted for is
        # a genuine fresh-from-disk segment (or a compaction-mid-crash
        # leftover; sync will upload it then orphan-delete in the next
        # round if it's truly stale).
        for path_str, p in local_files.items():
            if path_str in seen_paths:
                continue
            meta = _read_segment_metadata(p, key_column=key_column)
            self._local_segments.append(
                LocalSegment(
                    path=path_str,
                    size_bytes=p.stat().st_size,
                    level=meta.level,
                    min_seq=meta.min_seq,
                    max_seq=meta.max_seq,
                    row_count=meta.row_count,
                    created_at_ms=int(p.stat().st_mtime * 1000),
                    min_key_value=meta.min_key_value,
                    max_key_value=meta.max_key_value,
                    location=SegmentLocation.LOCAL,
                )
            )

        # Order the deque by min_seq so query iteration matches the
        # implicit "oldest first" expected by callers and the planner.
        self._local_segments = deque(sorted(self._local_segments, key=lambda s: s.min_seq))

        # Refresh the catalog from the deque so any drift in level / key
        # bounds (parquet footer is authoritative) is corrected. REMOTE
        # rows are left untouched. Each upsert is independent so no
        # transaction is required.
        for seg in self._local_segments:
            self._catalog.upsert_segment(self._segment_to_row(seg))

        # Wiped-catalog recovery: if the remote bucket has parquet files
        # that no row references, adopt them as ``REMOTE``. Without this,
        # the first ``_sync_step`` after a catalog wipe would treat the
        # whole bucket as orphan and ``fs.rm`` everything.
        if self._remote_namespace_dir:
            self._adopt_remote_segments(key_column=key_column)

        self._flush_rl = RateLimiter(flush_interval_sec)
        self._compaction_rl = RateLimiter(compaction_config.check_interval_sec)
        # Mark just-run so the bg loop doesn't fire a spurious tick at startup
        # and compact a partially-written set of segments.
        self._flush_rl.mark_run()
        self._compaction_rl.mark_run()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._flush_generation = 0
        self._flush_generation_cond = Condition(Lock())
        self._compaction_generation = 0
        self._compaction_generation_cond = Condition(Lock())
        self._bg_thread = threading.Thread(
            target=self._bg_loop,
            name=f"finelog_flush_{self.name}",
            daemon=True,
        )
        self._bg_thread.start()

    def _open_compaction_conn(self) -> duckdb.DuckDBPyConnection:
        """Fresh DuckDB connection for one compaction COPY.

        Opened per-merge so DuckDB's per-conn spill accountant resets
        (it leaks bytes after failed COPYs). ``preserve_insertion_order``
        is off because the COPY ``ORDER BY``s explicitly, and
        ``threads=2`` caps the per-thread sort buffer footprint.
        """
        self._compaction_tmp.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(
            config={
                "memory_limit": self._compaction_conn_memory_limit,
                "threads": "2",
                "temp_directory": str(self._compaction_tmp),
                "preserve_insertion_order": "false",
            }
        )

    def append_log_batch(self, items: list[tuple[str, list]]) -> None:
        """Log-namespace-only append for ``PushLogs`` RPCs.

        Pre-builds all five non-seq columns from the protobuf entries
        outside the lock — that's the bulk of the per-row Python work.
        Inside the lock we only allocate the seq range, materialize the
        seq array, assemble the Arrow table, and hand it to the buffer.
        """
        t_enter = time.monotonic()
        # Outside the lock: flatten items into one combined columnar batch.
        keys: list[str] = []
        sources: list[str] = []
        datas: list[str] = []
        epoch_ms: list[int] = []
        levels: list[int] = []
        for key, entries in items:
            if not entries:
                continue
            n = len(entries)
            keys.extend([key] * n)
            sources.extend(e.source for e in entries)
            datas.extend(e.data for e in entries)
            epoch_ms.extend(e.timestamp.epoch_ms for e in entries)
            levels.extend(int(e.level) for e in entries)
        total = len(keys)
        if total == 0:
            return
        keys_arr = pa.array(keys, type=pa.string())
        sources_arr = pa.array(sources, type=pa.string())
        datas_arr = pa.array(datas, type=pa.string())
        ts_arr = pa.array(epoch_ms, type=pa.int64())
        levels_arr = pa.array(levels, type=pa.int32())
        t_prepared = time.monotonic()

        wait_start = time.monotonic()
        with self._insertion_lock:
            critical_start = time.monotonic()
            first_seq = self._buffers.allocate_seq(total)
            seqs_arr = pa.array(range(first_seq, first_seq + total), type=pa.int64())
            self._buffers.append_table(
                pa.table(
                    [seqs_arr, keys_arr, sources_arr, datas_arr, ts_arr, levels_arr],
                    schema=self._arrow_schema,
                )
            )
            needs_drain = self._buffers.ram_bytes() >= self._segment_target_bytes
        critical_end = time.monotonic()
        if needs_drain:
            self._wake.set()

        total_ms = int((critical_end - t_enter) * 1000)
        if total_ms >= _SLOW_APPEND_THRESHOLD_MS:
            logger.warning(
                "slow append: items=%d rows=%d prepare_ms=%d lock_wait_ms=%d critical_ms=%d total_ms=%d",
                len(items),
                total,
                int((t_prepared - t_enter) * 1000),
                int((critical_start - wait_start) * 1000),
                int((critical_end - critical_start) * 1000),
                total_ms,
            )

    def append_record_batch(self, batch: pa.RecordBatch) -> None:
        """Stamp ``seq`` values onto ``batch`` and append it to the in-RAM chunks."""
        if batch.num_rows == 0:
            return
        with self._insertion_lock:
            first_seq = self._buffers.allocate_seq(batch.num_rows)
            self._buffers.append_table(_stamp_seq_column(batch, first_seq, self._arrow_schema))
            needs_drain = self._buffers.ram_bytes() >= self._segment_target_bytes
        if needs_drain:
            self._wake.set()

    def get_logs(
        self,
        key: str,
        *,
        match_scope: int = logging_pb2.MATCH_SCOPE_EXACT,
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        min_level_enum = str_to_log_level(min_level)
        where_parts, params, include_key_in_select, exact_key = _scope_query(key, cursor, match_scope)
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
        return self._execute_read(
            where_parts,
            params,
            max_lines,
            tail,
            cursor,
            include_key_in_select=include_key_in_select,
            exact_key=exact_key,
        )

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()
        # Flush RAM to L0 so a clean restart picks it up. We deliberately do
        # NOT promote L0 to L1 here — L0 is local-only by design.
        self._flush_step()
        # Final reconcile so the bucket matches the catalog at shutdown.
        self._sync_step()

    def stop_and_join(self) -> None:
        """Stop the bg thread without flushing or compacting (used by ``DropTable``)."""
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()

    def ram_bytes(self) -> int:
        return self._buffers.ram_bytes()

    def chunk_count(self) -> int:
        return self._buffers.chunk_count()

    def update_schema(self, new_schema: Schema) -> None:
        _assert_additive_schema_evolution(self.schema, new_schema)
        self.schema = new_schema
        self._arrow_schema = schema_to_arrow(new_schema)

    def _bg_loop(self) -> None:
        last_heartbeat = 0.0
        last_flush_at = time.monotonic()
        last_compact_at = time.monotonic()
        while not self._stop.is_set():
            with self._insertion_lock:
                ram_bytes = self._buffers.ram_bytes()
                chunk_count = self._buffers.chunk_count()
                next_seq = self._buffers.next_seq
                level_counts = _level_histogram(self._local_segments)
                level_bytes = _level_bytes_summary(self._local_segments, self._compactor.config.level_targets)
            force_drain = ram_bytes >= self._segment_target_bytes

            now = time.monotonic()
            if now - last_heartbeat >= _BG_HEARTBEAT_INTERVAL_SEC:
                logger.info(
                    "bg-loop tick ns=%s: chunks=%d ram_bytes=%d levels=%s level_bytes=%s "
                    "next_seq=%d since_flush_ms=%d since_compact_ms=%d",
                    self.name,
                    chunk_count,
                    ram_bytes,
                    level_counts,
                    level_bytes,
                    next_seq,
                    int((now - last_flush_at) * 1000),
                    int((now - last_compact_at) * 1000),
                )
                last_heartbeat = now

            if force_drain:
                self._flush_step()
                self._flush_rl.mark_run()
                last_flush_at = time.monotonic()
            elif self._flush_rl.should_run():
                self._flush_step()
                last_flush_at = time.monotonic()
            if self._compaction_rl.should_run():
                # Drain promotable runs back-to-back on the same tick: a
                # large flush burst can leave several tiers eligible at once,
                # and waiting another check_interval per merge needlessly
                # extends per-read fanout.
                while self._compaction_step():
                    pass
                self._compaction_rl.mark_run()
                last_compact_at = time.monotonic()
                self._sync_step()
                self._eviction_step()

            self._wake.wait(timeout=min(self._flush_rl.time_until_next(), 1.0))
            self._wake.clear()

    def _flush_step(self) -> None:
        with self._flush_lock:
            flush_start = time.monotonic()
            with self._insertion_lock:
                ram_bytes_before = self._buffers.ram_bytes()
                chunks_before = self._buffers.chunk_count()
                visible = self._buffers.seal()
            if visible is None:
                return

            logger.info(
                "flush starting: rows=%d ram_bytes=%d chunks=%d seq=[%d,%d]",
                visible.table.num_rows,
                ram_bytes_before,
                chunks_before,
                visible.min_seq,
                visible.max_seq,
            )

            sealed = _SealedBuffer(
                table=self._sort_for_flush(visible.table),
                min_seq=visible.min_seq,
                max_seq=visible.max_seq,
            )

            try:
                self._write_new_segment(sealed)
            except Exception:
                logger.warning(
                    "Flush failed after %d ms, restoring data to chunks",
                    int((time.monotonic() - flush_start) * 1000),
                    exc_info=True,
                )
                with self._insertion_lock:
                    self._buffers.restore_flush()
                return

            with self._flush_generation_cond:
                self._flush_generation += 1
                self._flush_generation_cond.notify_all()

    def _sort_for_flush(self, table: pa.Table) -> pa.Table:
        keys = compaction_sort_keys(self.schema)
        return table.sort_by([(name, "ascending") for name in keys])

    def _key_bounds_from_table(self, table: pa.Table) -> tuple[object | None, object | None]:
        """Compute (min, max) over the namespace's ``key_column`` in ``table``.

        Returns ``(None, None)`` when the schema has no key_column or the
        column is empty / all null. The returned values keep their Arrow
        Python type so ``aggregate_key_bounds`` can compare numerics
        natively; ``_segment_to_row`` stringifies them at the catalog
        boundary.
        """
        key_column = self.schema.key_column
        if not key_column or table.num_rows == 0:
            return None, None
        col = table.column(key_column)
        if col.null_count == col.length():
            return None, None
        result = pc.min_max(col)
        lo = result["min"].as_py()
        hi = result["max"].as_py()
        if lo is None or hi is None:
            return None, None
        return lo, hi

    def _segment_by_path(self, path: str) -> LocalSegment | None:
        """Look up the in-memory ``LocalSegment`` matching ``path``.

        Used to recover typed key bounds for a ``SegmentRow`` (the catalog
        round-trip stringifies them). Safe without ``_insertion_lock``
        when called from the per-namespace bg thread, which is the sole
        mutator of ``_local_segments`` post-construction.
        """
        for seg in self._local_segments:
            if seg.path == path:
                return seg
        return None

    def _segment_to_row(self, seg: LocalSegment) -> SegmentRow:
        """Build the catalog row that mirrors ``seg``.

        ``created_at_ms`` reflects the segment's stamped birth time, never
        ``now`` — overwriting it would defeat ``max_l0_age_sec`` aging,
        which the planner reads from the round-trip catalog snapshot every
        tick. Key bounds are stringified here because the catalog stores
        them in a generic TEXT column.
        """
        return SegmentRow(
            namespace=self.name,
            path=seg.path,
            level=seg.level,
            min_seq=seg.min_seq,
            max_seq=seg.max_seq,
            row_count=seg.row_count,
            byte_size=seg.size_bytes,
            created_at_ms=seg.created_at_ms,
            min_key_value=None if seg.min_key_value is None else str(seg.min_key_value),
            max_key_value=None if seg.max_key_value is None else str(seg.max_key_value),
            location=seg.location,
        )

    def _write_new_segment(self, sealed: _SealedBuffer) -> None:
        filename = seg_filename(level=0, min_seq=sealed.min_seq)
        write_start = time.monotonic()

        filepath = self._data_dir / filename
        staging_path = filepath.with_suffix(".parquet.tmp")
        # Materialize parquet in memory and flush in one write(). pyarrow's
        # path-based writer uses an unbuffered FileOutputStream that emits
        # ~40 syscalls per segment (most <100B — page headers, footer
        # fragments). On a contended boot disk those serialize against the
        # I/O queue and dominate flush latency.
        buf = pa.BufferOutputStream()
        # L0 segments do not get a parquet bloom filter on key_column:
        # pyarrow 22.0 does not yet expose ``bloom_filter_options`` to
        # Python (the C++ option exists; the binding hasn't shipped). L0
        # segments are short-lived (compacted within seconds) so the
        # missing bloom doesn't matter much in practice. Compacted output
        # written by the DuckDB COPY does get blooms via DuckDB ≥1.2's
        # default behavior.
        pq.write_table(
            sealed.table,
            buf,
            compression="zstd",
            row_group_size=_ROW_GROUP_SIZE,
            write_page_index=True,
        )
        with pa.OSFile(str(staging_path), "wb") as out:
            out.write(buf.getvalue())
        staging_path.rename(filepath)
        # Compute key_column bounds from the in-memory table — much cheaper
        # than re-opening the freshly written parquet and reading its footer.
        min_key_value, max_key_value = self._key_bounds_from_table(sealed.table)
        seg = LocalSegment(
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            level=0,
            min_seq=sealed.min_seq,
            max_seq=sealed.max_seq,
            row_count=sealed.table.num_rows,
            created_at_ms=int(time.time() * 1000),
            min_key_value=min_key_value,
            max_key_value=max_key_value,
        )

        with self._insertion_lock:
            self._local_segments.append(seg)
            self._buffers.commit_flush()
            self._catalog.upsert_segment(self._segment_to_row(seg))

        logger.info(
            "Wrote L0 segment %s: rows=%d bytes=%d seq=[%d,%d] elapsed_ms=%d",
            filename,
            sealed.table.num_rows,
            seg.size_bytes,
            sealed.min_seq,
            sealed.max_seq,
            int((time.monotonic() - write_start) * 1000),
        )

    def _compaction_step(self) -> bool:
        """Execute one planner-issued compaction job.

        Returns ``True`` when a job ran (caller may immediately tick again
        to drain pending work), ``False`` when the planner had nothing to
        do.
        """
        with self._insertion_lock:
            segment_rows = [self._segment_to_row(s) for s in self._local_segments]
        job = self._compactor.plan(segment_rows, now_ms=int(time.time() * 1000))
        if job is None:
            return False
        self._run_job(job)
        return True

    def _force_compact_l0(self) -> None:
        """Test-only hook: synthesize a one-shot L0 → L1 merge regardless of policy.

        Production code (including ``close()``) never calls this — the
        regular bg loop is planner-driven, and L0 segments are intentionally
        kept local-only between compactions. Tests use this when they need
        to assert against L1 state without configuring tiny ``level_targets``
        on every fixture.
        """
        with self._insertion_lock:
            l0 = sorted(
                [self._segment_to_row(s) for s in self._local_segments if s.level == 0],
                key=lambda r: r.min_seq,
            )
        if not l0:
            return
        job = CompactionJob(
            inputs=tuple(l0),
            output_level=1,
            output_min_seq=min(r.min_seq for r in l0),
            output_max_seq=max(r.max_seq for r in l0),
        )
        self._run_job(job)

    def _run_job(self, job: CompactionJob) -> None:
        # Single-input jobs skip the rewrite and just rename + bump level —
        # the rewrite would be a byte-for-byte copy.
        if len(job.inputs) == 1:
            self._apply_level_bump(job)
        else:
            self._apply_merge(job)
        with self._compaction_generation_cond:
            self._compaction_generation += 1
            self._compaction_generation_cond.notify_all()

    def _apply_level_bump(self, job: CompactionJob) -> None:
        old = job.inputs[0]
        new_filename = seg_filename(level=job.output_level, min_seq=old.min_seq)
        new_path = self._data_dir / new_filename

        # Recover the typed key bounds from the in-memory deque — the
        # SegmentRow on ``job.inputs`` carries the catalog's stringified
        # form, which loses ordering for numeric keys.
        old_local = self._segment_by_path(old.path)
        bumped = LocalSegment(
            path=str(new_path),
            size_bytes=old.byte_size,
            level=job.output_level,
            min_seq=old.min_seq,
            max_seq=old.max_seq,
            row_count=old.row_count,
            # Preserve the input's birth time across the level bump:
            # a single-input promotion is a rename, not a fresh write.
            created_at_ms=old.created_at_ms,
            min_key_value=old_local.min_key_value if old_local else None,
            max_key_value=old_local.max_key_value if old_local else None,
        )
        self._commit_swap(
            removed=[old.path],
            added=bumped,
            unlink_removed=False,
            pre_swap=lambda: os.rename(old.path, new_path),
        )
        logger.info("Level-bumped %s -> L%d (%s)", Path(old.path).name, job.output_level, new_filename)

    def _apply_merge(self, job: CompactionJob) -> None:
        merged_filename = seg_filename(level=job.output_level, min_seq=job.output_min_seq)
        merged_path = self._data_dir / merged_filename
        staging_path = merged_path.with_suffix(".parquet.tmp")
        sql = self._compactor.merge_sql(job, schema=self.schema, staging_path=staging_path)
        compaction_start = time.monotonic()
        # COPY reads input parquet files from disk and writes a staging file;
        # it touches neither ``_local_segments`` nor any other structure that
        # ``_insertion_lock`` protects, so we run it lock-free. A multi-second
        # COPY would otherwise stall every concurrent ``append_log_batch``.
        try:
            with self._open_compaction_conn() as conn:
                conn.execute(sql)
        except Exception:
            logger.warning("Compaction failed, leaving inputs in place", exc_info=True)
            staging_path.unlink(missing_ok=True)
            return

        # The catalog stores key bounds as TEXT; we compare on the typed
        # values held in the in-memory deque to preserve numeric ordering,
        # then stringify at the ``_segment_to_row`` boundary.
        input_locals = [self._segment_by_path(s.path) for s in job.inputs]
        merged_min_key, merged_max_key = aggregate_key_bounds(
            (loc.min_key_value, loc.max_key_value) for loc in input_locals if loc is not None
        )
        merged_seg = LocalSegment(
            path=str(merged_path),
            size_bytes=staging_path.stat().st_size,
            level=job.output_level,
            min_seq=job.output_min_seq,
            max_seq=job.output_max_seq,
            row_count=sum(s.row_count for s in job.inputs),
            created_at_ms=int(time.time() * 1000),
            min_key_value=merged_min_key,
            max_key_value=merged_max_key,
        )

        staging_path.rename(merged_path)
        self._commit_swap(removed=[s.path for s in job.inputs], added=merged_seg, unlink_removed=True)

        logger.info(
            "Merged %d L%d -> L%d %s: bytes=%d seq=[%d,%d] elapsed_ms=%d",
            len(job.inputs),
            job.inputs[0].level,
            job.output_level,
            merged_filename,
            merged_seg.size_bytes,
            merged_seg.min_seq,
            merged_seg.max_seq,
            int((time.monotonic() - compaction_start) * 1000),
        )

    def _commit_swap(
        self,
        *,
        removed: list[str],
        added: LocalSegment,
        unlink_removed: bool,
        pre_swap: Callable[[], None] | None = None,
    ) -> None:
        """Splice the deque + catalog: replace ``removed`` paths with ``added``.

        ``unlink_removed`` is False for level bumps (the file was renamed
        in place via ``pre_swap``, so the old path is already gone) and
        True for merges (the inputs are still on disk and need cleanup).

        The full transition runs under ``_query_visibility_lock`` write
        mode: readers hold the read lock for their entire query (DuckDB
        opens parquet files lazily), so renaming or unlinking a file
        under a stale snapshot path would surface as
        ``IOException: No files found``. Taking the write lock here
        drains existing readers before any rename/unlink and blocks new
        ones until the deque mirrors the post-swap state. ``pre_swap``
        runs inside the lock so a level-bump rename happens after readers
        have drained.
        """
        removed_set = set(removed)
        self._query_visibility_lock.write_acquire()
        try:
            if pre_swap is not None:
                pre_swap()
            with self._insertion_lock:
                new_segments: deque[LocalSegment] = deque()
                inserted = False
                for s in self._local_segments:
                    if s.path in removed_set:
                        if not inserted:
                            new_segments.append(added)
                            inserted = True
                    else:
                        new_segments.append(s)
                if not inserted:
                    new_segments.append(added)
                self._local_segments = new_segments
                self._catalog.replace_segments(
                    self.name,
                    removed_paths=removed,
                    added=[self._segment_to_row(added)],
                )
            if unlink_removed:
                for path in removed:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Failed to unlink merged input %s", path, exc_info=True)
        finally:
            self._query_visibility_lock.write_release()

    def _adopt_remote_segments(self, *, key_column: str | None) -> None:
        """Insert ``REMOTE`` catalog rows for bucket files with no row.

        Runs once at boot when ``_remote_namespace_dir`` is configured.
        After a catalog wipe the bucket is the only durable record of
        every L>=1 segment; we read each parquet footer remotely to
        reconstruct row metadata, then insert at ``location=REMOTE``.
        Files whose basename is already in the catalog are left alone
        (the local-disk pass already attached them).
        """
        try:
            fs, root = fsspec.core.url_to_fs(self._remote_namespace_dir)
            remote_paths = list(fs.find(root))
        except Exception:
            logger.warning("remote adopt list failed for %s", self.name, exc_info=True)
            return

        catalog_basenames = {Path(r.path).name for r in self._catalog.list_segments(self.name)}
        for fs_path in remote_paths:
            basename = Path(fs_path).name
            if basename in catalog_basenames:
                continue
            parsed = parse_seg_filename(basename)
            if parsed is None:
                logger.warning("ignoring unparseable remote file %s/%s", self.name, basename)
                continue
            level, min_seq = parsed
            try:
                with fs.open(fs_path, "rb") as f:
                    metadata = pq.read_metadata(f)
            except Exception:
                logger.warning("failed reading remote parquet footer %s/%s", self.name, basename, exc_info=True)
                continue
            num_rows = metadata.num_rows
            min_key, max_key = _key_bounds_from_parquet(metadata, key_column)
            try:
                byte_size = fs.info(fs_path).get("size", 0)
            except Exception:
                byte_size = 0
            local_path = self._data_dir / basename
            self._catalog.upsert_segment(
                SegmentRow(
                    namespace=self.name,
                    path=str(local_path),
                    level=level,
                    min_seq=min_seq,
                    max_seq=min_seq + max(num_rows - 1, 0),
                    row_count=num_rows,
                    byte_size=int(byte_size),
                    created_at_ms=int(time.time() * 1000),
                    location=SegmentLocation.REMOTE,
                    min_key_value=None if min_key is None else str(min_key),
                    max_key_value=None if max_key is None else str(max_key),
                )
            )
            logger.info("Adopted remote segment %s/%s as REMOTE", self.name, basename)

    def _sync_step(self) -> None:
        """Reconcile the remote namespace prefix with the catalog.

        The catalog is the source of truth for what should exist remotely.
        Phase 1 uploads every ``LOCAL`` row at L>=1 (or adopts a row whose
        file is already remote, which happens after a crash between
        ``fs`` write and the catalog flip). Phase 2 ``fs.rm``s any remote
        file with no catalog row — those are compaction inputs whose row
        has been dropped. The ordering is what makes this safe: by the
        time phase 2 runs, the merged output that subsumes those inputs
        has been uploaded in phase 1, so the durable copy is in place
        before any input remote bytes are deleted.

        Phase 2 is skipped entirely when any phase-1 upload failed: if
        the merged output isn't durable, the only remaining copies of
        its seq range are the compaction inputs sitting in the bucket
        (whose catalog rows were dropped at commit time). Deleting them
        as orphans before the replacement is durable would lose data.
        Cleanup is harmless to defer to the next tick.

        No-op when the namespace has no remote prefix configured.
        """
        if not self._remote_namespace_dir:
            return

        try:
            fs, root = fsspec.core.url_to_fs(self._remote_namespace_dir)
            remote_basenames = {Path(p).name for p in fs.find(root)}
        except Exception:
            logger.warning("remote sync list failed for %s", self.name, exc_info=True)
            return

        with self._insertion_lock:
            rows = self._catalog.list_segments(self.name, min_level=1)

        all_durable = True
        for row in rows:
            if row.location != SegmentLocation.LOCAL:
                continue
            basename = Path(row.path).name
            if basename in remote_basenames:
                # Crash recovery: the file was uploaded but the catalog
                # never flipped. Adopt without re-uploading.
                self._mark_uploaded(row.path)
                continue
            if self._upload(Path(row.path)):
                self._mark_uploaded(row.path)
            else:
                all_durable = False

        if not all_durable:
            return

        # Re-snapshot: phase 1 may have added basenames to the bucket; we
        # only want to delete files whose basename is genuinely orphan
        # (no catalog row at all).
        with self._insertion_lock:
            catalog_basenames = {Path(r.path).name for r in self._catalog.list_segments(self.name, min_level=1)}
        for basename in remote_basenames - catalog_basenames:
            try:
                fs.rm(f"{self._remote_namespace_dir}/{basename}")
                logger.info("Deleted orphan remote segment %s/%s", self.name, basename)
            except Exception:
                logger.warning("orphan delete failed: %s/%s", self.name, basename, exc_info=True)

    def _upload(self, local_path: Path) -> bool:
        """Stream ``local_path`` to remote. Returns True on success;
        the next sync retries on failure."""
        remote_path = f"{self._remote_namespace_dir}/{local_path.name}"
        upload_start = time.monotonic()
        try:
            with (
                fsspec.core.open(str(local_path), "rb") as f_src,
                fsspec.core.open(remote_path, "wb") as f_dst,
            ):
                shutil.copyfileobj(f_src, f_dst, length=_REMOTE_UPLOAD_CHUNK_BYTES)
        except Exception:
            logger.warning("Failed to copy %s to %s", local_path, remote_path, exc_info=True)
            return False
        try:
            size = local_path.stat().st_size
        except OSError:
            size = -1
        logger.info(
            "Copied %s to %s: bytes=%d elapsed_ms=%d",
            local_path.name,
            remote_path,
            size,
            int((time.monotonic() - upload_start) * 1000),
        )
        return True

    def _mark_uploaded(self, path: str) -> None:
        """Flip a segment's location to ``BOTH`` after a successful upload.

        Updates the in-memory deque and the catalog under the insertion
        lock so the planner / eviction queries see consistent state.
        """
        with self._insertion_lock:
            for s in self._local_segments:
                if s.path == path:
                    s.location = SegmentLocation.BOTH
                    break
            self._catalog.set_location(self.name, path, SegmentLocation.BOTH)

    def _eviction_step(self) -> None:
        """Evict the namespace's oldest L>=1 copied segments until under caps.

        Runs at the tail of every compaction tick. Per-namespace caps live
        on ``CompactionConfig`` (``max_segments_per_namespace``,
        ``max_bytes_per_namespace``); FIFO-by-min_seq picks the oldest
        eligible segment.
        """
        config = self._compactor.config
        while True:
            with self._insertion_lock:
                seg_count = len(self._local_segments)
                byte_total = sum(s.size_bytes for s in self._local_segments)
            if seg_count <= config.max_segments_per_namespace and byte_total <= config.max_bytes_per_namespace:
                return
            with self._insertion_lock:
                row = self._catalog.select_eviction_candidate(self.name)
            if row is None:
                # Over the cap but nothing eligible (everything still L0,
                # or terminal segments not yet copied). Bail and let the
                # next tick try again.
                return
            self.evict_segment(row.path)
            logger.info(
                "Evicted L%d segment %s (bytes=%d, remaining=%d segments)",
                row.level,
                Path(row.path).name,
                row.byte_size,
                seg_count - 1,
            )

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]:
        """Return all currently queryable local segments and RAM tables."""
        with self._insertion_lock:
            return list(self._local_segments), self._buffers.query_snapshot()

    def all_segments_unlocked(self) -> list[LocalSegment]:
        """Snapshot every locally-tracked segment. Caller MUST hold the insertion lock."""
        return list(self._local_segments)

    def evict_segment(self, path: str) -> int:
        """Drop ``path`` from the local deque and unlink the file. Returns
        bytes freed.

        For a ``BOTH`` segment, the catalog row stays in place at
        ``REMOTE`` and the bucket copy is the durable archive. For a
        ``LOCAL``-only segment, eviction is destructive: there's no
        durable copy, so the row is dropped entirely. Production eviction
        runs through ``select_eviction_candidate`` which gates on
        ``BOTH``; the destructive branch is here for direct callers
        (tests, manual recovery) that have already accepted the
        consequences.
        """
        with self._insertion_lock:
            new: deque[LocalSegment] = deque()
            removed_bytes = 0
            removed_location: SegmentLocation | None = None
            for s in self._local_segments:
                if s.path == path:
                    removed_bytes = s.size_bytes
                    removed_location = s.location
                    continue
                new.append(s)
            self._local_segments = new
            if removed_location is SegmentLocation.BOTH:
                self._catalog.set_location(self.name, path, SegmentLocation.REMOTE)
            else:
                self._catalog.remove_segment(self.name, path)
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete evicted segment %s", path, exc_info=True)
        return removed_bytes

    def remove_local_storage(self) -> None:
        """Delete every tracked segment file plus the namespace directory."""
        for s in list(self._local_segments):
            try:
                Path(s.path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to delete %s during drop", s.path, exc_info=True)
        self._local_segments.clear()
        # The catalog row for this namespace is dropped by ``Catalog.delete``
        # in :meth:`DuckDBLogStore.drop_table`; segments are cascaded there too.
        # Sweep stragglers (e.g. half-written .parquet.tmp) before rmdir.
        for p in list(self._data_dir.glob("*")):
            try:
                p.unlink()
            except OSError:
                logger.warning("Failed to delete stray file %s during drop", p, exc_info=True)
        try:
            self._data_dir.rmdir()
        except OSError:
            logger.warning("Namespace dir %s not empty after drop", self._data_dir)

    def stats(self) -> NamespaceStats:
        """Aggregate row/byte/seq stats over sealed segments + RAM buffer.

        Read straight from in-memory state (which is held in lockstep with
        the catalog) so the call is O(local_segments) and serves dashboard
        list requests without touching parquet.
        """
        with self._insertion_lock:
            if not self._local_segments and self._buffers.ram_rows() == 0:
                return NamespaceStats.empty()
            seg_rows = sum(s.row_count for s in self._local_segments)
            seg_bytes = sum(s.size_bytes for s in self._local_segments)
            seg_min = min((s.min_seq for s in self._local_segments if s.row_count > 0), default=0)
            seg_max = max((s.max_seq for s in self._local_segments if s.row_count > 0), default=0)
            ram_rows = self._buffers.ram_rows()
            ram_bytes = self._buffers.ram_bytes()
            next_seq = self._buffers.next_seq
        # The seq counter is the high-water mark; ``ram_rows`` rows occupy
        # ``[next_seq - ram_rows, next_seq - 1]``. If everything is in RAM
        # (cold start before any flush) this gives a correct seq window too.
        min_seq = seg_min if seg_min else (next_seq - ram_rows if ram_rows else 0)
        max_seq = max(seg_max, next_seq - 1) if (seg_max or ram_rows) else 0
        return NamespaceStats(
            row_count=seg_rows + ram_rows,
            byte_size=seg_bytes + ram_bytes,
            min_seq=min_seq,
            max_seq=max_seq,
            segment_count=len(self._local_segments),
        )

    def _execute_read(
        self,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        default_cursor: int,
        include_key_in_select: bool,
        exact_key: str | None = None,
    ) -> LogReadResult:
        # Hold the rwlock across the whole query so GC/compaction can't
        # unlink a file that DuckDB may still open lazily.
        self._query_visibility_lock.read_acquire()
        try:
            rows = self._run_read_locked(
                where_parts=where_parts,
                params=params,
                max_lines=max_lines,
                tail=tail,
                include_key_in_select=include_key_in_select,
            )
        finally:
            self._query_visibility_lock.read_release()

        return _shape_log_read_result(rows, tail, max_lines, default_cursor, include_key_in_select, exact_key)

    def _run_read_locked(
        self,
        *,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        include_key_in_select: bool,
    ) -> list[tuple]:
        with self._insertion_lock:
            segments = list(self._local_segments)
            ram_tables: list[pa.Table] = self._buffers.query_snapshot()

        segments = _cap_segments(segments)
        parquet_files = [s.path for s in segments]

        where_clause = " AND ".join(where_parts)
        select_cols = (
            "seq, key, source, data, epoch_ms, level" if include_key_in_select else "seq, source, data, epoch_ms, level"
        )
        order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
        limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

        with self._read_pool.cursor(buffers={"_ram": ram_tables} if ram_tables else None) as conn:
            source = _build_union_source(parquet_files, ["_ram"] if ram_tables else [], self._arrow_schema)
            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            return conn.execute(sql, params).fetchall()


class MemoryLogNamespace:
    """In-process Arrow-backed namespace for tests and embedded use.

    Holds every appended row in a single Arrow table; no segmentation,
    flush, compaction, eviction, or background thread. The registered
    schema may evolve (additive nullable extension); the backing table is
    reprojected on each :meth:`update_schema` call.
    """

    def __init__(
        self,
        *,
        name: str,
        schema: Schema,
        insertion_lock: Lock,
        query_visibility_lock: RWLock,
        read_pool: _ReadPoolProtocol,
    ) -> None:
        self.name = name
        self.schema = schema
        self._arrow_schema = schema_to_arrow(schema)
        self._insertion_lock = insertion_lock
        self._query_visibility_lock = query_visibility_lock
        self._read_pool = read_pool
        # Empty against the registered schema so consumers can register it
        # with DuckDB before any rows arrive.
        self._table: pa.Table = self._arrow_schema.empty_table()
        self._next_seq = 1

    def append_log_batch(self, items: list[tuple[str, list]]) -> None:
        with self._insertion_lock:
            new_tables: list[pa.Table] = [self._table]
            for key, entries in items:
                if not entries:
                    continue
                first_seq = self._next_seq
                self._next_seq += len(entries)
                rows = [
                    (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
                ]
                new_tables.append(_build_log_table(rows, self._arrow_schema))
            self._table = pa.concat_tables(new_tables) if len(new_tables) > 1 else self._table

    def append_record_batch(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        with self._insertion_lock:
            first_seq = self._next_seq
            self._next_seq += batch.num_rows
            stamped = _stamp_seq_column(batch, first_seq, self._arrow_schema)
            self._table = pa.concat_tables([self._table, stamped])

    def get_logs(
        self,
        key: str,
        *,
        match_scope: int = logging_pb2.MATCH_SCOPE_EXACT,
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        min_level_enum = str_to_log_level(min_level)
        where_parts, params, include_key_in_select, exact_key = _scope_query(key, cursor, match_scope)
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)

        # Insertion lock alone suffices; rwlock unneeded because there are
        # no files to unlink.
        with self._insertion_lock:
            table = self._table

        select_cols = (
            "seq, key, source, data, epoch_ms, level" if include_key_in_select else "seq, source, data, epoch_ms, level"
        )
        order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
        limit = f"LIMIT {max_lines}" if max_lines > 0 else ""
        where_clause = " AND ".join(where_parts)

        with self._read_pool.cursor(buffers={"_ram": [table]}) as conn:
            source = _build_union_source([], ["_ram"], self._arrow_schema)
            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            rows = conn.execute(sql, params).fetchall()

        return _shape_log_read_result(rows, tail, max_lines, cursor, include_key_in_select, exact_key)

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]:
        with self._insertion_lock:
            return [], [self._table]

    def all_segments_unlocked(self) -> list[LocalSegment]:
        return []

    def update_schema(self, new_schema: Schema) -> None:
        _assert_additive_schema_evolution(self.schema, new_schema)
        new_arrow = schema_to_arrow(new_schema)
        with self._insertion_lock:
            self._table = _project_to_schema(self._table, new_arrow)
            self.schema = new_schema
            self._arrow_schema = new_arrow

    def evict_segment(self, path: str) -> int:
        return 0

    def remove_local_storage(self) -> None:
        with self._insertion_lock:
            self._table = self._arrow_schema.empty_table()

    def close(self) -> None:
        return

    def stop_and_join(self) -> None:
        return

    def ram_bytes(self) -> int:
        return 0

    def chunk_count(self) -> int:
        return 0

    def stats(self) -> NamespaceStats:
        with self._insertion_lock:
            num_rows = self._table.num_rows
            byte_size = self._table.nbytes
            next_seq = self._next_seq
        if num_rows == 0:
            return NamespaceStats.empty()
        return NamespaceStats(
            row_count=num_rows,
            byte_size=byte_size,
            min_seq=max(next_seq - num_rows, 1),
            max_seq=next_seq - 1,
            segment_count=0,
        )


class _ReadPoolProtocol(Protocol):
    def cursor(
        self,
        buffers: dict[str, list[pa.Table]] | None = None,
    ) -> AbstractContextManager[duckdb.DuckDBPyConnection]: ...


def _assert_additive_schema_evolution(old: Schema, new: Schema) -> None:
    old_columns = {c.name: c for c in old.columns}
    new_columns = {c.name: c for c in new.columns}
    for name, old_col in old_columns.items():
        new_col = new_columns.get(name)
        assert new_col is not None, f"update_schema: column {name!r} dropped (must be additive)"
        assert (
            new_col.type == old_col.type
        ), f"update_schema: column {name!r} type changed {old_col.type}->{new_col.type}"
        assert new_col.nullable == old_col.nullable, f"update_schema: column {name!r} nullability changed"
    for name, new_col in new_columns.items():
        if name not in old_columns:
            assert new_col.nullable, f"update_schema: new column {name!r} must be nullable"


def _shape_log_read_result(
    rows: list[tuple],
    tail: bool,
    max_lines: int,
    default_cursor: int,
    include_key_in_select: bool,
    exact_key: str | None,
) -> LogReadResult:
    if tail and max_lines > 0:
        rows.reverse()

    if not rows:
        return LogReadResult(entries=[], cursor=default_cursor)

    max_seq = max(r[0] for r in rows)

    if include_key_in_select:
        # row: (seq, key, source, data, epoch_ms, level)
        entries = []
        for r in rows:
            entry = logging_pb2.LogEntry(source=r[2], data=r[3], level=r[5])
            entry.timestamp.epoch_ms = r[4]
            entry.key = r[1]
            entry.attempt_id = parse_attempt_id(r[1])
            entries.append(entry)
    else:
        # row: (seq, source, data, epoch_ms, level)
        entries = []
        attempt_id = parse_attempt_id(exact_key) if exact_key else 0
        for r in rows:
            entry = logging_pb2.LogEntry(source=r[1], data=r[2], level=r[4])
            entry.timestamp.epoch_ms = r[3]
            entry.attempt_id = attempt_id
            entries.append(entry)

    return LogReadResult(entries=entries, cursor=max_seq)


def _cap_segments(segments: list[LocalSegment]) -> list[LocalSegment]:
    if not segments:
        return segments
    newest_first = sorted(segments, key=lambda s: s.min_seq, reverse=True)
    capped: list[LocalSegment] = []
    total = 0
    for seg in newest_first:
        if capped and total + seg.size_bytes > _MAX_PARQUET_BYTES_PER_READ:
            break
        capped.append(seg)
        total += seg.size_bytes
    capped.sort(key=lambda s: s.min_seq)
    return capped


# Characters that hint a regex was passed where PREFIX was intended; used
# only for a friendlier error if a caller forgets to set match_scope=REGEX.
_REGEX_HINT_RE = re.compile(r"[.*+?\[\](){}^$|\\]")


def _regex_literal_prefix(pattern: str) -> str:
    match = _REGEX_HINT_RE.search(pattern)
    if match is None:
        return pattern
    return pattern[: match.start()]


def _scope_query(
    source: str,
    cursor: int,
    match_scope: int,
) -> tuple[list[str], dict, bool, str | None]:
    """Build the WHERE clause for a FetchLogs query.

    Returns ``(where_parts, params, include_key_in_select, exact_key)``.

    The in-process Python default is ``MATCH_SCOPE_EXACT``; the RPC server
    boundary maps wire-level ``MATCH_SCOPE_UNSPECIFIED`` to ``PREFIX`` before
    invoking ``get_logs``. Either of those resolves to one of the four
    branches below — ``UNSPECIFIED`` never reaches the query layer.
    """
    if match_scope == logging_pb2.MATCH_SCOPE_EXACT:
        where_parts = ["key = $key", "seq > $cursor"]
        params: dict = {"key": source, "cursor": cursor}
        return where_parts, params, False, source

    if match_scope == logging_pb2.MATCH_SCOPE_PREFIX:
        if not source:
            # Empty prefix would match every key in the store. Reads with no
            # source are almost always a caller bug (omitted/defaulted field);
            # fail fast instead of returning the first page of every stream.
            raise ValueError("FetchLogs source is required for MATCH_SCOPE_PREFIX")
        # `prefix(key, $p)` is DuckDB's literal-prefix predicate. It's pushed
        # into Parquet row-group min/max stats the same way an `=` is, so
        # PREFIX reads keep the pruning of EXACT.
        where_parts = ["seq > $cursor", "prefix(key, $prefix)"]
        params = {"cursor": cursor, "prefix": source}
        return where_parts, params, True, None

    if match_scope == logging_pb2.MATCH_SCOPE_REGEX:
        # Pull off any leading literal prefix to keep row-group pruning even
        # for regex queries. `prefix(key, $p)` is monotone, so it remains
        # correct as long as the regex requires that prefix to match.
        literal_prefix = _regex_literal_prefix(source)
        suffix = source[len(literal_prefix) :]
        # `^literal$`, `^literal`, `^literal.*` all reduce to the literal prefix
        # alone; we still need regexp_matches for any other suffix.
        is_pure_prefix = suffix in (".*", "")

        where_parts = ["seq > $cursor"]
        params = {"cursor": cursor}
        if literal_prefix:
            where_parts.append("prefix(key, $prefix_lo)")
            params["prefix_lo"] = literal_prefix
        if not is_pure_prefix:
            where_parts.append("regexp_matches(key, $key_pattern)")
            params["key_pattern"] = source
        return where_parts, params, True, None

    raise ValueError(f"unknown match_scope: {match_scope}")


def _add_common_filters(
    where_parts: list[str],
    params: dict,
    since_ms: int,
    substring_filter: str,
    min_level_enum: int,
) -> None:
    if since_ms > 0:
        where_parts.append("epoch_ms > $since_ms")
        params["since_ms"] = since_ms
    if substring_filter:
        where_parts.append("contains(data, $substring)")
        params["substring"] = substring_filter
    if min_level_enum > 0:
        where_parts.append("(level = 0 OR level >= $min_level)")
        params["min_level"] = min_level_enum


def _project_to_schema(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Cast/extend ``table`` to match ``target``: missing columns become nulls."""
    cols = []
    for field in target:
        if field.name in table.schema.names:
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type)
            cols.append(col)
        else:
            cols.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(cols, schema=target)


def _build_union_source(parquet_files: list[str], ram_table_names: list[str], arrow_schema: pa.Schema) -> str:
    # Both parquet paths and ram table names are self-generated, so f-string
    # embedding has no SQL-injection surface.
    parts: list[str] = []
    if parquet_files:
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        parts.append(f"SELECT * FROM read_parquet([{file_list}])")
    for name in ram_table_names:
        parts.append(f"SELECT * FROM {name}")
    if not parts:
        col_defs = ", ".join(f"NULL::{_arrow_to_duckdb_type(f.type)} AS {f.name}" for f in arrow_schema)
        return f"SELECT {col_defs} WHERE false"
    return " UNION ALL ".join(parts)


_ARROW_TO_DUCKDB: dict[pa.DataType, str] = {
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.string(): "VARCHAR",
    pa.float64(): "DOUBLE",
    pa.bool_(): "BOOLEAN",
    pa.timestamp("ms"): "TIMESTAMP_MS",
    pa.binary(): "BLOB",
}


def _arrow_to_duckdb_type(arrow_type: pa.DataType) -> str:
    return _ARROW_TO_DUCKDB[arrow_type]
