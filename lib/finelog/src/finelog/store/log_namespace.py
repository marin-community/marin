# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace log storage state.

:class:`DiskLogNamespace` is the production path; :class:`MemoryLogNamespace`
backs the in-memory store mode. Both satisfy :class:`LogNamespaceProtocol`.

The two global locks (insertion mutex + query-visibility rwlock) live on the
registry and are passed in at construction. The disk variant additionally
owns a per-namespace flush mutex preventing the test ``_force_flush`` hook
from racing the bg thread on the same ``tmp_*.parquet`` filename.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock
from typing import Protocol

import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.timing import RateLimiter

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.store.rwlock import RWLock
from finelog.store.schema import (
    IMPLICIT_SEQ_COLUMN,
    Column,
    Schema,
    duckdb_type_for,
    schema_to_arrow,
)
from finelog.store.sql_escape import quote_ident, quote_literal
from finelog.types import REGEX_META_RE, LogReadResult, parse_attempt_id, str_to_log_level

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

# Both prefixes keyed by min_seq, so sort-by-filename yields chronological order.
_TMP_PREFIX = "tmp_"
_LOG_PREFIX = "logs_"

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

_FILENAME_SEQ_RE = re.compile(rf"^(?:{_TMP_PREFIX}|{_LOG_PREFIX})(\d+)\.parquet$")


def _tmp_filename(min_seq: int) -> str:
    return f"{_TMP_PREFIX}{min_seq:019d}.parquet"


def _log_filename(min_seq: int) -> str:
    return f"{_LOG_PREFIX}{min_seq:019d}.parquet"


def _is_tmp_path(path: str) -> bool:
    return Path(path).name.startswith(_TMP_PREFIX)


def _min_seq_from_filename(name: str) -> int | None:
    match = _FILENAME_SEQ_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def _read_seq_bounds(path: Path) -> tuple[int, int]:
    """Compute (min_seq, max_seq) for a segment via filename + parquet metadata."""
    min_seq = _min_seq_from_filename(path.name)
    if min_seq is None:
        return 0, 0
    try:
        meta = pq.read_metadata(path)
        num_rows = meta.num_rows
    except Exception:
        logger.warning("Failed to read parquet metadata for %s; treating as empty", path, exc_info=True)
        return 0, 0
    if num_rows <= 0:
        return min_seq, min_seq
    return min_seq, min_seq + num_rows - 1


def _discover_segments(log_dir: Path) -> list[Path]:
    return sorted(list(log_dir.glob(f"{_TMP_PREFIX}*.parquet")) + list(log_dir.glob(f"{_LOG_PREFIX}*.parquet")))


def _recover_next_seq(log_dir: Path) -> int:
    next_seq = 1
    for p in _discover_segments(log_dir):
        _, max_seq = _read_seq_bounds(p)
        if max_seq + 1 > next_seq:
            next_seq = max_seq + 1
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
    min_seq: int = 0
    max_seq: int = 0


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

    def ram_bytes(self) -> int:
        flushing_b = self._flushing.table.nbytes if self._flushing is not None else 0
        return self._ram_bytes + flushing_b

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
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult: ...

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]: ...

    def sealed_segments(self) -> list[LocalSegment]: ...

    def all_segments_unlocked(self) -> list[LocalSegment]: ...

    def update_schema(self, new_schema: Schema) -> None: ...

    def evict_segment(self, path: str) -> int: ...

    def remove_local_storage(self) -> None: ...

    def close(self) -> None: ...

    def stop_and_join(self) -> None: ...


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
        compaction_interval_sec: float,
        max_tmp_segments_before_compact: int,
        insertion_lock: Lock,
        query_visibility_lock: RWLock,
        compaction_conn: duckdb.DuckDBPyConnection,
        read_pool: _ReadPoolProtocol,
        evict_hook: Callable[[], None] = lambda: None,
    ) -> None:
        self.name = name
        self.schema = schema
        self._arrow_schema = schema_to_arrow(schema)
        self._data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        self._remote_log_dir = remote_log_dir
        self._segment_target_bytes = segment_target_bytes
        self._max_tmp_segments_before_compact = max_tmp_segments_before_compact
        self._evict_hook = evict_hook

        self._insertion_lock = insertion_lock
        self._query_visibility_lock = query_visibility_lock
        # Prevents the test ``_force_flush`` hook racing the bg thread on
        # the same tmp filename (both derive from this namespace's _next_seq).
        self._flush_lock = Lock()

        self._compaction_conn = compaction_conn
        self._read_pool = read_pool

        self._buffers = RamBuffers(
            arrow_schema=self._arrow_schema,
            next_seq=_recover_next_seq(data_dir),
        )
        self._local_segments: deque[LocalSegment] = deque()

        # Drop stale tmp segments left behind by a prior compaction that
        # crashed between rename and unlink: any tmp whose [min, max] is
        # fully covered by a logs_ segment is a duplicate.
        discovered: list[LocalSegment] = []
        for p in _discover_segments(data_dir):
            min_seq, max_seq = _read_seq_bounds(p)
            discovered.append(
                LocalSegment(
                    path=str(p),
                    size_bytes=p.stat().st_size,
                    min_seq=min_seq,
                    max_seq=max_seq,
                )
            )
        log_ranges = [(s.min_seq, s.max_seq) for s in discovered if not _is_tmp_path(s.path)]
        for s in discovered:
            if _is_tmp_path(s.path) and any(lo <= s.min_seq and s.max_seq <= hi for lo, hi in log_ranges):
                logger.info("Dropping stale tmp segment %s covered by compacted logs_ range", s.path)
                try:
                    Path(s.path).unlink()
                except OSError:
                    logger.warning("Failed to unlink stale tmp segment %s", s.path, exc_info=True)
                continue
            self._local_segments.append(s)

        self._flush_rl = RateLimiter(flush_interval_sec)
        self._compaction_rl = RateLimiter(compaction_interval_sec)
        # Mark just-run so the bg loop doesn't fire a spurious tick at startup
        # and compact a partially-written set of tmp segments.
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
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        min_level_enum = str_to_log_level(min_level)
        is_pattern = bool(REGEX_META_RE.search(key))

        if not is_pattern:
            where_parts = ["key = $key", "seq > $cursor"]
            params: dict = {"key": key, "cursor": cursor}
            _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
            return self._execute_read(
                where_parts,
                params,
                max_lines,
                tail,
                cursor,
                include_key_in_select=False,
                exact_key=key,
            )

        where_parts, params = _regex_query(key, cursor)
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
        return self._execute_read(
            where_parts,
            params,
            max_lines,
            tail,
            cursor,
            include_key_in_select=True,
        )

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()
        self._flush_step()
        self._compaction_step(compact_single=True)

    def stop_and_join(self) -> None:
        """Stop the bg thread without flushing or compacting (used by ``DropTable``)."""
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()

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
                tmp_count = sum(1 for s in self._local_segments if _is_tmp_path(s.path))
                logs_count = len(self._local_segments) - tmp_count
            force_drain = ram_bytes >= self._segment_target_bytes
            force_compaction = tmp_count > self._max_tmp_segments_before_compact

            now = time.monotonic()
            if now - last_heartbeat >= _BG_HEARTBEAT_INTERVAL_SEC:
                logger.info(
                    "bg-loop tick: chunks=%d ram_bytes=%d tmp=%d logs=%d next_seq=%d "
                    "since_flush_ms=%d since_compact_ms=%d",
                    chunk_count,
                    ram_bytes,
                    tmp_count,
                    logs_count,
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
            if force_compaction or self._compaction_rl.should_run():
                self._compaction_step()
                self._compaction_rl.mark_run()
                last_compact_at = time.monotonic()

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

            self._evict_hook()

    def _derive_seq_bounds_locked(self, table: pa.Table) -> tuple[int, int]:
        seq_col = table.column(IMPLICIT_SEQ_COLUMN)
        return pc.min(seq_col).as_py(), pc.max(seq_col).as_py()

    def _sort_for_flush(self, table: pa.Table) -> pa.Table:
        keys = self._compaction_sort_keys()
        return table.sort_by([(name, "ascending") for name in keys])

    def _compaction_sort_keys(self) -> tuple[str, ...]:
        # key_column first so range scans on it prune row groups efficiently.
        if self.schema.key_column:
            return (self.schema.key_column, IMPLICIT_SEQ_COLUMN)
        return (IMPLICIT_SEQ_COLUMN,)

    def _write_new_segment(self, sealed: _SealedBuffer) -> None:
        filename = _tmp_filename(sealed.min_seq)
        write_start = time.monotonic()

        filepath = self._data_dir / filename
        tmp_path = filepath.with_suffix(".parquet.tmp")
        # Materialize parquet in memory and flush in one write(). pyarrow's
        # path-based writer uses an unbuffered FileOutputStream that emits
        # ~40 syscalls per segment (most <100B — page headers, footer
        # fragments). On a contended boot disk those serialize against the
        # I/O queue and dominate flush latency.
        buf = pa.BufferOutputStream()
        pq.write_table(
            sealed.table,
            buf,
            compression="zstd",
            row_group_size=_ROW_GROUP_SIZE,
            write_page_index=True,
        )
        with pa.OSFile(str(tmp_path), "wb") as out:
            out.write(buf.getvalue())
        tmp_path.rename(filepath)
        seg = LocalSegment(
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            min_seq=sealed.min_seq,
            max_seq=sealed.max_seq,
        )

        with self._insertion_lock:
            self._local_segments.append(seg)
            self._buffers.commit_flush()

        logger.info(
            "Wrote tmp segment %s: rows=%d bytes=%d seq=[%d,%d] elapsed_ms=%d",
            filename,
            sealed.table.num_rows,
            seg.size_bytes,
            sealed.min_seq,
            sealed.max_seq,
            int((time.monotonic() - write_start) * 1000),
        )

    def _compaction_step(self, *, compact_single: bool = False) -> None:
        with self._insertion_lock:
            tmps = [s for s in self._local_segments if _is_tmp_path(s.path)]
        if not tmps:
            return
        if len(tmps) < 2 and not compact_single:
            return

        tmps.sort(key=lambda s: s.min_seq)
        min_seq = tmps[0].min_seq
        max_seq = max(t.max_seq for t in tmps)
        merged_filename = _log_filename(min_seq)
        compaction_start = time.monotonic()

        merged_path = self._data_dir / merged_filename
        staging_path = merged_path.with_suffix(".parquet.tmp")
        sql = self._build_compaction_sql([Path(t.path) for t in tmps], staging_path)
        try:
            with self._insertion_lock:
                self._compaction_conn.execute(sql)
        except Exception:
            logger.warning("Compaction failed, leaving tmp segments in place", exc_info=True)
            staging_path.unlink(missing_ok=True)
            return
        merged_seg = LocalSegment(
            path=str(merged_path),
            size_bytes=staging_path.stat().st_size,
            min_seq=min_seq,
            max_seq=max_seq,
        )

        tmp_paths = {t.path for t in tmps}

        self._query_visibility_lock.write_acquire()
        try:
            staging_path.rename(merged_path)
            with self._insertion_lock:
                new_segments: deque[LocalSegment] = deque()
                merged_inserted = False
                for s in self._local_segments:
                    if s.path in tmp_paths:
                        if not merged_inserted:
                            new_segments.append(merged_seg)
                            merged_inserted = True
                    else:
                        new_segments.append(s)
                if not merged_inserted:
                    new_segments.append(merged_seg)
                self._local_segments = new_segments
            for t in tmps:
                try:
                    Path(t.path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to unlink tmp segment %s", t.path, exc_info=True)
        finally:
            self._query_visibility_lock.write_release()

        with self._compaction_generation_cond:
            self._compaction_generation += 1
            self._compaction_generation_cond.notify_all()

        logger.info(
            "Compacted %d tmp segments into %s: bytes=%d seq=[%d,%d] elapsed_ms=%d",
            len(tmps),
            merged_filename,
            merged_seg.size_bytes,
            min_seq,
            max_seq,
            int((time.monotonic() - compaction_start) * 1000),
        )
        self._offload_to_gcs(merged_filename, merged_path)
        self._evict_hook()

    def _build_compaction_sql(self, input_paths: list[Path], staging_path: Path) -> str:
        """Compose the COPY that merges ``input_paths`` to ``staging_path``.

        Columns missing from every input become ``NULL::TYPE AS name``.
        ``union_by_name=true`` fills NULL where a column is absent from
        individual segments.
        """
        present_columns = self._present_input_columns(input_paths)
        select_exprs: list[str] = []
        for col in self.schema.columns:
            ident = quote_ident(col.name)
            if col.name in present_columns:
                select_exprs.append(ident)
            else:
                select_exprs.append(f"NULL::{duckdb_type_for(col)} AS {ident}")

        # Self-generated paths from _tmp_filename — no SQL injection surface.
        paths_sql = ", ".join(quote_literal(str(p)) for p in input_paths)
        select_clause = ", ".join(select_exprs)
        order_clause = self._compaction_order_clause()
        return (
            f"COPY (SELECT {select_clause} "
            f"FROM read_parquet([{paths_sql}], union_by_name=true) "
            f"{order_clause}) "
            f"TO {quote_literal(str(staging_path))} "
            f"(FORMAT 'parquet', ROW_GROUP_SIZE {_ROW_GROUP_SIZE}, COMPRESSION 'zstd', COMPRESSION_LEVEL 1)"
        )

    def _present_input_columns(self, input_paths: list[Path]) -> set[str]:
        present: set[str] = set()
        for p in input_paths:
            schema = pq.read_schema(p)
            present.update(schema.names)
        return present

    def _compaction_order_clause(self) -> str:
        cols = ", ".join(quote_ident(name) for name in self._compaction_sort_keys())
        return f"ORDER BY {cols}"

    def sealed_segments(self) -> list[LocalSegment]:
        """Return flushed (logs_*) segments only, oldest first."""
        with self._insertion_lock:
            return [s for s in self._local_segments if not _is_tmp_path(s.path)]

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]:
        """Return all currently queryable local segments and RAM tables."""
        with self._insertion_lock:
            return list(self._local_segments), self._buffers.query_snapshot()

    def all_segments_unlocked(self) -> list[LocalSegment]:
        """Snapshot every locally-tracked segment. Caller MUST hold the insertion lock."""
        return list(self._local_segments)

    def evict_segment(self, path: str) -> int:
        """Remove ``path`` from tracking and unlink the file. Returns bytes freed."""
        with self._insertion_lock:
            new: deque[LocalSegment] = deque()
            removed_bytes = 0
            for s in self._local_segments:
                if s.path == path:
                    removed_bytes = s.size_bytes
                    continue
                new.append(s)
            self._local_segments = new
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

    def _offload_to_gcs(self, filename: str, filepath: Path) -> None:
        if not self._remote_log_dir:
            return

        remote_path = f"{self._remote_log_dir.rstrip('/')}/{self.name}/{filename}"
        upload_start = time.monotonic()
        try:
            with fsspec.core.open(str(filepath), "rb") as f_src, fsspec.core.open(remote_path, "wb") as f_dst:
                f_dst.write(f_src.read())
        except Exception:
            logger.warning("Failed to offload %s to GCS", filepath, exc_info=True)
            return
        logger.info(
            "Offloaded %s to %s: bytes=%d elapsed_ms=%d",
            filename,
            remote_path,
            filepath.stat().st_size,
            int((time.monotonic() - upload_start) * 1000),
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

        with self._read_pool.checkout(ram_tables) as (conn, ram_names):
            source = _build_union_source(parquet_files, ram_names, self._arrow_schema)
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
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        min_level_enum = str_to_log_level(min_level)
        is_pattern = bool(REGEX_META_RE.search(key))

        if not is_pattern:
            where_parts = ["key = $key", "seq > $cursor"]
            params: dict = {"key": key, "cursor": cursor}
            _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
            include_key_in_select = False
            exact_key: str | None = key
        else:
            where_parts, params = _regex_query(key, cursor)
            _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
            include_key_in_select = True
            exact_key = None

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

        with self._read_pool.checkout([table]) as (conn, ram_names):
            source = _build_union_source([], ram_names, self._arrow_schema)
            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            rows = conn.execute(sql, params).fetchall()

        return _shape_log_read_result(rows, tail, max_lines, cursor, include_key_in_select, exact_key)

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]:
        with self._insertion_lock:
            return [], [self._table]

    def sealed_segments(self) -> list[LocalSegment]:
        return []

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


class _ReadPoolProtocol(Protocol):
    def checkout(
        self, buffer_tables: list[pa.Table]
    ) -> AbstractContextManager[tuple[duckdb.DuckDBPyConnection, list[str]]]: ...


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


def _regex_literal_prefix(pattern: str) -> str:
    match = REGEX_META_RE.search(pattern)
    if match is None:
        return pattern
    return pattern[: match.start()]


def _regex_query(pattern: str, cursor: int) -> tuple[list[str], dict]:
    literal_prefix = _regex_literal_prefix(pattern)
    suffix = pattern[len(literal_prefix) :]
    is_pure_prefix = suffix in (".*", "")

    where_parts = ["seq > $cursor"]
    params: dict = {"cursor": cursor}

    if literal_prefix:
        where_parts.append("prefix(key, $prefix_lo)")
        params["prefix_lo"] = literal_prefix

    if not is_pure_prefix:
        where_parts.append("regexp_matches(key, $key_pattern)")
        params["key_pattern"] = pattern

    return where_parts, params


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
