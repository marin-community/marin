# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store backed by rotating RAM buffers + Parquet segments + DuckDB reads.

Lifecycle of a log entry:

    1. Appended to the *head* RAM buffer (a plain Python list).
    2. Every ``flush_interval_sec`` (default 10 min), the head buffer is
       *sealed* and a background thread flushes it to a local Parquet file.
       If the head buffer exceeds ``SEGMENT_TARGET_BYTES`` it is sealed
       immediately.
    3. When the background thread flushes a sealed buffer:
       a. If the newest local Parquet segment is small enough, it reads it,
          concatenates the new rows, and writes a replacement file. This
          keeps the file count low (no thousands of tiny files).
       b. If the newest segment is already large (>= ``SEGMENT_TARGET_BYTES``),
          a new Parquet file is created (named by the new buffer's min_seq).
       c. The sealed RAM buffer is removed (readers no longer need it).
       d. The new/updated file is copied to GCS (best-effort).
       e. GC drops oldest local segments past count/byte limits.

Read path: DuckDB ``read_parquet()`` over the snapshot of local Parquet files
UNION ALL in-memory pyarrow tables for each RAM buffer (head + sealed).

Locking:
    ``_lock``  --protects all mutable state (head buffer, sealed deque, local
    segments list, and the sequence counter). Held briefly for snapshots.

    ``_segments_rwlock`` --readers hold a *shared* read lock while DuckDB has
    parquet files open; GC holds the *exclusive* write lock before unlinking
    files. This prevents GC from deleting a file that an in-progress query
    still references.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock


import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from iris.cluster.log_store._types import _EST_BYTES_PER_ROW, REGEX_META_RE, LogReadResult
from iris.cluster.types import TaskAttempt
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

logger = logging.getLogger(__name__)

_PARQUET_SCHEMA = pa.schema(
    [
        ("seq", pa.int64()),
        ("key", pa.string()),
        ("source", pa.string()),
        ("data", pa.string()),
        ("epoch_ms", pa.int64()),
        ("level", pa.int32()),
    ]
)

# Arrow type → DuckDB type mapping for empty-source fallback SQL.
_DUCKDB_TYPE_MAP: dict[pa.DataType, str] = {
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.string(): "VARCHAR",
}

# ---------------------------------------------------------------------------
# Heuristic thresholds
# ---------------------------------------------------------------------------

# Target size for a single Parquet segment on disk. New data is concatenated
# onto the latest segment until it reaches this size, then a new file starts.
SEGMENT_TARGET_BYTES = 100 * 1024 * 1024  # 100 MB

# Seal the head buffer after this many seconds (even if small).
DEFAULT_FLUSH_INTERVAL_SEC = 600.0  # 10 minutes

# Default caps for local Parquet retention.
DEFAULT_MAX_LOCAL_SEGMENTS = 50
DEFAULT_MAX_LOCAL_BYTES = 5 * 1024**3  # 5 GB

_ROW_GROUP_SIZE = 16_384

# When doing a tail query, we optimistically restrict seq to the most recent
# (max_lines * margin) entries. If the result is short, we fall back to a
# full scan. 10x covers most realistic filter selectivities.
_TAIL_SEQ_MARGIN = 10


def _prefix_upper_bound(prefix: str) -> str | None:
    """Return the exclusive upper bound for a prefix range, or None if unbounded.

    All strings starting with ``prefix`` satisfy ``prefix <= s < upper``.
    This lets DuckDB use range predicates on Parquet row-group statistics
    instead of regexp_matches, which isn't pushed down through parameterized queries.
    """
    if not prefix:
        return None
    # Increment the last byte. Works for all realistic UTF-8 key prefixes.
    last = ord(prefix[-1])
    if last >= 0x10FFFF:
        return None
    return prefix[:-1] + chr(last + 1)


def _fsspec_copy(src: str, dst: str) -> None:
    """Copy a file using fsspec so either path can be remote (e.g. GCS)."""
    with fsspec.core.open(src, "rb") as f_src, fsspec.core.open(dst, "wb") as f_dst:
        f_dst.write(f_src.read())


def _read_seq_bounds(path: Path) -> tuple[int, int]:
    """Read min/max seq from Parquet row-group statistics."""
    try:
        meta = pq.read_metadata(path)
        schema = meta.schema.to_arrow_schema()
        seq_idx = schema.get_field_index("seq")
        min_seq = 0
        max_seq = 0
        for i in range(meta.num_row_groups):
            col = meta.row_group(i).column(seq_idx)
            if col.statistics is not None and col.statistics.has_min_max:
                if not min_seq or col.statistics.min < min_seq:
                    min_seq = col.statistics.min
                if col.statistics.max > max_seq:
                    max_seq = col.statistics.max
        return min_seq, max_seq
    except Exception:
        return 0, 0


def _recover_max_seq(log_dir: Path) -> int:
    """Recover the max sequence number by reading Parquet row-group statistics.

    Filenames are ``logs_{min_seq:019d}.parquet``.
    Returns max_seq + 1 so the counter can resume, or 1 if no files exist.
    """
    max_seen = -1
    for p in log_dir.glob("logs_*.parquet"):
        _, max_seq = _read_seq_bounds(p)
        if max_seq > max_seen:
            max_seen = max_seq
    return max_seen + 1 if max_seen >= 0 else 1


def _read_key_bounds(path: Path) -> tuple[str, str]:
    """Read min/max key from Parquet row-group statistics."""
    try:
        meta = pq.read_metadata(path)
        schema = meta.schema.to_arrow_schema()
        key_idx = schema.get_field_index("key")
        min_key = ""
        max_key = ""
        for i in range(meta.num_row_groups):
            col = meta.row_group(i).column(key_idx)
            if col.statistics is not None and col.statistics.has_min_max:
                rg_min = col.statistics.min
                rg_max = col.statistics.max
                if not min_key or rg_min < min_key:
                    min_key = rg_min
                if not max_key or rg_max > max_key:
                    max_key = rg_max
        return min_key, max_key
    except Exception:
        return "", ""


def _segment_overlaps_key(seg: _LocalSegment, key: str) -> bool:
    """Check if a segment's key range could contain the exact key."""
    if not seg.min_key:
        return True  # no metadata, must scan
    return seg.min_key <= key <= seg.max_key


def _segment_overlaps_prefix(seg: _LocalSegment, prefix: str) -> bool:
    """Check if a segment's key range could contain keys starting with prefix.

    A prefix like "/user/job-50/" overlaps a segment if:
    - segment's max_key >= prefix (some key could start at or after prefix)
    - segment's min_key starts with prefix, OR min_key < prefix_upper_bound
    """
    if not seg.min_key:
        return True  # no metadata, must scan
    # Fast check: segment ends before prefix starts
    if seg.max_key < prefix:
        return False
    # Compute the exclusive upper bound for the prefix by incrementing the last
    # character. All strings matching the prefix satisfy: prefix <= s < upper.
    # If we can't compute an upper bound (empty prefix or all-\xff), scan.
    if not prefix:
        return True
    upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    if seg.min_key >= upper:
        return False
    return True


def _build_buffer_table(buffer: list[tuple]) -> pa.Table:
    """Convert a list of row tuples into a pyarrow Table with the log schema."""
    if not buffer:
        return _PARQUET_SCHEMA.empty_table()
    cols: list[list] = [[] for _ in range(6)]
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
    return pa.table(arrays, schema=_PARQUET_SCHEMA)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


_CHUNK_THRESHOLD = 1024

# Minimum number of rows in a chunk before we merge it with another of the
# same size (power-of-2 compaction).  Chunks below this size sit in the list
# until the next append pushes a new chunk that triggers merging.
_MIN_MERGE_ROWS = _CHUNK_THRESHOLD


def _merge_chunks(chunks: list[pa.Table]) -> list[pa.Table]:
    """Compact the chunk list by merging adjacent same-order-of-magnitude tables.

    Maintains the invariant: each chunk is at least 2x the size of the
    previous one (like a log-structured merge). This keeps len(chunks)
    logarithmic in total row count.
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
    """A RAM buffer that has been sealed (no more writes) and is pending flush."""

    table: pa.Table
    min_seq: int
    max_seq: int
    flushed: bool = False


@dataclass
class _LocalSegment:
    """Metadata for a Parquet file on local disk."""

    path: str
    size_bytes: int
    min_seq: int = 0
    max_seq: int = 0
    min_key: str = ""
    max_key: str = ""


@dataclass(frozen=True)
class _SegmentFilter:
    """Filters segments by key bounds before building a DuckDB query."""

    exact_key: str | None = None
    prefix: str | None = None

    def apply(self, segments: list[_LocalSegment]) -> list[_LocalSegment]:
        if self.exact_key is not None:
            return [s for s in segments if _segment_overlaps_key(s, self.exact_key)]
        if self.prefix is not None:
            return [s for s in segments if _segment_overlaps_prefix(s, self.prefix)]
        return segments


# Sentinel: include all segments (no filtering).
_SEGMENT_FILTER_ALL = _SegmentFilter()


class _RWLock:
    """Simple readers-writer lock.

    Multiple readers can hold the lock concurrently. A writer must wait for
    all readers to release before acquiring exclusive access. Used to prevent
    GC from unlinking parquet files while DuckDB reads are in flight.
    """

    def __init__(self):
        self._cond = Condition(Lock())
        self._readers = 0
        self._writer = False

    def read_acquire(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def read_release(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self) -> None:
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def write_release(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


_DEFAULT_DUCKDB_MEMORY_LIMIT = "256MB"


_cursor_counter = 0
_cursor_counter_lock = Lock()


def _next_cursor_id() -> int:
    global _cursor_counter
    with _cursor_counter_lock:
        _cursor_counter += 1
        return _cursor_counter


class _ConnectionPool:
    """Single DuckDB database with cursor-based concurrency.

    One ``duckdb.connect()`` call creates the shared buffer pool. Callers
    get cursors via ``conn.cursor()`` which share that pool, keeping total
    memory bounded by a single ``memory_limit``.

    RAM tables are registered with unique names (incorporating a monotonic
    counter) so concurrent cursors don't collide on table names.
    """

    def __init__(self, memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT):
        self._conn = duckdb.connect(config={"memory_limit": memory_limit, "threads": "2"})

    @contextmanager
    def checkout(self, buffer_tables: list[pa.Table]) -> Iterator[tuple[duckdb.DuckDBPyConnection, list[str]]]:
        """Create a cursor and register each RAM table under a unique name.

        Yields ``(cursor, list_of_table_names)`` so callers can UNION ALL
        the names into their SQL without a ``pa.concat_tables`` copy.
        """
        cid = _next_cursor_id()
        cursor = self._conn.cursor()
        names: list[str] = []
        try:
            for i, table in enumerate(buffer_tables):
                name = f"_ram_{cid}_{i}"
                cursor.register(name, table)
                names.append(name)
            yield cursor, names
        finally:
            for name in names:
                cursor.unregister(name)
            cursor.close()

    def close(self) -> None:
        self._conn.close()


class DuckDBLogStore:
    """Log store backed by rotating RAM buffers + Parquet segments.

    Thread-safe. One lock protects all mutable state: the head buffer,
    the sealed-buffer deque, and the local-segment deque.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        max_local_segments: int = DEFAULT_MAX_LOCAL_SEGMENTS,
        max_local_bytes: int = DEFAULT_MAX_LOCAL_BYTES,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        segment_target_bytes: int = SEGMENT_TARGET_BYTES,
        duckdb_memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT,
    ):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if log_dir is not None:
            self._log_dir = log_dir
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            self._log_dir = Path(self._temp_dir.name) / "parquet_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._remote_log_dir = remote_log_dir
        self._max_local_segments = max_local_segments
        self._max_local_bytes = max_local_bytes
        self._flush_interval_sec = flush_interval_sec
        self._segment_target_bytes = segment_target_bytes

        # ---- shared mutable state (all guarded by _lock) ----
        self._lock = Lock()
        self._next_seq = _recover_max_seq(self._log_dir)  # guarded by _lock
        self._pending: list[tuple] = []  # hot write list, converted to arrow at _CHUNK_THRESHOLD
        self._chunks: list[pa.Table] = []  # power-of-2 merged arrow tables
        self._sealed: deque[_SealedBuffer] = deque()  # sealed, pending flush
        self._local_segments: deque[_LocalSegment] = deque()  # flushed parquet files
        self._last_flush_time = time.monotonic()

        # RWLock: readers hold shared lock during DuckDB queries;
        # GC holds exclusive lock before unlinking files.
        self._segments_rwlock = _RWLock()

        # Discover pre-existing Parquet files from a previous run.
        for p in sorted(self._log_dir.glob("logs_*.parquet")):
            min_seq, max_seq = _read_seq_bounds(p)
            min_key, max_key = _read_key_bounds(p)
            self._local_segments.append(
                _LocalSegment(
                    path=str(p),
                    size_bytes=p.stat().st_size,
                    min_seq=min_seq,
                    max_seq=max_seq,
                    min_key=min_key,
                    max_key=max_key,
                )
            )

        self._pool = _ConnectionPool(memory_limit=duckdb_memory_limit)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        with self._lock:
            first_seq = self._next_seq
            self._next_seq += len(entries)
            rows = [(first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)]
            self._pending.extend(rows)
            self._maybe_compact_pending()
        self._maybe_seal()

    def append_batch(self, items: list[tuple[str, list]]) -> None:
        """Write log entries from multiple keys in a single operation."""
        with self._lock:
            all_rows: list[tuple] = []
            for key, entries in items:
                if not entries:
                    continue
                first_seq = self._next_seq
                self._next_seq += len(entries)
                all_rows.extend(
                    (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
                )
            if not all_rows:
                return
            self._pending.extend(all_rows)
            self._maybe_compact_pending()
        self._maybe_seal()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

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
        """Fetch logs for a key or regex pattern.

        If the key contains regex metacharacters, it is interpreted as a
        regular expression and matched with DuckDB's ``regexp_matches()``.
        Otherwise it is treated as an exact key lookup.
        """
        min_level_enum = str_to_log_level(min_level) if min_level else 0
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
                segment_filter=_SegmentFilter(exact_key=key),
                exact_key=key,
            )

        # Regex pattern path.
        where_parts, params, segment_filter = _regex_query(key, cursor)
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
        return self._execute_read(
            where_parts,
            params,
            max_lines,
            tail,
            cursor,
            include_key_in_select=True,
            segment_filter=segment_filter,
        )

    def has_logs(self, key: str) -> bool:
        """Check whether any log entries exist for the given key."""
        result = self.get_logs(key, max_lines=1)
        return len(result.entries) > 0

    def cursor(self, key: str):
        """Return a stateful cursor for incremental reads on *key*."""
        from iris.cluster.log_store import LogCursor

        return LogCursor(self, key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining buffer, shut down background executor, clean up temp dir."""
        self._seal_head()
        self._executor.shutdown(wait=True)
        # Flush any sealed buffers that didn't get submitted (edge case).
        with self._lock:
            remaining = list(self._sealed)
        for sb in remaining:
            if not sb.flushed:
                self._flush_sealed_buffer(sb)
        self._pool.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    # ------------------------------------------------------------------
    # Internal: seal and flush
    # ------------------------------------------------------------------

    def _maybe_compact_pending(self) -> None:
        """Convert _pending to an arrow chunk when it exceeds _CHUNK_THRESHOLD. Must hold _lock."""
        if len(self._pending) < _CHUNK_THRESHOLD:
            return
        self._chunks.append(_build_buffer_table(self._pending))
        self._pending = []
        self._chunks = _merge_chunks(self._chunks)

    def _head_row_count(self) -> int:
        """Total rows in the head buffer (chunks + pending). Must hold _lock."""
        return sum(c.num_rows for c in self._chunks) + len(self._pending)

    def _maybe_seal(self) -> None:
        """Seal the head buffer if it's big enough or old enough."""
        with self._lock:
            est_size = self._head_row_count() * _EST_BYTES_PER_ROW
            elapsed = time.monotonic() - self._last_flush_time
            should_seal = est_size >= self._segment_target_bytes or (
                self._head_row_count() > 0 and elapsed >= self._flush_interval_sec
            )
        if should_seal:
            self._seal_head()

    def _seal_head(self) -> None:
        """Move the head buffer to the sealed deque and submit a flush task."""
        with self._lock:
            if not self._chunks and not self._pending:
                return
            tables = list(self._chunks)
            if self._pending:
                tables.append(_build_buffer_table(self._pending))
            table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

            seq_col = table.column("seq")
            sealed = _SealedBuffer(
                table=table,
                min_seq=pc.min(seq_col).as_py(),
                max_seq=pc.max(seq_col).as_py(),
            )
            self._chunks = []
            self._pending = []
            self._sealed.append(sealed)
            self._last_flush_time = time.monotonic()
        logger.info(
            "Sealed head buffer: rows=%d seq=[%d,%d] est_bytes=%d",
            sealed.table.num_rows,
            sealed.min_seq,
            sealed.max_seq,
            sealed.table.num_rows * _EST_BYTES_PER_ROW,
        )
        self._executor.submit(self._flush_sealed_buffer, sealed)

    def _flush_sealed_buffer(self, sealed: _SealedBuffer) -> None:
        """Write a sealed buffer to Parquet, possibly consolidating with the
        latest segment if it's small. Runs on the background executor thread.

        Consolidation heuristic: if the newest local segment is smaller than
        ``_segment_target_bytes``, read it, concatenate the new rows, and
        write a replacement file. This avoids accumulating thousands of tiny
        Parquet files when log volume is low.
        """
        if sealed.table.num_rows == 0:
            return

        new_min_seq = sealed.min_seq
        new_max_seq = sealed.max_seq
        # Sort by (key, seq) so row-group statistics on `key` are tight,
        # enabling DuckDB to skip row groups that don't contain the target key.
        new_table = sealed.table.sort_by([("key", "ascending"), ("seq", "ascending")])

        # Decide whether to consolidate with the latest segment.
        with self._lock:
            latest = self._local_segments[-1] if self._local_segments else None
            can_consolidate = latest is not None and latest.size_bytes < self._segment_target_bytes

        if can_consolidate:
            assert latest is not None
            try:
                consolidate_start = time.monotonic()
                existing_table = pq.read_table(latest.path)
                combined = pa.concat_tables([existing_table, new_table])
                combined = combined.sort_by([("key", "ascending"), ("seq", "ascending")])
                combined_min_seq = latest.min_seq
                combined_max_seq = new_max_seq
                # Reuse the same filename (keyed only on min_seq) so the GCS
                # upload overwrites the old object in place — no deletion needed.
                filename = f"logs_{combined_min_seq:019d}.parquet"
                filepath = self._log_dir / filename

                # Write combined data to a temp file. Then hold the write lock
                # across both the rename and the segment-list update so that no
                # reader can snapshot a state where the file has the combined
                # content AND the sealed buffer is still in RAM tables (which
                # would cause double-counting).
                tmp_path = filepath.with_suffix(".parquet.tmp")
                pq.write_table(
                    combined, tmp_path, compression="zstd", row_group_size=_ROW_GROUP_SIZE, write_page_index=True
                )
                key_col = combined.column("key")
                seg = _LocalSegment(
                    path=str(filepath),
                    size_bytes=tmp_path.stat().st_size,
                    min_seq=combined_min_seq,
                    max_seq=combined_max_seq,
                    min_key=key_col[0].as_py(),
                    max_key=key_col[-1].as_py(),
                )

                self._segments_rwlock.write_acquire()
                try:
                    tmp_path.rename(filepath)
                    with self._lock:
                        for i, s in enumerate(self._local_segments):
                            if s.path == latest.path:
                                del self._local_segments[i]
                                break
                        self._local_segments.append(seg)
                        try:
                            self._sealed.remove(sealed)
                        except ValueError:
                            pass
                        sealed.flushed = True
                finally:
                    self._segments_rwlock.write_release()

                logger.info(
                    "Wrote consolidated segment %s: rows=%d (added=%d) bytes=%d seq=[%d,%d] elapsed_ms=%d",
                    filename,
                    combined.num_rows,
                    new_table.num_rows,
                    seg.size_bytes,
                    combined_min_seq,
                    combined_max_seq,
                    int((time.monotonic() - consolidate_start) * 1000),
                )

                # GCS upload overwrites the same object (same filename).
                self._offload_to_gcs(filename, filepath)
                self._gc_local_segments()
                return

            except Exception:
                logger.warning("Consolidation failed, writing as new segment", exc_info=True)
                # Fall through to write as a new standalone segment.

        # Write as a new standalone segment.
        filename = f"logs_{new_min_seq:019d}.parquet"
        filepath = self._log_dir / filename

        write_start = time.monotonic()
        try:
            tmp_path = filepath.with_suffix(".parquet.tmp")
            pq.write_table(
                new_table, tmp_path, compression="zstd", row_group_size=_ROW_GROUP_SIZE, write_page_index=True
            )
            tmp_path.rename(filepath)
        except Exception:
            logger.warning("Failed to write Parquet segment %s", filepath, exc_info=True)
            # Leave the sealed buffer in the deque so reads still see the data.
            return

        key_col = new_table.column("key")
        seg = _LocalSegment(
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            min_seq=new_min_seq,
            max_seq=new_max_seq,
            min_key=key_col[0].as_py(),
            max_key=key_col[-1].as_py(),
        )

        with self._lock:
            self._local_segments.append(seg)
            try:
                self._sealed.remove(sealed)
            except ValueError:
                pass
            sealed.flushed = True

        logger.info(
            "Wrote segment %s: rows=%d bytes=%d seq=[%d,%d] elapsed_ms=%d",
            filename,
            new_table.num_rows,
            seg.size_bytes,
            new_min_seq,
            new_max_seq,
            int((time.monotonic() - write_start) * 1000),
        )

        self._offload_to_gcs(filename, filepath)
        self._gc_local_segments()

    def _gc_local_segments(self) -> None:
        """Drop oldest local Parquet segments if count or size exceeds limits.

        Takes the _segments_rwlock exclusively before unlinking files so that
        in-progress DuckDB reads (which hold the shared read lock) are not
        disrupted by file deletion.
        """
        with self._lock:
            total_bytes = sum(s.size_bytes for s in self._local_segments)
            to_delete: list[tuple[str, int]] = []
            remaining_count = len(self._local_segments)
            remaining_bytes = total_bytes

            while self._local_segments and (
                len(self._local_segments) > self._max_local_segments or total_bytes > self._max_local_bytes
            ):
                oldest = self._local_segments.popleft()
                total_bytes -= oldest.size_bytes
                to_delete.append((oldest.path, oldest.size_bytes))
                remaining_count -= 1
                remaining_bytes -= oldest.size_bytes

        if not to_delete:
            return

        # Hold the write lock while deleting local files so concurrent reads
        # (which hold the read lock) finish before we unlink anything.
        self._segments_rwlock.write_acquire()
        try:
            for path, _ in to_delete:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to delete old segment %s", path, exc_info=True)
        finally:
            self._segments_rwlock.write_release()

        logger.info(
            "GC'd %d local segment(s), freed=%d bytes, remaining=%d segments / %d bytes",
            len(to_delete),
            sum(b for _, b in to_delete),
            remaining_count,
            remaining_bytes,
        )

    def _drop_missing_local_segments(self, paths: list[str]) -> list[str]:
        """Filter ``paths`` to those that still exist on disk.

        If any are missing they're also pruned from ``_local_segments`` so
        future reads don't repeatedly hit the same DuckDB error. Vanishing
        files normally come from out-of-band deletion (manual ``rm``, disk
        pressure outside our GC, leftover entries from an old filename
        format) — anything our own GC removes is gone from the in-memory
        list before the file is unlinked.
        """
        existing: list[str] = []
        missing: list[str] = []
        for p in paths:
            if Path(p).exists():
                existing.append(p)
            else:
                missing.append(p)
        if not missing:
            return existing

        missing_set = set(missing)
        with self._lock:
            self._local_segments = deque(s for s in self._local_segments if s.path not in missing_set)
        logger.warning(
            "Pruned %d missing local segment(s) from in-memory index (e.g. %s)",
            len(missing),
            missing[:3],
        )
        return existing

    def _offload_to_gcs(self, filename: str, filepath: Path) -> None:
        """Copy a Parquet file to GCS (best-effort)."""
        if not self._remote_log_dir:
            return
        remote_path = f"{self._remote_log_dir.rstrip('/')}/{filename}"
        upload_start = time.monotonic()
        try:
            _fsspec_copy(str(filepath), remote_path)
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

    # ------------------------------------------------------------------
    # Internal: read
    # ------------------------------------------------------------------

    def _execute_read(
        self,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        default_cursor: int,
        include_key_in_select: bool,
        segment_filter: _SegmentFilter = _SEGMENT_FILTER_ALL,
        exact_key: str | None = None,
    ) -> LogReadResult:
        # Acquire the segments read lock BEFORE snapshotting paths. This
        # guarantees that no file in our snapshot can be deleted (by GC or
        # consolidation) until we release the lock after DuckDB is done.
        self._segments_rwlock.read_acquire()
        try:
            with self._lock:
                segments = list(self._local_segments)
                ram_tables: list[pa.Table] = [sb.table for sb in self._sealed]
                ram_tables.extend(self._chunks)
                if self._pending:
                    ram_tables.append(_build_buffer_table(self._pending))
                current_max_seq = self._next_seq - 1

            parquet_files = [s.path for s in segment_filter.apply(segments)]
            parquet_files = self._drop_missing_local_segments(parquet_files)

            where_clause = " AND ".join(where_parts)

            if include_key_in_select:
                select_cols = "seq, key, source, data, epoch_ms, level"
            else:
                select_cols = "seq, source, data, epoch_ms, level"

            order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
            limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

            # Register each RAM table individually so DuckDB scans them
            # via zero-copy Arrow, avoiding a pa.concat_tables() copy.
            with self._pool.checkout(ram_tables) as (conn, ram_names):
                source = _build_union_source(parquet_files, ram_names)

                # For tail+prefix queries, try a seq-bounded scan first. The
                # last max_lines rows must have seq within a bounded range of
                # the global max. Use a generous margin and fall back to a
                # full scan if we get fewer rows than requested. Skip this
                # for exact-key queries since row-group pruning already
                # narrows the scan sufficiently.
                rows = None
                if tail and max_lines > 0 and include_key_in_select:
                    seq_lower = max(0, current_max_seq - max_lines * _TAIL_SEQ_MARGIN)
                    bounded_where = f"{where_clause} AND seq > {seq_lower}"
                    sql = f"SELECT {select_cols} FROM ({source}) WHERE {bounded_where} {order} {limit}"
                    rows = conn.execute(sql, params).fetchall()
                    if len(rows) < max_lines:
                        rows = None  # margin was too tight, fall back

                if rows is None:
                    sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
                    rows = conn.execute(sql, params).fetchall()
        finally:
            self._segments_rwlock.read_release()

        if tail and max_lines > 0:
            rows.reverse()

        if not rows:
            return LogReadResult(entries=[], cursor=default_cursor)

        max_seq = max(r[0] for r in rows)

        if include_key_in_select:
            entries = []
            for r in rows:
                # r: (seq, key, source, data, epoch_ms, level)
                parsed = TaskAttempt.from_wire(r[1])
                entry = logging_pb2.LogEntry(source=r[2], data=r[3], level=r[5])
                entry.timestamp.epoch_ms = r[4]
                entry.key = r[1]
                entry.attempt_id = parsed.attempt_id if parsed.attempt_id is not None else 0
                entries.append(entry)
        else:
            entries = []
            # Parse attempt_id from the exact key once for all entries.
            attempt_id = 0
            if exact_key and ":" in exact_key:
                try:
                    parsed = TaskAttempt.from_wire(exact_key)
                    attempt_id = parsed.attempt_id if parsed.attempt_id is not None else 0
                except ValueError:
                    pass
            for r in rows:
                # r: (seq, source, data, epoch_ms, level)
                entry = logging_pb2.LogEntry(source=r[1], data=r[2], level=r[4])
                entry.timestamp.epoch_ms = r[3]
                entry.attempt_id = attempt_id
                entries.append(entry)

        return LogReadResult(entries=entries, cursor=max_seq)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regex_literal_prefix(pattern: str) -> str:
    """Extract the literal prefix from a regex pattern.

    Returns the leading portion of *pattern* that contains no regex
    metacharacters, so it can be used for Parquet range pushdown.
    """
    match = REGEX_META_RE.search(pattern)
    if match is None:
        return pattern
    return pattern[: match.start()]


def _regex_query(pattern: str, cursor: int) -> tuple[list[str], dict, _SegmentFilter]:
    """Build WHERE clauses and segment filter for a regex pattern.

    Extracts a literal prefix for Parquet range pushdown. If the pattern
    is purely a prefix (literal text followed only by ``.*``), range
    predicates alone suffice. Otherwise ``regexp_matches()`` is added.
    """
    literal_prefix = _regex_literal_prefix(pattern)

    # Pure-prefix pattern: "some/prefix/.*" — the literal prefix covers
    # everything and the trailing .* matches any suffix.
    suffix = pattern[len(literal_prefix) :]
    is_pure_prefix = suffix in (".*", "")

    if is_pure_prefix and literal_prefix:
        where_parts = ["key >= $prefix_lo", "seq > $cursor"]
        params: dict = {"prefix_lo": literal_prefix, "cursor": cursor}
        upper = _prefix_upper_bound(literal_prefix)
        if upper is not None:
            where_parts.append("key < $prefix_hi")
            params["prefix_hi"] = upper
        return where_parts, params, _SegmentFilter(prefix=literal_prefix)

    # General regex pattern — use regexp_matches with optional prefix pushdown.
    where_parts = ["regexp_matches(key, $key_pattern)", "seq > $cursor"]
    params = {"key_pattern": pattern, "cursor": cursor}
    if literal_prefix:
        upper = _prefix_upper_bound(literal_prefix)
        where_parts.append("key >= $prefix_lo")
        params["prefix_lo"] = literal_prefix
        if upper is not None:
            where_parts.append("key < $prefix_hi")
            params["prefix_hi"] = upper
        segment_filter = _SegmentFilter(prefix=literal_prefix)
    else:
        segment_filter = _SEGMENT_FILTER_ALL
    return where_parts, params, segment_filter


def _add_common_filters(
    where_parts: list[str],
    params: dict,
    since_ms: int,
    substring_filter: str,
    min_level_enum: int,
) -> None:
    """Append shared WHERE clauses for since_ms, substring, and min_level."""
    if since_ms > 0:
        where_parts.append("epoch_ms > $since_ms")
        params["since_ms"] = since_ms
    if substring_filter:
        where_parts.append("contains(data, $substring)")
        params["substring"] = substring_filter
    if min_level_enum > 0:
        where_parts.append("(level = 0 OR level >= $min_level)")
        params["min_level"] = min_level_enum


def _build_union_source(parquet_files: list[str], ram_table_names: list[str]) -> str:
    """Build a SQL source expression: local Parquet files UNION ALL ram tables.

    File paths are self-generated (``logs_{seq}_{seq}.parquet``) so no SQL
    injection risk from the f-string embedding. RAM table names are generated
    internally (``_ram_0``, ``_ram_1``, …).
    """
    parts: list[str] = []
    if parquet_files:
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        parts.append(f"SELECT * FROM read_parquet([{file_list}])")
    for name in ram_table_names:
        parts.append(f"SELECT * FROM {name}")
    if not parts:
        # No data sources at all; return an empty selection from the schema.
        col_defs = ", ".join(f"NULL::{_DUCKDB_TYPE_MAP[f.type]} AS {f.name}" for f in _PARQUET_SCHEMA)
        return f"SELECT {col_defs} WHERE false"
    return " UNION ALL ".join(parts)
