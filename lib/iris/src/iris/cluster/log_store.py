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
          a new Parquet file is created.
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
import queue
import tempfile
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock

import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq

from iris.cluster.types import TaskAttempt
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

logger = logging.getLogger(__name__)

_LIKE_ESCAPE_TABLE = str.maketrans({"%": "\\%", "_": "\\_", "\\": "\\\\"})

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

# ---------------------------------------------------------------------------
# Heuristic thresholds
# ---------------------------------------------------------------------------

# Target size for a single Parquet segment on disk. New data is concatenated
# onto the latest segment until it reaches this size, then a new file starts.
SEGMENT_TARGET_BYTES = 100 * 1024 * 1024  # 100 MB

# Seal the head buffer after this many seconds (even if small).
DEFAULT_FLUSH_INTERVAL_SEC = 600.0  # 10 minutes

# Estimated bytes per buffered row (used to decide when to force-seal).
_EST_BYTES_PER_ROW = 256

# Default caps for local Parquet retention.
DEFAULT_MAX_LOCAL_SEGMENTS = 50
DEFAULT_MAX_LOCAL_BYTES = 5 * 1024**3  # 5 GB

_ROW_GROUP_SIZE = 16_384


def _escape_like(s: str) -> str:
    """Escape SQL LIKE wildcards so the string matches literally."""
    return s.translate(_LIKE_ESCAPE_TABLE)


PROCESS_LOG_KEY = "/system/process"


def task_log_key(task_attempt: TaskAttempt) -> str:
    """Build a hierarchical key for task attempt logs."""
    task_attempt.require_attempt()
    return task_attempt.to_wire()


def _fsspec_copy(src: str, dst: str) -> None:
    """Copy a file using fsspec so either path can be remote (e.g. GCS)."""
    with fsspec.core.open(src, "rb") as f_src, fsspec.core.open(dst, "wb") as f_dst:
        f_dst.write(f_src.read())


def _recover_max_seq(log_dir: Path) -> int:
    """Parse the max sequence number from existing Parquet segment filenames.

    Filenames are ``logs_{min_seq:019d}_{max_seq:019d}.parquet``.
    Returns max_seq + 1 so the counter can resume, or 1 if no files exist.
    """
    max_seen = -1
    for p in log_dir.glob("logs_*_*.parquet"):
        parts = p.stem.split("_")
        if len(parts) >= 3:
            try:
                max_seen = max(max_seen, int(parts[2]))
            except ValueError:
                continue
    return max_seen + 1 if max_seen >= 0 else 1


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


@dataclass
class _SealedBuffer:
    """A RAM buffer that has been sealed (no more writes) and is pending flush."""

    rows: list[tuple]
    # Set once the Parquet file is written and registered.
    flushed: bool = False


@dataclass
class _LocalSegment:
    """Metadata for a Parquet file on local disk."""

    path: str
    size_bytes: int
    min_seq: int = 0
    max_seq: int = 0


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


class _ConnectionPool:
    """Pool of reusable DuckDB connections.

    Each connection is checked out by exactly one thread at a time.
    The buffer table is registered per-use since it changes on every read.
    """

    def __init__(self, size: int = 8):
        self._pool: queue.SimpleQueue[duckdb.DuckDBPyConnection] = queue.SimpleQueue()
        for _ in range(size):
            self._pool.put(duckdb.connect())

    @contextmanager
    def checkout(self, buffer_table: pa.Table) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = self._pool.get()
        try:
            conn.register("ram_buffer", buffer_table)
            yield conn
        finally:
            conn.unregister("ram_buffer")
            self._pool.put(conn)

    def close(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry] = field(default_factory=list)
    cursor: int = 0  # max seq seen


class LogStore:
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
        pool_size: int = 8,
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
        self._head: list[tuple] = []  # current write buffer
        self._sealed: deque[_SealedBuffer] = deque()  # sealed, pending flush
        self._local_segments: deque[_LocalSegment] = deque()  # flushed parquet files
        self._last_flush_time = time.monotonic()

        # RWLock: readers hold shared lock during DuckDB queries;
        # GC holds exclusive lock before unlinking files.
        self._segments_rwlock = _RWLock()

        # Discover pre-existing Parquet files from a previous run.
        for p in sorted(self._log_dir.glob("logs_*_*.parquet")):
            parts = p.stem.split("_")
            min_seq = int(parts[1]) if len(parts) >= 3 else 0
            max_seq = int(parts[2]) if len(parts) >= 3 else 0
            self._local_segments.append(
                _LocalSegment(path=str(p), size_bytes=p.stat().st_size, min_seq=min_seq, max_seq=max_seq)
            )

        self._pool = _ConnectionPool(size=pool_size)
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
            self._head.extend(rows)
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
            self._head.extend(all_rows)
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
        """Fetch logs for a single key."""
        min_level_enum = str_to_log_level(min_level) if min_level else 0

        where_parts = ["key = $key", "seq > $cursor"]
        params: dict = {"key": key, "cursor": cursor}
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)

        return self._execute_read(where_parts, params, max_lines, tail, cursor, include_key_in_select=False)

    def get_logs_by_prefix(
        self,
        prefix: str,
        *,
        cursor: int = 0,
        since_ms: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
        shallow: bool = False,
    ) -> LogReadResult:
        """Fetch logs for all keys matching prefix, ordered by seq."""
        min_level_enum = str_to_log_level(min_level) if min_level else 0

        where_parts = ["key LIKE $prefix ESCAPE '\\'", "seq > $cursor"]
        params: dict = {"prefix": _escape_like(prefix) + "%", "cursor": cursor}
        if shallow:
            where_parts.append("key NOT LIKE $shallow_exclude ESCAPE '\\'")
            params["shallow_exclude"] = _escape_like(prefix) + "%/%"
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)

        return self._execute_read(where_parts, params, max_lines, tail, cursor, include_key_in_select=True)

    def has_logs(self, key: str) -> bool:
        """Check whether any log entries exist for the given key."""
        result = self.get_logs(key, max_lines=1)
        return len(result.entries) > 0

    def cursor(self, key: str) -> LogCursor:
        """Return a stateful cursor for incremental reads on *key*."""
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

    def _maybe_seal(self) -> None:
        """Seal the head buffer if it's big enough or old enough."""
        with self._lock:
            est_size = len(self._head) * _EST_BYTES_PER_ROW
            elapsed = time.monotonic() - self._last_flush_time
            should_seal = est_size >= self._segment_target_bytes or (
                len(self._head) > 0 and elapsed >= self._flush_interval_sec
            )
        if should_seal:
            self._seal_head()

    def _seal_head(self) -> None:
        """Move the head buffer to the sealed deque and submit a flush task."""
        with self._lock:
            if not self._head:
                return
            sealed = _SealedBuffer(rows=self._head)
            self._head = []
            self._sealed.append(sealed)
            self._last_flush_time = time.monotonic()
        self._executor.submit(self._flush_sealed_buffer, sealed)

    def _flush_sealed_buffer(self, sealed: _SealedBuffer) -> None:
        """Write a sealed buffer to Parquet, possibly consolidating with the
        latest segment if it's small. Runs on the background executor thread.

        Consolidation heuristic: if the newest local segment is smaller than
        ``_segment_target_bytes``, read it, concatenate the new rows, and
        write a replacement file. This avoids accumulating thousands of tiny
        Parquet files when log volume is low.
        """
        rows = sealed.rows
        if not rows:
            return

        # Sort by (key, seq) so row-group statistics on `key` are tight,
        # enabling DuckDB to skip row groups that don't contain the target key.
        new_min_seq = min(r[0] for r in rows)
        new_max_seq = max(r[0] for r in rows)
        rows = sorted(rows, key=lambda r: (r[1], r[0]))
        new_table = _build_buffer_table(rows)

        # Decide whether to consolidate with the latest segment.
        with self._lock:
            latest = self._local_segments[-1] if self._local_segments else None
            can_consolidate = latest is not None and latest.size_bytes < self._segment_target_bytes

        if can_consolidate:
            assert latest is not None
            try:
                existing_table = pq.read_table(latest.path)
                combined = pa.concat_tables([existing_table, new_table])
                combined = combined.sort_by([("key", "ascending"), ("seq", "ascending")])
                combined_min_seq = latest.min_seq
                combined_max_seq = new_max_seq
                filename = f"logs_{combined_min_seq:019d}_{combined_max_seq:019d}.parquet"
                filepath = self._log_dir / filename

                # Write to a temp file first, then rename for atomicity.
                tmp_path = filepath.with_suffix(".parquet.tmp")
                pq.write_table(
                    combined, tmp_path, compression="zstd", row_group_size=_ROW_GROUP_SIZE, write_page_index=True
                )
                tmp_path.rename(filepath)

                seg = _LocalSegment(
                    path=str(filepath),
                    size_bytes=filepath.stat().st_size,
                    min_seq=combined_min_seq,
                    max_seq=combined_max_seq,
                )

                with self._lock:
                    # Replace the old segment with the consolidated one.
                    try:
                        # Find and remove the old segment from the deque.
                        for i, s in enumerate(self._local_segments):
                            if s.path == latest.path:
                                del self._local_segments[i]
                                break
                    except (ValueError, IndexError):
                        pass
                    self._local_segments.append(seg)
                    try:
                        self._sealed.remove(sealed)
                    except ValueError:
                        pass
                    sealed.flushed = True

                # Delete the old segment file (now replaced). Hold the write lock
                # so concurrent reads that snapshotted the old path finish first.
                if str(filepath) != latest.path:
                    self._segments_rwlock.write_acquire()
                    try:
                        Path(latest.path).unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Failed to delete old segment %s", latest.path, exc_info=True)
                    finally:
                        self._segments_rwlock.write_release()

                # GCS offload for the consolidated file.
                self._offload_to_gcs(filename, filepath)
                self._gc_local_segments()
                return

            except Exception:
                logger.warning("Consolidation failed, writing as new segment", exc_info=True)
                # Fall through to write as a new standalone segment.

        # Write as a new standalone segment.
        filename = f"logs_{new_min_seq:019d}_{new_max_seq:019d}.parquet"
        filepath = self._log_dir / filename

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

        seg = _LocalSegment(
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            min_seq=new_min_seq,
            max_seq=new_max_seq,
        )

        with self._lock:
            self._local_segments.append(seg)
            try:
                self._sealed.remove(sealed)
            except ValueError:
                pass
            sealed.flushed = True

        self._offload_to_gcs(filename, filepath)
        self._gc_local_segments()

    def _offload_to_gcs(self, filename: str, filepath: Path) -> None:
        """Copy a Parquet file to GCS (best-effort)."""
        if not self._remote_log_dir:
            return
        remote_path = f"{self._remote_log_dir.rstrip('/')}/{filename}"
        try:
            _fsspec_copy(str(filepath), remote_path)
        except Exception:
            logger.warning("Failed to offload %s to GCS", filepath, exc_info=True)

    def _gc_local_segments(self) -> None:
        """Drop oldest local Parquet segments if count or size exceeds limits.

        Takes the _segments_rwlock exclusively before unlinking files so that
        in-progress DuckDB reads (which hold the shared read lock) are not
        disrupted by file deletion.
        """
        with self._lock:
            total_bytes = sum(s.size_bytes for s in self._local_segments)
            to_delete: list[str] = []

            while self._local_segments and (
                len(self._local_segments) > self._max_local_segments or total_bytes > self._max_local_bytes
            ):
                oldest = self._local_segments.popleft()
                total_bytes -= oldest.size_bytes
                to_delete.append(oldest.path)

        if not to_delete:
            return

        # Hold the write lock while deleting files so concurrent reads
        # (which hold the read lock) finish before we unlink anything.
        self._segments_rwlock.write_acquire()
        try:
            for path in to_delete:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to delete old segment %s", path, exc_info=True)
        finally:
            self._segments_rwlock.write_release()

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
    ) -> LogReadResult:
        # Snapshot all state atomically.
        with self._lock:
            parquet_files = [s.path for s in self._local_segments]
            # Collect all RAM data: sealed buffers + head.
            ram_rows: list[tuple] = []
            for sb in self._sealed:
                ram_rows.extend(sb.rows)
            ram_rows.extend(self._head)

        ram_table = _build_buffer_table(ram_rows)

        # Hold the segments read lock while DuckDB has files open so that
        # GC (which takes the write lock) cannot delete them mid-query.
        self._segments_rwlock.read_acquire()
        try:
            source = _build_union_source(parquet_files)
            where_clause = " AND ".join(where_parts)

            if include_key_in_select:
                select_cols = "seq, key, source, data, epoch_ms, level"
            else:
                select_cols = "seq, source, data, epoch_ms, level"

            order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
            limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            with self._pool.checkout(ram_table) as conn:
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
                entry.attempt_id = parsed.attempt_id if parsed.attempt_id is not None else 0
                entries.append(entry)
        else:
            entries = []
            for r in rows:
                # r: (seq, source, data, epoch_ms, level)
                entry = logging_pb2.LogEntry(source=r[1], data=r[2], level=r[4])
                entry.timestamp.epoch_ms = r[3]
                entries.append(entry)

        return LogReadResult(entries=entries, cursor=max_seq)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        where_parts.append("data LIKE $substring ESCAPE '\\'")
        params["substring"] = f"%{_escape_like(substring_filter)}%"
    if min_level_enum > 0:
        where_parts.append("(level = 0 OR level >= $min_level)")
        params["min_level"] = min_level_enum


def _build_union_source(parquet_files: list[str]) -> str:
    """Build a SQL source expression: local Parquet files UNION ALL ram_buffer.

    File paths are self-generated (``logs_{seq}_{seq}.parquet``) so no SQL
    injection risk from the f-string embedding.
    """
    parts: list[str] = []
    if parquet_files:
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        parts.append(f"SELECT * FROM read_parquet([{file_list}])")
    parts.append("SELECT * FROM ram_buffer")
    return " UNION ALL ".join(parts)


# ---------------------------------------------------------------------------
# Cursor helper
# ---------------------------------------------------------------------------


class LogCursor:
    """Stateful incremental reader for a single LogStore key.

    Tracks a seq cursor across calls to read() so callers don't need to
    manage cursor bookkeeping.
    """

    def __init__(self, store: LogStore, key: str) -> None:
        self._store = store
        self._key = key
        self._cursor: int = 0

    def read(self, max_entries: int = 5000) -> list[logging_pb2.LogEntry]:
        """Return new entries since the last call, advancing the cursor."""
        result = self._store.get_logs(self._key, cursor=self._cursor, max_lines=max_entries)
        self._cursor = result.cursor
        return result.entries


# ---------------------------------------------------------------------------
# Logging handler bridge
# ---------------------------------------------------------------------------


class LogStoreHandler(logging.Handler):
    """Logging handler that writes formatted records directly into a LogStore."""

    def __init__(self, log_store: LogStore, key: str = PROCESS_LOG_KEY):
        super().__init__()
        self._log_store = log_store
        self._key = key
        self._closed = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return
        try:
            entry = logging_pb2.LogEntry(
                source="process",
                data=self.format(record),
                level=str_to_log_level(record.levelname),
            )
            entry.timestamp.epoch_ms = int(record.created * 1000)
            self._log_store.append(self._key, [entry])
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._closed = True
        super().close()
