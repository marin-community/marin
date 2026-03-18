# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store backed by rotating RAM buffers + Parquet segments + DuckDB reads.

Lifecycle of a log entry:

    1. Appended to the *head* RAM buffer (a plain Python list).
    2. When the head buffer exceeds ``SEGMENT_TARGET_BYTES`` or
       ``flush_interval_sec`` elapses, it is *sealed*: moved to a deque of
       sealed RAM buffers and a background thread flushes it to a local
       Parquet file.
    3. When the background thread finishes writing the Parquet file it:
       a. Registers the file in ``_local_segments`` (a deque).
       b. Removes the corresponding sealed RAM buffer (readers no longer need it).
       c. Kicks off a GCS copy of the new file.
       d. Runs the GC cycle: drops the oldest local Parquet segment if
          ``len(_local_segments) > max_local_segments`` or total local bytes
          exceed ``max_local_bytes``.

Read path: DuckDB ``read_parquet()`` over the snapshot of local Parquet files
UNION ALL in-memory pyarrow tables for each RAM buffer (head + sealed).
All snapshots are taken atomically under ``_lock``.

The design keeps exactly one lock for all shared state so ordering is trivial.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

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

# Target size for a single Parquet segment. We estimate RAM row size at ~256
# bytes; a buffer is flushed when its estimated size exceeds this threshold.
SEGMENT_TARGET_BYTES = 100 * 1024 * 1024  # 100 MB

# If the head buffer has data but hasn't reached SEGMENT_TARGET_BYTES, flush
# after this many seconds anyway so logs are durable on disk.
DEFAULT_FLUSH_INTERVAL_SEC = 60.0

# Estimated bytes per buffered row (used to decide when to flush).
_EST_BYTES_PER_ROW = 256

# Default caps for local Parquet retention.
DEFAULT_MAX_LOCAL_SEGMENTS = 50
DEFAULT_MAX_LOCAL_BYTES = 5 * 1024**3  # 5 GB


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


class _SequenceCounter:
    """Thread-safe monotonically increasing counter."""

    def __init__(self, start: int = 1):
        self._value = start
        self._lock = Lock()

    def next_batch(self, count: int) -> int:
        """Return the first seq of a batch of ``count`` consecutive numbers."""
        with self._lock:
            first = self._value
            self._value += count
            return first


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

        self._seq = _SequenceCounter(start=_recover_max_seq(self._log_dir))

        # ---- shared mutable state (all guarded by _lock) ----
        self._lock = Lock()
        self._head: list[tuple] = []  # current write buffer
        self._sealed: deque[_SealedBuffer] = deque()  # sealed, pending flush
        self._local_segments: deque[_LocalSegment] = deque()  # flushed parquet files
        self._last_flush_time = time.monotonic()

        # Discover pre-existing Parquet files from a previous run.
        for p in sorted(self._log_dir.glob("logs_*_*.parquet")):
            parts = p.stem.split("_")
            min_seq = int(parts[1]) if len(parts) >= 3 else 0
            max_seq = int(parts[2]) if len(parts) >= 3 else 0
            self._local_segments.append(
                _LocalSegment(path=str(p), size_bytes=p.stat().st_size, min_seq=min_seq, max_seq=max_seq)
            )

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        first_seq = self._seq.next_batch(len(entries))
        rows = [(first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)]
        with self._lock:
            self._head.extend(rows)
        self._maybe_seal()

    def append_batch(self, items: list[tuple[str, list]]) -> None:
        """Write log entries from multiple keys in a single operation."""
        all_rows: list[tuple] = []
        for key, entries in items:
            if not entries:
                continue
            first_seq = self._seq.next_batch(len(entries))
            all_rows.extend(
                (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
            )
        if not all_rows:
            return
        with self._lock:
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
        """Write a sealed buffer to Parquet, register it, GCS-copy, and GC.

        Runs on the background executor thread.
        """
        rows = sealed.rows
        if not rows:
            return

        table = _build_buffer_table(rows)
        min_seq = rows[0][0]
        max_seq = rows[-1][0]
        filename = f"logs_{min_seq:019d}_{max_seq:019d}.parquet"
        filepath = self._log_dir / filename

        try:
            pq.write_table(table, filepath, compression="zstd")
        except Exception:
            logger.warning("Failed to write Parquet segment %s", filepath, exc_info=True)
            # Leave the sealed buffer in the deque so reads still see the data.
            return

        seg = _LocalSegment(
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            min_seq=min_seq,
            max_seq=max_seq,
        )

        with self._lock:
            self._local_segments.append(seg)
            # Remove the sealed buffer — data is now in the Parquet file.
            try:
                self._sealed.remove(sealed)
            except ValueError:
                pass
            sealed.flushed = True

        # Copy to GCS (best-effort).
        if self._remote_log_dir:
            remote_path = f"{self._remote_log_dir.rstrip('/')}/{filename}"
            try:
                _fsspec_copy(str(filepath), remote_path)
            except Exception:
                logger.warning("Failed to offload %s to GCS", filepath, exc_info=True)

        # GC: drop oldest segments if over budget.
        self._gc_local_segments()

    def _gc_local_segments(self) -> None:
        """Drop oldest local Parquet segments if count or size exceeds limits."""
        with self._lock:
            total_bytes = sum(s.size_bytes for s in self._local_segments)
            to_delete: list[str] = []

            while self._local_segments and (
                len(self._local_segments) > self._max_local_segments or total_bytes > self._max_local_bytes
            ):
                oldest = self._local_segments.popleft()
                total_bytes -= oldest.size_bytes
                to_delete.append(oldest.path)

        # Delete files outside the lock.
        for path in to_delete:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete old segment %s", path, exc_info=True)

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
        conn = duckdb.connect()
        try:
            conn.register("ram_buffer", ram_table)
            source = _build_union_source(parquet_files)
            where_clause = " AND ".join(where_parts)

            if include_key_in_select:
                select_cols = "seq, key, source, data, epoch_ms, level"
            else:
                select_cols = "seq, source, data, epoch_ms, level"

            order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
            limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

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
