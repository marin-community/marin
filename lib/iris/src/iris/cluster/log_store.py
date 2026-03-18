# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store backed by in-memory buffer + rotating Parquet segments + DuckDB reads.

Stores log entries keyed by an arbitrary string, suitable for task attempt logs,
process logs, autoscaler logs, or any other log stream.

Write path: Python list buffer -> Parquet segment (ZSTD) -> optional GCS offload.
Read path: DuckDB read_parquet() over local segments UNION ALL in-memory buffer.

Local parquet file list is maintained in RAM (updated on flush and archival)
rather than listing the filesystem on every query. The background offload thread
handles GCS upload, local budget enforcement, and metadata updates atomically.

Also provides LogStoreHandler, a logging.Handler that bridges Python's logging
framework into the LogStore so process-level logs (controller, worker) are
queryable through the same FetchLogs RPC as task logs.
"""

from __future__ import annotations

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    Returns max_seq + 1 so the counter can resume, or 0 if no files exist.
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


def _discover_parquet_files(log_dir: Path) -> list[str]:
    """List existing parquet segment paths on disk, sorted by name."""
    return sorted(str(p) for p in log_dir.glob("logs_*_*.parquet"))


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
    entries: list[logging_pb2.LogEntry]
    cursor: int  # max seq seen


class LogStore:
    """Log store backed by in-memory buffer + rotating Parquet segments.

    Thread-safe: writers append to a buffer protected by a lock, readers
    query via DuckDB over local Parquet files and the current buffer.

    The in-memory list of local parquet files is maintained under ``_write_lock``
    and updated only on flush (add) or archival (remove). No filesystem listing
    happens during reads.

    The background offload thread handles GCS upload and local budget enforcement.
    Files are only deleted after successful upload (or when no remote dir is
    configured), preventing data loss from premature eviction.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        local_budget_bytes: int = 5 * 1024**3,
        flush_row_threshold: int = 1_000_000,
        flush_time_threshold: float = 60.0,
    ):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if log_dir is not None:
            self._log_dir = log_dir
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            self._log_dir = Path(self._temp_dir.name) / "parquet_logs"

        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._remote_log_dir = remote_log_dir
        self._local_budget_bytes = local_budget_bytes
        self._flush_row_threshold = flush_row_threshold
        self._flush_time_threshold = flush_time_threshold

        self._seq = _SequenceCounter(start=_recover_max_seq(self._log_dir))
        self._buffer: list[tuple] = []
        # Protects _buffer and _parquet_files.
        self._write_lock = Lock()
        self._parquet_files: list[str] = _discover_parquet_files(self._log_dir)
        self._last_flush_time = time.monotonic()

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_gcs_offload")

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        first_seq = self._seq.next_batch(len(entries))
        rows = [(first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)]
        with self._write_lock:
            self._buffer.extend(rows)
        self._maybe_flush()

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
        with self._write_lock:
            self._buffer.extend(all_rows)
        self._maybe_flush()

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

        return self._execute_read(
            where_parts,
            params,
            max_lines,
            tail,
            cursor,
            include_key_in_select=False,
        )

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

        return self._execute_read(
            where_parts,
            params,
            max_lines,
            tail,
            cursor,
            include_key_in_select=True,
        )

    def has_logs(self, key: str) -> bool:
        """Check whether any log entries exist for the given key."""
        result = self.get_logs(key, max_lines=1)
        return len(result.entries) > 0

    def cursor(self, key: str) -> LogCursor:
        """Return a stateful cursor for incremental reads on *key*."""
        return LogCursor(self, key)

    def close(self) -> None:
        """Flush remaining buffer, shut down background executor, clean up temp dir."""
        self._flush_to_parquet()
        self._executor.shutdown(wait=True)
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def _maybe_flush(self) -> None:
        with self._write_lock:
            buf_len = len(self._buffer)
        elapsed = time.monotonic() - self._last_flush_time
        if buf_len >= self._flush_row_threshold or (buf_len > 0 and elapsed >= self._flush_time_threshold):
            self._flush_to_parquet()

    def _flush_to_parquet(self) -> None:
        with self._write_lock:
            if not self._buffer:
                return
            snapshot = self._buffer
            self._buffer = []

        self._last_flush_time = time.monotonic()

        table = _build_buffer_table(snapshot)
        min_seq = snapshot[0][0]
        max_seq = snapshot[-1][0]
        filename = f"logs_{min_seq:019d}_{max_seq:019d}.parquet"
        filepath = self._log_dir / filename

        pq.write_table(table, filepath, compression="zstd")

        # Add to in-memory file list under lock so readers see it immediately.
        with self._write_lock:
            self._parquet_files.append(str(filepath))

        # Background thread handles GCS offload + local budget enforcement.
        self._executor.submit(self._offload_and_enforce_budget, filepath)

    def _offload_and_enforce_budget(self, filepath: Path) -> None:
        """Upload to GCS, then enforce local disk budget.

        Runs on the single-worker executor so offloads are serialized.
        Files are only deleted after successful upload (or when no remote is configured).
        """
        if self._remote_log_dir:
            remote_path = f"{self._remote_log_dir.rstrip('/')}/{filepath.name}"
            try:
                _fsspec_copy(str(filepath), remote_path)
            except Exception:
                logger.warning("Failed to offload %s to GCS", filepath, exc_info=True)
                # Don't evict files that failed to upload.
                return

        self._enforce_local_budget()

    def _enforce_local_budget(self) -> None:
        """Delete oldest local segments past the disk budget.

        Updates the in-memory file list under ``_write_lock`` so readers
        never see a stale path.
        """
        with self._write_lock:
            files = list(self._parquet_files)

        # Compute sizes for local files that still exist.
        file_sizes: list[tuple[str, int]] = []
        for f in files:
            p = Path(f)
            try:
                file_sizes.append((f, p.stat().st_size))
            except FileNotFoundError:
                continue

        total = sum(s for _, s in file_sizes)
        to_remove: set[str] = set()
        while total > self._local_budget_bytes and file_sizes:
            oldest_path, oldest_size = file_sizes.pop(0)
            total -= oldest_size
            Path(oldest_path).unlink(missing_ok=True)
            to_remove.add(oldest_path)

        if to_remove:
            with self._write_lock:
                self._parquet_files = [f for f in self._parquet_files if f not in to_remove]

    def _execute_read(
        self,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        default_cursor: int,
        include_key_in_select: bool,
    ) -> LogReadResult:
        # Snapshot both file list and buffer atomically under the lock.
        with self._write_lock:
            parquet_files = list(self._parquet_files)
            buffer_snapshot = list(self._buffer)

        buf_table = _build_buffer_table(buffer_snapshot)
        conn = duckdb.connect()
        try:
            conn.register("write_buffer", buf_table)
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
    """Build a SQL source expression that unions parquet files with the write_buffer table.

    Always includes write_buffer. Only includes read_parquet when files exist.
    File paths are self-generated (``logs_{seq}_{seq}.parquet``) so no SQL injection risk.
    """
    parts: list[str] = []
    if parquet_files:
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        parts.append(f"SELECT * FROM read_parquet([{file_list}])")
    parts.append("SELECT * FROM write_buffer")
    return " UNION ALL ".join(parts)


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
