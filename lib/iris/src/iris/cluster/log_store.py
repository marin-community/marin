# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store backed by in-memory buffer + rotating Parquet segments + DuckDB reads.

Stores log entries keyed by an arbitrary string, suitable for task attempt logs,
process logs, autoscaler logs, or any other log stream.

Write path: Python list buffer → Parquet segment (ZSTD, sorted by key) → optional GCS offload.
Read path: DuckDB connection pool over local segments UNION ALL in-memory buffer.

Also provides LogStoreHandler, a logging.Handler that bridges Python's logging
framework into the LogStore so process-level logs (controller, worker) are
queryable through the same FetchLogs RPC as task logs.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from collections.abc import Iterator

import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq

from iris.cluster.types import TaskAttempt
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

logger = logging.getLogger(__name__)

_LIKE_ESCAPE_TABLE = str.maketrans({"%": "\\%", "_": "\\_", "\\": "\\\\"})

_ROW_GROUP_SIZE = 16_384

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
            conn.register("write_buffer", buffer_table)
            yield conn
        finally:
            conn.unregister("write_buffer")
            self._pool.put(conn)

    def close(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry]
    cursor: int  # max seq seen


class LogStore:
    """Log store backed by in-memory buffer + rotating Parquet segments.

    Thread-safe: writers append to a buffer protected by a lock, readers
    query via a pool of DuckDB connections over local Parquet files and
    the current buffer.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        db_path: Path | None = None,
        max_records: int = 0,
        remote_log_dir: str = "",
        local_budget_bytes: int = 5 * 1024**3,
        flush_row_threshold: int = 1_000_000,
        flush_time_threshold: float = 5.0,
        pool_size: int = 8,
    ):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if db_path is not None:
            self._log_dir = db_path.parent / "parquet_logs"
        elif log_dir is not None:
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
        self._write_lock = Lock()
        self._last_flush_time = time.monotonic()

        # Cached file list, updated on flush. Avoids filesystem glob per read.
        self._parquet_files: list[str] = sorted(str(p) for p in self._log_dir.glob("logs_*_*.parquet"))
        self._files_lock = Lock()

        self._pool = _ConnectionPool(size=pool_size)
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
        with self._files_lock:
            parquet_files = list(self._parquet_files)
        with self._write_lock:
            buffer_snapshot = list(self._buffer)

        buf_table = _build_buffer_table(buffer_snapshot)
        source = _build_union_source(parquet_files)
        with self._pool.checkout(buf_table) as conn:
            row = conn.execute(
                f"SELECT 1 FROM ({source}) WHERE key = $key LIMIT 1",
                {"key": key},
            ).fetchone()
        return row is not None

    def clear(self, key: str) -> None:
        """Remove matching entries from the in-memory buffer only."""
        with self._write_lock:
            self._buffer = [row for row in self._buffer if row[1] != key]

    def cursor(self, key: str) -> LogCursor:
        """Return a stateful cursor for incremental reads on *key*."""
        return LogCursor(self, key)

    def close(self) -> None:
        """Flush remaining buffer, shut down background executor, clean up temp dir."""
        self._flush_to_parquet()
        self._executor.shutdown(wait=True)
        self._pool.close()
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

        # Sort by (key, seq) so row-group statistics on `key` are tight,
        # enabling DuckDB to skip row groups that don't contain the target key.
        snapshot.sort(key=lambda row: (row[1], row[0]))

        table = _build_buffer_table(snapshot)
        min_seq = min(row[0] for row in snapshot)
        max_seq = max(row[0] for row in snapshot)
        filename = f"logs_{min_seq:019d}_{max_seq:019d}.parquet"
        filepath = self._log_dir / filename
        pq.write_table(
            table,
            filepath,
            compression="zstd",
            row_group_size=_ROW_GROUP_SIZE,
            write_page_index=True,
        )

        # Update cached file list
        with self._files_lock:
            self._parquet_files.append(str(filepath))
            self._parquet_files.sort()

        self._executor.submit(self._offload_to_gcs, filepath)
        self._enforce_local_budget()

    def _offload_to_gcs(self, filepath: Path) -> None:
        if not self._remote_log_dir:
            return
        remote_path = f"{self._remote_log_dir.rstrip('/')}/{filepath.name}"
        try:
            _fsspec_copy(str(filepath), remote_path)
        except Exception:
            logger.warning("Failed to offload %s to GCS", filepath, exc_info=True)

    def _enforce_local_budget(self) -> None:
        with self._files_lock:
            files = list(self._parquet_files)

        total = sum(Path(f).stat().st_size for f in files)
        removed: list[str] = []
        while total > self._local_budget_bytes and files:
            oldest = files.pop(0)
            total -= Path(oldest).stat().st_size
            Path(oldest).unlink(missing_ok=True)
            removed.append(oldest)

        if removed:
            with self._files_lock:
                self._parquet_files = [f for f in self._parquet_files if f not in removed]

    def _execute_read(
        self,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        default_cursor: int,
        include_key_in_select: bool,
    ) -> LogReadResult:
        with self._files_lock:
            parquet_files = list(self._parquet_files)

        with self._write_lock:
            buffer_snapshot = list(self._buffer)

        buf_table = _build_buffer_table(buffer_snapshot)
        source = _build_union_source(parquet_files)
        where_clause = " AND ".join(where_parts)

        if include_key_in_select:
            select_cols = "seq, key, source, data, epoch_ms, level"
        else:
            select_cols = "seq, source, data, epoch_ms, level"

        order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
        limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

        sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"

        with self._pool.checkout(buf_table) as conn:
            rows = conn.execute(sql, params).fetchall()

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
