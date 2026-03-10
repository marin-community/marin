# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed log store with string keys.

Stores log entries keyed by an arbitrary string, suitable for task attempt logs,
process logs, autoscaler logs, or any other log stream. Uses WAL mode for
concurrent reader/writer access.

Also provides LogStoreHandler, a logging.Handler that bridges Python's logging
framework into the LogStore so process-level logs (controller, worker) are
queryable through the same FetchLogs RPC as task logs.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from iris.cluster.types import JobName, TaskName
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    source TEXT NOT NULL,
    data TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL,
    level INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_key ON logs(key, id);
"""

_MAX_RECORDS = 5_000_000

_LIKE_ESCAPE_TABLE = str.maketrans({"%": "\\%", "_": "\\_", "\\": "\\\\"})


def _escape_like(s: str) -> str:
    """Escape SQL LIKE wildcards so the string matches literally."""
    return s.translate(_LIKE_ESCAPE_TABLE)


PROCESS_LOG_KEY = "/system/process"


def task_log_key(task_id: JobName | TaskName, attempt_id: int | None = None) -> str:
    """Build a hierarchical key for task attempt logs.

    Accepts either a TaskName (preferred) or a JobName + explicit attempt_id.
    """
    if isinstance(task_id, TaskName):
        return task_id.with_attempt(attempt_id if attempt_id is not None else task_id.require_attempt()).to_wire()
    if attempt_id is None:
        raise ValueError("attempt_id is required when task_id is a JobName")
    return TaskName(task_id=task_id, attempt_id=attempt_id).to_wire()


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry]
    cursor: int  # max autoincrement id seen


class LogStore:
    """SQLite-backed log store keyed by arbitrary strings.

    Thread-safe: writers and RPC readers may run concurrently. WAL mode
    allows readers to proceed without blocking the writer.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        db_path: Path | None = None,
        max_records: int = _MAX_RECORDS,
    ):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path_str = str(db_path)
        elif log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            db_path_str = str(log_dir / "logs.db")
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            db_path_str = str(Path(self._temp_dir.name) / "logs.db")

        # Separate connections for writers and readers so WAL concurrency
        # actually works: readers never block the writer and vice-versa.
        self._write_conn = self._make_conn(db_path_str)
        self._write_conn.executescript(_SCHEMA)
        self._write_conn.commit()

        self._read_conn = self._make_conn(db_path_str)

        self._write_lock = Lock()
        self._read_lock = Lock()
        self._max_records = max_records
        self._rows_since_eviction_check = 0
        self._eviction_check_interval = max_records // 10

    @staticmethod
    def _make_conn(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return

        rows = [(key, e.source, e.data, e.timestamp.epoch_ms, e.level) for e in entries]
        with self._write_lock:
            self._write_conn.executemany(
                "INSERT INTO logs (key, source, data, epoch_ms, level) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._write_conn.commit()
            self._post_write_maintenance(len(rows))

    def append_batch(self, items: list[tuple[str, list]]) -> None:
        """Write log entries from multiple keys in a single transaction.

        Each item is (key, entries). All rows are inserted with a single
        executemany + commit, avoiding per-key commit overhead and allowing
        callers to release other locks before flushing logs.
        """
        all_rows = []
        for key, entries in items:
            all_rows.extend((key, e.source, e.data, e.timestamp.epoch_ms, e.level) for e in entries)
        if not all_rows:
            return
        with self._write_lock:
            self._write_conn.executemany(
                "INSERT INTO logs (key, source, data, epoch_ms, level) VALUES (?, ?, ?, ?, ?)",
                all_rows,
            )
            self._write_conn.commit()
            self._post_write_maintenance(len(all_rows))

    def _post_write_maintenance(self, rows_written: int) -> None:
        """Run eviction as needed. Must hold self._write_lock."""
        self._rows_since_eviction_check += rows_written
        if self._rows_since_eviction_check >= self._eviction_check_interval:
            self._evict_if_needed()
            self._rows_since_eviction_check = 0

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
        """Fetch logs for a single key.

        All filtering (substring, min_level, since_ms) is pushed into SQL so
        that LIMIT works correctly and we never fetch millions of rows into Python.
        """
        min_level_enum = str_to_log_level(min_level) if min_level else 0

        params: list = [key, cursor]
        where_extra = ""
        if since_ms > 0:
            where_extra += " AND epoch_ms > ?"
            params.append(since_ms)
        if substring_filter:
            where_extra += " AND data LIKE ? ESCAPE '\\'"
            params.append(f"%{_escape_like(substring_filter)}%")
        if min_level_enum > 0:
            where_extra += " AND (level = 0 OR level >= ?)"
            params.append(min_level_enum)

        if tail and max_lines > 0:
            params.append(max_lines)
            with self._read_lock:
                rows = self._read_conn.execute(
                    f"SELECT id, source, data, epoch_ms, level FROM logs "
                    f"WHERE key = ? AND id > ?{where_extra} ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall()
            rows.reverse()
        else:
            limit_clause = ""
            if max_lines > 0:
                limit_clause = " LIMIT ?"
                params.append(max_lines)
            with self._read_lock:
                rows = self._read_conn.execute(
                    f"SELECT id, source, data, epoch_ms, level FROM logs "
                    f"WHERE key = ? AND id > ?{where_extra} ORDER BY id{limit_clause}",
                    params,
                ).fetchall()

        if not rows:
            return LogReadResult(entries=[], cursor=cursor)

        max_id = max(r[0] for r in rows)
        entries = [self._row_to_entry(r) for r in rows]
        return LogReadResult(entries=entries, cursor=max_id)

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
        """Fetch logs for all keys matching prefix, ordered by autoincrement id.

        All filtering is pushed into SQL so LIMIT works correctly.

        Args:
            tail: If True and max_lines > 0, return the *last* N entries instead
                  of the first N (uses DESC ordering then reverses).
            shallow: If True, only match keys one level deep (no nested '/' after prefix).
                     This excludes child job logs when fetching by job prefix.
        """
        min_level_enum = str_to_log_level(min_level) if min_level else 0

        params: list = [prefix + "%", cursor]
        where = "WHERE key LIKE ? AND id > ?"
        if shallow:
            where += " AND key NOT LIKE ?"
            params.append(prefix + "%/%")
        if since_ms > 0:
            where += " AND epoch_ms > ?"
            params.append(since_ms)
        if substring_filter:
            where += " AND data LIKE ? ESCAPE '\\'"
            params.append(f"%{_escape_like(substring_filter)}%")
        if min_level_enum > 0:
            where += " AND (level = 0 OR level >= ?)"
            params.append(min_level_enum)

        if tail and max_lines > 0:
            params.append(max_lines)
            with self._read_lock:
                rows = self._read_conn.execute(
                    f"SELECT id, key, source, data, epoch_ms, level FROM logs {where} ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall()
            rows.reverse()
        else:
            limit_clause = ""
            if max_lines > 0:
                limit_clause = " LIMIT ?"
                params.append(max_lines)

            with self._read_lock:
                rows = self._read_conn.execute(
                    f"SELECT id, key, source, data, epoch_ms, level FROM logs {where} ORDER BY id{limit_clause}",
                    params,
                ).fetchall()

        max_id = max((r[0] for r in rows), default=cursor)
        entries = []
        for r in rows:
            # Parse attempt_id from key using TaskName wire format: "/user/job/0:attempt_id"
            key = r[1]
            parsed = TaskName.from_wire(key)
            entry = logging_pb2.LogEntry(source=r[2], data=r[3], level=r[5])
            entry.timestamp.epoch_ms = r[4]
            entry.attempt_id = parsed.attempt_id or 0
            entries.append(entry)

        return LogReadResult(entries=entries, cursor=max_id)

    def has_logs(self, key: str) -> bool:
        with self._read_lock:
            row = self._read_conn.execute(
                "SELECT 1 FROM logs WHERE key = ? LIMIT 1",
                (key,),
            ).fetchone()
        return row is not None

    def clear(self, key: str) -> None:
        with self._write_lock:
            self._write_conn.execute("DELETE FROM logs WHERE key = ?", (key,))
            self._write_conn.commit()

    def close(self) -> None:
        """Close database connections and clean up temp dir if applicable."""
        self._read_conn.close()
        self._write_conn.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    @staticmethod
    def _row_to_entry(row: tuple) -> logging_pb2.LogEntry:
        # row: (id, source, data, epoch_ms, level)
        entry = logging_pb2.LogEntry(source=row[1], data=row[2], level=row[4])
        entry.timestamp.epoch_ms = row[3]
        return entry

    def cursor(self, key: str) -> LogCursor:
        """Return a stateful cursor for incremental reads on *key*."""
        return LogCursor(self, key)

    def _evict_if_needed(self) -> None:
        """Delete oldest rows when the total count exceeds the cap. Must hold self._write_lock.

        Uses MAX(id) - MIN(id) as a cheap approximation of row count (autoincrement
        IDs are nearly contiguous since we only delete from the bottom). Evicts down
        to max_records // 2 so we don't pay eviction cost on every subsequent append.

        After bulk deletion we truncate the WAL to prevent it from growing unboundedly.
        Without this, the DELETE pages accumulate in the WAL and the next automatic
        checkpoint stalls the writer while flushing gigabytes of data.
        """
        row = self._write_conn.execute("SELECT MIN(id), MAX(id) FROM logs").fetchone()
        min_id, max_id = row
        if min_id is None:
            return
        approx_count = max_id - min_id + 1
        if approx_count <= self._max_records:
            return
        # Keep the most recent max_records // 2 rows. Delete in batches to
        # avoid a single enormous transaction that bloats the WAL.
        target_cutoff = max_id - (self._max_records // 2)
        batch_size = 100_000
        cursor = min_id - 1
        while cursor < target_cutoff:
            batch_end = min(cursor + batch_size, target_cutoff)
            self._write_conn.execute("DELETE FROM logs WHERE id > ? AND id <= ?", (cursor, batch_end))
            self._write_conn.commit()
            self._write_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            cursor = batch_end


class LogCursor:
    """Stateful incremental reader for a single LogStore key.

    Tracks an autoincrement id cursor across calls to read() so callers don't
    need to manage cursor bookkeeping. Useful for streaming new log entries
    incrementally (e.g. heartbeat log forwarding).
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
    """Logging handler that writes formatted records directly into a LogStore.

    Each log record is written to SQLite immediately. WAL mode handles
    concurrent writes without contention on low-volume process logs.
    """

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
