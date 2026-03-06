# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed log store for controller task attempts.

Replaces per-attempt JSONL files with a single SQLite database, eliminating
the per-file FD pressure that can exhaust process limits when many tasks
run concurrently. Uses WAL mode for concurrent reader/writer access.
"""

import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from iris.cluster.types import JobName
from iris.rpc import logging_pb2

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_wire TEXT NOT NULL,
    attempt_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    data TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_attempt ON logs(task_wire, attempt_id, id);
"""

_MAX_RECORDS = 1_000_000
_EVICT_CHECK_INTERVAL = 100  # check eviction every N appends


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry]
    lines_read: int  # total line count at end of read (cursor for next poll)


class ControllerLogStore:
    """SQLite-backed log store for task attempts.

    Thread-safe: heartbeat writers (complete_heartbeat) and RPC readers
    (get_task_logs) may run concurrently. WAL mode allows readers to
    proceed without blocking the writer.
    """

    def __init__(self, log_dir: Path | None = None, *, max_records: int = _MAX_RECORDS):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(log_dir / "logs.db")
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            db_path = str(Path(self._temp_dir.name) / "logs.db")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._lock = RLock()
        self._max_records = max_records
        self._append_count = 0

    @staticmethod
    def _entry_to_row(task_wire: str, attempt_id: int, entry: logging_pb2.LogEntry) -> tuple:
        return (
            task_wire,
            attempt_id,
            entry.source,
            entry.data,
            entry.timestamp.epoch_ms,
        )

    @staticmethod
    def _row_to_entry(row: tuple) -> logging_pb2.LogEntry:
        # row: (source, data, epoch_ms)
        entry = logging_pb2.LogEntry(source=row[0], data=row[1])
        entry.timestamp.epoch_ms = row[2]
        return entry

    def append(self, task_id: JobName, attempt_id: int, entries: list) -> None:
        if not entries:
            return
        task_wire = task_id.to_wire()
        rows = [self._entry_to_row(task_wire, attempt_id, e) for e in entries]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO logs (task_wire, attempt_id, source, data, epoch_ms) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            self._append_count += 1
            if self._append_count % _EVICT_CHECK_INTERVAL == 0:
                self._evict_if_needed()

    def get_logs(
        self,
        task_id: JobName,
        attempt_id: int,
        *,
        since_ms: int = 0,
        skip_lines: int = 0,
        regex_filter: re.Pattern[str] | None = None,
        max_lines: int = 0,
        tail: bool = False,
    ) -> LogReadResult:
        task_wire = task_id.to_wire()
        total = self._count_lines(task_wire, attempt_id)
        if total == 0:
            return LogReadResult(entries=[], lines_read=0)

        has_filter = regex_filter is not None or since_ms > 0

        # Build the query depending on mode
        if tail and max_lines > 0 and not has_filter:
            # Optimized tail without filters: skip to last N rows via OFFSET
            effective_skip = max(skip_lines, total - max_lines)
            rows = self._conn.execute(
                "SELECT source, data, epoch_ms FROM logs "
                "WHERE task_wire = ? AND attempt_id = ? "
                "ORDER BY id LIMIT -1 OFFSET ?",
                (task_wire, attempt_id, effective_skip),
            ).fetchall()
        elif tail and max_lines > 0 and has_filter:
            # Tail with filters: fetch all candidate rows, filter in Python, take last N
            if since_ms > 0:
                rows = self._conn.execute(
                    "SELECT source, data, epoch_ms FROM logs "
                    "WHERE task_wire = ? AND attempt_id = ? AND epoch_ms > ? "
                    "ORDER BY id",
                    (task_wire, attempt_id, since_ms),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT source, data, epoch_ms FROM logs "
                    "WHERE task_wire = ? AND attempt_id = ? "
                    "ORDER BY id LIMIT -1 OFFSET ?",
                    (task_wire, attempt_id, skip_lines),
                ).fetchall()
            if regex_filter:
                rows = [r for r in rows if regex_filter.search(r[1])]
            rows = rows[-max_lines:]
        else:
            # Forward mode
            params: list = [task_wire, attempt_id]
            where_extra = ""
            if since_ms > 0:
                where_extra += " AND epoch_ms > ?"
                params.append(since_ms)

            query = (
                "SELECT source, data, epoch_ms FROM logs "
                f"WHERE task_wire = ? AND attempt_id = ?{where_extra} "
                "ORDER BY id LIMIT -1 OFFSET ?"
            )
            params.append(skip_lines)
            rows = self._conn.execute(query, params).fetchall()

            if regex_filter:
                rows = [r for r in rows if regex_filter.search(r[1])]

            if max_lines > 0:
                rows = rows[:max_lines]

        entries = [self._row_to_entry(r) for r in rows]
        return LogReadResult(entries=entries, lines_read=total)

    def has_logs(self, task_id: JobName, attempt_id: int) -> bool:
        task_wire = task_id.to_wire()
        row = self._conn.execute(
            "SELECT 1 FROM logs WHERE task_wire = ? AND attempt_id = ? LIMIT 1",
            (task_wire, attempt_id),
        ).fetchone()
        return row is not None

    def clear_attempt(self, task_id: JobName, attempt_id: int) -> None:
        task_wire = task_id.to_wire()
        with self._lock:
            self._conn.execute(
                "DELETE FROM logs WHERE task_wire = ? AND attempt_id = ?",
                (task_wire, attempt_id),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection and clean up temp dir if applicable."""
        self._conn.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def _count_lines(self, task_wire: str, attempt_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM logs WHERE task_wire = ? AND attempt_id = ?",
            (task_wire, attempt_id),
        ).fetchone()
        return row[0]

    def _evict_if_needed(self) -> None:
        """Delete oldest rows when the total count exceeds the cap. Must hold self._lock."""
        count = self._conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        if count > self._max_records:
            excess = count - self._max_records
            self._conn.execute(
                "DELETE FROM logs WHERE id IN (SELECT id FROM logs ORDER BY id LIMIT ?)",
                (excess,),
            )
            self._conn.commit()
