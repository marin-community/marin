# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed log store for controller task attempts.

Appends log entries to per-attempt JSONL files. Maintains an in-memory
list of byte offsets per line so that polling readers can skip to their
last-seen position in O(1) instead of re-scanning from byte 0.

Supports an optional persistent directory so that controller restarts
preserve logs. When no directory is given, a temp dir is used (tests).
"""

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from google.protobuf import json_format

from iris.cluster.types import JobName
from iris.rpc import logging_pb2


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry]
    lines_read: int  # total line count at end of read (cursor for next poll)


class ControllerLogStore:
    """Disk-backed log store for task attempts.

    Directory layout:
        {log_dir}/{job_wire}/{attempt_id}/logs.jsonl

    Thread-safe: heartbeat writers (complete_heartbeat) and RPC readers
    (get_task_logs) may run concurrently.
    """

    def __init__(self, log_dir: Path | None = None):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir = log_dir
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            self._log_dir = Path(self._temp_dir.name)
        self._lock = RLock()
        self._line_offsets: dict[tuple[JobName, int], list[int]] = {}

    def _attempt_path(self, task_id: JobName, attempt_id: int) -> Path:
        task_wire = task_id.to_wire().lstrip("/")
        return self._log_dir / task_wire / str(attempt_id) / "logs.jsonl"

    @staticmethod
    def _entry_to_json(entry: logging_pb2.LogEntry) -> str:
        return json.dumps(
            json_format.MessageToDict(entry, preserving_proto_field_name=True),
            ensure_ascii=False,
        )

    @staticmethod
    def _json_to_entry(line: str) -> logging_pb2.LogEntry:
        entry = logging_pb2.LogEntry()
        json_format.Parse(line, entry)
        return entry

    def append(self, task_id: JobName, attempt_id: int, entries: list) -> None:
        if not entries:
            return
        key = (task_id, attempt_id)
        path = self._attempt_path(task_id, attempt_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        new_offsets: list[int] = []
        with open(path, "ab") as f:
            for e in entries:
                new_offsets.append(f.tell())
                f.write((self._entry_to_json(e) + "\n").encode("utf-8"))

        with self._lock:
            self._line_offsets.setdefault(key, []).extend(new_offsets)

    def _ensure_line_offsets(self, key: tuple[JobName, int]) -> list[int]:
        """Return the offset array, rebuilding from disk if needed (restart recovery)."""
        with self._lock:
            offsets = self._line_offsets.get(key)
            if offsets is not None:
                return offsets

        path = self._attempt_path(*key)
        if not path.exists():
            return []

        offsets = []
        with open(path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)

        with self._lock:
            self._line_offsets[key] = offsets
        return offsets

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
        key = (task_id, attempt_id)
        path = self._attempt_path(task_id, attempt_id)
        if not path.exists():
            return LogReadResult(entries=[], lines_read=0)

        offsets = self._ensure_line_offsets(key)

        # When filtering (regex/since_ms), we can't predict how many raw lines
        # yield max_lines matches, so we scan everything and take the tail slice.
        has_filter = regex_filter is not None or since_ms > 0
        if tail and max_lines > 0 and not has_filter:
            skip_lines = max(skip_lines, len(offsets) - max_lines)

        if skip_lines >= len(offsets):
            return LogReadResult(entries=[], lines_read=len(offsets))

        result: list[logging_pb2.LogEntry] = []
        with open(path, "rb") as f:
            if skip_lines > 0:
                f.seek(offsets[skip_lines])
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = self._json_to_entry(line.decode("utf-8"))
                except Exception:
                    continue
                if since_ms > 0 and entry.timestamp.epoch_ms <= since_ms:
                    continue
                if regex_filter and not regex_filter.search(entry.data):
                    continue
                result.append(entry)
                if max_lines > 0 and len(result) >= max_lines and not (tail and has_filter):
                    break

        # For tail + filter, take only the last max_lines matches.
        if tail and has_filter and max_lines > 0 and len(result) > max_lines:
            result = result[-max_lines:]

        with self._lock:
            total = len(self._line_offsets.get(key, []))
        return LogReadResult(entries=result, lines_read=total)

    def has_logs(self, task_id: JobName, attempt_id: int) -> bool:
        return self._attempt_path(task_id, attempt_id).exists()

    def clear_attempt(self, task_id: JobName, attempt_id: int) -> None:
        key = (task_id, attempt_id)
        with self._lock:
            self._line_offsets.pop(key, None)
        path = self._attempt_path(task_id, attempt_id)
        if path.exists():
            path.unlink()

    def close(self) -> None:
        """Clean up the temporary directory (no-op for persistent dirs)."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
