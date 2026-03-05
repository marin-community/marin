# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed log store for controller task attempts.

Appends log entries to per-attempt JSONL files in a self-managed temp
directory so the controller doesn't accumulate unbounded log data in memory.
Workers already persist the authoritative copy to GCS/S3.
"""

import json
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from threading import RLock

from google.protobuf import json_format

from iris.cluster.types import JobName
from iris.rpc import logging_pb2


def _reverse_read_lines(path: Path, block_size: int = 25 * 1024) -> Iterator[str]:
    """Yield non-empty lines from a file in reverse order, reading in block_size chunks.

    Reads backwards from the end of the file so that only the tail is touched
    for bounded queries, giving O(N) cost for the last N lines instead of
    O(total).
    """
    with open(path, "rb") as f:
        f.seek(0, 2)
        remaining = f.tell()
        buf = b""
        while remaining > 0:
            read_size = min(block_size, remaining)
            remaining -= read_size
            f.seek(remaining)
            chunk = f.read(read_size)
            buf = chunk + buf
            lines = buf.split(b"\n")
            buf = lines[0]  # incomplete line carried to next iteration
            for line in reversed(lines[1:]):
                stripped = line.strip()
                if stripped:
                    yield stripped.decode("utf-8")
        if buf.strip():
            yield buf.strip().decode("utf-8")


class ControllerLogStore:
    """Disk-backed log store for task attempts.

    Directory layout:
        {temp_dir}/{job_wire}/{task_index}/{attempt_id}/logs.jsonl

    Thread-safe: heartbeat writers (complete_heartbeat) and RPC readers
    (get_task_logs) may run concurrently.
    """

    def __init__(self):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
        self._log_dir = Path(self._temp_dir.name)
        self._lock = RLock()
        # Track which attempts have logs (avoids stat calls on every get_logs)
        self._known_attempts: set[tuple[JobName, int]] = set()

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
        with self._lock:
            self._known_attempts.add(key)

        path = self._attempt_path(task_id, attempt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(self._entry_to_json(e) + "\n" for e in entries)
        with open(path, "a") as f:
            f.write(lines)

    def get_logs(
        self,
        task_id: JobName,
        attempt_id: int,
        *,
        since_ms: int = 0,
        regex_filter: re.Pattern[str] | None = None,
        max_lines: int = 0,
        tail: bool = False,
    ) -> list:
        path = self._attempt_path(task_id, attempt_id)
        if not path.exists():
            return []

        if tail and max_lines > 0:
            return self._get_logs_tail(path, since_ms=since_ms, regex_filter=regex_filter, max_lines=max_lines)

        return self._get_logs_forward(path, since_ms=since_ms, regex_filter=regex_filter, max_lines=max_lines)

    def _get_logs_forward(
        self,
        path: Path,
        *,
        since_ms: int,
        regex_filter: re.Pattern[str] | None,
        max_lines: int,
    ) -> list[logging_pb2.LogEntry]:
        result: list[logging_pb2.LogEntry] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = self._json_to_entry(line)
                except Exception:
                    continue
                if since_ms > 0 and entry.timestamp.epoch_ms <= since_ms:
                    continue
                if regex_filter and not regex_filter.search(entry.data):
                    continue
                result.append(entry)
                if max_lines > 0 and len(result) >= max_lines:
                    break
        return result

    def _get_logs_tail(
        self,
        path: Path,
        *,
        since_ms: int,
        regex_filter: re.Pattern[str] | None,
        max_lines: int,
    ) -> list[logging_pb2.LogEntry]:
        result: list[logging_pb2.LogEntry] = []
        for line_str in _reverse_read_lines(path):
            try:
                entry = self._json_to_entry(line_str)
            except Exception:
                continue
            if since_ms > 0 and entry.timestamp.epoch_ms <= since_ms:
                continue
            if regex_filter and not regex_filter.search(entry.data):
                continue
            result.append(entry)
            if len(result) >= max_lines:
                break
        result.reverse()
        return result

    def has_logs(self, task_id: JobName, attempt_id: int) -> bool:
        key = (task_id, attempt_id)
        if key in self._known_attempts:
            return True
        return self._attempt_path(task_id, attempt_id).exists()

    def clear_attempt(self, task_id: JobName, attempt_id: int) -> None:
        key = (task_id, attempt_id)
        with self._lock:
            self._known_attempts.discard(key)
        path = self._attempt_path(task_id, attempt_id)
        if path.exists():
            path.unlink()

    def close(self) -> None:
        """Clean up the temporary directory."""
        self._temp_dir.cleanup()
