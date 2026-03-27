# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure in-memory LogStore for tests and CI. No DuckDB, no Parquet, no background threads."""

from __future__ import annotations

import re
from pathlib import Path
from threading import Lock

from iris.cluster.log_store._types import LogReadResult
from iris.cluster.types import TaskAttempt
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

# Row layout: (seq, key, source, data, epoch_ms, level)
_Row = tuple[int, str, str, str, int, int]


def _matches_common_filters(row: _Row, since_ms: int, substring_filter: str, min_level_enum: int) -> bool:
    _, _, _, data, epoch_ms, level = row
    if since_ms > 0 and epoch_ms <= since_ms:
        return False
    if substring_filter and substring_filter not in data:
        return False
    if min_level_enum > 0 and level != 0 and level < min_level_enum:
        return False
    return True


def _rows_to_entries(rows: list[_Row], include_key: bool, exact_key: str | None = None) -> list[logging_pb2.LogEntry]:
    entries: list[logging_pb2.LogEntry] = []
    # For exact-key queries, parse attempt_id once from the key.
    fixed_attempt_id = 0
    if not include_key and exact_key and ":" in exact_key:
        try:
            parsed = TaskAttempt.from_wire(exact_key)
            fixed_attempt_id = parsed.attempt_id if parsed.attempt_id is not None else 0
        except ValueError:
            pass
    for row in rows:
        _seq, key, source, data, epoch_ms, level = row
        entry = logging_pb2.LogEntry(source=source, data=data, level=level)
        entry.timestamp.epoch_ms = epoch_ms
        if include_key:
            entry.key = key
            parsed = TaskAttempt.from_wire(key)
            entry.attempt_id = parsed.attempt_id if parsed.attempt_id is not None else 0
        else:
            entry.attempt_id = fixed_attempt_id
        entries.append(entry)
    return entries


def _like_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a SQL LIKE pattern (with % and _ wildcards) to a compiled regex."""
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            # Escaped character: treat next char literally.
            parts.append(re.escape(pattern[i + 1]))
            i += 2
        elif ch == "%":
            parts.append(".*")
            i += 1
        elif ch == "_":
            parts.append(".")
            i += 1
        else:
            parts.append(re.escape(ch))
            i += 1
    return re.compile("".join(parts), re.DOTALL)


class MemStore:
    """In-memory LogStore drop-in for tests. Thread-safe, zero dependencies beyond protobuf."""

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        max_local_segments: int = 50,
        max_local_bytes: int = 5 * 1024**3,
        flush_interval_sec: float = 600.0,
        segment_target_bytes: int = 100 * 1024 * 1024,
        duckdb_memory_limit: str | None = None,
    ):
        self._lock = Lock()
        self._rows: list[_Row] = []
        self._next_seq = 1

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        with self._lock:
            first_seq = self._next_seq
            self._next_seq += len(entries)
            self._rows.extend(
                (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
            )

    def append_batch(self, items: list[tuple[str, list]]) -> None:
        with self._lock:
            for key, entries in items:
                if not entries:
                    continue
                first_seq = self._next_seq
                self._next_seq += len(entries)
                self._rows.extend(
                    (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
                )

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
        min_level_enum = str_to_log_level(min_level) if min_level else 0
        is_pattern = "%" in key or "_" in key

        if is_pattern:
            pat = _like_to_regex(key)
            key_match = lambda k: pat.fullmatch(k) is not None  # noqa: E731
        else:
            key_match = lambda k: k == key  # noqa: E731

        with self._lock:
            rows = [
                r
                for r in self._rows
                if key_match(r[1])
                and r[0] > cursor
                and _matches_common_filters(r, since_ms, substring_filter, min_level_enum)
            ]

        rows.sort(key=lambda r: r[0])

        if tail and max_lines > 0:
            rows = rows[-max_lines:]
        elif max_lines > 0:
            rows = rows[:max_lines]

        if not rows:
            return LogReadResult(entries=[], cursor=cursor)

        max_seq = max(r[0] for r in rows)
        return LogReadResult(
            entries=_rows_to_entries(rows, include_key=is_pattern, exact_key=key if not is_pattern else None),
            cursor=max_seq,
        )

    def has_logs(self, key: str) -> bool:
        with self._lock:
            return any(r[1] == key for r in self._rows)

    def cursor(self, key: str):
        from iris.cluster.log_store import LogCursor

        return LogCursor(self, key)

    def close(self) -> None:
        pass
