# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure in-memory LogStore for tests and CI. No DuckDB, no Parquet, no background threads."""

from __future__ import annotations

import re
from pathlib import Path
from threading import Lock

from finelog.rpc import logging_pb2
from finelog.types import REGEX_META_RE, LogReadResult, parse_attempt_id, str_to_log_level

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
    fixed_attempt_id = parse_attempt_id(exact_key) if (not include_key and exact_key) else 0
    for row in rows:
        _seq, key, source, data, epoch_ms, level = row
        entry = logging_pb2.LogEntry(source=source, data=data, level=level)
        entry.timestamp.epoch_ms = epoch_ms
        if include_key:
            entry.key = key
            entry.attempt_id = parse_attempt_id(key)
        else:
            entry.attempt_id = fixed_attempt_id
        entries.append(entry)
    return entries


class MemStore:
    """In-memory LogStore drop-in for tests. Thread-safe, zero dependencies beyond protobuf.

    ``max_rows`` caps total retained entries; oldest are evicted FIFO when the
    cap is exceeded. ``None`` (the default) is unbounded — fine for tests, not
    for any path that ingests untrusted volumes (e.g. the controller's bundled
    fallback log server, which sets an explicit cap).
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        max_rows: int | None = None,
        max_local_segments: int = 50,
        max_local_bytes: int = 5 * 1024**3,
        flush_interval_sec: float = 600.0,
        segment_target_bytes: int = 100 * 1024 * 1024,
        duckdb_memory_limit: str | None = None,
    ):
        self._lock = Lock()
        self._rows: list[_Row] = []
        self._next_seq = 1
        self._max_rows = max_rows

    def _trim_locked(self) -> None:
        if self._max_rows is None:
            return
        excess = len(self._rows) - self._max_rows
        if excess > 0:
            del self._rows[:excess]

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        self.append_batch([(key, entries)])

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
            self._trim_locked()

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
        min_level_enum = str_to_log_level(min_level)
        is_pattern = bool(REGEX_META_RE.search(key))

        if is_pattern:
            pat = re.compile(key)
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
        from finelog.store import LogCursor

        return LogCursor(self, key)

    def close(self) -> None:
        pass

    def _compact_step(self) -> None:
        # MemStore has no pending buffer — writes are visible to readers
        # immediately. Present for interface parity with DuckDBLogStore so
        # test fixtures can force a sync point uniformly.
        pass
