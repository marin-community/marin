# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store package.

Exports ``LogStore`` (an alias for :class:`DuckDBLogStore`) plus the
``LogCursor`` and ``LogStoreHandler`` helpers that wrap it. Constructed
without a ``log_dir`` it owns a tempdir that is cleaned up on ``close``,
which is the form tests use.
"""

from __future__ import annotations

import logging

from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore as LogStore
from finelog.types import _EST_BYTES_PER_ROW, LogReadResult, str_to_log_level


class LogCursor:
    """Stateful incremental reader for a single LogStore key."""

    def __init__(self, store: LogStore, key: str) -> None:
        self._store = store
        self._key = key
        self._cursor: int = 0

    def read(self, max_entries: int = 5000) -> list[logging_pb2.LogEntry]:
        result = self._store.get_logs(self._key, cursor=self._cursor, max_lines=max_entries)
        self._cursor = result.cursor
        return result.entries


class LogStoreHandler(logging.Handler):
    """Logging handler that writes formatted records directly into a LogStore."""

    def __init__(self, log_store: LogStore, key: str):
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


__all__ = [
    "_EST_BYTES_PER_ROW",
    "LogCursor",
    "LogReadResult",
    "LogStore",
    "LogStoreHandler",
]
