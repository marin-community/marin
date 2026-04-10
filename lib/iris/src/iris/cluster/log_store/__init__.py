# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store package.

Exports ``LogStore`` as the environment-appropriate implementation:
- Tests / CI (``PYTEST_CURRENT_TEST`` or ``CI`` set): in-memory ``MemStore``
- Production: DuckDB-backed ``DuckDBLogStore``

All consumers should import from this package:
``from iris.cluster.log_store import LogStore, LogCursor, ...``
"""

from __future__ import annotations

import logging
import os

from iris.cluster.log_store._types import (
    CONTROLLER_LOG_KEY,
    LogReadResult,
    _EST_BYTES_PER_ROW,
    build_log_source,
    task_log_key,
    worker_log_key,
)
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2


def _is_test_environment() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "CI" in os.environ


def _has_duckdb() -> bool:
    try:
        import duckdb

        return True
    except ImportError:
        return False


if _is_test_environment() or not _has_duckdb():
    from iris.cluster.log_store.mem_store import MemStore as LogStore
else:
    from iris.cluster.log_store.duckdb_store import DuckDBLogStore as LogStore


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
    "CONTROLLER_LOG_KEY",
    "_EST_BYTES_PER_ROW",
    "LogCursor",
    "LogReadResult",
    "LogStore",
    "LogStoreHandler",
    "build_log_source",
    "task_log_key",
    "worker_log_key",
]
