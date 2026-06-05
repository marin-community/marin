# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateful incremental reader for a single LogStore key."""

from __future__ import annotations

from typing import Protocol

from finelog.rpc import logging_pb2
from finelog.types import LogReadResult


class _LogReader(Protocol):
    def get_logs(self, key: str, *, cursor: int = ..., max_lines: int = ...) -> LogReadResult: ...


class LogCursor:
    """Stateful incremental reader for a single LogStore key."""

    def __init__(self, store: _LogReader, key: str) -> None:
        self._store = store
        self._key = key
        self._cursor: int = 0

    def read(self, max_entries: int = 5000) -> list[logging_pb2.LogEntry]:
        result = self._store.get_logs(self._key, cursor=self._cursor, max_lines=max_entries)
        self._cursor = result.cursor
        return result.entries
