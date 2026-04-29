# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types and helpers for finelog stores and clients.

Finelog treats keys as opaque strings — any structure (e.g.
``/user/<job>/<task>:<attempt>``) is caller convention, not a finelog
concern. The only exception is ``parse_attempt_id`` below, which the
DuckDB store uses to populate ``LogEntry.attempt_id`` for entries fetched
through pattern queries (best-effort; falls back to 0 on parse failure).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from finelog.rpc import logging_pb2

# Characters that indicate a regex pattern (vs. a literal key).
REGEX_META_RE = re.compile(r"[.*+?\[\](){}^$|\\]")

# Rough per-row size estimate for backpressure accounting in DuckDBLogStore.
_EST_BYTES_PER_ROW = 256


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry] = field(default_factory=list)
    cursor: int = 0  # max seq seen


class LogStoreProtocol(Protocol):
    """Minimal interface for log storage used by background collectors."""

    def append_batch(self, items: list[tuple[str, list]]) -> None: ...


class LogWriterProtocol(Protocol):
    """Minimal interface for writing log entries to the LogService.

    Satisfied by :class:`finelog.client.LogClient` (via ``write_batch``) and
    by lightweight test fakes that don't want to subclass LogClient. The
    protocol exists so collectors can accept either without import-cycling
    through ``finelog.client``.
    """

    def write_batch(self, key: str, messages: list[logging_pb2.LogEntry]) -> None: ...


_STR_TO_ENUM = {
    "DEBUG": logging_pb2.LOG_LEVEL_DEBUG,
    "INFO": logging_pb2.LOG_LEVEL_INFO,
    "WARNING": logging_pb2.LOG_LEVEL_WARNING,
    "ERROR": logging_pb2.LOG_LEVEL_ERROR,
    "CRITICAL": logging_pb2.LOG_LEVEL_CRITICAL,
}


def str_to_log_level(level_name: str | None) -> int:
    """Convert a level name (e.g. ``"INFO"``) to the LogLevel proto enum value.

    Returns ``LOG_LEVEL_UNKNOWN`` (0) for ``None``, empty strings, or
    unrecognized names.
    """
    if not level_name:
        return logging_pb2.LOG_LEVEL_UNKNOWN
    return _STR_TO_ENUM.get(level_name.upper(), logging_pb2.LOG_LEVEL_UNKNOWN)


def parse_attempt_id(key: str) -> int:
    """Best-effort attempt-id extraction from a structured key.

    Convention: keys ending in ``...:<int>`` carry an attempt id (e.g.
    ``/user/job/0:3``). Returns 0 when the suffix is missing or non-numeric.
    """
    if ":" not in key:
        return 0
    suffix = key.rsplit(":", 1)[1]
    if not suffix:
        return 0
    try:
        return int(suffix)
    except ValueError:
        return 0


def is_retryable_error(exc: Exception) -> bool:
    """True for transient ConnectRPC errors worth retrying.

    Retries on UNAVAILABLE / INTERNAL / DEADLINE_EXCEEDED / RESOURCE_EXHAUSTED.
    Application errors (NOT_FOUND, INVALID_ARGUMENT, etc.) are not retried.
    """
    if isinstance(exc, ConnectError):
        return exc.code in (
            Code.UNAVAILABLE,
            Code.INTERNAL,
            Code.DEADLINE_EXCEEDED,
            Code.RESOURCE_EXHAUSTED,
        )
    return False
