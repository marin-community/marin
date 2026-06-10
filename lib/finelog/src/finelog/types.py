# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types and helpers for finelog clients.

Finelog treats keys as opaque strings — any structure (e.g.
``/user/<job>/<task>:<attempt>``) is caller convention, not a finelog
concern.
"""

from __future__ import annotations

from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from finelog.rpc import logging_pb2


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
