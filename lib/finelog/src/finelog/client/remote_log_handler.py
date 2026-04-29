# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``logging.Handler`` that ships records through a :class:`LogClient`.

Replaces the old ``RemoteLogHandler``-on-``LogPusher`` shape. The handler
formats records with its formatter, builds a :class:`LogEntry`, and calls
``client.write_batch(key, [entry])``. Batching, retries, and backoff live
inside the LogClient's per-namespace Table.
"""

from __future__ import annotations

import logging
from typing import Protocol

from finelog.rpc import logging_pb2
from finelog.types import str_to_log_level


class _WriteBatchClient(Protocol):
    """Subset of :class:`LogClient` the handler consumes.

    Declared structurally so test fakes don't have to subclass LogClient.
    """

    def write_batch(self, key: str, messages: list[logging_pb2.LogEntry]) -> None: ...

    def flush(self, timeout: float | None = None) -> bool: ...


class RemoteLogHandler(logging.Handler):
    """Ship Python log records to a finelog LogService through ``client``.

    Args:
        client: A :class:`LogClient` (or any object implementing the
            ``write_batch`` / ``flush`` subset).
        key: Logical log-source key (e.g. ``worker_log_key(worker_id)``).
            Mutable post-construction via the ``key`` property; useful when
            the controller assigns a worker_id post-registration.
    """

    def __init__(self, client: _WriteBatchClient, key: str) -> None:
        super().__init__()
        self._client = client
        self._key = key
        self._closed = False

    @property
    def key(self) -> str:
        return self._key

    @key.setter
    def key(self, value: str) -> None:
        self._key = value

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
            self._client.write_batch(self._key, [entry])
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        try:
            self._client.flush(timeout=0.5)
        except Exception:
            # Drain failure is logged inside the client; handler.flush() must
            # not raise (callers may invoke it from atexit / shutdown paths).
            pass

    def close(self) -> None:
        self._closed = True
        self.flush()
        super().close()
