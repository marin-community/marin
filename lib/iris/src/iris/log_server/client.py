# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clients for the push-based LogService.

LogPusher: low-level client for pushing log entries (used by workers for
    task stdout/stderr and process logs).
RemoteLogHandler: Python logging.Handler that batches and pushes log
    records to the LogService via RPC.
"""

from __future__ import annotations

import logging
import threading

from iris.logging import str_to_log_level
from iris.rpc import logging_pb2
from iris.rpc.logging_connect import LogServiceClientSync


class LogPusher:
    """Pushes log entries to a remote LogService via Connect/RPC."""

    def __init__(self, server_url: str, timeout_ms: int = 10_000) -> None:
        self._client = LogServiceClientSync(address=server_url, timeout_ms=timeout_ms)

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        if entries:
            self._client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))


class RemoteLogHandler(logging.Handler):
    """Python logging.Handler that batches records and pushes to a LogService.

    Records are buffered in memory and flushed either when the buffer reaches
    ``batch_size`` or when ``flush_interval`` seconds elapse, whichever comes
    first. A background daemon thread handles periodic flushing.

    Push failures are logged locally (via handleError) and never raise.
    """

    def __init__(
        self,
        pusher: LogPusher,
        key: str,
        batch_size: int = 50,
        flush_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self._pusher = pusher
        self._key = key
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: list[logging_pb2.LogEntry] = []
        self._lock = threading.Lock()
        self._closed = False

        # Periodic flush thread
        self._flush_timer: threading.Timer | None = None
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._closed:
            return
        self._flush_timer = threading.Timer(self._flush_interval, self._periodic_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _periodic_flush(self) -> None:
        with self._lock:
            self._do_flush()
        self._schedule_flush()

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
            with self._lock:
                self._buffer.append(entry)
                if len(self._buffer) >= self._batch_size:
                    self._do_flush()
        except Exception:
            self.handleError(record)

    def _do_flush(self) -> None:
        """Flush buffered entries. Caller must hold self._lock."""
        if not self._buffer:
            return
        entries = self._buffer
        self._buffer = []
        try:
            self._pusher.push(self._key, entries)
        except Exception:
            # Log locally but don't lose the entries silently.
            # We could re-buffer, but that risks unbounded growth if the
            # server is permanently down. Drop and let handleError report.
            logging.getLogger(__name__).debug(
                "Failed to push %d log entries for key %s", len(entries), self._key, exc_info=True
            )

    def flush(self) -> None:
        with self._lock:
            self._do_flush()

    def close(self) -> None:
        self._closed = True
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self.flush()
        super().close()
