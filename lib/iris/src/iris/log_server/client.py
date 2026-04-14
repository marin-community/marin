# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clients for the push-based LogService.

LogPusher: buffered client for pushing log entries. Batches entries and
    flushes every ``flush_interval`` seconds or ``batch_size`` entries,
    whichever comes first.
RemoteLogHandler: Python logging.Handler that formats records and pushes
    them through a LogPusher.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

from connectrpc.interceptor import Interceptor

from iris.logging import str_to_log_level
from iris.rpc import logging_pb2
from iris.rpc.logging_connect import LogServiceClientSync
from rigging.timing import RateLimiter

logger = logging.getLogger(__name__)

# Internal logger for LogPusher diagnostics.  Writes directly to stderr and
# does NOT propagate to the root logger, which may have a RemoteLogHandler
# attached in worker mode.  This avoids re-entrant deadlocks when _send()
# logs a warning that would otherwise be fed back through push() → _send().
_pusher_logger = logging.getLogger(__name__ + "._internal")
_pusher_logger.propagate = False
_pusher_logger.setLevel(logging.DEBUG)
if not _pusher_logger.handlers:
    _pusher_logger.addHandler(logging.StreamHandler())

# Minimum interval between repeated failure warnings (seconds).
_WARN_INTERVAL = 60

# Per-key cap on buffered entries to prevent unbounded memory growth when the
# log server is down for extended periods.
_MAX_BUFFER_PER_KEY = 10_000


class LogPusher:
    """Buffered client for pushing log entries to a remote LogService.

    Entries are buffered per-key and flushed when either ``batch_size``
    entries accumulate or ``flush_interval`` seconds elapse.
    """

    def __init__(
        self,
        server_url: str,
        timeout_ms: int = 10_000,
        batch_size: int = 1000,
        flush_interval: float = 5.0,
        interceptors: Iterable[Interceptor] = (),
    ) -> None:
        self._client = LogServiceClientSync(address=server_url, timeout_ms=timeout_ms, interceptors=tuple(interceptors))
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffers: dict[str, list[logging_pb2.LogEntry]] = {}
        self._lock = threading.Lock()
        # Serializes RPC sends so close() can wait for in-flight flushes.
        self._send_lock = threading.Lock()
        self._closed = False

        self._flush_timer: threading.Timer | None = None
        self._consecutive_failures = 0
        self._warn_limiter = RateLimiter(interval_seconds=_WARN_INTERVAL)
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._closed:
            return
        self._flush_timer = threading.Timer(self._flush_interval, self._periodic_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _periodic_flush(self) -> None:
        self._flush_all()
        self._schedule_flush()

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        if not entries:
            return
        to_send: list[logging_pb2.LogEntry] | None = None
        with self._lock:
            buf = self._buffers.get(key)
            if buf is None:
                buf = []
                self._buffers[key] = buf
            buf.extend(entries)
            if len(buf) >= self._batch_size:
                to_send = self._buffers.pop(key, None)
        if to_send:
            self._send(key, to_send)

    def _flush_all(self) -> None:
        with self._lock:
            snapshot = self._buffers
            self._buffers = {}
        for key, entries in snapshot.items():
            self._send(key, entries)

    def _send(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        with self._send_lock:
            if self._closed:
                return
            try:
                self._client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))
            except Exception:
                self._consecutive_failures += 1
                if self._warn_limiter.should_run():
                    _pusher_logger.warning(
                        "Failed to push %d log entries for key %s (%d consecutive failures)",
                        len(entries),
                        key,
                        self._consecutive_failures,
                        exc_info=True,
                    )
                # Retain failed entries for retry on the next flush cycle.
                self._requeue(key, entries)
                return
            if self._consecutive_failures > 0:
                _pusher_logger.info("Log push recovered after %d consecutive failures", self._consecutive_failures)
                self._consecutive_failures = 0
                self._warn_limiter.reset()

    def _requeue(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        """Re-buffer entries that failed to send, for retry on the next flush."""
        with self._lock:
            buf = self._buffers.setdefault(key, [])
            buf.extend(entries)
            overflow = len(buf) - _MAX_BUFFER_PER_KEY
            if overflow > 0:
                del buf[:overflow]
                _pusher_logger.warning(
                    "Dropped %d oldest log entries for key %s (buffer cap %d)",
                    overflow,
                    key,
                    _MAX_BUFFER_PER_KEY,
                )

    def flush(self) -> None:
        """Force-flush all buffered entries."""
        self._flush_all()

    def close(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self.flush()
        with self._send_lock:
            self._closed = True
        self._client.close()


class LogServiceProxy:
    """Protocol adapter that forwards push_logs/fetch_logs to a remote LogService over RPC.

    Bridges ``LogServiceClientSync`` (an RPC client with kwargs-only,
    ctx-less methods) to the ``LogServiceSync`` protocol (positional
    ``ctx`` arg) expected by ``LogServiceWSGIApplication`` and the
    controller/dashboard call sites. Used in place of ``LogServiceImpl``
    when the log service is hosted in a separate process.
    """

    def __init__(
        self,
        address: str,
        timeout_ms: int = 10_000,
        interceptors: Iterable[Interceptor] = (),
    ) -> None:
        self._client = LogServiceClientSync(address=address, timeout_ms=timeout_ms, interceptors=tuple(interceptors))

    def push_logs(
        self,
        request: logging_pb2.PushLogsRequest,
        ctx: object,
    ) -> logging_pb2.PushLogsResponse:
        return self._client.push_logs(request)

    def fetch_logs(
        self,
        request: logging_pb2.FetchLogsRequest,
        ctx: object,
    ) -> logging_pb2.FetchLogsResponse:
        return self._client.fetch_logs(request)

    def close(self) -> None:
        self._client.close()


class RemoteLogHandler(logging.Handler):
    """Python logging.Handler that pushes records through a LogPusher.

    The LogPusher handles batching and periodic flushing, so this handler
    simply converts each record to a LogEntry and pushes it.
    """

    def __init__(self, pusher: LogPusher, key: str) -> None:
        super().__init__()
        self._pusher = pusher
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
            self._pusher.push(self._key, [entry])
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self._pusher.flush()

    def close(self) -> None:
        self._closed = True
        self.flush()
        super().close()
