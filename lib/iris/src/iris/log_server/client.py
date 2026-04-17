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
from collections.abc import Callable, Iterable

from connectrpc.interceptor import Interceptor

from iris.logging import str_to_log_level
from iris.rpc import logging_pb2
from iris.rpc.errors import is_retryable_error
from iris.rpc.logging_connect import LogServiceClientSync

logger = logging.getLogger(__name__)


class LogPusher:
    """Buffered client for pushing log entries to a remote LogService.

    Entries are buffered per-key and flushed when either ``batch_size``
    entries accumulate or ``flush_interval`` seconds elapse.

    Endpoint resolution:
        If ``resolver`` is None, ``server_url`` is used as a direct http
        address and the RPC client is built eagerly.

        If ``resolver`` is set, ``server_url`` is passed to it on each
        resolution to obtain the actual http address — e.g. the worker
        passes ``server_url="iris://system/log-server"`` with a resolver
        that calls the controller's ``list_endpoints``. The underlying
        client is built lazily on first push. On a retryable RPC error
        (UNAVAILABLE / INTERNAL / DEADLINE_EXCEEDED) the cached client
        is invalidated and the next push re-invokes the resolver.
    """

    def __init__(
        self,
        server_url: str,
        timeout_ms: int = 10_000,
        batch_size: int = 1000,
        flush_interval: float = 5.0,
        interceptors: Iterable[Interceptor] = (),
        *,
        resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._server_url = server_url
        self._resolver = resolver
        self._timeout_ms = timeout_ms
        self._interceptors = tuple(interceptors)

        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffers: dict[str, list[logging_pb2.LogEntry]] = {}
        self._lock = threading.Lock()
        # Serializes RPC sends so close() can wait for in-flight flushes.
        self._send_lock = threading.Lock()
        self._closed = False

        # Without a resolver, build the client eagerly so we fail fast on
        # bad URLs. With a resolver, defer construction until first push —
        # the controller may not be reachable at construction time.
        self._client: LogServiceClientSync | None = None
        if self._resolver is None:
            self._client = self._build_client(self._server_url)

        self._flush_timer: threading.Timer | None = None
        self._schedule_flush()

    def _build_client(self, address: str) -> LogServiceClientSync:
        return LogServiceClientSync(
            address=address,
            timeout_ms=self._timeout_ms,
            interceptors=self._interceptors,
        )

    def _get_client(self) -> LogServiceClientSync:
        """Return the cached RPC client, resolving + constructing on demand.

        Must be called under ``_send_lock`` so invalidation is race-free
        against other senders.
        """
        if self._client is not None:
            return self._client
        assert self._resolver is not None  # cleared only by _invalidate
        address = self._resolver(self._server_url)
        if not address:
            raise ConnectionError(f"LogPusher resolver returned empty address for {self._server_url!r}")
        self._client = self._build_client(address)
        logger.info("LogPusher resolved %s -> %s", self._server_url, address)
        return self._client

    def _invalidate(self, reason: str) -> None:
        """Drop the cached RPC client so the next send re-resolves.

        Must be called under ``_send_lock``. No-op when no resolver is
        configured (a static-URL pusher has nothing to re-resolve to).
        """
        if self._resolver is None or self._client is None:
            return
        logger.warning("LogPusher: invalidating cached endpoint for %s (%s)", self._server_url, reason)
        try:
            self._client.close()
        except Exception:
            logger.debug("LogPusher _invalidate: cached client close raised", exc_info=True)
        self._client = None

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
                client = self._get_client()
                client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))
            except Exception as exc:
                logger.debug("Failed to push %d log entries for key %s", len(entries), key, exc_info=True)
                if is_retryable_error(exc):
                    self._invalidate(str(exc))

    def flush(self) -> None:
        """Force-flush all buffered entries."""
        self._flush_all()

    def close(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self.flush()
        with self._send_lock:
            self._closed = True
            if self._client is not None:
                self._client.close()
                self._client = None


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
