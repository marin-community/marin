# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clients for the push-based LogService.

IrisLogClient: unified client for push + fetch with a caller-supplied
    endpoint resolver and failure-driven re-resolution. Buffers push entries
    per-key with a global cap; on retryable RPC failure the cached
    connection is invalidated and re-resolved on the next call.

RemoteLogHandler: logging.Handler that formats records and pushes them
    through an IrisLogClient.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import TypeVar

from iris.logging import str_to_log_level
from iris.rpc import logging_pb2
from iris.rpc.errors import call_with_retry, is_retryable_error
from iris.rpc.logging_connect import LogServiceClientSync

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


EndpointResolver = Callable[[], LogServiceClientSync]
"""Builds and returns a ``LogServiceClientSync`` pointed at the current log
service. Called lazily on first push/fetch and whenever the cached client
is invalidated by a retryable failure.

Callers own every construction detail (address lookup, auth interceptors,
timeouts) — ``IrisLogClient`` only treats the returned object as an opaque
handle with ``push_logs`` / ``fetch_logs`` / ``close``.

Raises on resolution failure; the client treats raises as a retryable
error and will re-invoke on the next call.
"""


class IrisLogClient:
    """Unified log client: push entries, fetch logs, re-resolve on failure.

    Thread-safe. One instance can be shared across threads; pushes from
    multiple threads are serialized at the RPC boundary.

    Re-resolution policy: failure-driven. A retryable RPC error
    (``UNAVAILABLE`` / ``INTERNAL`` / ``DEADLINE_EXCEEDED``) invalidates
    the cached connection; the next call re-resolves via the resolver.
    There is no periodic re-resolution tick: when the remote endpoint is
    healthy, zero resolver traffic is generated.

    Push buffering: entries are buffered per-key and flushed when either
    ``batch_size`` entries accumulate for a key or ``flush_interval``
    seconds elapse. A global ``max_buffered_entries`` cap bounds total
    memory; once hit, oldest entries are dropped (across all keys) and
    a sampled WARNING is logged.

    On a failed send, the batch is re-buffered at the head of the key's
    deque so the next successful push drains it first.
    """

    def __init__(
        self,
        resolver: EndpointResolver,
        *,
        batch_size: int = 1000,
        flush_interval: float = 5.0,
        max_buffered_entries: int = 100_000,
    ) -> None:
        self._resolver = resolver

        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_buffered_entries = max_buffered_entries

        # --- push buffering (guarded by _buf_lock) ---
        self._buf_lock = threading.Lock()
        self._buffers: dict[str, deque[logging_pb2.LogEntry]] = {}
        self._buffered_count = 0
        self._dropped_since_warn = 0
        self._warn_every = 1  # first drop logs immediately; backoff after

        # --- client cache (guarded by _client_lock) ---
        self._client_lock = threading.Lock()
        self._cached_client: LogServiceClientSync | None = None

        # Serializes RPC sends so close() can wait for in-flight flushes.
        self._send_lock = threading.Lock()
        self._closed = False

        self._flush_timer: threading.Timer | None = None
        self._schedule_flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        """Buffer ``entries`` under ``key``; flush if the key's batch is full."""
        if not entries:
            return
        to_send: list[logging_pb2.LogEntry] | None = None
        with self._buf_lock:
            buf = self._buffers.setdefault(key, deque())
            for entry in entries:
                self._evict_if_full_locked()
                buf.append(entry)
                self._buffered_count += 1
            if len(buf) >= self._batch_size:
                to_send = list(buf)
                buf.clear()
                self._buffered_count -= len(to_send)
        if to_send is not None:
            self._send(key, to_send)

    def fetch(self, request: logging_pb2.FetchLogsRequest) -> logging_pb2.FetchLogsResponse:
        """Fetch logs through the current endpoint, re-resolving on failure."""
        return self._call("fetch_logs", lambda client: client.fetch_logs(request))

    def flush(self) -> None:
        """Drain all per-key buffers through the remote LogService."""
        with self._buf_lock:
            snapshot = {k: list(v) for k, v in self._buffers.items() if v}
            for buf in self._buffers.values():
                buf.clear()
            self._buffered_count = 0
        for key, entries in snapshot.items():
            self._send(key, entries)

    def close(self) -> None:
        """Cancel periodic flush, drain buffers, close the cached RPC client."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self.flush()
        with self._send_lock:
            self._closed = True
        with self._client_lock:
            if self._cached_client is not None:
                try:
                    self._cached_client.close()
                except Exception:
                    logger.debug("IrisLogClient close: cached client raise ignored", exc_info=True)
                self._cached_client = None

    def as_logging_handler(
        self,
        key: str,
        *,
        level: int = logging.INFO,
        formatter: logging.Formatter | None = None,
    ) -> RemoteLogHandler:
        """Create a ``logging.Handler`` that emits records through this client.

        The returned handler's ``key`` attribute is mutable so callers can
        rename it mid-stream (e.g. worker rename-on-register) without
        rebuilding.
        """
        handler = RemoteLogHandler(self, key=key)
        handler.setLevel(level)
        if formatter is not None:
            handler.setFormatter(formatter)
        return handler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _schedule_flush(self) -> None:
        if self._closed:
            return
        self._flush_timer = threading.Timer(self._flush_interval, self._periodic_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _periodic_flush(self) -> None:
        try:
            self.flush()
        finally:
            self._schedule_flush()

    def _evict_if_full_locked(self) -> None:
        """Drop oldest entry across all keys if the buffer is full. Must hold ``_buf_lock``."""
        if self._buffered_count < self._max_buffered_entries:
            return
        # Find the deque with the oldest entry. ``dict`` preserves insertion
        # order, and we only append to existing deques, so the oldest entry
        # lives in the earliest-inserted non-empty deque.
        for buf in self._buffers.values():
            if buf:
                buf.popleft()
                self._buffered_count -= 1
                self._dropped_since_warn += 1
                if self._dropped_since_warn >= self._warn_every:
                    logger.warning(
                        "IrisLogClient buffer full (max=%d); dropped %d oldest entries",
                        self._max_buffered_entries,
                        self._dropped_since_warn,
                    )
                    self._dropped_since_warn = 0
                    # Exponential backoff on WARN frequency so a persistent
                    # full buffer doesn't spam at every drop.
                    self._warn_every = min(self._warn_every * 2, 10_000)
                return

    def _send(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        with self._send_lock:
            if self._closed:
                return
            try:
                self._call(
                    f"push_logs(key={key})",
                    lambda client: client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries)),
                )
            except Exception as exc:
                # Re-buffer at the head for retry on next successful send.
                if is_retryable_error(exc):
                    self._rebuffer(key, entries)
                    logger.warning(
                        "IrisLogClient: push failed for key=%s (%d entries); re-buffered for retry: %s",
                        key,
                        len(entries),
                        exc,
                    )
                else:
                    logger.warning(
                        "IrisLogClient: push failed for key=%s (%d entries) with non-retryable error; dropping: %s",
                        key,
                        len(entries),
                        exc,
                    )

    def _rebuffer(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        """Put ``entries`` back at the head of ``key``'s deque, honoring the global cap."""
        with self._buf_lock:
            buf = self._buffers.setdefault(key, deque())
            for entry in reversed(entries):
                self._evict_if_full_locked()
                if self._buffered_count >= self._max_buffered_entries:
                    # evict couldn't free space (buffer empty edge case); stop
                    break
                buf.appendleft(entry)
                self._buffered_count += 1

    def _call(
        self,
        operation: str,
        fn: Callable[[LogServiceClientSync], _T],
    ) -> _T:
        """Run ``fn`` against the cached client; on retryable error, invalidate and retry."""
        return call_with_retry(
            operation,
            lambda: fn(self._get_client()),
            on_retry=lambda exc: self._invalidate(str(exc)),
            max_attempts=3,
        )

    def _get_client(self) -> LogServiceClientSync:
        with self._client_lock:
            if self._cached_client is not None:
                return self._cached_client
            self._cached_client = self._resolver()
            logger.info("IrisLogClient resolved log service client")
            return self._cached_client

    def _invalidate(self, reason: str) -> None:
        """Drop the cached client so the next ``_get_client`` re-resolves."""
        with self._client_lock:
            if self._cached_client is None:
                return
            logger.warning("IrisLogClient: invalidating cached endpoint (%s)", reason)
            try:
                self._cached_client.close()
            except Exception:
                logger.debug("IrisLogClient _invalidate: cached client close raised", exc_info=True)
            self._cached_client = None


class RemoteLogHandler(logging.Handler):
    """Python logging.Handler that pushes records through an IrisLogClient.

    Created via ``IrisLogClient.as_logging_handler(key)``. The ``key``
    attribute is mutable so callers can rename mid-stream (worker
    rename-on-register) without rebuilding.
    """

    def __init__(self, pusher: IrisLogClient, key: str) -> None:
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
