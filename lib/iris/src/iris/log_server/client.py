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
import time
from collections import deque
from collections.abc import Callable, Iterable

from connectrpc.interceptor import Interceptor
from rigging.timing import ExponentialBackoff

from iris.logging import str_to_log_level
from iris.rpc import logging_pb2
from iris.rpc.errors import is_retryable_error
from iris.rpc.logging_connect import LogServiceClientSync

logger = logging.getLogger(__name__)


MAX_LOG_BUFFER_SIZE = 10_000
"""Global cap on buffered entries across all keys. Older entries are
dropped first when the cap is exceeded."""

# Exponential backoff bounds between send failures. Prevents the drain
# loop from hot-spinning when the log server is unreachable. Reset after
# any successful batch send.
_BACKOFF_INITIAL_SEC = 0.5
_BACKOFF_MAX_SEC = 30.0


class LogPusher:
    """Buffered non-blocking client for pushing log entries to a remote LogService.

    ``push`` always returns immediately — entries land in an in-memory
    deque and a dedicated background thread drains them to the LogService
    in batches. The thread:

    - sleeps on a condition variable, waking whenever ``batch_size``
      entries accumulate, ``flush()`` is called, or after
      ``flush_interval`` seconds (whichever fires first);
    - on any send failure, re-buffers the batch at the head of the key's
      deque and backs off briefly before retrying. Retryable errors
      additionally invalidate the cached RPC client so the next attempt
      re-resolves the endpoint;
    - when total buffered entries exceed ``MAX_LOG_BUFFER_SIZE``, drops
      the oldest entries across keys. This is the only path that discards
      log entries — send failures never drop.

    Endpoint resolution:
        If ``resolver`` is None, ``server_url`` is used as a direct http
        address and the RPC client is built eagerly.

        If ``resolver`` is set, ``server_url`` is passed to it on each
        resolution to obtain the actual http address. The client is built
        lazily and invalidated on retryable failures; the next send will
        re-invoke the resolver.
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
        max_buffer_size: int = MAX_LOG_BUFFER_SIZE,
    ) -> None:
        self._server_url = server_url
        self._resolver = resolver
        self._timeout_ms = timeout_ms
        self._interceptors = tuple(interceptors)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_buffer_size = max_buffer_size

        # All shared state is guarded by _cond. The drain thread is the
        # only owner of _client, so no separate client lock. ``_queue`` is
        # a single FIFO of (key, entry); the drain thread groups by key
        # just before sending. Trimming on overflow is one popleft.
        self._cond = threading.Condition()
        self._queue: deque[tuple[str, logging_pb2.LogEntry]] = deque()
        self._closed = False

        # Without a resolver, build the client eagerly so we fail fast on
        # bad URLs. With a resolver, defer until the drain thread needs it.
        self._client: LogServiceClientSync | None = None
        if self._resolver is None:
            self._client = self._build_client(self._server_url)

        # Owned by the drain thread; reset after any successful send.
        self._backoff = ExponentialBackoff(initial=_BACKOFF_INITIAL_SEC, maximum=_BACKOFF_MAX_SEC, factor=2.0)

        self._thread = threading.Thread(target=self._run, name="log-pusher", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        """Append ``entries`` to the outbound queue. Never blocks the caller."""
        if not entries:
            return
        with self._cond:
            if self._closed:
                return
            for e in entries:
                self._queue.append((key, e))
            self._trim_oldest_locked()
            if len(self._queue) >= self._batch_size:
                self._cond.notify()

    def flush(self) -> None:
        """Poke the drain thread to send whatever is buffered now.

        Non-blocking. For draining on shutdown, use ``close``.
        """
        with self._cond:
            if self._queue:
                self._cond.notify()

    def close(self) -> None:
        """Stop the drain thread after one best-effort drain, close the RPC client."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify()
        # Join the drain thread; it will send what it can and exit.
        self._thread.join(timeout=max(self._flush_interval * 2, 10.0))
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("LogPusher close: cached client close raised", exc_info=True)
            self._client = None

    # ------------------------------------------------------------------
    # Internal — buffer management (callers hold ``_cond``)
    # ------------------------------------------------------------------

    def _trim_oldest_locked(self) -> None:
        """Drop oldest entries until under ``_max_buffer_size``."""
        dropped = 0
        while len(self._queue) > self._max_buffer_size:
            self._queue.popleft()
            dropped += 1
        if dropped:
            logger.warning(
                "LogPusher buffer overflow: dropped %d oldest entries (cap=%d)",
                dropped,
                self._max_buffer_size,
            )

    def _take_queue_locked(self) -> list[tuple[str, logging_pb2.LogEntry]]:
        """Drain the entire queue, preserving arrival order."""
        items = list(self._queue)
        self._queue.clear()
        return items

    def _rebuffer_at_head_locked(self, items: list[tuple[str, logging_pb2.LogEntry]]) -> None:
        """Put unsent items back at the head of the queue (original order)."""
        for pair in reversed(items):
            self._queue.appendleft(pair)
        self._trim_oldest_locked()

    # ------------------------------------------------------------------
    # Internal — drain thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Drain loop: wait for buffered entries, send them, retry failures with backoff."""
        while True:
            with self._cond:
                while not self._closed and not self._queue:
                    self._cond.wait(timeout=self._flush_interval)
                if not self._queue:
                    # _closed must be True here; nothing left to flush.
                    return
                items = self._take_queue_locked()

            unsent = self._send_items(items)
            if not unsent:
                self._backoff.reset()
                continue

            with self._cond:
                self._rebuffer_at_head_locked(unsent)
                if self._closed:
                    # Best-effort on close: don't loop forever against an
                    # unreachable server. Unsent entries are left in the
                    # queue and lost on process exit.
                    return
            # Sleep on the backoff clock. Push-triggered notifies would
            # re-wake us immediately and defeat the backoff, so we loop
            # on _cond.wait until the deadline passes — only ``close()``
            # can (correctly) shortcut by setting ``_closed``.
            deadline = time.monotonic() + self._backoff.next_interval()
            with self._cond:
                while not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)
                if self._closed:
                    return

    def _send_items(
        self,
        items: list[tuple[str, logging_pb2.LogEntry]],
    ) -> list[tuple[str, logging_pb2.LogEntry]]:
        """Group ``items`` by key (stable on first occurrence) and push one
        RPC per key. On any failure, return every item from that key onward
        so the caller can re-buffer it at the head of the queue.

        Every failure mode — resolver error, retryable RPC error, or
        non-retryable RPC error — re-buffers so no log entries are silently
        dropped. Retryable errors additionally invalidate the cached client
        so the next attempt re-resolves the endpoint.
        """
        groups: dict[str, list[logging_pb2.LogEntry]] = {}
        for key, entry in items:
            groups.setdefault(key, []).append(entry)

        sent_keys: set[str] = set()
        for key, entries in groups.items():
            try:
                client = self._get_client()
            except Exception as exc:
                logger.warning("LogPusher: endpoint resolution failed: %s", exc)
                return [p for p in items if p[0] not in sent_keys]
            try:
                client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))
                sent_keys.add(key)
            except Exception as exc:
                retryable = is_retryable_error(exc)
                logger.warning(
                    "LogPusher: send failure for key=%s (%d entries, retryable=%s): %s",
                    key,
                    len(entries),
                    retryable,
                    exc,
                )
                if retryable:
                    self._invalidate(str(exc))
                return [p for p in items if p[0] not in sent_keys]
        return []

    def _build_client(self, address: str) -> LogServiceClientSync:
        return LogServiceClientSync(
            address=address,
            timeout_ms=self._timeout_ms,
            interceptors=self._interceptors,
        )

    def _get_client(self) -> LogServiceClientSync:
        """Return the cached RPC client, resolving on demand. Drain-thread only."""
        if self._client is not None:
            return self._client
        assert self._resolver is not None
        address = self._resolver(self._server_url)
        if not address:
            raise ConnectionError(f"LogPusher resolver returned empty address for {self._server_url!r}")
        self._client = self._build_client(address)
        logger.info("LogPusher resolved %s -> %s", self._server_url, address)
        return self._client

    def _invalidate(self, reason: str) -> None:
        """Drop the cached RPC client so the next send re-resolves. Drain-thread only."""
        if self._resolver is None or self._client is None:
            return
        logger.info("LogPusher: invalidating cached endpoint for %s (%s)", self._server_url, reason)
        try:
            self._client.close()
        except Exception:
            logger.debug("LogPusher _invalidate: cached client close raised", exc_info=True)
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
