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
import sys
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

# Detached from the root logger: ``RemoteLogHandler`` lives on the root
# logger and calls ``LogPusher.push``, so if our own diagnostics reached
# the root they'd be enqueued right back into the pusher — a re-entrant
# loop that silently amplifies during failure storms. We send to stderr
# directly and set ``propagate = False`` so nothing here can feed the
# handler we serve.
logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_stderr_handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


MAX_LOG_BUFFER_SIZE = 10_000
"""Global cap on buffered entries across all keys. Older entries are
dropped first when the cap is exceeded."""

# Exponential backoff bounds between send failures. Prevents the drain
# loop from hot-spinning when the log server is unreachable. Reset after
# any successful batch send.
_BACKOFF_INITIAL_SEC = 0.5
_BACKOFF_MAX_SEC = 30.0


class LogPusher:
    """Buffered client for pushing log entries to a remote LogService.

    ``push`` is non-blocking: it appends to an in-memory queue and returns.
    A background thread drains the queue in per-key batches. Send failures
    re-buffer and back off exponentially — only the ``MAX_LOG_BUFFER_SIZE``
    overflow path drops entries.

    ``flush`` blocks until every entry enqueued before the call has been
    processed (sent or overflow-dropped). Use the ``timeout`` argument to
    bound the wait — by default ``flush`` waits indefinitely.

    ``server_url`` is passed to ``resolver`` (default: identity) to obtain
    the actual http address. Retryable failures invalidate the cached RPC
    client so the next send re-resolves.
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
        self._resolver: Callable[[str], str] = resolver if resolver is not None else (lambda url: url)
        self._timeout_ms = timeout_ms
        self._interceptors = tuple(interceptors)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_buffer_size = max_buffer_size

        # All shared state is guarded by _cond. The drain thread is the
        # only owner of _client, so no separate client lock. ``_queue`` is
        # a single FIFO of (seq, key, entry); the drain thread groups by
        # key just before sending. Trimming on overflow is one popleft.
        # ``seq`` is a monotonic per-entry counter used by blocking flush.
        self._cond = threading.Condition()
        self._queue: deque[tuple[int, str, logging_pb2.LogEntry]] = deque()
        self._closed = False

        # Monotonic counters for blocking flush(). ``_pushed_seq`` advances
        # on every entry enqueued. ``_processed_seq`` advances when the
        # drain thread acks an entry as either successfully sent or
        # overflow-dropped — both terminal states from flush's POV.
        self._pushed_seq = 0
        self._processed_seq = 0

        # Built lazily by the drain thread on first send; invalidated on
        # any failure so the next attempt re-resolves.
        self._client: LogServiceClientSync | None = None

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
                self._pushed_seq += 1
                self._queue.append((self._pushed_seq, key, e))
            self._trim_oldest_locked()
            if len(self._queue) >= self._batch_size:
                self._cond.notify_all()

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every entry enqueued before this call has been processed.

        "Processed" means either successfully sent or overflow-dropped —
        both terminal states. Returns ``True`` if the drain caught up,
        ``False`` on timeout. ``timeout=None`` waits indefinitely.

        For shutdown drain, prefer ``close`` (best-effort, won't block on
        a stuck server).
        """
        with self._cond:
            target = self._pushed_seq
            if target == 0 or self._processed_seq >= target:
                return True
            self._cond.notify_all()
            deadline = (time.monotonic() + timeout) if timeout is not None else None
            while self._processed_seq < target:
                if self._closed:
                    return self._processed_seq >= target
                if deadline is None:
                    # Re-check periodically so a wedged drain still surfaces.
                    self._cond.wait(timeout=1.0)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)
            return True

    def close(self) -> None:
        """Stop the drain thread after one best-effort drain, close the RPC client.

        Best-effort: if a send is in flight when ``close()`` returns the
        join timeout, we still close the cached client. Use ``flush()``
        first if you need to guarantee final delivery.
        """
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
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
        """Drop oldest entries until under ``_max_buffer_size``.

        Dropped entries advance ``_processed_seq`` so blocking ``flush``
        doesn't wait forever on entries that will never reach the server.
        """
        dropped = 0
        max_dropped_seq = 0
        while len(self._queue) > self._max_buffer_size:
            seq, _key, _entry = self._queue.popleft()
            if seq > max_dropped_seq:
                max_dropped_seq = seq
            dropped += 1
        if dropped:
            logger.warning(
                "LogPusher buffer overflow: dropped %d oldest entries (cap=%d)",
                dropped,
                self._max_buffer_size,
            )
            if max_dropped_seq > self._processed_seq:
                self._processed_seq = max_dropped_seq
                self._cond.notify_all()

    def _take_queue_locked(self) -> list[tuple[int, str, logging_pb2.LogEntry]]:
        """Drain the entire queue, preserving arrival order."""
        items = list(self._queue)
        self._queue.clear()
        return items

    def _rebuffer_at_head_locked(self, items: list[tuple[int, str, logging_pb2.LogEntry]]) -> None:
        """Put unsent items back at the head of the queue (original order)."""
        for triple in reversed(items):
            self._queue.appendleft(triple)
        self._trim_oldest_locked()

    # ------------------------------------------------------------------
    # Internal — drain thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Drain loop: wait for buffered entries, send them, retry failures with backoff.

        Close semantics: on ``close()`` the drain thread performs exactly
        one best-effort send pass over whatever is buffered, then exits.
        A second attempt against a presumably-dead server on close would
        just delay shutdown by the RPC timeout.
        """
        while not self._closed:
            with self._cond:
                while not self._closed and not self._queue:
                    self._cond.wait(timeout=self._flush_interval)
                if not self._queue:
                    return
                items = self._take_queue_locked()

            sent_max_seq, unsent = self._send_items(items)
            with self._cond:
                if sent_max_seq > self._processed_seq:
                    self._processed_seq = sent_max_seq
                    self._cond.notify_all()
            if not unsent:
                self._backoff.reset()
                continue

            with self._cond:
                self._rebuffer_at_head_locked(unsent)
            # Sleep on the backoff clock. Push-triggered notifies would
            # re-wake us immediately and defeat the backoff, so we loop
            # on _cond.wait until the deadline passes. ``close()`` breaks
            # the wait via ``_closed``; the outer ``while not self._closed``
            # then terminates the drain loop.
            deadline = time.monotonic() + self._backoff.next_interval()
            with self._cond:
                while not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)

    def _send_items(
        self,
        items: list[tuple[int, str, logging_pb2.LogEntry]],
    ) -> tuple[int, list[tuple[int, str, logging_pb2.LogEntry]]]:
        """Group ``items`` by key (stable on first occurrence) and push one
        RPC per key. Returns ``(max_sent_seq, unsent_items)``.

        On any failure, every item from that key onward is returned as
        unsent so the caller can re-buffer it at the head of the queue.
        Every failure mode — resolver error, retryable RPC error, or
        non-retryable RPC error — re-buffers so no log entries are silently
        dropped. Retryable errors additionally invalidate the cached client
        so the next attempt re-resolves the endpoint.
        """
        groups: dict[str, list[tuple[int, logging_pb2.LogEntry]]] = {}
        for seq, key, entry in items:
            groups.setdefault(key, []).append((seq, entry))

        sent_keys: set[str] = set()
        max_sent_seq = 0
        for key, seq_entries in groups.items():
            try:
                client = self._get_client()
            except Exception as exc:
                logger.warning("LogPusher: endpoint resolution failed: %s", exc)
                return max_sent_seq, [p for p in items if p[1] not in sent_keys]
            try:
                entries = [e for _s, e in seq_entries]
                client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))
                sent_keys.add(key)
                for seq, _e in seq_entries:
                    if seq > max_sent_seq:
                        max_sent_seq = seq
            except Exception as exc:
                retryable = is_retryable_error(exc)
                logger.warning(
                    "LogPusher: send failure for key=%s (%d entries, retryable=%s): %s",
                    key,
                    len(seq_entries),
                    retryable,
                    exc,
                )
                if retryable:
                    self._invalidate(str(exc))
                return max_sent_seq, [p for p in items if p[1] not in sent_keys]
        return max_sent_seq, []

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
        address = self._resolver(self._server_url)
        if not address:
            raise ConnectionError(f"LogPusher resolver returned empty address for {self._server_url!r}")
        self._client = self._build_client(address)
        logger.info("LogPusher resolved %s -> %s", self._server_url, address)
        return self._client

    def _invalidate(self, reason: str) -> None:
        """Drop the cached RPC client so the next send re-resolves. Drain-thread only."""
        if self._client is None:
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
