# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Buffered client for pushing log entries to a remote LogService.

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
from rigging.timing import ExponentialBackoff, RateLimiter

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from finelog.types import is_retryable_error, str_to_log_level


# Detached from the root logger: ``RemoteLogHandler`` lives on the root
# logger and calls ``LogPusher.push``, so if our own diagnostics reached
# the root they'd be enqueued right back into the pusher — a re-entrant
# loop that silently amplifies during failure storms. We send to stderr
# directly and set ``propagate = False`` so nothing here can feed the
# handler we serve.
class _QuietStreamHandler(logging.StreamHandler):
    """StreamHandler that drops emit failures silently.

    This logger only carries LogPusher's own diagnostics. The drain thread
    is a daemon that outlives pytest's stderr capture (and interpreter
    shutdown), so any emit failure is a dead-stream symptom of teardown,
    not a LogPusher bug we could react to. Swallowing avoids the cascade
    of "--- Logging error ---" tracebacks during test teardown.
    """

    def handleError(self, record: logging.LogRecord) -> None:
        pass


logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    _stderr_handler = _QuietStreamHandler(sys.stderr)
    _stderr_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_stderr_handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


def _format_exc_summary(exc: BaseException) -> str:
    """Collapse a ConnectError-style exception to ``ClassName(CODE)``.

    The raw str(ConnectError) repeats the endpoint URL that's already
    visible from configuration and log context; a short summary keeps the
    drain-thread diagnostics readable during failure storms.
    """
    code = getattr(exc, "code", None)
    code_name = getattr(code, "name", None) or getattr(code, "value", None)
    if code_name is not None:
        return f"{type(exc).__name__}({code_name})"
    return f"{type(exc).__name__}: {exc}"


MAX_LOG_BUFFER_SIZE = 10_000
"""Global cap on buffered entries across all keys. Older entries are
dropped first when the cap is exceeded."""

# Exponential backoff bounds between send failures. Prevents the drain
# loop from hot-spinning when the log server is unreachable. Reset after
# any successful batch send.
_BACKOFF_INITIAL_SEC = 0.5
_BACKOFF_MAX_SEC = 30.0

# Minimum seconds between overflow warnings. Without throttling, every push
# to a full buffer emits its own warning — with the RemoteLogHandler pushing
# one entry per record, that is one stderr line per log record indefinitely.
_OVERFLOW_LOG_INTERVAL_SEC = 5.0


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

        self._cond = threading.Condition()
        self._queue: deque[tuple[int, str, logging_pb2.LogEntry]] = deque()
        self._closed = False

        self._pushed_seq = 0
        self._processed_seq = 0

        self._client: LogServiceClientSync | None = None

        self._backoff = ExponentialBackoff(initial=_BACKOFF_INITIAL_SEC, maximum=_BACKOFF_MAX_SEC, factor=2.0)

        self._overflow_dropped_pending = 0
        self._overflow_log_limiter = RateLimiter(interval_seconds=_OVERFLOW_LOG_INTERVAL_SEC)

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
        """Block until every entry enqueued before this call has been processed."""
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
                    self._cond.wait(timeout=1.0)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)
            return True

    def close(self) -> None:
        """Stop the drain thread after one best-effort drain, close the RPC client."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
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
        max_dropped_seq = 0
        while len(self._queue) > self._max_buffer_size:
            seq, _key, _entry = self._queue.popleft()
            if seq > max_dropped_seq:
                max_dropped_seq = seq
            dropped += 1
        if dropped:
            self._overflow_dropped_pending += dropped
            if self._overflow_log_limiter.should_run():
                logger.warning(
                    "LogPusher buffer overflow: dropped %d oldest entries (cap=%d)",
                    self._overflow_dropped_pending,
                    self._max_buffer_size,
                )
                self._overflow_dropped_pending = 0
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
        """Drain loop: wait for buffered entries, send them, retry failures with backoff."""
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
        """Group ``items`` by key and push one RPC per key."""
        groups: dict[str, list[tuple[int, logging_pb2.LogEntry]]] = {}
        for seq, key, entry in items:
            groups.setdefault(key, []).append((seq, entry))

        sent_keys: set[str] = set()
        max_sent_seq = 0
        for key, seq_entries in groups.items():
            try:
                client = self._get_client()
            except Exception as exc:
                logger.warning("LogPusher: endpoint resolution failed: %s", _format_exc_summary(exc))
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
                summary = _format_exc_summary(exc)
                logger.warning(
                    "LogPusher: send failure for key=%s (%d entries, retryable=%s): %s",
                    key,
                    len(seq_entries),
                    retryable,
                    summary,
                )
                if retryable:
                    self._invalidate(summary)
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
        self._pusher.flush(timeout=0.5)

    def close(self) -> None:
        self._closed = True
        self.flush()
        super().close()
