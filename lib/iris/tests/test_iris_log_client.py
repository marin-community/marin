# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for IrisLogClient: resolver caching, failure-driven re-resolution,
buffer overflow, and handler integration.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.log_server.client import IrisLogClient
from iris.rpc import logging_pb2


class FakeLogServiceClient:
    """Stand-in for LogServiceClientSync.

    Records all pushes/fetches. Each call may raise an exception drawn from
    ``errors``; once ``errors`` is exhausted calls succeed normally.
    """

    def __init__(self, label: str = "fake") -> None:
        self.label = label
        self.pushes: list[logging_pb2.PushLogsRequest] = []
        self.fetches: list[logging_pb2.FetchLogsRequest] = []
        self.errors: list[Exception] = []
        self.closed = False

    def _maybe_raise(self) -> None:
        if self.errors:
            raise self.errors.pop(0)

    def push_logs(self, request: logging_pb2.PushLogsRequest) -> logging_pb2.PushLogsResponse:
        self._maybe_raise()
        self.pushes.append(request)
        return logging_pb2.PushLogsResponse()

    def fetch_logs(self, request: logging_pb2.FetchLogsRequest) -> logging_pb2.FetchLogsResponse:
        self._maybe_raise()
        self.fetches.append(request)
        return logging_pb2.FetchLogsResponse()

    def close(self) -> None:
        self.closed = True


class TrackingResolver:
    """Counts calls and hands out pre-created FakeLogServiceClients in order.

    If ``fixed`` is set, every call returns the same client.
    """

    def __init__(self, *, fixed: FakeLogServiceClient | None = None) -> None:
        self.calls = 0
        self.created: list[FakeLogServiceClient] = []
        self._fixed = fixed

    def __call__(self) -> FakeLogServiceClient:
        self.calls += 1
        if self._fixed is not None:
            if not self.created:
                self.created.append(self._fixed)
            return self._fixed
        c = FakeLogServiceClient(label=f"fake#{len(self.created)}")
        self.created.append(c)
        return c


def _entry(data: str = "hello", level: int = logging_pb2.LOG_LEVEL_INFO) -> logging_pb2.LogEntry:
    e = logging_pb2.LogEntry(source="test", data=data, level=level)
    e.timestamp.epoch_ms = 1
    return e


def _unavailable() -> ConnectError:
    return ConnectError(Code.UNAVAILABLE, "server unavailable")


# ---------------------------------------------------------------------------
# Resolution and caching
# ---------------------------------------------------------------------------


def test_resolver_called_lazily_and_cached():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver)
    try:
        # No push yet → no resolution yet.
        assert resolver.calls == 0

        client.push("k", [_entry()])
        client.flush()
        assert resolver.calls == 1
        assert len(resolver.created[0].pushes) == 1

        # Second push reuses the cached client.
        client.push("k", [_entry()])
        client.flush()
        assert resolver.calls == 1
        assert len(resolver.created[0].pushes) == 2
    finally:
        client.close()


def test_on_retry_invalidates_cached_client():
    resolver = TrackingResolver()
    client = IrisLogClient(resolver, batch_size=1)
    try:
        # First push — forces resolution → fake#0.
        client.push("k", [_entry()])
        assert resolver.calls == 1
        assert len(resolver.created[0].pushes) == 1

        # Inject an UNAVAILABLE on the *next* push. call_with_retry will
        # invoke on_retry → _invalidate, then re-resolve for the retry,
        # creating fake#1, which handles the retry successfully.
        resolver.created[0].errors.append(_unavailable())

        client.push("k", [_entry()])
        assert resolver.calls == 2
        assert resolver.created[0].closed is True  # invalidated client was closed
        assert len(resolver.created[1].pushes) == 1
    finally:
        client.close()


def test_non_retryable_error_does_not_invalidate():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=1)
    try:
        client.push("k", [_entry()])
        assert resolver.calls == 1

        # NOT_FOUND is not retryable.
        resolver.created[0].errors.append(ConnectError(Code.NOT_FOUND, "missing"))
        client.push("k", [_entry()])

        # No re-resolution, cached client still in place.
        assert resolver.calls == 1
        assert resolver.created[0].closed is False
    finally:
        client.close()


def test_resolver_raising_is_treated_as_failure():
    calls = 0

    def resolver():
        nonlocal calls
        calls += 1
        raise ConnectionError("no endpoint")

    client = IrisLogClient(resolver, batch_size=1)
    try:
        # Push triggers resolve; resolve raises (non-retryable) → swallowed.
        client.push("k", [_entry()])
        # With non-retryable error the client is NOT re-invoked.
        # (ConnectionError is not a ConnectError, so is_retryable_error → False.)
        assert calls == 1
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Fetch path
# ---------------------------------------------------------------------------


def test_fetch_uses_same_client_cache():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver)
    try:
        client.fetch(logging_pb2.FetchLogsRequest(source="k"))
        assert resolver.calls == 1

        # Push reuses the cached client.
        client.push("k", [_entry()])
        client.flush()
        assert resolver.calls == 1
        assert len(resolver.created[0].fetches) == 1
        assert len(resolver.created[0].pushes) == 1
    finally:
        client.close()


def test_fetch_reresolves_on_unavailable():
    resolver = TrackingResolver()
    client = IrisLogClient(resolver)
    try:
        # First fetch succeeds on fake#0.
        client.fetch(logging_pb2.FetchLogsRequest(source="k"))
        assert resolver.calls == 1

        # Next fetch: UNAVAILABLE on #0 → invalidate → #1 handles retry.
        resolver.created[0].errors.append(_unavailable())
        client.fetch(logging_pb2.FetchLogsRequest(source="k"))
        assert resolver.calls == 2
        assert resolver.created[0].closed is True
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Buffering and overflow
# ---------------------------------------------------------------------------


def test_buffer_batches_and_flushes():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=3)
    try:
        client.push("k", [_entry("a")])
        client.push("k", [_entry("b")])
        # No flush yet — below batch_size.
        assert resolver.calls == 0

        client.push("k", [_entry("c")])
        # batch_size hit → pushed.
        assert resolver.calls == 1
        assert len(resolver.created[0].pushes) == 1
        assert [e.data for e in resolver.created[0].pushes[0].entries] == ["a", "b", "c"]
    finally:
        client.close()


def test_buffer_overflow_drops_oldest_with_warning(caplog):
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=1000, max_buffered_entries=3)
    try:
        with caplog.at_level(logging.WARNING, logger="iris.log_server.client"):
            client.push("k", [_entry(f"{i}") for i in range(5)])
        # Below batch_size, nothing sent yet.
        assert resolver.calls == 0
        client.flush()
        remaining = [e.data for e in resolver.created[0].pushes[0].entries]
        # Oldest 2 dropped; "2","3","4" survive in order.
        assert remaining == ["2", "3", "4"]
        assert any("buffer full" in r.message for r in caplog.records)
    finally:
        client.close()


def test_send_failure_rebuffers():
    # Factory: first client always fails, second succeeds.
    created: list[FakeLogServiceClient] = []

    def resolver() -> FakeLogServiceClient:
        c = FakeLogServiceClient(label=f"fake#{len(created)}")
        if not created:
            c.errors = [_unavailable() for _ in range(10)]
        created.append(c)
        return c

    client = IrisLogClient(resolver, batch_size=1)
    try:
        # First push fails on fake#0 all 3 retry attempts → re-buffered.
        client.push("k", [_entry("a")])
        # Next push re-resolves; fake#1 succeeds and drains both entries.
        client.push("k", [_entry("b")])
        client.flush()

        healthy_pushes = [p for c in created if not c.errors for p in c.pushes]
        all_data = [e.data for p in healthy_pushes for e in p.entries]
        assert "a" in all_data
        assert "b" in all_data
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Handler integration
# ---------------------------------------------------------------------------


def test_as_logging_handler_emits_through_client():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=1)
    try:
        handler = client.as_logging_handler("worker/w0")
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        assert resolver.created[0].pushes[0].key == "worker/w0"
        assert resolver.created[0].pushes[0].entries[0].data == "hello world"
    finally:
        client.close()


def test_handler_key_rename():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=1)
    try:
        handler = client.as_logging_handler("worker/old")
        handler.setFormatter(logging.Formatter("%(message)s"))

        handler.emit(logging.LogRecord("t", logging.INFO, "", 0, "a", (), None))
        handler.key = "worker/new"
        handler.emit(logging.LogRecord("t", logging.INFO, "", 0, "b", (), None))

        keys = [p.key for p in resolver.created[0].pushes]
        assert keys == ["worker/old", "worker/new"]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_pushes_do_not_lose_entries():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=50)
    try:
        n_threads = 8
        per_thread = 200

        def worker(tid: int) -> None:
            for i in range(per_thread):
                client.push(f"k{tid}", [_entry(f"{tid}:{i}")])

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        client.flush()

        total_pushed = sum(len(p.entries) for p in resolver.created[0].pushes)
        assert total_pushed == n_threads * per_thread
    finally:
        client.close()


def test_close_stops_periodic_flush():
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, flush_interval=0.05)
    client.push("k", [_entry()])
    client.close()

    # After close, periodic flush is cancelled; no new RPC calls should go out.
    pushes_at_close = len(resolver.created[0].pushes) if resolver.created else 0
    time.sleep(0.2)
    pushes_after_wait = len(resolver.created[0].pushes) if resolver.created else 0
    assert pushes_after_wait == pushes_at_close


@pytest.mark.parametrize("batch_size", [1, 5])
def test_resolver_not_reinvoked_on_steady_state(batch_size: int):
    resolver = TrackingResolver(fixed=FakeLogServiceClient())
    client = IrisLogClient(resolver, batch_size=batch_size)
    try:
        for i in range(50):
            client.push("k", [_entry(f"{i}")])
        client.flush()
        # One and only one resolve across 50 pushes.
        assert resolver.calls == 1
    finally:
        client.close()
