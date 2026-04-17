# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogPusher buffering and RemoteLogHandler."""

import logging
import threading

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.log_server import client as client_mod
from iris.log_server.client import LogPusher, RemoteLogHandler
from iris.rpc import logging_pb2


class FakeLogPusher:
    """Implements the LogPusherProtocol interface, recording calls."""

    def __init__(self, *, fail: bool = False) -> None:
        self.pushed: list[tuple[str, list[logging_pb2.LogEntry]]] = []
        self._fail = fail

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        self.pushed.append((key, list(entries)))
        if self._fail:
            raise ConnectionError("server unavailable")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture()
def fake_pusher():
    return FakeLogPusher()


@pytest.fixture()
def handler(fake_pusher):
    h = RemoteLogHandler(fake_pusher, key="test")
    yield h
    h.close()


def test_handler_pushes_entries(handler: RemoteLogHandler, fake_pusher: FakeLogPusher):
    logger = logging.getLogger("test_handler_push")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("hello")
        assert len(fake_pusher.pushed) == 1
        assert fake_pusher.pushed[0][1][0].data.endswith("hello")
    finally:
        logger.removeHandler(handler)


def test_handler_flush_delegates_to_pusher():
    """RemoteLogHandler.flush() calls pusher.flush()."""
    pusher = FakeLogPusher()
    handler = RemoteLogHandler(pusher, key="test")
    logger = logging.getLogger("test_handler_flush")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("msg")
        handler.flush()
        assert len(pusher.pushed) == 1
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_no_deadlock_on_push_failure():
    """When push fails, the error log must not deadlock by re-entering emit().

    We verify this completes within 2 seconds (a deadlock would hang forever).
    """
    pusher = FakeLogPusher(fail=True)
    handler = RemoteLogHandler(pusher, key="test")
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    done = threading.Event()

    def log_one():
        try:
            logging.getLogger("test_deadlock").info("trigger flush")
        finally:
            done.set()

    t = threading.Thread(target=log_one)
    t.start()
    finished = done.wait(timeout=2.0)
    root.removeHandler(handler)
    handler.close()
    t.join(timeout=1.0)
    assert finished, "RemoteLogHandler deadlocked on push failure"


def _wait_for(predicate, timeout: float = 5.0) -> None:
    """Poll ``predicate`` for up to ``timeout`` seconds. Assert on timeout."""
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return
        _time.sleep(0.01)
    raise AssertionError(f"predicate {predicate!r} never became true within {timeout}s")


def test_log_pusher_buffers_and_flushes_on_demand(tracked_log_service_client):
    """Entries buffered below batch_size are drained on flush()."""
    pusher = LogPusher(
        "http://h:1",
        batch_size=1000,
        flush_interval=999.0,  # don't rely on periodic wake
    )
    try:
        entry = logging_pb2.LogEntry(source="test", data="line1")
        pusher.push("key-a", [entry, entry, entry])
        pusher.push("key-b", [entry])
        pusher.flush()
        _wait_for(lambda: len(tracked_log_service_client[0].pushes) >= 2)

        totals = {p.key: len(p.entries) for p in tracked_log_service_client[0].pushes}
        assert totals == {"key-a": 3, "key-b": 1}
    finally:
        pusher.close()


def test_log_pusher_flushes_at_batch_size(tracked_log_service_client):
    """Reaching batch_size wakes the drain thread without waiting for a timer."""
    pusher = LogPusher(
        "http://h:1",
        batch_size=2,
        flush_interval=999.0,  # isolate the batch-size-triggered send
    )
    try:
        entry = logging_pb2.LogEntry(source="test", data="line")
        pusher.push("k", [entry])
        # One entry is below batch_size — nothing should ship yet.
        # Small sleep lets any bogus wake happen (test will still catch).
        import time as _time

        _time.sleep(0.05)
        assert tracked_log_service_client[0].pushes == []

        pusher.push("k", [entry])
        _wait_for(lambda: len(tracked_log_service_client[0].pushes) == 1)
        assert tracked_log_service_client[0].pushes[0].key == "k"
        assert len(tracked_log_service_client[0].pushes[0].entries) == 2
    finally:
        pusher.close()


def test_close_drains_pending_entries(tracked_log_service_client):
    """close() must drain buffered entries before returning."""
    pusher = LogPusher(
        "http://h:1",
        batch_size=1000,
        flush_interval=999.0,
    )
    entry = logging_pb2.LogEntry(source="test", data="line")
    pusher.push("k", [entry, entry])
    pusher.close()
    # Drain thread must have sent the pending batch before close() returned.
    assert len(tracked_log_service_client[0].pushes) == 1
    assert len(tracked_log_service_client[0].pushes[0].entries) == 2


def test_overflow_drops_oldest(tracked_log_service_client, caplog):
    """Exceeding max_buffer_size drops oldest entries across keys."""
    pusher = LogPusher(
        "http://h:1",
        batch_size=1000,  # large — don't ship mid-test
        flush_interval=999.0,
        max_buffer_size=3,
    )
    try:
        with caplog.at_level(logging.WARNING, logger="iris.log_server.client"):
            entries = [
                logging_pb2.LogEntry(source="test", data=str(i), level=logging_pb2.LOG_LEVEL_INFO) for i in range(5)
            ]
            pusher.push("k", entries)
        pusher.flush()
        pusher.close()
        datas = [e.data for p in tracked_log_service_client[0].pushes for e in p.entries]
        # Oldest 2 dropped; "2","3","4" survive in order.
        assert datas == ["2", "3", "4"]
        assert any("buffer overflow" in r.message for r in caplog.records)
    except Exception:
        # Best-effort cleanup for test failures.
        pusher.close()
        raise


def test_retryable_failure_rebuffers_at_head(monkeypatch):
    """On a retryable send failure the batch is re-buffered at the head and
    drained on the next successful attempt."""
    created: list[_FakeLogServiceClient] = []

    def factory(address, timeout_ms=10_000, interceptors=()):
        c = _FakeLogServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        if not created:
            c.errors = [ConnectError(Code.UNAVAILABLE, "gone") for _ in range(3)]
        created.append(c)
        return c

    monkeypatch.setattr(client_mod, "LogServiceClientSync", factory)

    addresses = iter(["http://a:1", "http://b:2"])
    pusher = LogPusher(
        "/system/log-server",
        batch_size=1,
        flush_interval=999.0,
        resolver=lambda _url: next(addresses),
    )
    try:
        pusher.push("k", [logging_pb2.LogEntry(source="test", data="a", level=logging_pb2.LOG_LEVEL_INFO)])
        # Wait for the drain thread to fail on client #0 and re-resolve to #1.
        _wait_for(lambda: len(created) == 2)
        _wait_for(lambda: len(created[1].pushes) == 1)
        assert created[0].closed is True
        assert created[1].pushes[0].entries[0].data == "a"
    finally:
        pusher.close()


# ---------------------------------------------------------------------------
# Resolver-based LogPusher (self-healing on log-server failover)
# ---------------------------------------------------------------------------


class _FakeLogServiceClient:
    """Records pushes and can be seeded with errors to raise."""

    def __init__(self, address, **_kwargs):
        self.address = address
        self.pushes: list[logging_pb2.PushLogsRequest] = []
        self.errors: list[Exception] = []
        self.closed = False

    def push_logs(self, request):
        if self.errors:
            raise self.errors.pop(0)
        self.pushes.append(request)
        return logging_pb2.PushLogsResponse()

    def close(self):
        self.closed = True


@pytest.fixture
def tracked_log_service_client(monkeypatch):
    """Patch LogServiceClientSync to track every constructed instance."""
    created: list[_FakeLogServiceClient] = []

    def factory(address, timeout_ms=10_000, interceptors=()):
        c = _FakeLogServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        created.append(c)
        return c

    monkeypatch.setattr(client_mod, "LogServiceClientSync", factory)
    return created


def _entry(data="line"):
    e = logging_pb2.LogEntry(source="test", data=data, level=logging_pb2.LOG_LEVEL_INFO)
    e.timestamp.epoch_ms = 1
    return e


def test_resolver_called_lazily_and_cached(tracked_log_service_client):
    calls = []

    def resolver(url):
        calls.append(url)
        return "http://resolved:1"

    pusher = LogPusher("iris://system/log-server", batch_size=1, resolver=resolver)
    try:
        # Drain thread hasn't needed a client yet.
        assert calls == []
        assert tracked_log_service_client == []

        pusher.push("k", [_entry("a")])
        _wait_for(lambda: len(tracked_log_service_client) == 1)
        assert calls == ["iris://system/log-server"]

        pusher.push("k", [_entry("b")])
        _wait_for(lambda: len(tracked_log_service_client[0].pushes) == 2)
        assert calls == ["iris://system/log-server"]
        assert len(tracked_log_service_client) == 1
    finally:
        pusher.close()


def test_retryable_error_invalidates_and_reresolves(tracked_log_service_client):
    addresses = iter(["http://a:1", "http://b:2"])
    pusher = LogPusher("iris://system/log-server", batch_size=1, resolver=lambda _url: next(addresses))
    try:
        pusher.push("k", [_entry("a")])
        _wait_for(lambda: len(tracked_log_service_client) == 1 and tracked_log_service_client[0].pushes)
        assert tracked_log_service_client[0].address == "http://a:1"

        # Seed the cached client with a retryable failure; next send
        # invalidates, re-resolves to #b, and delivers.
        tracked_log_service_client[0].errors.append(ConnectError(Code.UNAVAILABLE, "gone"))
        pusher.push("k", [_entry("b")])
        _wait_for(lambda: len(tracked_log_service_client) == 2)
        assert tracked_log_service_client[0].closed is True
        assert tracked_log_service_client[1].address == "http://b:2"
        _wait_for(lambda: len(tracked_log_service_client[1].pushes) == 1)
    finally:
        pusher.close()


def test_non_retryable_error_rebuffers_without_invalidating(tracked_log_service_client):
    """Non-retryable errors re-buffer the entry (no drops) but keep the cached
    client — the next attempt retries on the same endpoint."""
    pusher = LogPusher("iris://system/log-server", batch_size=1, resolver=lambda _url: "http://a:1")
    try:
        pusher.push("k", [_entry("a")])
        _wait_for(lambda: len(tracked_log_service_client) == 1 and tracked_log_service_client[0].pushes)

        # Seed one NOT_FOUND: the entry is re-buffered and then delivered on
        # the next attempt against the same (not invalidated) client.
        tracked_log_service_client[0].errors.append(ConnectError(Code.NOT_FOUND, "missing"))
        pusher.push("k", [_entry("b")])
        _wait_for(lambda: len(tracked_log_service_client[0].errors) == 0)
        _wait_for(lambda: len(tracked_log_service_client[0].pushes) == 2)

        assert len(tracked_log_service_client) == 1
        assert tracked_log_service_client[0].closed is False
        assert [p.entries[0].data for p in tracked_log_service_client[0].pushes] == ["a", "b"]
    finally:
        pusher.close()


def test_resolver_raising_is_retried(tracked_log_service_client):
    """A resolver failure re-buffers the pending entries; the next
    successful resolution drains them."""
    attempts = []
    lock = threading.Lock()

    def resolver(_url):
        with lock:
            attempts.append(1)
            n = len(attempts)
        if n == 1:
            raise ConnectionError("controller down")
        return "http://good:1"

    pusher = LogPusher(
        "iris://system/log-server",
        batch_size=1,
        flush_interval=0.1,
        resolver=resolver,
    )
    try:
        pusher.push("k", [_entry("a")])
        # First attempt raises, backoff, second attempt succeeds.
        _wait_for(lambda: len(tracked_log_service_client) == 1, timeout=10.0)
        _wait_for(lambda: len(tracked_log_service_client[0].pushes) == 1)
        with lock:
            assert len(attempts) >= 2
    finally:
        pusher.close()


def test_static_url_pusher_retries_without_invalidating(tracked_log_service_client):
    """Without a resolver, retryable failures re-buffer and retry on the same client."""
    pusher = LogPusher(
        "http://h:1",
        batch_size=1,
        flush_interval=0.1,
    )
    try:
        assert len(tracked_log_service_client) == 1
        assert tracked_log_service_client[0].address == "http://h:1"

        # One UNAVAILABLE, then success on retry.
        tracked_log_service_client[0].errors.append(ConnectError(Code.UNAVAILABLE, "gone"))
        pusher.push("k", [_entry("a")])
        _wait_for(lambda: len(tracked_log_service_client[0].pushes) == 1)
        # Same client — no resolver means no re-resolution.
        assert tracked_log_service_client[0].closed is False
        assert len(tracked_log_service_client) == 1
    finally:
        pusher.close()
