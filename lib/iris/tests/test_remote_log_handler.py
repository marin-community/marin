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

    def flush(self, timeout: float | None = None) -> bool:
        return True

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
    """Entries buffered below batch_size are drained on flush().

    flush() blocks until every entry enqueued before the call has shipped,
    so the assertions can run immediately without polling.
    """
    pusher = LogPusher(
        "http://h:1",
        batch_size=1000,
        flush_interval=999.0,  # don't rely on periodic wake
    )
    try:
        entry = logging_pb2.LogEntry(source="test", data="line1")
        pusher.push("key-a", [entry, entry, entry])
        pusher.push("key-b", [entry])
        assert pusher.flush(timeout=5.0)

        totals = {p.key: len(p.entries) for p in tracked_log_service_client[0].pushes}
        assert totals == {"key-a": 3, "key-b": 1}
    finally:
        pusher.close()


def test_log_pusher_flush_is_blocking(tracked_log_service_client):
    """flush() returns only after every previously-pushed entry has been sent."""
    pusher = LogPusher(
        "http://h:1",
        batch_size=1000,
        flush_interval=999.0,
    )
    try:
        entry = logging_pb2.LogEntry(source="test", data="line")
        pusher.push("k", [entry, entry])
        # No polling — flush must block until shipped.
        assert pusher.flush(timeout=5.0) is True
        assert len(tracked_log_service_client[0].pushes) == 1
        assert len(tracked_log_service_client[0].pushes[0].entries) == 2
    finally:
        pusher.close()


def test_log_pusher_flush_timeout_returns_false(monkeypatch):
    """flush(timeout=...) returns False when the drain can't catch up in time.

    Seeds a non-retryable error so the drain rebuffers and enters the
    backoff window; flush is given less time than the backoff interval.
    """
    created: list[_FakeLogServiceClient] = []

    def factory(address, timeout_ms=10_000, interceptors=()):
        c = _FakeLogServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        created.append(c)
        return c

    monkeypatch.setattr(client_mod, "LogServiceClientSync", factory)

    pusher = LogPusher("http://h:1", batch_size=1, flush_interval=999.0)
    try:
        entry = logging_pb2.LogEntry(source="test", data="primer")
        pusher.push("k", [entry])
        # Wait for the cached client to exist, then seed a non-retryable
        # error so the next send rebuffers and the drain enters backoff.
        assert pusher.flush(timeout=5.0) is True
        created[0].errors.append(ConnectError(Code.NOT_FOUND, "missing"))
        pusher.push("k", [logging_pb2.LogEntry(source="test", data="stuck")])
        # Backoff is 0.5s; a 0.05s flush cannot catch up.
        assert pusher.flush(timeout=0.05) is False
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
        import time as _time

        _time.sleep(0.05)
        # Length checks only — don't let a (buggy) populated pushes list dump its
        # entire proto repr into the failure output.
        n_clients = len(tracked_log_service_client)
        n_pushes = len(tracked_log_service_client[0].pushes) if n_clients else 0
        assert n_clients == 0 or n_pushes == 0, f"expected no pushes before batch_size reached, got {n_pushes}"

        pusher.push("k", [entry])
        _wait_for(lambda: len(tracked_log_service_client) == 1 and len(tracked_log_service_client[0].pushes) == 1)
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
    # The pusher's own logger is detached from the root (to avoid
    # re-entry via RemoteLogHandler); attach caplog's handler directly.
    client_logger = logging.getLogger("iris.log_server.client")
    client_logger.addHandler(caplog.handler)
    client_logger.setLevel(logging.WARNING)
    try:
        entries = [logging_pb2.LogEntry(source="test", data=str(i), level=logging_pb2.LOG_LEVEL_INFO) for i in range(5)]
        pusher.push("k", entries)
        pusher.flush()
        pusher.close()
        datas = [e.data for p in tracked_log_service_client[0].pushes for e in p.entries]
        # Oldest 2 dropped; "2","3","4" survive in order.
        assert datas == ["2", "3", "4"]
        assert any("buffer overflow" in r.message for r in caplog.records)
    finally:
        client_logger.removeHandler(caplog.handler)
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
