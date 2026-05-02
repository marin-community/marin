# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogPusher buffering and RemoteLogHandler."""

import logging
import threading
import time

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.client import LogPusher, RemoteLogHandler
from finelog.client import pusher as pusher_mod
from finelog.rpc import logging_pb2


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


def test_handler_pushes_entries():
    pusher = FakeLogPusher()
    handler = RemoteLogHandler(pusher, key="test")
    log = logging.getLogger("test_handler_push")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        log.info("hello")
        assert len(pusher.pushed) == 1
        assert pusher.pushed[0][1][0].data.endswith("hello")
    finally:
        log.removeHandler(handler)
        handler.close()


def test_no_deadlock_on_push_failure():
    """When push fails, the error log must not deadlock by re-entering emit().

    Verified by completing within 2 seconds (a deadlock would hang forever).
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


class _FakeLogServiceClient:
    """Records pushes; can be seeded with errors to raise."""

    def __init__(self, address, **_kwargs):
        self.address = address
        self.pushes: list[logging_pb2.PushLogsRequest] = []
        self.errors: list[Exception] = []

    def push_logs(self, request):
        if self.errors:
            raise self.errors.pop(0)
        self.pushes.append(request)
        return logging_pb2.PushLogsResponse()

    def close(self):
        pass


@pytest.fixture
def tracked_clients(monkeypatch):
    """Patch LogServiceClientSync to track every constructed instance."""
    created: list[_FakeLogServiceClient] = []

    def factory(address, timeout_ms=10_000, interceptors=()):
        c = _FakeLogServiceClient(address, timeout_ms=timeout_ms, interceptors=interceptors)
        created.append(c)
        return c

    monkeypatch.setattr(pusher_mod, "LogServiceClientSync", factory)
    return created


def test_buffers_and_flushes_on_demand(tracked_clients):
    """Entries below batch_size drain on flush(); flush blocks until shipped."""
    pusher = LogPusher("http://h:1", batch_size=1000, flush_interval=999.0)
    try:
        entry = logging_pb2.LogEntry(source="test", data="line")
        pusher.push("key-a", [entry, entry, entry])
        pusher.push("key-b", [entry])
        assert pusher.flush(timeout=5.0) is True
        totals = {p.key: len(p.entries) for p in tracked_clients[0].pushes}
        assert totals == {"key-a": 3, "key-b": 1}
    finally:
        pusher.close()


def test_flush_timeout_returns_false(tracked_clients):
    """flush(timeout=...) returns False when the drain can't catch up.

    Seeds a non-retryable error so the drain rebuffers and enters the
    backoff window; flush is given less time than the backoff interval.
    """
    pusher = LogPusher("http://h:1", batch_size=1, flush_interval=999.0)
    try:
        pusher.push("k", [logging_pb2.LogEntry(source="test", data="primer")])
        assert pusher.flush(timeout=5.0) is True
        tracked_clients[0].errors.append(ConnectError(Code.NOT_FOUND, "missing"))
        pusher.push("k", [logging_pb2.LogEntry(source="test", data="stuck")])
        # Backoff is 0.5s; a 0.05s flush cannot catch up.
        assert pusher.flush(timeout=0.05) is False
    finally:
        pusher.close()


def test_flushes_at_batch_size(tracked_clients):
    """Reaching batch_size wakes the drain thread without waiting for the timer."""
    pusher = LogPusher("http://h:1", batch_size=2, flush_interval=999.0)
    try:
        entry = logging_pb2.LogEntry(source="test", data="line")
        pusher.push("k", [entry])
        time.sleep(0.05)
        assert not tracked_clients or not tracked_clients[0].pushes, "no push expected before batch_size"

        pusher.push("k", [entry])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tracked_clients and tracked_clients[0].pushes:
                break
            time.sleep(0.01)
        assert tracked_clients[0].pushes[0].key == "k"
        assert len(tracked_clients[0].pushes[0].entries) == 2
    finally:
        pusher.close()


def test_close_drains_pending_entries(tracked_clients):
    """close() drains buffered entries before returning."""
    pusher = LogPusher("http://h:1", batch_size=1000, flush_interval=999.0)
    entry = logging_pb2.LogEntry(source="test", data="line")
    pusher.push("k", [entry, entry])
    pusher.close()
    assert len(tracked_clients[0].pushes) == 1
    assert len(tracked_clients[0].pushes[0].entries) == 2


def test_overflow_drops_oldest(tracked_clients, caplog):
    """Exceeding max_buffer_size drops oldest entries across keys."""
    pusher = LogPusher("http://h:1", batch_size=1000, flush_interval=999.0, max_buffer_size=3)
    # The pusher's own logger is detached from the root (to avoid re-entry
    # via RemoteLogHandler); attach caplog's handler directly.
    client_logger = logging.getLogger("finelog.client.pusher")
    client_logger.addHandler(caplog.handler)
    client_logger.setLevel(logging.WARNING)
    try:
        entries = [logging_pb2.LogEntry(source="test", data=str(i)) for i in range(5)]
        pusher.push("k", entries)
        pusher.flush()
        pusher.close()
        datas = [e.data for p in tracked_clients[0].pushes for e in p.entries]
        # Oldest 2 dropped; "2","3","4" survive in order.
        assert datas == ["2", "3", "4"]
        assert any("buffer overflow" in r.message for r in caplog.records)
    finally:
        client_logger.removeHandler(caplog.handler)
        pusher.close()
