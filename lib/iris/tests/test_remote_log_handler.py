# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogPusher buffering and RemoteLogHandler."""

import logging
import threading

import pytest

from iris.log_server.client import LogPusher, RemoteLogHandler, _WARN_INTERVAL
from iris.rpc import logging_pb2
from rigging.timing import RateLimiter


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


def test_log_pusher_buffers_and_flushes():
    """LogPusher buffers entries and flushes on flush() call."""
    sent: list[tuple[str, int]] = []

    class RecordingPusher(LogPusher):
        """Override _send to record without RPC."""

        def __init__(self):
            # Skip real __init__ to avoid RPC client setup.
            self._batch_size = 1000
            self._flush_interval = 999.0
            self._buffers: dict[str, list[logging_pb2.LogEntry]] = {}
            self._lock = threading.Lock()
            self._send_lock = threading.Lock()
            self._closed = False
            self._flush_timer = None

        def _send(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
            sent.append((key, len(entries)))

    pusher = RecordingPusher()

    entry = logging_pb2.LogEntry(source="test", data="line1")
    pusher.push("key-a", [entry])
    pusher.push("key-a", [entry, entry])
    pusher.push("key-b", [entry])

    assert len(sent) == 0  # Still buffered

    pusher.flush()

    assert ("key-a", 3) in sent
    assert ("key-b", 1) in sent


def test_log_pusher_flushes_at_batch_size():
    """LogPusher flushes automatically when batch_size is reached."""
    sent: list[tuple[str, int]] = []

    class RecordingPusher(LogPusher):
        def __init__(self):
            self._batch_size = 2
            self._flush_interval = 999.0
            self._buffers: dict[str, list[logging_pb2.LogEntry]] = {}
            self._lock = threading.Lock()
            self._send_lock = threading.Lock()
            self._closed = False
            self._flush_timer = None

        def _send(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
            sent.append((key, len(entries)))

    pusher = RecordingPusher()
    entry = logging_pb2.LogEntry(source="test", data="line")

    pusher.push("k", [entry])
    assert len(sent) == 0

    pusher.push("k", [entry])  # Hits batch_size=2
    assert len(sent) == 1
    assert sent[0] == ("k", 2)


def test_send_warns_on_rpc_failure(caplog):
    """LogPusher._send() emits a warning (not debug) on RPC failure."""

    class FailingClient:
        def push_logs(self, request):
            raise ConnectionError("server unavailable")

        def close(self):
            pass

    pusher = LogPusher.__new__(LogPusher)
    pusher._client = FailingClient()
    pusher._batch_size = 1000
    pusher._flush_interval = 999.0
    pusher._buffers = {}
    pusher._lock = threading.Lock()
    pusher._send_lock = threading.Lock()
    pusher._closed = False
    pusher._flush_timer = None
    pusher._consecutive_failures = 0
    pusher._warn_limiter = RateLimiter(interval_seconds=_WARN_INTERVAL)

    entry = logging_pb2.LogEntry(source="test", data="line")
    with caplog.at_level(logging.WARNING, logger="iris.log_server.client"):
        pusher._send("k", [entry])

    assert any("Failed to push" in r.message and r.levelno == logging.WARNING for r in caplog.records)
    assert pusher._consecutive_failures == 1


def test_send_warns_rate_limited(caplog):
    """Repeated failures only warn on the first, then respect the rate limit."""

    class FailingClient:
        def push_logs(self, request):
            raise ConnectionError("server unavailable")

        def close(self):
            pass

    pusher = LogPusher.__new__(LogPusher)
    pusher._client = FailingClient()
    pusher._batch_size = 1000
    pusher._flush_interval = 999.0
    pusher._buffers = {}
    pusher._lock = threading.Lock()
    pusher._send_lock = threading.Lock()
    pusher._closed = False
    pusher._flush_timer = None
    pusher._consecutive_failures = 0
    pusher._warn_limiter = RateLimiter(interval_seconds=_WARN_INTERVAL)

    entry = logging_pb2.LogEntry(source="test", data="line")
    with caplog.at_level(logging.WARNING, logger="iris.log_server.client"):
        # First failure should warn
        pusher._send("k", [entry])
        # Subsequent failures within the interval should not warn
        pusher._send("k", [entry])
        pusher._send("k", [entry])

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "Failed to push" in r.message]
    assert len(warnings) == 1, f"Expected 1 warning, got {len(warnings)}"
    assert pusher._consecutive_failures == 3


def test_send_logs_recovery(caplog):
    """LogPusher logs info when sends recover after failures."""
    call_count = 0

    class FlakeyClient:
        def push_logs(self, request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("server unavailable")

        def close(self):
            pass

    pusher = LogPusher.__new__(LogPusher)
    pusher._client = FlakeyClient()
    pusher._batch_size = 1000
    pusher._flush_interval = 999.0
    pusher._buffers = {}
    pusher._lock = threading.Lock()
    pusher._send_lock = threading.Lock()
    pusher._closed = False
    pusher._flush_timer = None
    pusher._consecutive_failures = 0
    pusher._warn_limiter = RateLimiter(interval_seconds=_WARN_INTERVAL)

    entry = logging_pb2.LogEntry(source="test", data="line")
    with caplog.at_level(logging.INFO, logger="iris.log_server.client"):
        pusher._send("k", [entry])  # fail
        pusher._send("k", [entry])  # fail
        pusher._send("k", [entry])  # succeed

    assert any("recovered" in r.message and r.levelno == logging.INFO for r in caplog.records)
    assert pusher._consecutive_failures == 0


def test_close_waits_for_inflight_send():
    """close() must not destroy the client while _send() is in flight.

    Simulates a slow RPC by blocking inside _send. close() must wait for it
    to finish before closing the client. Without the _send_lock this would
    close the client while the send is still running.
    """
    send_entered = threading.Event()
    send_may_proceed = threading.Event()
    client_closed_during_send = False

    class SlowPusher(LogPusher):
        def __init__(self):
            self._batch_size = 1000
            self._flush_interval = 0.01  # fast timer to trigger the race
            self._buffers: dict[str, list[logging_pb2.LogEntry]] = {}
            self._lock = threading.Lock()
            self._send_lock = threading.Lock()
            self._closed = False
            self._flush_timer = None
            self._client_alive = True
            self._schedule_flush()

        def _send(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
            nonlocal client_closed_during_send
            with self._send_lock:
                if self._closed:
                    return
                send_entered.set()
                send_may_proceed.wait(timeout=5.0)
                if not self._client_alive:
                    client_closed_during_send = True

        def close(self) -> None:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
            self.flush()
            with self._send_lock:
                self._closed = True
            self._client_alive = False

    pusher = SlowPusher()
    entry = logging_pb2.LogEntry(source="test", data="line")
    pusher.push("k", [entry])

    # Wait for the periodic flush timer to trigger _send
    assert send_entered.wait(timeout=5.0), "timer-triggered _send never started"

    # Now call close() from another thread — it should block on _send_lock
    close_done = threading.Event()

    def do_close():
        pusher.close()
        close_done.set()

    t = threading.Thread(target=do_close)
    t.start()

    # Give close() a moment to potentially race ahead (it shouldn't)
    assert not close_done.wait(timeout=0.1), "close() returned before send finished"

    # Let the send complete
    send_may_proceed.set()

    assert close_done.wait(timeout=5.0), "close() never completed"
    t.join(timeout=1.0)
    assert not client_closed_during_send, "client was destroyed while _send was in flight"
