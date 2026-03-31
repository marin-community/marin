# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoteLogHandler, particularly the re-entrancy deadlock fix."""

import logging
import threading

import pytest

from iris.log_server.client import LogPusher, RemoteLogHandler
from iris.rpc import logging_pb2


class FakeLogPusher(LogPusher):
    """LogPusher that records calls instead of making RPCs."""

    def __init__(self, *, fail: bool = False) -> None:
        # Skip real __init__ — we don't need a real RPC client.
        self.pushed: list[list[logging_pb2.LogEntry]] = []
        self._fail = fail

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None:
        self.pushed.append(list(entries))
        if self._fail:
            raise ConnectionError("server unavailable")


@pytest.fixture()
def handler():
    pusher = FakeLogPusher()
    h = RemoteLogHandler(pusher, key="test", batch_size=100, flush_interval=999)
    yield h
    h.close()


def test_flush_sends_buffered_entries(handler: RemoteLogHandler):
    logger = logging.getLogger("test_flush_sends")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("hello")
        handler.flush()
        assert len(handler._pusher.pushed) == 1
        assert len(handler._pusher.pushed[0]) == 1
        assert handler._pusher.pushed[0][0].data.endswith("hello")
    finally:
        logger.removeHandler(handler)


def test_no_deadlock_on_push_failure():
    """When push fails, the error log must not deadlock by re-entering emit().

    Before the fix, _do_flush logged via the root logger while holding
    self._lock. If this handler was on the root logger, the log call would
    re-enter emit() and block on the same non-reentrant lock.

    We verify this completes within 2 seconds (a deadlock would hang forever).
    """
    pusher = FakeLogPusher(fail=True)
    handler = RemoteLogHandler(pusher, key="test", batch_size=1, flush_interval=999)
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    done = threading.Event()

    def log_one():
        try:
            # batch_size=1 means emit() will call _do_flush() immediately.
            # _do_flush() will fail and try to log the error via root logger,
            # which would re-enter emit(). Without the fix, this deadlocks.
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
