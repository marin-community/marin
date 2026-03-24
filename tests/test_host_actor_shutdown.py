# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4098: _host_actor blocks forever after SHUTDOWN."""

import threading

from iris.actor.server import ActorServer
from iris.managed_thread import thread_container_scope

from fray.v2.actor import _clear_shutdown_event, _set_shutdown_event, request_shutdown


class _Noop:
    def ping(self) -> str:
        return "pong"


def test_request_shutdown_unblocks_wait():
    event = threading.Event()
    _set_shutdown_event(event)
    try:
        assert not event.is_set()
        request_shutdown()
        assert event.is_set()
    finally:
        _clear_shutdown_event()


def test_request_shutdown_noop_outside_actor():
    """No-op when not in a hosted actor — supports Ray/local backends."""
    request_shutdown()  # should not raise


def test_host_actor_shutdown_stops_server():
    """Shutdown signal unblocks the host thread and tears down the ActorServer."""
    with thread_container_scope("test-shutdown") as threads:
        server = ActorServer(host="127.0.0.1", port=0, threads=threads)
        server.register("test-actor", _Noop())
        server.serve_background()

        shutdown_event = threading.Event()
        host_done = threading.Event()

        def host_main():
            shutdown_event.wait()
            server.stop()
            host_done.set()

        host_thread = threading.Thread(target=host_main, daemon=True)
        host_thread.start()
        assert threads.is_alive

        shutdown_event.set()
        assert host_done.wait(timeout=5.0)
        host_thread.join(timeout=2.0)
        assert not host_thread.is_alive()
        assert not threads.is_alive


def test_request_shutdown_works_from_child_thread():
    """request_shutdown() must work from threads spawned by the actor (e.g. polling thread)."""
    event = threading.Event()
    _set_shutdown_event(event)
    try:
        triggered = threading.Event()

        def child():
            request_shutdown()
            triggered.set()

        t = threading.Thread(target=child, daemon=True)
        t.start()
        assert triggered.wait(timeout=2.0)
        assert event.is_set()
    finally:
        _clear_shutdown_event()
