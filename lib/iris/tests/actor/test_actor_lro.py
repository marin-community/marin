# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for long-running operations (LRO) on actor RPC."""

import threading
import time

import cloudpickle
import pytest
from connectrpc.errors import ConnectError
from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from iris.rpc import actor_pb2


class SlowActor:
    """Actor with a method that takes a while to complete."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def slow_add(self, a: int, b: int, delay: float = 0.5) -> int:
        time.sleep(delay)
        return a + b

    def fail(self):
        raise ValueError("intentional failure")


class BlockingActor:
    def __init__(self, expected_calls: int):
        self._expected_calls = expected_calls
        self._started_calls = 0
        self._lock = threading.Lock()
        self.all_started = threading.Event()
        self.release = threading.Event()

    def block(self) -> None:
        with self._lock:
            self._started_calls += 1
            if self._started_calls == self._expected_calls:
                self.all_started.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("test did not release blocking actor calls")

    def ping(self) -> str:
        return "pong"


def _make_client(port: int, name: str = "actor") -> ActorClient:
    resolver = FixedResolver({name: f"http://127.0.0.1:{port}"})
    return ActorClient(resolver, name)


def test_lro_basic():
    """Start an operation, poll until done, get result."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        op_id = client.start_operation("add", 2, 3)
        assert isinstance(op_id, str)

        # get_operation auto-polls until done
        op = client.get_operation(op_id)
        assert op.state == actor_pb2.Operation.SUCCEEDED

        assert cloudpickle.loads(op.serialized_result) == 5
    finally:
        server.stop()


def test_actor_server_honors_concurrency_above_default():
    blocking_calls = 40
    actor = BlockingActor(blocking_calls)
    server = ActorServer(host="127.0.0.1", max_concurrency=blocking_calls + 1)
    server.register("actor", actor)
    port = server.serve_background()

    try:
        client = _make_client(port)
        operations = [client.start_operation("block") for _ in range(blocking_calls)]
        assert actor.all_started.wait(timeout=5)
        assert client.ping() == "pong"

        actor.release.set()
        assert all(client.get_operation(operation).state == actor_pb2.Operation.SUCCEEDED for operation in operations)
    finally:
        actor.release.set()
        server.stop()


def test_lro_failure():
    """Operation that raises should report FAILED with the exception."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        op_id = client.start_operation("fail")

        op = client.get_operation(op_id)
        assert op.state == actor_pb2.Operation.FAILED
        assert "intentional failure" in op.error.message
    finally:
        server.stop()


def test_lro_cancel():
    """Cancelling an operation sets the cancelled flag."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        # Start a slow operation
        op_id = client.start_operation("slow_add", 1, 2, delay=0.1)

        # Cancel immediately
        op = client.cancel_operation(op_id)
        # State may still be RUNNING (cooperative cancellation)
        assert op.state in (actor_pb2.Operation.RUNNING, actor_pb2.Operation.CANCELLED)

        # get_operation auto-polls until completion; should be CANCELLED
        op = client.get_operation(op_id)
        assert op.state == actor_pb2.Operation.CANCELLED
    finally:
        server.stop()


def test_lro_not_found():
    """Polling a nonexistent operation returns NOT_FOUND."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)

        with pytest.raises(ConnectError, match="not found"):
            client.poll_operation_status("nonexistent")
    finally:
        server.stop()
