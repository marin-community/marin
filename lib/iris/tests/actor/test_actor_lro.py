# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for long-running operations (LRO) on actor RPC."""

import time

import pytest

from iris.actor import ActorClient, ActorServer
from iris.actor.resolver import FixedResolver
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

        # Poll until done
        for _ in range(50):
            op = client.get_operation(op_id)
            if op.state == actor_pb2.Operation.SUCCEEDED:
                break
            time.sleep(0.05)

        assert op.state == actor_pb2.Operation.SUCCEEDED
        import cloudpickle

        assert cloudpickle.loads(op.serialized_result) == 5
    finally:
        server.stop()


def test_lro_failure():
    """Operation that raises should report FAILED with the exception."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        op_id = client.start_operation("fail")

        for _ in range(50):
            op = client.get_operation(op_id)
            if op.state != actor_pb2.Operation.RUNNING:
                break
            time.sleep(0.05)

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
        op_id = client.start_operation("slow_add", 1, 2, delay=1.0)

        # Cancel immediately
        op = client.cancel_operation(op_id)
        # State may still be RUNNING (cooperative cancellation)
        assert op.state in (actor_pb2.Operation.RUNNING, actor_pb2.Operation.CANCELLED)

        # Wait for the operation to finish (it should complete since cancellation is cooperative)
        for _ in range(100):
            op = client.get_operation(op_id)
            if op.state != actor_pb2.Operation.RUNNING:
                break
            time.sleep(0.1)

        # Should be CANCELLED since the cancelled flag was set before completion
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
        from connectrpc.errors import ConnectError

        with pytest.raises(ConnectError, match="not found"):
            client.get_operation("nonexistent")
    finally:
        server.stop()
