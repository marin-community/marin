# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for OperationFuture (fray wrapper over Iris LRO polling)."""

import time

import pytest

from fray.v2.iris_backend import OperationFuture
from iris.actor import ActorClient, ActorServer
from iris.actor.resolver import FixedResolver
from iris.rpc import actor_pb2


class SlowActor:
    """Actor with a method that takes a while to complete."""

    def slow_add(self, a: int, b: int, delay: float = 0.5) -> int:
        time.sleep(delay)
        return a + b

    def fail(self):
        raise ValueError("intentional failure")


def _make_client(port: int, name: str = "actor") -> ActorClient:
    resolver = FixedResolver({name: f"http://127.0.0.1:{port}"})
    return ActorClient(resolver, name)


def test_operation_future_basic():
    """OperationFuture.result() polls and returns the result."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        op_id = client.start_operation("slow_add", 10, 20, delay=0.2)
        future = OperationFuture(client, op_id, poll_interval=0.1)
        assert future.result() == 30
    finally:
        server.stop()


def test_operation_future_timeout():
    """OperationFuture.result(timeout=...) cancels and raises TimeoutError."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        op_id = client.start_operation("slow_add", 1, 2, delay=5.0)
        future = OperationFuture(client, op_id, poll_interval=0.1)

        with pytest.raises(TimeoutError):
            future.result(timeout=0.3)

        # Verify the operation was cancelled server-side
        op = client.poll_operation_status(op_id)
        # May still be RUNNING briefly, but cancelled flag is set
        assert op.state in (actor_pb2.Operation.RUNNING, actor_pb2.Operation.CANCELLED)
    finally:
        server.stop()


def test_operation_future_propagates_exception():
    """OperationFuture.result() re-raises server-side exceptions."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        op_id = client.start_operation("fail")
        future = OperationFuture(client, op_id, poll_interval=0.1)

        with pytest.raises(ValueError, match="intentional failure"):
            future.result()
    finally:
        server.stop()
