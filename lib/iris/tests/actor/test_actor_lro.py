# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for long-running operations (LRO) on actor RPC."""

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


def test_retry_with_same_idempotency_key_reuses_the_operation():
    """A retry whose first response was lost must not start the work twice.

    The client cannot tell "request never arrived" from "reply was lost", so it
    retries either way. Without deduplication the second attempt starts a second
    operation: the work runs twice and the first operation is left running with
    nobody holding its id.
    """
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        call = actor_pb2.ActorCall(
            method_name="slow_add",
            actor_name="actor",
            serialized_args=cloudpickle.dumps((1, 2)),
            serialized_kwargs=cloudpickle.dumps({"delay": 0.2}),
            idempotency_key="fixed-key-for-one-logical-call",
        )
        rpc = client.rpc_client()

        first = rpc.start_operation(call)
        # Simulate the lost reply: the client never saw `first`, so it retries
        # the identical call.
        second = rpc.start_operation(call)
        assert second.operation_id == first.operation_id, "retry started a second operation"

        op = client.get_operation(first.operation_id)
        assert op.state == actor_pb2.Operation.SUCCEEDED
        assert cloudpickle.loads(op.serialized_result) == 3

        # The key is released with the operation, so a genuinely new call after
        # the result was read is not answered with the stale one.
        third = rpc.start_operation(call)
        assert third.operation_id != first.operation_id
    finally:
        server.stop()


def test_the_same_key_on_two_actors_is_two_operations():
    """A key means "this call again", so it is scoped to the actor and method.

    Nothing stops two unrelated callers from picking the same key. If the index
    were global, one of them would be handed the other's operation and would
    read a result computed from arguments it never sent.
    """
    server = ActorServer(host="127.0.0.1")
    server.register("a", SlowActor())
    server.register("b", SlowActor())
    port = server.serve_background()

    try:
        rpc = _make_client(port).rpc_client()

        def _call(actor: str, method: str) -> actor_pb2.ActorCall:
            return actor_pb2.ActorCall(
                method_name=method,
                actor_name=actor,
                serialized_args=cloudpickle.dumps((1, 2)),
                serialized_kwargs=cloudpickle.dumps({}),
                idempotency_key="a-key-two-callers-happened-to-share",
            )

        on_a = rpc.start_operation(_call("a", "add"))
        on_b = rpc.start_operation(_call("b", "add"))
        other_method = rpc.start_operation(_call("a", "slow_add"))

        assert len({on_a.operation_id, on_b.operation_id, other_method.operation_id}) == 3
    finally:
        server.stop()


def test_calls_without_an_idempotency_key_are_independent():
    """An empty key opts out: two calls are two operations."""
    server = ActorServer(host="127.0.0.1")
    server.register("actor", SlowActor())
    port = server.serve_background()

    try:
        client = _make_client(port)
        call = actor_pb2.ActorCall(
            method_name="add",
            actor_name="actor",
            serialized_args=cloudpickle.dumps((1, 2)),
            serialized_kwargs=cloudpickle.dumps({}),
        )
        rpc = client.rpc_client()
        assert rpc.start_operation(call).operation_id != rpc.start_operation(call).operation_id
    finally:
        server.stop()
