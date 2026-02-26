# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for actor client and pool retry on transient errors."""

import threading

import pytest
from connectrpc.errors import ConnectError

from iris.actor import ActorClient, ActorPool
from iris.actor.resolver import FixedResolver, ResolveResult, Resolver
from iris.actor.server import ActorServer


class Counter:
    """Test actor with a call counter."""

    def __init__(self):
        self._count = 0

    def get_count(self) -> int:
        return self._count

    def increment(self) -> int:
        self._count += 1
        return self._count


class FailingResolver:
    """Resolver that simulates transient resolution failures.

    Returns empty results for the first N calls, then returns real endpoints.
    This simulates the case where a worker hasn't registered yet or has
    temporarily disappeared.
    """

    def __init__(self, real_resolver: Resolver, fail_count: int = 2):
        self._real = real_resolver
        self._fail_count = fail_count
        self._call_count = 0
        self._lock = threading.Lock()

    def resolve(self, name: str) -> ResolveResult:
        with self._lock:
            self._call_count += 1
            if self._call_count <= self._fail_count:
                return ResolveResult(name=name, endpoints=[])
        return self._real.resolve(name)


class SwitchingResolver:
    """Resolver that switches endpoints after a certain number of calls.

    Simulates the case where the first resolved endpoint becomes unavailable,
    and the resolver returns a new endpoint on re-resolution.
    """

    def __init__(self, endpoints_sequence: list[dict[str, str | list[str]]]):
        self._sequence = endpoints_sequence
        self._call_count = 0
        self._lock = threading.Lock()

    def resolve(self, name: str) -> ResolveResult:
        with self._lock:
            idx = min(self._call_count, len(self._sequence) - 1)
            self._call_count += 1
        resolver = FixedResolver(self._sequence[idx])
        return resolver.resolve(name)


def test_actor_client_retries_on_transient_rpc_error():
    """ActorClient should retry with re-resolution when an RPC call fails
    with a transient error (UNAVAILABLE, INTERNAL)."""
    server = ActorServer(host="127.0.0.1")
    server.register("counter", Counter())
    port = server.serve_background()

    try:
        # First endpoint is a dead address that will cause UNAVAILABLE,
        # second endpoint is the real server.
        # The SwitchingResolver returns the dead endpoint first, then the real one.
        switching = SwitchingResolver(
            [
                {"counter": "http://127.0.0.1:1"},  # Dead endpoint
                {"counter": f"http://127.0.0.1:{port}"},  # Real endpoint
            ]
        )

        client = ActorClient(
            switching,
            "counter",
            resolve_timeout=5.0,
            max_call_attempts=3,
            initial_backoff=0.05,
            max_backoff=0.1,
        )

        # First call should fail to reach dead endpoint, re-resolve to real
        # endpoint, and succeed on retry.
        result = client.increment()
        assert result == 1
    finally:
        server.stop()


def test_actor_client_does_not_retry_on_application_error():
    """ActorClient should NOT retry when the actor method itself raises."""
    server = ActorServer(host="127.0.0.1")

    class Divider:
        def divide(self, a: int, b: int) -> float:
            return a / b

    server.register("divider", Divider())
    port = server.serve_background()

    try:
        resolver = FixedResolver({"divider": f"http://127.0.0.1:{port}"})
        client = ActorClient(resolver, "divider", max_call_attempts=3)

        # ZeroDivisionError is an application error, not a transient RPC error.
        # It should propagate immediately without retry.
        with pytest.raises(ZeroDivisionError):
            client.divide(1, 0)
    finally:
        server.stop()


def test_actor_pool_retries_on_transient_rpc_error():
    """ActorPool should retry with re-resolution when an RPC call fails
    with a transient error."""
    server = ActorServer(host="127.0.0.1")
    server.register("counter", Counter())
    port = server.serve_background()

    try:
        # First resolve returns dead + real endpoints; after the dead one fails,
        # re-resolution should route to the real one.
        switching = SwitchingResolver(
            [
                {"counter": "http://127.0.0.1:1"},  # Dead endpoint
                {"counter": f"http://127.0.0.1:{port}"},  # Real endpoint
            ]
        )

        pool = ActorPool(
            switching,
            "counter",
            timeout=2.0,
            max_call_attempts=3,
            initial_backoff=0.05,
            max_backoff=0.1,
        )

        result = pool.call().increment()
        assert result == 1
    finally:
        server.stop()


def test_actor_client_exhausts_retries():
    """ActorClient should raise after exhausting all retry attempts."""
    # All endpoints are dead - no server running on port 1
    resolver = FixedResolver({"ghost": "http://127.0.0.1:1"})
    client = ActorClient(
        resolver,
        "ghost",
        resolve_timeout=2.0,
        max_call_attempts=2,
        initial_backoff=0.05,
        max_backoff=0.1,
    )

    with pytest.raises(ConnectError):
        client.increment()
