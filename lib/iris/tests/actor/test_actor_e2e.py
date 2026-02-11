# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for actor server and client."""

import pytest

from iris.actor import ActorClient, ActorServer
from iris.actor.resolver import FixedResolver


class Calculator:
    """Test actor with basic arithmetic operations."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: int, b: int) -> float:
        return a / b  # May raise ZeroDivisionError


def test_basic_actor_call():
    """Test basic actor method calls work correctly."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    try:
        resolver = FixedResolver({"calc": f"http://127.0.0.1:{port}"})
        client = ActorClient(resolver, "calc")
        assert client.add(2, 3) == 5
        assert client.multiply(4, 5) == 20
    finally:
        server.stop()


def test_actor_exception_propagation():
    """Test that exceptions from actor methods propagate to the client."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    try:
        resolver = FixedResolver({"calc": f"http://127.0.0.1:{port}"})
        client = ActorClient(resolver, "calc")
        with pytest.raises(ZeroDivisionError):
            client.divide(1, 0)
    finally:
        server.stop()
