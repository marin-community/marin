# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for actor server and client."""

import errno
import socket

import httpx
import pytest
from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


class Calculator:
    """Test actor with basic arithmetic operations."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: int, b: int) -> float:
        return a / b  # May raise ZeroDivisionError

    def increment(self, value: int, *, amount: int = 1) -> int:
        return value + amount


class CalculatorWithDashboard(Calculator):
    @property
    def web_application(self) -> Starlette:
        async def status(_request):
            return JSONResponse({"dashboard": "ready"})

        return Starlette(routes=[Route("/", status)])


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


def test_actor_call_with_kwargs():
    """Test actor method calls preserve keyword arguments."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    try:
        resolver = FixedResolver({"calc": f"http://127.0.0.1:{port}"})
        client = ActorClient(resolver, "calc")
        assert client.increment(2, amount=3) == 5
    finally:
        server.stop()


def test_actor_web_application_shares_endpoint_with_actor_rpc():
    server = ActorServer(host="127.0.0.1")
    server.register("calc", CalculatorWithDashboard())
    port = server.serve_background()

    try:
        resolver = FixedResolver({"calc": f"http://127.0.0.1:{port}"})
        actor_client = ActorClient(resolver, "calc")
        dashboard_response = httpx.get(f"http://127.0.0.1:{port}/")

        assert actor_client.add(2, 3) == 5
        assert dashboard_response.status_code == 200
        assert dashboard_response.json() == {"dashboard": "ready"}
    finally:
        server.stop()


def test_actor_web_application_must_register_before_server_starts():
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    server.serve_background()

    try:
        with pytest.raises(RuntimeError, match="before the actor server starts"):
            server.register("dashboard", CalculatorWithDashboard())
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


def test_serve_background_with_unavailable_port_raises():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        server = ActorServer(host="127.0.0.1", port=port)
        with pytest.raises(OSError) as exc_info:
            server.serve_background()

    assert exc_info.value.errno == errno.EADDRINUSE
