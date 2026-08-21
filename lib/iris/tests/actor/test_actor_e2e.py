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
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp


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


class DashboardCounter:
    def __init__(self) -> None:
        self._value = 0
        self._web_application = Starlette(
            routes=[
                Route("/", self._dashboard),
                Route("/increment", self._increment_from_http, methods=["POST"]),
            ]
        )

    @property
    def web_application(self) -> ASGIApp:
        return self._web_application

    def increment(self, amount: int = 1) -> int:
        self._value += amount
        return self._value

    def current(self) -> int:
        return self._value

    async def _dashboard(self, _request: Request) -> JSONResponse:
        return JSONResponse({"value": self.current()})

    async def _increment_from_http(self, _request: Request) -> JSONResponse:
        return JSONResponse({"value": self.increment()})


class WebApplicationMethodActor:
    def web_application(self) -> str:
        return "RPC method"


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


def test_actor_web_application_shares_state_with_rpc():
    """RPC and HTTP requests operate on one actor instance."""
    server = ActorServer(host="127.0.0.1")
    server.register("counter", DashboardCounter())
    port = server.serve_background()

    try:
        address = f"http://127.0.0.1:{port}"
        client = ActorClient(FixedResolver({"counter": address}), "counter")

        assert client.increment(2) == 2
        assert httpx.get(f"{address}/", timeout=2).json() == {"value": 2}
        assert httpx.post(f"{address}/increment", timeout=2).json() == {"value": 3}
        assert client.current() == 3
    finally:
        server.stop()


def test_web_application_method_remains_available_through_rpc():
    server = ActorServer(host="127.0.0.1")
    server.register("method", WebApplicationMethodActor())
    port = server.serve_background()

    try:
        address = f"http://127.0.0.1:{port}"
        client = ActorClient(FixedResolver({"method": address}), "method")
        assert client.web_application() == "RPC method"
    finally:
        server.stop()


def test_actor_web_application_must_register_before_server_start():
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    server.serve_background()

    try:
        with pytest.raises(RuntimeError, match="before the actor server starts"):
            server.register("counter", DashboardCounter())
    finally:
        server.stop()


def test_actor_server_rejects_second_web_application():
    server = ActorServer(host="127.0.0.1")
    try:
        server.register("first", DashboardCounter())
        with pytest.raises(RuntimeError, match="one actor web application"):
            server.register("second", DashboardCounter())
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
