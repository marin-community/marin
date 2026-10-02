# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for actor server and client."""

import errno
import socket
from typing import Any

import httpx
import pytest
from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from iris.actor.web import web_endpoint
from starlette.requests import Request


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

    @web_endpoint("/increment", method="POST")
    def increment(self, amount: int = 1) -> int:
        self._value += amount
        return self._value

    def current(self) -> int:
        return self._value

    @web_endpoint("/counter/{label}")
    def labeled_counter(self, label: str, prefix: str = "Counter") -> dict[str, int | str]:
        return {"label": f"{prefix} {label}", "value": self.current()}

    @web_endpoint("/")
    def _dashboard(self) -> dict[str, int]:
        return {"value": self.current()}

    @web_endpoint("/request")
    def _request_method(self, request: Request) -> str:
        return request.method


class DuplicateWebEndpointActor:
    @web_endpoint("/")
    @web_endpoint("/")
    def dashboard(self) -> None:
        pass


class RpcPathWebEndpointActor:
    @web_endpoint("/iris.actor.ActorService/Call")
    def call(self) -> None:
        pass


class PrivatePropertyActor:
    @property
    def _failure(self) -> None:
        raise AssertionError("private properties must not run during registration")

    def ping(self) -> str:
        return "pong"


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


def test_actor_web_endpoint_shares_state_with_rpc():
    server = ActorServer(host="127.0.0.1")
    server.register("counter", DashboardCounter())
    port = server.serve_background()

    try:
        address = f"http://127.0.0.1:{port}"
        client = ActorClient(FixedResolver({"counter": address}), "counter")

        assert client.increment(2) == 2
        assert httpx.get(f"{address}/", timeout=2).json() == {"value": 2}
        assert httpx.get(f"{address}/counter/main?prefix=Current", timeout=2).json() == {
            "label": "Current main",
            "value": 2,
        }
        assert httpx.post(f"{address}/increment", json={"amount": 1}, timeout=2).json() == 3
        assert client.current() == 3
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs", "status_code"),
    [
        ("GET", "/counter/main?label=other", {}, 422),
        ("POST", "/increment?amount=1", {"json": {"amount": 2}}, 422),
        ("POST", "/increment", {"content": "{"}, 400),
        ("POST", "/increment", {"json": []}, 422),
        ("GET", "/request?request=spoof", {}, 422),
        ("GET", "/counter/main?unknown=value", {}, 422),
    ],
)
def test_actor_web_endpoint_rejects_invalid_arguments(
    method: str,
    path: str,
    request_kwargs: dict[str, Any],
    status_code: int,
):
    server = ActorServer(host="127.0.0.1")
    server.register("counter", DashboardCounter())
    port = server.serve_background()

    try:
        response = httpx.request(method, f"http://127.0.0.1:{port}{path}", timeout=2, **request_kwargs)
        assert response.status_code == status_code
    finally:
        server.stop()


def test_actor_server_rejects_conflicting_web_endpoints():
    server = ActorServer(host="127.0.0.1")

    with pytest.raises(ValueError):
        server.register("duplicate", DuplicateWebEndpointActor())
    with pytest.raises(ValueError):
        server.register("rpc-path", RpcPathWebEndpointActor())


def test_actor_registration_does_not_read_private_properties():
    server = ActorServer(host="127.0.0.1")
    server.register("private-property", PrivatePropertyActor())
    port = server.serve_background()

    try:
        address = f"http://127.0.0.1:{port}"
        client = ActorClient(FixedResolver({"private-property": address}), "private-property")
        assert client.ping() == "pong"
    finally:
        server.stop()


def test_actor_web_endpoint_must_register_before_server_start():
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    server.serve_background()

    try:
        with pytest.raises(RuntimeError, match="before the actor server starts"):
            server.register("counter", DashboardCounter())
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
