# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for actor RPC via the path-based endpoint proxy.

Tests the full round-trip: ActorClient → ProxyResolver → EndpointProxy →
actor server → response. All requests route through the
``/proxy/<encoded-name>/<sub-path>`` pattern introduced by the EndpointProxy
(PR #5336). There is no special actor-routing header; the encoded actor name
lives in the URL path.
"""

import socket

import pytest
import uvicorn
from connectrpc.errors import ConnectError
from iris.actor.client import ActorClient
from iris.actor.resolver import ProxyResolver
from iris.actor.server import ActorServer
from iris.cluster.controller.endpoint_proxy import ALLOWED_METHODS, PROXY_ROUTE, EndpointProxy
from iris.cluster.dashboard_common import on_shutdown
from iris.managed_thread import ThreadContainer
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route


class StatusActor:
    """Minimal actor that returns status counters (mimics Zephyr Coordinator)."""

    def __init__(self):
        self.documents_processed = 0

    def get_status(self) -> dict:
        self.documents_processed += 1
        return {"documents_processed": self.documents_processed, "healthy": True}

    def echo(self, message: str) -> str:
        return f"echo: {message}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(server: uvicorn.Server) -> None:
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
        lambda: server.started,
        timeout=Duration.from_seconds(5.0),
    )


def _build_proxy_app(proxy: EndpointProxy) -> Starlette:
    """Build a minimal Starlette app hosting EndpointProxy with the standard route pair.

    Mirrors the wiring in ControllerDashboard._create_app so these tests
    exercise the same dispatch logic as production.
    """

    async def _redirect_to_slash(request: Request) -> Response:
        name = request.path_params["endpoint_name"]
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"/proxy/{name}/{query}", status_code=307)

    async def _proxy_route(request: Request) -> Response:
        name = request.path_params["endpoint_name"]
        return await proxy.dispatch(
            request,
            encoded_name=name,
            sub_path=request.path_params["sub_path"],
            proxy_prefix=f"/proxy/{name}",
        )

    app = Starlette(
        routes=[
            Route("/proxy/{endpoint_name:str}", _redirect_to_slash, methods=list(ALLOWED_METHODS)),
            Route(PROXY_ROUTE, _proxy_route, methods=list(ALLOWED_METHODS)),
        ],
        lifespan=on_shutdown(proxy.close),
    )
    app.router.redirect_slashes = False
    return app


def _start_proxy(
    threads: ThreadContainer,
    *,
    endpoints: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Spin up an EndpointProxy-backed Starlette server.

    Returns ``(base_url, endpoints)`` where ``endpoints`` is the mutable dict
    the proxy resolver closes over. Tests can register or remove entries at
    runtime.
    """
    endpoints = endpoints if endpoints is not None else {}
    ep_proxy = EndpointProxy(endpoints.get)
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)
    return f"http://127.0.0.1:{port}", endpoints


def test_proxy_round_trip():
    """Full round-trip: ActorClient → ProxyResolver → EndpointProxy → actor server."""
    threads = ThreadContainer()

    actor_name = "test-ns/status"
    actor_server = ActorServer(host="127.0.0.1", threads=threads)
    actor_server.register(actor_name, StatusActor())
    actor_port = actor_server.serve_background()

    try:
        proxy_url, _ = _start_proxy(threads, endpoints={actor_name: f"127.0.0.1:{actor_port}"})

        resolver = ProxyResolver(proxy_url)
        client = ActorClient(resolver, actor_name, max_call_attempts=1)

        result = client.get_status()
        assert result["documents_processed"] == 1
        assert result["healthy"] is True

        result = client.echo("hello")
        assert result == "echo: hello"

        # Second call increments the counter.
        result = client.get_status()
        assert result["documents_processed"] == 2
    finally:
        threads.stop()


def test_proxy_namespaced_actor():
    """ProxyResolver encodes slash-prefixed namespaced names with dot substitution.

    Mirrors real Iris backend behavior where actors are registered under paths
    like /user/job/coordinator/actor-0 and the address includes the http:// scheme.
    """
    threads = ThreadContainer()

    actor_name = "/user/my-job/coordinator/status-0"
    actor_server = ActorServer(host="127.0.0.1", threads=threads)
    actor_server.register(actor_name, StatusActor())
    actor_port = actor_server.serve_background()

    try:
        proxy_url, _ = _start_proxy(threads, endpoints={actor_name: f"http://127.0.0.1:{actor_port}"})

        resolver = ProxyResolver(proxy_url)
        client = ActorClient(resolver, actor_name, max_call_attempts=1)

        result = client.get_status()
        assert result["documents_processed"] == 1
    finally:
        threads.stop()


def test_proxy_unknown_endpoint():
    """EndpointProxy returns an error when the actor endpoint is not registered.

    The proxy returns 404; ConnectClientSync translates this to a ConnectError
    that propagates to the caller.
    """
    threads = ThreadContainer()

    try:
        proxy_url, _ = _start_proxy(threads, endpoints={})

        resolver = ProxyResolver(proxy_url)
        client = ActorClient(resolver, "no-such-ns/no-such-actor", max_call_attempts=1)

        with pytest.raises((ConnectError, Exception)):
            client.get_status()
    finally:
        threads.stop()
