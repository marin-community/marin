# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the actor proxy.

Tests the full round-trip: external client → proxy → actor server.
"""

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route

from iris.actor import ActorClient, ActorServer
from iris.actor.resolver import ACTOR_ENDPOINT_HEADER, ProxyResolver
from iris.cluster.controller.actor_proxy import PROXY_ROUTE
from iris.managed_thread import ThreadContainer
from iris.time_utils import Duration, ExponentialBackoff


class StatusActor:
    """Minimal actor that returns status counters (mimics Zephyr Coordinator)."""

    def __init__(self):
        self.documents_processed = 0

    def get_status(self) -> dict:
        self.documents_processed += 1
        return {"documents_processed": self.documents_processed, "healthy": True}

    def echo(self, message: str) -> str:
        return f"echo: {message}"


class FakeEndpointDB:
    """Minimal fake that satisfies ActorProxy._resolve_endpoint via the same interface as ControllerDB."""

    def __init__(self):
        self._endpoints: dict[str, str] = {}

    def register(self, name: str, address: str) -> None:
        self._endpoints[name] = address

    def resolve(self, name: str) -> str | None:
        return self._endpoints.get(name)


class StandaloneActorProxy:
    """A standalone proxy for testing without a full controller.

    Wraps ActorProxy's forwarding logic but uses a simple dict-based
    endpoint registry instead of ControllerDB.
    """

    def __init__(self):
        self._endpoints: dict[str, str] = {}
        self._client = httpx.AsyncClient(timeout=60.0)

    def register(self, name: str, address: str) -> None:
        self._endpoints[name] = address

    async def close(self) -> None:
        await self._client.aclose()

    async def handle(self, request):
        from starlette.responses import JSONResponse, Response

        method = request.path_params["method"]
        endpoint_name = request.headers.get(ACTOR_ENDPOINT_HEADER)
        if not endpoint_name:
            return JSONResponse(
                {"error": f"Missing {ACTOR_ENDPOINT_HEADER} header"},
                status_code=400,
            )

        address = self._endpoints.get(endpoint_name)
        if address is None:
            return JSONResponse(
                {"error": f"No endpoint found for '{endpoint_name}'"},
                status_code=404,
            )

        upstream_url = f"http://{address}/iris.actor.ActorService/{method}"
        body = await request.body()
        # Forward all headers except hop-by-hop and routing headers.
        skip = frozenset({"host", "transfer-encoding", "connection", "keep-alive", ACTOR_ENDPOINT_HEADER})
        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}

        upstream_resp = await self._client.post(
            upstream_url,
            content=body,
            headers=forward_headers,
        )

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
        )


def _start_proxy_server(proxy: StandaloneActorProxy, threads: ThreadContainer) -> int:
    """Start a standalone proxy Starlette app and return the port."""
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    app = Starlette(
        routes=[Route(PROXY_ROUTE, proxy.handle, methods=["POST"])],
        on_shutdown=[proxy.close],
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-server-{port}")
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
        lambda: server.started,
        timeout=Duration.from_seconds(5.0),
    )
    return port


def test_proxy_round_trip():
    """Full round-trip: client → proxy → actor server → response."""
    threads = ThreadContainer()
    actor_server = ActorServer(host="127.0.0.1", threads=threads)
    actor_server.register("status", StatusActor())
    actor_port = actor_server.serve_background()

    proxy = StandaloneActorProxy()
    proxy.register("test-ns/status", f"127.0.0.1:{actor_port}")
    proxy_port = _start_proxy_server(proxy, threads)

    try:
        resolver = ProxyResolver(f"http://127.0.0.1:{proxy_port}", namespace="test-ns")
        client = ActorClient(resolver, "status")

        result = client.get_status()
        assert result["documents_processed"] == 1
        assert result["healthy"] is True

        result = client.echo("hello")
        assert result == "echo: hello"

        # Second call increments the counter
        result = client.get_status()
        assert result["documents_processed"] == 2
    finally:
        threads.stop()


def test_proxy_missing_endpoint_header():
    """Proxy returns 400 when the endpoint header is missing."""
    threads = ThreadContainer()
    proxy = StandaloneActorProxy()
    proxy_port = _start_proxy_server(proxy, threads)

    try:
        # Call the proxy directly without the header
        with httpx.Client() as client:
            resp = client.post(
                f"http://127.0.0.1:{proxy_port}/iris.actor.ActorService/Call",
                content=b"",
                headers={"content-type": "application/proto"},
            )
            assert resp.status_code == 400
            assert ACTOR_ENDPOINT_HEADER in resp.json()["error"]
    finally:
        threads.stop()


def test_proxy_unknown_endpoint():
    """Proxy returns 404 when the endpoint name is not registered."""
    threads = ThreadContainer()
    proxy = StandaloneActorProxy()
    proxy_port = _start_proxy_server(proxy, threads)

    try:
        with httpx.Client() as client:
            resp = client.post(
                f"http://127.0.0.1:{proxy_port}/iris.actor.ActorService/Call",
                content=b"",
                headers={
                    "content-type": "application/proto",
                    ACTOR_ENDPOINT_HEADER: "no-such-ns/no-such-actor",
                },
            )
            assert resp.status_code == 404
    finally:
        threads.stop()
