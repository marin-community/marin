# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the endpoint proxy.

Spin up a real upstream Starlette app on 127.0.0.1:0, route through a real
EndpointProxy hosted on its own Starlette app, and verify the full
round-trip: method, path suffix, query string, headers, and streaming
bodies. Mirrors the structure of test_actor_proxy.py.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest
import uvicorn
from iris.cluster.controller.endpoint_proxy import (
    ALLOWED_METHODS,
    PROXY_ROUTE,
    EndpointProxy,
    _rewrite_location,
)
from iris.cluster.controller.schema import EndpointRow
from iris.cluster.dashboard_common import on_shutdown
from iris.cluster.types import JobName
from iris.managed_thread import ThreadContainer
from rigging.timing import Duration, ExponentialBackoff, Timestamp
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

# Endpoint name registered in the fake store; reachable at /proxy/user.jobX.dash/...
ENDPOINT_NAME = "/user/jobX/dash"
ENDPOINT_URL_NAME = "user.jobX.dash"


class _FakeEndpointStore:
    """In-memory stand-in for ``EndpointStore`` that exposes ``resolve``.

    Only the methods used by ``EndpointProxy`` are implemented. This mirrors
    the ``StandaloneActorProxy`` pattern in ``test_actor_proxy.py``: keep
    the proxy under test real, swap out the persistent store.
    """

    def __init__(self) -> None:
        self._rows: dict[str, EndpointRow] = {}

    def register(self, name: str, address: str) -> None:
        self._rows[name] = EndpointRow(
            endpoint_id=f"e-{len(self._rows)}",
            name=name,
            address=address,
            task_id=JobName.from_wire("/user/jobX/dash"),
            metadata={},
            registered_at=Timestamp.now(),
        )

    def resolve(self, name: str) -> EndpointRow | None:
        return self._rows.get(name)


class _FakeStore:
    """Duck-typed stand-in for ``ControllerStore`` used by the proxy.

    ``EndpointProxy`` only touches ``store.endpoints.resolve``; we ignore the
    declared ``ControllerStore`` type at the construction sites.
    """

    def __init__(self) -> None:
        self.endpoints = _FakeEndpointStore()


@dataclass
class UpstreamHandle:
    port: int
    received_headers: list[dict[str, str]]
    received_bodies: list[bytes]
    received_paths: list[str]
    received_methods: list[str]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(server: uvicorn.Server) -> None:
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
        lambda: server.started,
        timeout=Duration.from_seconds(5.0),
    )


def _build_upstream_app(handle: UpstreamHandle) -> Starlette:
    """Starlette app exposing the routes used in the test contract."""

    async def _record(request: Request) -> None:
        handle.received_headers.append({k.lower(): v for k, v in request.headers.items()})
        handle.received_bodies.append(await request.body())
        handle.received_paths.append(request.url.path + (f"?{request.url.query}" if request.url.query else ""))
        handle.received_methods.append(request.method)

    async def echo(request: Request) -> Response:
        await _record(request)
        body = handle.received_bodies[-1]
        # Echo back path, query, body length, method, and a sentinel header.
        return JSONResponse(
            {
                "path": request.url.path,
                "query": request.url.query,
                "body_len": len(body),
                "method": request.method,
                "x_custom_in": request.headers.get("x-custom"),
            },
            headers={"x-upstream-saw": request.headers.get("x-custom", "<missing>")},
        )

    async def upstream_500(request: Request) -> Response:
        await _record(request)
        return PlainTextResponse("upstream blew up", status_code=500)

    async def slow(request: Request) -> Response:
        await _record(request)
        # Sleep long enough to outlast any reasonable test-side proxy timeout.
        # The timeout test uses a 0.5s proxy timeout, so 5s here is plenty.
        await asyncio.sleep(5.0)
        return PlainTextResponse("late", status_code=200)

    async def large(request: Request) -> Response:
        await _record(request)
        # 9 MiB streamed in 64 KiB chunks. Reading on the client side before
        # the upstream finishes producing it demonstrates streaming.
        chunk = b"x" * 65536

        async def gen():
            for _ in range(144):  # 144 * 64 KiB = 9 MiB
                yield chunk

        return StreamingResponse(gen(), media_type="application/octet-stream")

    async def cookie_setter(request: Request) -> Response:
        await _record(request)
        return PlainTextResponse(
            "ok",
            headers={"set-cookie": "upstream_session=abc; Path=/"},
        )

    async def redirect_absolute(request: Request) -> Response:
        # Mirrors what Starlette / many WSGI apps emit for canonical-slash
        # redirects: an absolute URL containing the upstream's bind host.
        await _record(request)
        return PlainTextResponse(
            "",
            status_code=302,
            headers={"location": f"http://127.0.0.1:{handle.port}/echo?from=abs"},
        )

    async def redirect_path(request: Request) -> Response:
        # Absolute-path redirect (no scheme/host). Common for ``/`` -> ``/login``.
        await _record(request)
        return PlainTextResponse(
            "",
            status_code=302,
            headers={"location": "/echo?from=path"},
        )

    async def redirect_external(request: Request) -> Response:
        # Cross-origin redirect — proxy must NOT rewrite this.
        await _record(request)
        return PlainTextResponse(
            "",
            status_code=302,
            headers={"location": "https://other.example/landing"},
        )

    routes = [
        Route("/echo", echo, methods=list(ALLOWED_METHODS)),
        Route("/500", upstream_500),
        Route("/slow", slow),
        Route("/large", large),
        Route("/cookie", cookie_setter),
        Route("/redirect-abs", redirect_absolute),
        Route("/redirect-path", redirect_path),
        Route("/redirect-ext", redirect_external),
    ]
    return Starlette(routes=routes)


@pytest.fixture
def threads() -> Iterator[ThreadContainer]:
    container = ThreadContainer()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def upstream(threads: ThreadContainer) -> UpstreamHandle:
    handle = UpstreamHandle(
        port=_free_port(), received_headers=[], received_bodies=[], received_paths=[], received_methods=[]
    )
    app = _build_upstream_app(handle)
    config = uvicorn.Config(app, host="127.0.0.1", port=handle.port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"upstream-{handle.port}")
    _wait_for_server(server)
    return handle


@dataclass
class ProxyHandle:
    base_url: str
    upstream: UpstreamHandle
    store: _FakeStore


def _build_proxy_app(proxy: EndpointProxy) -> Starlette:
    # redirect_slashes=False mirrors the controller dashboard: Starlette's
    # default slash-redirect builds an absolute URL from the request's Host
    # header / bind address, which leaks the internal backend IP back to the
    # browser when the controller sits behind IAP / a load balancer.
    app = Starlette(
        routes=[Route(PROXY_ROUTE, proxy.handle, methods=list(ALLOWED_METHODS))],
        lifespan=on_shutdown(proxy.close),
    )
    app.router.redirect_slashes = False
    return app


@pytest.fixture
def proxy(upstream: UpstreamHandle, threads: ThreadContainer) -> ProxyHandle:
    store = _FakeStore()
    store.endpoints.register(ENDPOINT_NAME, f"127.0.0.1:{upstream.port}")
    ep_proxy = EndpointProxy(store)  # type: ignore[arg-type]
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)
    return ProxyHandle(base_url=f"http://127.0.0.1:{port}", upstream=upstream, store=store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_round_trip_get(proxy: ProxyHandle) -> None:
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/echo",
            params={"q": "1"},
            headers={"x-custom": "hello"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/echo"
    assert body["query"] == "q=1"
    assert body["method"] == "GET"
    assert body["x_custom_in"] == "hello"
    assert resp.headers["x-upstream-saw"] == "hello"
    # Upstream actually saw the request.
    assert proxy.upstream.received_methods[-1] == "GET"
    assert proxy.upstream.received_paths[-1] == "/echo?q=1"


def test_round_trip_post_body(proxy: ProxyHandle) -> None:
    payload = (b"a" * 1024) * 1024  # 1 MiB
    with httpx.Client() as client:
        resp = client.post(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/echo",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "POST"
    assert body["body_len"] == len(payload)
    assert proxy.upstream.received_bodies[-1] == payload


def test_streams_large_response(proxy: ProxyHandle) -> None:
    # Stream-read; assert we can pull bytes incrementally and that the total
    # equals the upstream's 9 MiB without tripping any internal cap.
    total = 0
    first_chunk_at: float | None = None
    started = time.monotonic()
    with httpx.Client(timeout=10.0) as client:
        with client.stream("GET", f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/large") as resp:
            assert resp.status_code == 200
            for chunk in resp.iter_bytes():
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                total += len(chunk)
    assert total == 9 * 1024 * 1024
    # Sanity: we received the first byte well before the full transfer would
    # complete on a buffered (non-streaming) implementation. This is a weak
    # signal but it's the best we can do without a memory profile.
    assert first_chunk_at is not None
    assert first_chunk_at - started < 5.0


def test_unknown_endpoint_returns_404(proxy: ProxyHandle) -> None:
    with httpx.Client() as client:
        resp = client.get(f"{proxy.base_url}/proxy/no.such.endpoint/whatever")
    assert resp.status_code == 404
    assert "no.such.endpoint" in resp.json()["error"]


def test_upstream_5xx_passes_through(proxy: ProxyHandle) -> None:
    with httpx.Client() as client:
        resp = client.get(f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/500")
    assert resp.status_code == 500
    assert resp.text == "upstream blew up"


def test_upstream_connection_refused_returns_502(threads: ThreadContainer) -> None:
    store = _FakeStore()
    # Bind a port and immediately release it; the address is dead by the time
    # the proxy connects.
    dead_port = _free_port()
    store.endpoints.register(ENDPOINT_NAME, f"127.0.0.1:{dead_port}")
    ep_proxy = EndpointProxy(store)  # type: ignore[arg-type]
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)

    with httpx.Client() as client:
        resp = client.get(f"http://127.0.0.1:{port}/proxy/{ENDPOINT_URL_NAME}/anything")
    assert resp.status_code == 502
    assert "Upstream error" in resp.json()["error"]


def test_upstream_timeout_returns_504(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    # Use a short proxy timeout so the test runs quickly. The /slow upstream
    # sleeps far longer than the proxy timeout, guaranteeing a ReadTimeout.
    store = _FakeStore()
    store.endpoints.register(ENDPOINT_NAME, f"127.0.0.1:{upstream.port}")
    ep_proxy = EndpointProxy(store, timeout_seconds=0.5)  # type: ignore[arg-type]
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"http://127.0.0.1:{port}/proxy/{ENDPOINT_URL_NAME}/slow")
    assert resp.status_code == 504
    assert "timeout" in resp.json()["error"].lower()


def test_cookies_stripped_both_directions(proxy: ProxyHandle) -> None:
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/cookie",
            headers={"cookie": "session=secret"},
        )
    assert resp.status_code == 200
    # Upstream did not see the inbound Cookie.
    last_in = proxy.upstream.received_headers[-1]
    assert "cookie" not in last_in
    # Client did not see the outbound Set-Cookie.
    assert "set-cookie" not in {k.lower() for k in resp.headers.keys()}


def test_authorization_stripped(proxy: ProxyHandle) -> None:
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/echo",
            headers={"authorization": "Bearer abc"},
        )
    assert resp.status_code == 200
    last_in = proxy.upstream.received_headers[-1]
    assert "authorization" not in last_in


def test_dot_to_slash_transform(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """``.`` in the URL maps to ``/`` at lookup; literal-``.`` names are unreachable."""
    store = _FakeStore()
    store.endpoints.register("/user/jobX/dash", f"127.0.0.1:{upstream.port}")
    # A name with a literal '.' would only be reachable via /proxy/literal.dot/...,
    # but that URL transforms to 'literal/dot' on lookup and won't match.
    store.endpoints.register("literal.dot", f"127.0.0.1:{upstream.port}")
    ep_proxy = EndpointProxy(store)  # type: ignore[arg-type]
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)

    with httpx.Client() as client:
        # Slash-substituted name reaches the upstream.
        ok = client.get(f"http://127.0.0.1:{port}/proxy/user.jobX.dash/echo")
        assert ok.status_code == 200

        # Literal-dot name is unreachable: 'literal.dot' -> 'literal/dot' on lookup.
        miss = client.get(f"http://127.0.0.1:{port}/proxy/literal.dot/echo")
        assert miss.status_code == 404


def test_method_not_allowed_returns_405(proxy: ProxyHandle) -> None:
    # Starlette filters by registered methods before the handler runs.
    # TRACE / CONNECT are not in ALLOWED_METHODS.
    with httpx.Client() as client:
        resp = client.request(
            "TRACE",
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/echo",
        )
    assert resp.status_code == 405


def test_disallowed_methods_not_listed() -> None:
    """ALLOWED_METHODS should not include CONNECT or TRACE."""
    assert "CONNECT" not in ALLOWED_METHODS
    assert "TRACE" not in ALLOWED_METHODS
    assert set(ALLOWED_METHODS) == {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


# ---------------------------------------------------------------------------
# Location-rewrite unit tests
# ---------------------------------------------------------------------------


_UPSTREAM_BASE = "http://10.0.0.1:8080"
_PROXY_PREFIX = "/proxy/myep"


@pytest.mark.parametrize(
    "loc, expected",
    [
        # Absolute URL with same origin → rewritten to dashboard-relative path.
        ("http://10.0.0.1:8080/foo", "/proxy/myep/foo"),
        ("http://10.0.0.1:8080/foo?a=1&b=2", "/proxy/myep/foo?a=1&b=2"),
        ("http://10.0.0.1:8080/foo#frag", "/proxy/myep/foo#frag"),
        ("http://10.0.0.1:8080/", "/proxy/myep/"),
        # No path on the absolute URL — treat as root.
        ("http://10.0.0.1:8080", "/proxy/myep/"),
        # Protocol-relative on same netloc → rewritten.
        ("//10.0.0.1:8080/foo", "/proxy/myep/foo"),
        # Absolute path → prepended.
        ("/foo", "/proxy/myep/foo"),
        ("/foo?x=1", "/proxy/myep/foo?x=1"),
        ("/", "/proxy/myep/"),
        # Cross-origin absolute URL → passthrough.
        ("http://other.host/foo", "http://other.host/foo"),
        # Different scheme on same host → passthrough (HTTPS upstream is a
        # different origin and we should not silently downgrade).
        ("https://10.0.0.1:8080/foo", "https://10.0.0.1:8080/foo"),
        # Protocol-relative on a different netloc → passthrough.
        ("//other.host/foo", "//other.host/foo"),
        # Relative path → browser resolves against current proxy URL.
        ("foo", "foo"),
        ("./foo", "./foo"),
        ("../foo", "../foo"),
        # Fragment-only and empty → passthrough.
        ("#anchor", "#anchor"),
        ("", ""),
    ],
)
def test_rewrite_location(loc: str, expected: str) -> None:
    assert _rewrite_location(loc, upstream_base=_UPSTREAM_BASE, proxy_prefix=_PROXY_PREFIX) == expected


# ---------------------------------------------------------------------------
# Integration tests: redirects round-trip through the proxy
# ---------------------------------------------------------------------------


def test_absolute_redirect_rewritten_to_proxy(proxy: ProxyHandle) -> None:
    """Upstream emits absolute self-URL Location; proxy must keep us inside."""
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/redirect-abs",
            follow_redirects=False,
        )
    assert resp.status_code == 302
    # Browser would follow this; it must point back at the proxy, not at the
    # upstream's bind address (which is unreachable from outside the cluster).
    assert resp.headers["location"] == f"/proxy/{ENDPOINT_URL_NAME}/echo?from=abs"


def test_path_redirect_rewritten_to_proxy(proxy: ProxyHandle) -> None:
    """Upstream emits ``Location: /foo``; proxy prepends the /proxy/<name> prefix."""
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/redirect-path",
            follow_redirects=False,
        )
    assert resp.status_code == 302
    assert resp.headers["location"] == f"/proxy/{ENDPOINT_URL_NAME}/echo?from=path"


def test_external_redirect_passthrough(proxy: ProxyHandle) -> None:
    """Cross-origin Location must NOT be rewritten; upstream may legitimately send users away."""
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/redirect-ext",
            follow_redirects=False,
        )
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://other.example/landing"


def test_system_endpoint_resolves_via_in_memory_map(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """``/system/...`` endpoints aren't in the SQL store; the proxy must consult
    the service's in-memory ``system_endpoints`` dict — the same dict
    ``ListEndpoints`` uses. This is how ``/system/log-server`` reaches finelog.
    """
    store = _FakeStore()  # empty: no rows for /system/log-server
    system_endpoints = {"/system/log-server": f"127.0.0.1:{upstream.port}"}
    ep_proxy = EndpointProxy(store, system_endpoints=system_endpoints)  # type: ignore[arg-type]
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)

    with httpx.Client() as client:
        resp = client.get(f"http://127.0.0.1:{port}/proxy/system.log-server/echo")
    assert resp.status_code == 200
    assert resp.json()["path"] == "/echo"


def test_system_endpoints_dict_mutation_visible(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """Controller registers ``/system/log-server`` after the dashboard is built;
    the dict is shared by reference so post-construction updates take effect.
    """
    store = _FakeStore()
    system_endpoints: dict[str, str] = {}
    ep_proxy = EndpointProxy(store, system_endpoints=system_endpoints)  # type: ignore[arg-type]
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)

    with httpx.Client() as client:
        # Before registration: 404.
        miss = client.get(f"http://127.0.0.1:{port}/proxy/system.log-server/echo")
        assert miss.status_code == 404

        # Mutate the dict the proxy holds a reference to.
        system_endpoints["/system/log-server"] = f"127.0.0.1:{upstream.port}"

        ok = client.get(f"http://127.0.0.1:{port}/proxy/system.log-server/echo")
    assert ok.status_code == 200


def test_no_trailing_slash_does_not_redirect(proxy: ProxyHandle) -> None:
    """``/proxy/<name>`` (no trailing slash) must NOT trigger Starlette's
    slash-redirect. That redirect's Location is built from the request's
    Host header (or scope["server"]); behind GCP IAP / a load balancer
    that resolves to the internal bind IP, so the browser gets sent to
    an unreachable backend address (ERR_ADDRESS_UNREACHABLE).
    """
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}",
            follow_redirects=False,
        )
    # Critical: 404, never 307. A 307 here means the dashboard is configured
    # with redirect_slashes=True and will leak the internal IP.
    assert resp.status_code == 404


def test_redirect_followed_through_proxy_lands_on_upstream(proxy: ProxyHandle) -> None:
    """End-to-end: client follows the rewritten Location and reaches the upstream's /echo."""
    with httpx.Client(base_url=proxy.base_url) as client:
        resp = client.get(
            f"/proxy/{ENDPOINT_URL_NAME}/redirect-abs",
            follow_redirects=True,
        )
    assert resp.status_code == 200
    assert resp.json()["path"] == "/echo"
    # The follow-up GET hit the upstream's /echo, not /redirect-abs again.
    assert proxy.upstream.received_paths[-1] == "/echo?from=abs"
