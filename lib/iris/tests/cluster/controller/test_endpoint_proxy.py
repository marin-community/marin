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
from iris.cluster.controller.dashboard import _extract_proxy_subdomain, _SubdomainProxyMiddleware
from iris.cluster.controller.endpoint_proxy import (
    ALLOWED_METHODS,
    PROXY_ROUTE,
    EndpointProxy,
    _rewrite_location,
)
from iris.cluster.dashboard_common import on_shutdown
from iris.managed_thread import ThreadContainer
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

# Endpoint name registered with the proxy resolver; reachable at /proxy/user.jobX.dash/...
ENDPOINT_NAME = "/user/jobX/dash"
ENDPOINT_URL_NAME = "user.jobX.dash"


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
    # Mutable name->address mapping that backs the proxy's resolver. Tests can
    # add or remove entries at runtime to exercise the post-construction
    # registration path used by the controller's system endpoints.
    endpoints: dict[str, str]


def _build_proxy_app(proxy: EndpointProxy) -> Starlette:
    # Mirrors the controller dashboard's wiring:
    # - ``/proxy/<name>`` (no trailing slash) -> path-only 307 to ``/proxy/<name>/``.
    #   We can't use Starlette's ``redirect_slashes=True`` because it builds
    #   an *absolute* Location from scope["server"] / Host, leaking the
    #   internal bind IP behind IAP. A path-only Location resolves against
    #   the browser's current origin instead.
    # - ``/proxy/<name>/<sub_path>`` -> the proxy itself.
    async def _redirect_to_slash(request: Request) -> Response:
        from starlette.responses import RedirectResponse

        name = request.path_params["endpoint_name"]
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"/proxy/{name}/{query}", status_code=307)

    app = Starlette(
        routes=[
            Route("/proxy/{endpoint_name:str}", _redirect_to_slash, methods=list(ALLOWED_METHODS)),
            Route(PROXY_ROUTE, proxy.handle, methods=list(ALLOWED_METHODS)),
        ],
        lifespan=on_shutdown(proxy.close),
    )
    app.router.redirect_slashes = False
    return app


def _start_proxy(
    threads: ThreadContainer,
    *,
    endpoints: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[str, dict[str, str], EndpointProxy]:
    """Spin up an EndpointProxy + its hosting Starlette server.

    Returns ``(base_url, endpoints, proxy)``. ``endpoints`` is the mutable
    dict the proxy resolves against — callers can register or remove names
    at runtime. The proxy uses ``endpoints.get`` directly as its resolver,
    keeping the test wiring identical to production: a single ``name ->
    address`` callable, no fakes for the persistent stores.
    """
    endpoints = endpoints if endpoints is not None else {}
    ep_proxy = EndpointProxy(endpoints.get, timeout_seconds=timeout_seconds)
    app = _build_proxy_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"proxy-{port}")
    _wait_for_server(server)
    return f"http://127.0.0.1:{port}", endpoints, ep_proxy


@pytest.fixture
def proxy(upstream: UpstreamHandle, threads: ThreadContainer) -> ProxyHandle:
    base_url, endpoints, _ = _start_proxy(threads, endpoints={ENDPOINT_NAME: f"127.0.0.1:{upstream.port}"})
    return ProxyHandle(base_url=base_url, upstream=upstream, endpoints=endpoints)


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
    # Bind a port and immediately release it; the address is dead by the time
    # the proxy connects.
    dead_port = _free_port()
    base_url, _, _ = _start_proxy(threads, endpoints={ENDPOINT_NAME: f"127.0.0.1:{dead_port}"})

    with httpx.Client() as client:
        resp = client.get(f"{base_url}/proxy/{ENDPOINT_URL_NAME}/anything")
    assert resp.status_code == 502
    assert "Upstream error" in resp.json()["error"]


def test_upstream_timeout_returns_504(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    # Use a short proxy timeout so the test runs quickly. The /slow upstream
    # sleeps far longer than the proxy timeout, guaranteeing a ReadTimeout.
    base_url, _, _ = _start_proxy(
        threads,
        endpoints={ENDPOINT_NAME: f"127.0.0.1:{upstream.port}"},
        timeout_seconds=0.5,
    )

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{base_url}/proxy/{ENDPOINT_URL_NAME}/slow")
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
    # A name with a literal '.' would only be reachable via /proxy/literal.dot/...,
    # but that URL transforms to 'literal/dot' on lookup and won't match.
    base_url, _, _ = _start_proxy(
        threads,
        endpoints={
            "/user/jobX/dash": f"127.0.0.1:{upstream.port}",
            "literal.dot": f"127.0.0.1:{upstream.port}",
        },
    )

    with httpx.Client() as client:
        # Slash-substituted name reaches the upstream.
        ok = client.get(f"{base_url}/proxy/user.jobX.dash/echo")
        assert ok.status_code == 200

        # Literal-dot name is unreachable: 'literal.dot' -> 'literal/dot' on lookup.
        miss = client.get(f"{base_url}/proxy/literal.dot/echo")
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


def test_post_construction_registration_visible(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """The resolver is a callable closed over a mutable map — registrations
    after the proxy is built must take effect on the next request. This is
    how the controller's ``/system/log-server`` registration during start()
    becomes visible to a dashboard already constructed in ``__init__``.
    """
    base_url, endpoints, _ = _start_proxy(threads, endpoints={})

    with httpx.Client() as client:
        miss = client.get(f"{base_url}/proxy/{ENDPOINT_URL_NAME}/echo")
        assert miss.status_code == 404

        # Mutate the dict the resolver closes over.
        endpoints[ENDPOINT_NAME] = f"127.0.0.1:{upstream.port}"

        ok = client.get(f"{base_url}/proxy/{ENDPOINT_URL_NAME}/echo")
    assert ok.status_code == 200


def test_no_trailing_slash_redirects_with_path_only_location(proxy: ProxyHandle) -> None:
    """``/proxy/<name>`` (no trailing slash) must 307 to ``/proxy/<name>/``
    with a *path-only* Location.

    Starlette's built-in ``redirect_slashes=True`` would emit an absolute
    Location built from scope["server"] / the Host header — behind IAP that
    is the internal bind IP, sending the browser to an unreachable address.
    Our handler emits a path-only Location instead, which the browser
    resolves against its current origin.
    """
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}",
            follow_redirects=False,
        )
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert location == f"/proxy/{ENDPOINT_URL_NAME}/"
    # Critical: no scheme, no netloc — anything else risks leaking the
    # internal address (e.g. ``http://10.x.x.x:10000/...``).
    assert "://" not in location


def test_no_trailing_slash_redirect_preserves_query_string(proxy: ProxyHandle) -> None:
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}?a=1&b=2",
            follow_redirects=False,
        )
    assert resp.status_code == 307
    assert resp.headers["location"] == f"/proxy/{ENDPOINT_URL_NAME}/?a=1&b=2"


def test_no_trailing_slash_redirect_followed_lands_on_upstream(proxy: ProxyHandle) -> None:
    """End-to-end: client follows the slash redirect and reaches the upstream."""
    with httpx.Client(base_url=proxy.base_url) as client:
        resp = client.get(f"/proxy/{ENDPOINT_URL_NAME}/echo", follow_redirects=True)
    # /echo doesn't need the slash redirect; just sanity-check the round trip works.
    assert resp.status_code == 200
    # And following ``/proxy/<name>`` (bare) -> ``/proxy/<name>/`` actually hits
    # the upstream root. The upstream's "/" returns 404 (no route registered),
    # so what we're really proving is that the redirect was followed at all.
    with httpx.Client(base_url=proxy.base_url) as client:
        bare = client.get(f"/proxy/{ENDPOINT_URL_NAME}", follow_redirects=True)
    # The redirect target (/proxy/<name>/) maps to upstream "/", which the
    # test upstream doesn't define -> 404 from upstream, NOT a connection error.
    assert bare.status_code == 404


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


# ---------------------------------------------------------------------------
# X-Forwarded-* headers
# ---------------------------------------------------------------------------


def test_forwarded_headers_path_mode(proxy: ProxyHandle) -> None:
    """Path-mode requests forward X-Forwarded-Host/Proto/Prefix to the upstream.

    Frameworks like Starlette/FastAPI (`root_path`) and Werkzeug
    (`ProxyFix`) rely on these to mount themselves under ``/proxy/<name>``
    and emit public-facing self-URLs.
    """
    with httpx.Client() as client:
        resp = client.get(
            f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/echo",
            headers={"x-forwarded-host": "iris-dev.oa.dev", "x-forwarded-proto": "https"},
        )
    assert resp.status_code == 200
    upstream = proxy.upstream.received_headers[-1]
    # Inbound forwarded values flow through unchanged (we are one hop in a
    # multi-hop chain).
    assert upstream["x-forwarded-host"] == "iris-dev.oa.dev"
    assert upstream["x-forwarded-proto"] == "https"
    # The proxy adds its own prefix on top.
    assert upstream["x-forwarded-prefix"] == f"/proxy/{ENDPOINT_URL_NAME}"


def test_forwarded_headers_default_to_inbound_host(proxy: ProxyHandle) -> None:
    """Without explicit X-Forwarded-* the proxy synthesizes them from the inbound request."""
    with httpx.Client() as client:
        resp = client.get(f"{proxy.base_url}/proxy/{ENDPOINT_URL_NAME}/echo")
    assert resp.status_code == 200
    upstream = proxy.upstream.received_headers[-1]
    # Proxy is bound on http://127.0.0.1:<port>; that's what the upstream sees.
    assert upstream["x-forwarded-proto"] == "http"
    assert "x-forwarded-host" in upstream
    assert "127.0.0.1" in upstream["x-forwarded-host"]
    assert upstream["x-forwarded-prefix"] == f"/proxy/{ENDPOINT_URL_NAME}"


# ---------------------------------------------------------------------------
# Subdomain dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host, expected",
    [
        # Single-label name (most common — flat endpoint name).
        ("foo.proxy.iris-dev.oa.dev", "foo"),
        # Multi-label name (Iris path-style names with embedded ``.``).
        ("user.job1.dash.proxy.iris-dev.oa.dev", "user.job1.dash"),
        # Port stripped from the inbound Host header.
        ("foo.proxy.iris-dev.oa.dev:443", "foo"),
        # DNS labels are case-insensitive — both marker and encoded name
        # are returned lowercased. Endpoints registered with mixed case
        # are unreachable via subdomain dispatch (use the path-style
        # ``/proxy/<name>`` route instead).
        ("FOO.PROXY.iris-dev.oa.dev", "foo"),
        # First ``proxy`` label wins — anything to its right is the public domain.
        ("foo.proxy.proxy.example.com", "foo"),
        # No ``proxy`` label -> not a subdomain request.
        ("iris-dev.oa.dev", None),
        ("iris.oa.dev", None),
        # ``proxy`` as the leftmost label means there is no encoded name.
        ("proxy.iris-dev.oa.dev", None),
        # Empty / missing host.
        ("", None),
    ],
)
def test_extract_proxy_subdomain(host: str, expected: str | None) -> None:
    assert _extract_proxy_subdomain(host) == expected


def _build_subdomain_app(proxy: EndpointProxy):
    """Wrap an EndpointProxy with subdomain dispatch + a fall-through inner app.

    The inner app responds 418 to any request that the middleware does not
    capture — so tests can distinguish "fell through to the dashboard" from
    "subdomain dispatch fired".
    """

    async def _inner(request: Request) -> Response:
        return PlainTextResponse("inner", status_code=418)

    inner = Starlette(routes=[Route("/{path:path}", _inner, methods=list(ALLOWED_METHODS))])
    inner.router.redirect_slashes = False
    return _SubdomainProxyMiddleware(inner, endpoint_proxy=proxy)


def _start_subdomain_proxy(
    threads: ThreadContainer,
    *,
    endpoints: dict[str, str],
) -> str:
    ep_proxy = EndpointProxy(endpoints.get)
    app = _build_subdomain_app(ep_proxy)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"subdomain-{port}")
    _wait_for_server(server)
    return f"http://127.0.0.1:{port}"


# DNS labels are case-insensitive, so subdomain mode lowercases the encoded
# name before resolving. Use an all-lowercase Iris name for these tests.
_SUB_ENDPOINT_NAME = "/user/job1/dash"
_SUB_ENDPOINT_URL_NAME = "user.job1.dash"


def test_subdomain_dispatch_round_trip(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """A request whose Host has a ``proxy`` label routes to the upstream."""
    base_url = _start_subdomain_proxy(
        threads,
        endpoints={_SUB_ENDPOINT_NAME: f"127.0.0.1:{upstream.port}"},
    )
    with httpx.Client() as client:
        # The Host header drives dispatch; the actual TCP target stays 127.0.0.1.
        resp = client.get(
            f"{base_url}/echo?q=1",
            headers={"host": f"{_SUB_ENDPOINT_URL_NAME}.proxy.iris-dev.oa.dev"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/echo"
    assert body["query"] == "q=1"
    # Subdomain mode does not set X-Forwarded-Prefix — the upstream owns the origin.
    upstream_headers = upstream.received_headers[-1]
    assert "x-forwarded-prefix" not in upstream_headers


def test_subdomain_unknown_endpoint_returns_404(threads: ThreadContainer) -> None:
    base_url = _start_subdomain_proxy(threads, endpoints={})
    with httpx.Client() as client:
        resp = client.get(
            f"{base_url}/anything",
            headers={"host": "no-such.proxy.iris-dev.oa.dev"},
        )
    assert resp.status_code == 404


def test_subdomain_passthrough_when_no_proxy_label(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """Hosts without a ``proxy`` label fall through to the inner app."""
    base_url = _start_subdomain_proxy(
        threads,
        endpoints={_SUB_ENDPOINT_NAME: f"127.0.0.1:{upstream.port}"},
    )
    with httpx.Client() as client:
        resp = client.get(
            f"{base_url}/echo",
            headers={"host": "iris-dev.oa.dev"},
        )
    # Inner app returned 418 — confirms the middleware passed through.
    assert resp.status_code == 418
    assert resp.text == "inner"


def test_subdomain_redirect_rewrites_to_path_only(threads: ThreadContainer, upstream: UpstreamHandle) -> None:
    """Absolute upstream redirects in subdomain mode strip back to a path on the same origin.

    The browser is already on ``<name>.proxy.iris-dev.oa.dev``, so a path-only
    Location lands it on the right host without rewriting through ``/proxy/<name>``.
    """
    base_url = _start_subdomain_proxy(
        threads,
        endpoints={_SUB_ENDPOINT_NAME: f"127.0.0.1:{upstream.port}"},
    )
    with httpx.Client() as client:
        resp = client.get(
            f"{base_url}/redirect-abs",
            headers={"host": f"{_SUB_ENDPOINT_URL_NAME}.proxy.iris-dev.oa.dev"},
            follow_redirects=False,
        )
    assert resp.status_code == 302
    # No /proxy/<name> prefix — just the upstream path on the current origin.
    assert resp.headers["location"] == "/echo?from=abs"
