# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the federated endpoint proxy.

Spin up a real "child controller" Starlette app on 127.0.0.1:0 serving the
path-style ``/proxy/<name>/<sub_path>`` route, drive requests through a real
``FederatedEndpointProxy`` hosted on its own Starlette app, and verify the
cross-cluster forward: path/method/query preservation, the injected federation
bearer, the stripped browser credentials, X-Forwarded-* propagation, verbatim
Location relay, the 401->502 translation, unknown-peer handling, and body
streaming.

The "upstream" here plays the role of the child controller: this controller
cannot reach the child's pod, so it forwards to the child's own ``/proxy``.
"""

import socket
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest
import uvicorn
from iris.cluster.controller.endpoint_proxy import (
    ALLOWED_METHODS,
    PROXY_ROUTE,
    FederatedEndpointProxy,
    PeerAddressResolver,
)
from iris.cluster.dashboard_common import on_shutdown
from iris.managed_thread import ThreadContainer
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

# The peer whose child controller owns the federated endpoint.
PEER_ID = "cw-rno2a"
# Proxy-encoded endpoint name (``.`` for ``/``); the child sees it as a path segment.
ENCODED_NAME = "serve.foo"
# The parent's browser-facing prefix, forwarded so the child rewrites pod redirects to it.
PROXY_PREFIX = "/proxy/t/tok-abc/serve.foo"
# What ``mint_token`` returns: the only credential the child should see.
FEDERATION_TOKEN = "federation-bearer-xyz"


@dataclass
class ChildHandle:
    """Records what the fake child controller observed on its ``/proxy`` route."""

    port: int
    received_headers: list[dict[str, str]]
    received_bodies: list[bytes]
    received_paths: list[str]
    received_methods: list[str]
    received_names: list[str]
    received_sub_paths: list[str]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(server: uvicorn.Server) -> None:
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
        lambda: server.started,
        timeout=Duration.from_seconds(5.0),
    )


def _build_child_app(handle: ChildHandle) -> Starlette:
    """A fake child controller serving the path-style ``/proxy/<name>/<sub_path>`` route.

    The response varies by ``sub_path`` so one route exercises every contract:
    ``echo`` reflects the observed request, ``mirror`` streams the request body back,
    ``unauthorized`` refuses with a 401, and ``redirect`` emits a self-relative Location.
    """

    async def child_proxy(request: Request) -> Response:
        name = request.path_params["endpoint_name"]
        sub_path = request.path_params["sub_path"]
        body = await request.body()
        handle.received_headers.append({k.lower(): v for k, v in request.headers.items()})
        handle.received_bodies.append(body)
        handle.received_paths.append(request.url.path + (f"?{request.url.query}" if request.url.query else ""))
        handle.received_methods.append(request.method)
        handle.received_names.append(name)
        handle.received_sub_paths.append(sub_path)

        if sub_path == "unauthorized":
            return PlainTextResponse(
                "child refused the parent",
                status_code=401,
                headers={"www-authenticate": 'Bearer realm="child"'},
            )
        if sub_path == "redirect":
            # A path already rewritten to the browser-facing prefix; the parent must
            # relay it verbatim (no second rewrite).
            return PlainTextResponse("", status_code=302, headers={"location": f"/proxy/{name}/next"})
        if sub_path == "mirror":
            return Response(content=body, media_type="application/octet-stream")
        return JSONResponse(
            {
                "name": name,
                "sub_path": sub_path,
                "query": request.url.query,
                "method": request.method,
                "authorization": request.headers.get("authorization"),
                "cookie": request.headers.get("cookie"),
                "body_len": len(body),
                "x_forwarded_prefix": request.headers.get("x-forwarded-prefix"),
                "x_forwarded_host": request.headers.get("x-forwarded-host"),
            }
        )

    app = Starlette(routes=[Route(PROXY_ROUTE, child_proxy, methods=list(ALLOWED_METHODS))])
    app.router.redirect_slashes = False
    return app


def _build_fed_proxy_app(proxy: FederatedEndpointProxy, *, proxy_prefix: str) -> Starlette:
    """Host ``FederatedEndpointProxy.dispatch`` behind the parent's path-style route."""

    async def _route(request: Request) -> Response:
        name = request.path_params["endpoint_name"]
        return await proxy.dispatch(
            request,
            peer_id=PEER_ID,
            encoded_name=name,
            sub_path=request.path_params["sub_path"],
            proxy_prefix=proxy_prefix,
        )

    app = Starlette(
        routes=[Route(PROXY_ROUTE, _route, methods=list(ALLOWED_METHODS))],
        lifespan=on_shutdown(proxy.close),
    )
    app.router.redirect_slashes = False
    return app


@pytest.fixture
def threads() -> Iterator[ThreadContainer]:
    container = ThreadContainer()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def child(threads: ThreadContainer) -> ChildHandle:
    handle = ChildHandle(
        port=_free_port(),
        received_headers=[],
        received_bodies=[],
        received_paths=[],
        received_methods=[],
        received_names=[],
        received_sub_paths=[],
    )
    app = _build_child_app(handle)
    config = uvicorn.Config(app, host="127.0.0.1", port=handle.port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"child-{handle.port}")
    _wait_for_server(server)
    return handle


def _start_fed_proxy(
    threads: ThreadContainer,
    *,
    peer_address: PeerAddressResolver,
    mint_token=lambda: FEDERATION_TOKEN,
    proxy_prefix: str = PROXY_PREFIX,
    timeout_seconds: float = 30.0,
) -> str:
    proxy = FederatedEndpointProxy(peer_address, mint_token, timeout_seconds=timeout_seconds)
    app = _build_fed_proxy_app(proxy, proxy_prefix=proxy_prefix)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=f"fed-proxy-{port}")
    _wait_for_server(server)
    return f"http://127.0.0.1:{port}"


@dataclass
class FedProxyHandle:
    base_url: str
    child: ChildHandle


@pytest.fixture
def fed_proxy(child: ChildHandle, threads: ThreadContainer) -> FedProxyHandle:
    child_base = f"http://127.0.0.1:{child.port}"

    def peer_address(peer_id: str) -> str | None:
        return child_base if peer_id == PEER_ID else None

    base_url = _start_fed_proxy(threads, peer_address=peer_address)
    return FedProxyHandle(base_url=base_url, child=child)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forwards_to_child_proxy_route_preserving_method_subpath_and_query(fed_proxy: FedProxyHandle) -> None:
    """The parent reaches the child at ``/proxy/<name>/<sub_path>`` with method and query intact."""
    with httpx.Client() as client:
        resp = client.get(f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/echo/deeper", params={"q": "1", "n": "2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == ENCODED_NAME
    assert body["sub_path"] == "echo/deeper"
    assert body["query"] == "q=1&n=2"
    assert body["method"] == "GET"
    # The child actually saw the path-style forward.
    assert fed_proxy.child.received_paths[-1] == f"/proxy/{ENCODED_NAME}/echo/deeper?q=1&n=2"
    assert fed_proxy.child.received_methods[-1] == "GET"
    assert fed_proxy.child.received_sub_paths[-1] == "echo/deeper"


def test_injects_federation_bearer(fed_proxy: FedProxyHandle) -> None:
    """The child is authorized by the parent's minted federation bearer, nothing else."""
    with httpx.Client() as client:
        resp = client.get(f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/echo")
    assert resp.status_code == 200
    assert fed_proxy.child.received_headers[-1]["authorization"] == f"Bearer {FEDERATION_TOKEN}"
    assert resp.json()["authorization"] == f"Bearer {FEDERATION_TOKEN}"


def test_strips_inbound_cookie_and_authorization(fed_proxy: FedProxyHandle) -> None:
    """The browser's own Cookie/Authorization never cross to the child; only the
    federation bearer does — no parent session or user token leaks across the boundary."""
    with httpx.Client() as client:
        resp = client.get(
            f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/echo",
            headers={"authorization": "Bearer user-session-token", "cookie": "session=secret"},
        )
    assert resp.status_code == 200
    child_headers = fed_proxy.child.received_headers[-1]
    # The user's Authorization is replaced by the federation bearer, not forwarded.
    assert child_headers["authorization"] == f"Bearer {FEDERATION_TOKEN}"
    assert "cookie" not in child_headers


def test_forwards_prefix_and_passes_inbound_forwarded_host(fed_proxy: FedProxyHandle) -> None:
    """X-Forwarded-Prefix carries the parent's browser-facing prefix so the child
    rewrites the pod's self-URLs to it; an inbound X-Forwarded-Host flows through unchanged."""
    with httpx.Client() as client:
        resp = client.get(
            f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/echo",
            headers={"x-forwarded-host": "iris-dev.oa.dev"},
        )
    assert resp.status_code == 200
    child_headers = fed_proxy.child.received_headers[-1]
    assert child_headers["x-forwarded-prefix"] == PROXY_PREFIX
    assert child_headers["x-forwarded-host"] == "iris-dev.oa.dev"


def test_relays_child_location_header_verbatim(fed_proxy: FedProxyHandle) -> None:
    """The child already rewrote Location to the parent's prefix; the parent must not
    rewrite it a second time — it relays the header exactly as the child sent it."""
    with httpx.Client() as client:
        resp = client.get(
            f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/redirect",
            follow_redirects=False,
        )
    assert resp.status_code == 302
    assert resp.headers["location"] == f"/proxy/{ENCODED_NAME}/next"


def test_child_401_translated_to_502_with_body_folded_in(fed_proxy: FedProxyHandle) -> None:
    """A 401 from the child is not a browser auth challenge (the browser never
    authenticated to the child), so the parent translates it to 502, folds the child's
    body into the error, and drops the child's challenge header."""
    with httpx.Client() as client:
        resp = client.get(f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/unauthorized")
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert PEER_ID in error
    assert "child refused the parent" in error
    assert "www-authenticate" not in {k.lower() for k in resp.headers}


def test_unknown_peer_returns_502_without_dialing(child: ChildHandle, threads: ThreadContainer) -> None:
    """An unresolvable peer (peer_address -> None) fails fast with a 502 and never
    reaches any child controller."""
    base_url = _start_fed_proxy(threads, peer_address=lambda _peer_id: None)
    with httpx.Client() as client:
        resp = client.get(f"{base_url}/proxy/{ENCODED_NAME}/echo")
    assert resp.status_code == 502
    assert PEER_ID in resp.json()["error"]
    # No child was dialed.
    assert child.received_methods == []


def test_post_body_streams_through_both_directions(fed_proxy: FedProxyHandle) -> None:
    """A POST body reaches the child intact and the child's response body streams back."""
    payload = (b"payload-" * 4096) + b"tail"  # ~32 KiB, non-round size
    with httpx.Client() as client:
        resp = client.post(
            f"{fed_proxy.base_url}/proxy/{ENCODED_NAME}/mirror",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
    assert resp.status_code == 200
    # Request body forwarded verbatim; response body relayed verbatim.
    assert fed_proxy.child.received_bodies[-1] == payload
    assert fed_proxy.child.received_methods[-1] == "POST"
    assert resp.content == payload
