# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic HTTP reverse proxy for registered task endpoints.

Two equivalent dispatch styles share one forwarding pipeline. Both call
:meth:`EndpointProxy.dispatch`; the caller is responsible for computing
``encoded_name`` / ``sub_path`` / ``proxy_prefix`` from whichever source
identifies the request:

- Path-style: ``/proxy/<encoded_name>/<sub_path>`` on the controller's
  base host. The Starlette route handler reads ``encoded_name`` /
  ``sub_path`` from path params and passes ``proxy_prefix=/proxy/<name>``
  so upstream-emitted absolute URLs get re-prefixed back into the proxy.
- Subdomain-style: ``<encoded_name>.<base_host>/<sub_path>``. The
  dashboard's ``_SubdomainProxyMiddleware`` extracts ``encoded_name``
  from the Host header, takes ``sub_path`` from ``request.url.path``,
  and passes ``proxy_prefix=""`` because the browser already sees the
  upstream as the entire origin.

In both cases the encoded name maps to an Iris endpoint name with ``.``
-> ``/`` substitution (so ``user.jobX.dash`` -> ``/user/jobX/dash``). The
proxy resolves the name via a caller-supplied ``resolve: (name) ->
address | None`` callable, then forwards method, path, query string, and
filtered headers to the upstream's ``address``. Bodies are streamed in
both directions with no size cap; the only backstop is the upstream timeout
(:data:`PROXY_TIMEOUT_SECONDS` by default, or a per-endpoint override the caller
passes to :meth:`EndpointProxy.dispatch`).

Hop-by-hop headers, ``Cookie`` / ``Set-Cookie``, and ``Authorization`` are
stripped (in both directions for cookies; client -> upstream for
``Authorization``). Forwarding the controller's session JWT to an arbitrary
upstream would be a credential leak, and dashboards that maintain their own
cookie state would shadow the controller session — both are intentionally
prevented here.

``X-Forwarded-Host`` / ``X-Forwarded-Proto`` are set so upstreams that build
self-URLs (e.g. Starlette ``url_for``, FastAPI ``request.url_for``) emit
public-facing URLs. ``X-Forwarded-Prefix`` is set in path-style mode only,
which Starlette/FastAPI (`root_path`), Werkzeug (`ProxyFix`), and most
modern Python frameworks honor to mount themselves under the ``/proxy/<name>``
prefix. Subdomain-style mode does not set ``X-Forwarded-Prefix``: the
upstream effectively owns the whole origin.

An upstream ``401`` is translated to a ``502``, keeping the first
:data:`_MAX_UPSTREAM_ERROR_DETAIL_BYTES` of the upstream body as the error
detail. The browser never authenticates to the upstream directly, so an
upstream 401 is not an auth challenge to the browser; relaying it verbatim makes
the dashboard mistake it for an iris auth challenge and pop its login modal. The
controller's own ``/proxy`` access check runs before a request reaches this
module, so a genuine iris auth challenge still reaches the browser as a ``401``.

``Location`` and ``Content-Location`` response headers are rewritten so 3xx
redirects (and any other absolute-URL hints) keep the browser inside the
proxy instead of escaping to the upstream's bind address. Without this,
upstreams that emit absolute self-URLs (Starlette canonical-slash redirects,
``/`` -> ``/login`` flows, ...) would navigate the user out of
``iris-dev.oa.dev/proxy/<name>/`` straight to the upstream IP.
"""

import logging
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from iris.cluster.controller.endpoint_service import proxy_name_to_endpoint_names

logger = logging.getLogger(__name__)

# Resolves an endpoint wire name (e.g. ``/system/log-server``) to an
# upstream address (``host:port`` or ``http(s)://host:port``), or None if
# unknown. Decoupled from the storage layer so the proxy doesn't need to
# know whether the address came from the SQL endpoint store, the
# controller's in-memory ``system_endpoints`` map, or anywhere else.
EndpointResolver = Callable[[str], str | None]

PROXY_ROUTE = "/proxy/{endpoint_name:str}/{sub_path:path}"

# Fallback upstream timeout for an endpoint that registers no override. Well above
# a web round trip because the proxy also fronts model servers, where a single
# non-streaming completion sends no bytes until the whole generation is done and
# runs for minutes. Endpoints override it via PROXY_TIMEOUT_METADATA_KEY.
PROXY_TIMEOUT_SECONDS: float = 120.0

# Connect stays short whatever the read budget, so an unreachable upstream fails
# fast with a 502 instead of hanging for the full read timeout.
PROXY_CONNECT_TIMEOUT_SECONDS: float = 30.0

# Methods exposed via the proxy. CONNECT and TRACE are intentionally absent —
# CONNECT has no meaningful proxy semantics here, TRACE is a recurring source
# of header-disclosure issues.
ALLOWED_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

# Headers stripped on the request (client -> upstream) and response
# (upstream -> client) hops. The bottom three (cookie / set-cookie /
# authorization) are a deliberate security choice; the rest are standard
# hop-by-hop per RFC 7230 §6.1 and RFC 9110 §7.6.1.
_HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "host",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        "te",
        "trailer",
        "proxy-authorization",
        "proxy-authenticate",
        "cookie",
        "set-cookie",
        "authorization",
    }
)

# Response headers carrying URLs that point at the upstream. Rewritten so a
# redirect (or content-negotiation hint) does not navigate the browser out of
# the proxy. Other URL-bearing headers (Refresh, Link, ...) are uncommon for
# the dashboards we proxy and are left alone for now.
_LOCATION_HEADERS: frozenset[str] = frozenset({"location", "content-location"})

# Bound the connection pool explicitly so httpx default drift cannot silently
# change resource usage on the controller.
#
# max_keepalive_connections=0 disables connection reuse to upstreams. Reuse
# races with upstream keepalive lifecycle — the browser fires a burst of asset
# requests and cancels in-flight ones on refresh, leaving a pooled connection
# half-read; the next request on it fails mid-stream with httpx.ReadError, which
# surfaces as an uncaught 500. Dashboard-proxy traffic is low, so a fresh
# connection per request is a fine trade for eliminating that flakiness.
_HTTPX_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=0)

# How much of a rejected upstream's response body is echoed back as the proxy's
# error detail. Bounds the buffering an upstream can force on the controller.
_MAX_UPSTREAM_ERROR_DETAIL_BYTES = 2048


def _build_timeout(seconds: float) -> httpx.Timeout:
    """Give reads/writes/pool the full ``seconds`` budget but cap connect (see
    :data:`PROXY_CONNECT_TIMEOUT_SECONDS`), honoring a per-endpoint budget smaller
    than the cap."""
    return httpx.Timeout(seconds, connect=min(seconds, PROXY_CONNECT_TIMEOUT_SECONDS))


async def _read_error_detail(response: httpx.Response) -> str:
    """Read the head of a streamed error body as text, capped and stripped."""
    chunks: list[bytes] = []
    read = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        read += len(chunk)
        if read >= _MAX_UPSTREAM_ERROR_DETAIL_BYTES:
            break
    body = b"".join(chunks)[:_MAX_UPSTREAM_ERROR_DETAIL_BYTES]
    return body.decode("utf-8", errors="replace").strip()


def _build_forwarded_headers(request: Request, *, proxy_prefix: str) -> dict[str, str]:
    """Compute X-Forwarded-* headers to send upstream.

    Existing X-Forwarded-Host / X-Forwarded-Proto from the inbound chain
    are preserved (so a multi-hop chain — IAP -> controller -> upstream —
    keeps the originating values). X-Forwarded-Prefix is always set to
    *this* hop's prefix (or omitted in subdomain mode where the upstream
    owns the whole origin).

    These headers let frameworks like Starlette/FastAPI (`root_path`),
    Werkzeug (`ProxyFix`), and others mount themselves under the proxy
    prefix and emit public-facing self-URLs.
    """
    fh: dict[str, str] = {}
    inbound_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if inbound_host:
        fh["x-forwarded-host"] = inbound_host
    fh["x-forwarded-proto"] = request.headers.get("x-forwarded-proto") or request.url.scheme
    if proxy_prefix:
        fh["x-forwarded-prefix"] = proxy_prefix
    return fh


def _rewrite_location(loc: str, *, upstream_base: str, proxy_prefix: str) -> str:
    """Rewrite a Location-style URL so it stays inside the proxy.

    ``proxy_prefix`` is the request path prefix used by the dashboard,
    e.g. ``/proxy/system.log-server`` (no trailing slash). ``upstream_base``
    is the upstream origin the proxy forwards to, e.g.
    ``http://10.128.0.31:10001``.

    Cases:

    - Absolute URL whose origin matches ``upstream_base`` -> path on the
      dashboard origin, with ``proxy_prefix`` prepended.
    - Protocol-relative URL (``//host/...``) on the same netloc -> same
      treatment.
    - Absolute path (``/foo``) -> ``proxy_prefix`` prepended.
    - Anything else (cross-origin URL, relative path, fragment-only,
      empty) -> passed through unchanged. Relative paths resolve against
      the browser's current URL, which is already inside the proxy.

    Upstream addresses with a non-empty path component (rare in this
    codebase — endpoints register ``host:port`` only) are not stripped:
    callers should register origin-only addresses.
    """
    if not loc:
        return loc

    parsed = urlsplit(loc)
    base = urlsplit(upstream_base)

    if parsed.netloc:
        scheme_matches = not parsed.scheme or parsed.scheme == base.scheme
        if scheme_matches and parsed.netloc == base.netloc:
            new_path = f"{proxy_prefix}{parsed.path or '/'}"
            return urlunsplit(("", "", new_path, parsed.query, parsed.fragment))
        return loc

    if parsed.path.startswith("/"):
        new_path = f"{proxy_prefix}{parsed.path}"
        return urlunsplit(("", "", new_path, parsed.query, parsed.fragment))

    return loc


def _request_has_body(request: Request) -> bool:
    """Whether the inbound request carries a body to forward upstream.

    A body is present when Content-Length is non-zero or Transfer-Encoding is
    set. Bodyless requests (typically GET/HEAD) must NOT be forwarded with a
    streamed body: ``content=request.stream()`` makes httpx frame an empty
    chunked body, which some upstreams (e.g. hyper) answer by closing the
    connection — poisoning a reused keepalive connection for the next request.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length != "0":
        return True
    return "transfer-encoding" in request.headers


class _UpstreamError(Exception):
    """A forwarded request that the caller answers with ``status_code`` and this message.

    Carries a transport failure (502/504) or an upstream 401 folded to 502.
    """

    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code

    def as_response(self) -> JSONResponse:
        return JSONResponse({"error": str(self)}, status_code=self.status_code)


async def _send_upstream(
    client: httpx.AsyncClient,
    request: Request,
    upstream_url: str,
    forward_headers: dict[str, str],
    *,
    name: str,
    kind: str,
    timeout_seconds: float,
) -> httpx.Response:
    """Forward ``request`` to ``upstream_url``, returning the streaming upstream response.

    ``kind`` labels the upstream in error text ("Upstream" for a direct pod, "Peer" for a
    federated hop) and ``name`` identifies the target (endpoint name / peer id). ``timeout_seconds``
    is the per-request upstream budget (endpoint override or proxy default) applied to this hop and
    named in the 504 text. A transport failure raises :class:`_UpstreamError` carrying the terminal
    502/504 the caller relays; a 401 does too, folded to a 502 with the upstream body appended — the
    upstream refused *this controller*, not the browser (whose Authorization is stripped), so relaying
    the challenge verbatim would misfire the dashboard's login modal. A bodyless request forwards no
    body.
    """
    body = request.stream() if _request_has_body(request) else None
    upstream_req = client.build_request(
        request.method, upstream_url, headers=forward_headers, content=body, timeout=_build_timeout(timeout_seconds)
    )
    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectTimeout as exc:
        # Connect has its own (shorter) budget, so name that one, not the read budget.
        connect_budget = min(timeout_seconds, PROXY_CONNECT_TIMEOUT_SECONDS)
        logger.warning("Proxy connect timeout for %s: %s", name, exc)
        raise _UpstreamError(f"{kind} connect timeout after {connect_budget:g}s", status_code=504) from exc
    except httpx.ReadTimeout as exc:
        logger.warning("Proxy timeout for %s: %s", name, exc)
        raise _UpstreamError(f"{kind} timeout after {timeout_seconds:g}s", status_code=504) from exc
    except httpx.HTTPError as exc:
        logger.warning("Proxy %s error for %s: %s", kind.lower(), name, exc)
        raise _UpstreamError(f"{kind} error: {exc!r}", status_code=502) from exc

    if upstream_resp.status_code == 401:
        detail = await _read_error_detail(upstream_resp)
        await upstream_resp.aclose()
        logger.warning("Proxy 401 from %s -> 502: %s", name, detail)
        refused = f"{kind} '{name}' refused the controller (401)"
        raise _UpstreamError(f"{refused}: {detail}" if detail else refused, status_code=502)
    return upstream_resp


class EndpointProxy:
    """Forwards arbitrary HTTP requests to a registered endpoint.

    The proxy resolves the endpoint name (with ``.`` -> ``/`` substitution)
    via the caller-supplied ``resolve`` callable, then forwards request
    method, path suffix, query string, and filtered headers to the
    upstream's ``address``. Bodies are streamed in both directions with no
    size cap. Hop-by-hop headers, ``Cookie`` / ``Set-Cookie``, and
    ``Authorization`` are stripped (see :data:`_HOP_BY_HOP`).

    The dashboard wires ``resolve`` to consult both the SQL endpoint store
    (task-registered endpoints) and the controller service's in-memory
    ``system_endpoints`` map (``/system/...`` entries such as
    ``/system/log-server``), mirroring ``ListEndpoints``.

    Lifecycle: construct once on dashboard startup; await :meth:`close` on
    shutdown to drain the underlying httpx connection pool. The proxy is
    safe for concurrent use across requests.
    """

    def __init__(
        self,
        resolve: EndpointResolver,
        *,
        timeout_seconds: float = PROXY_TIMEOUT_SECONDS,
    ) -> None:
        self._resolve = resolve
        self._timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(
            timeout=_build_timeout(timeout_seconds),
            follow_redirects=False,
            limits=_HTTPX_LIMITS,
        )

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient. Idempotent."""
        await self._client.aclose()

    async def dispatch(
        self,
        request: Request,
        *,
        encoded_name: str,
        sub_path: str,
        proxy_prefix: str,
        address: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Response:
        """Forward ``request`` to ``encoded_name`` and stream the response back.

        ``encoded_name`` uses ``.`` for path separators (resolved with both
        slash-prefixed and bare forms). ``sub_path`` is the upstream path
        with no leading slash. ``proxy_prefix`` is prepended to rewritten
        ``Location`` / ``Content-Location`` values and forwarded as
        ``X-Forwarded-Prefix``; pass ``""`` when the public URL already
        roots the upstream (subdomain style).

        ``address`` is the upstream for a caller that already resolved (and
        authorized) the endpoint — authorization and forwarding must use the
        same resolution. When omitted, the endpoint is resolved via the
        injected ``resolve`` callable, using the same decode.

        ``timeout_seconds`` overrides the upstream timeout for this one request
        (the endpoint's registered override); ``None`` uses the proxy default.
        """
        if address is None:
            slashed, bare = proxy_name_to_endpoint_names(encoded_name)
            address = self._resolve(slashed) or self._resolve(bare)
        if address is None:
            logger.warning("Proxy %s %s -> no endpoint %r", request.method, request.url.path, encoded_name)
            return JSONResponse(
                {"error": f"No endpoint '{encoded_name}'"},
                status_code=404,
            )

        base = address if "://" in address else f"http://{address}"
        upstream_url = f"{base}/{sub_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        logger.info("Proxy %s %s -> %s", request.method, request.url.path, upstream_url)

        # Honor a prefix set by an upstream proxy in the chain (a parent controller
        # forwarding a federated endpoint sends its own browser-facing prefix, which
        # differs from this hop's ``/proxy/<name>``). Using it for BOTH the pod's
        # X-Forwarded-Prefix and Location rewriting keeps self-URLs and redirects
        # pointing at the browser-visible prefix. Absent (direct browser hit) → this
        # hop's own prefix.
        inbound_prefix = request.headers.get("x-forwarded-prefix")
        effective_prefix = inbound_prefix if inbound_prefix is not None else proxy_prefix

        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        forward_headers.update(_build_forwarded_headers(request, proxy_prefix=effective_prefix))

        try:
            upstream_resp = await _send_upstream(
                self._client,
                request,
                upstream_url,
                forward_headers,
                name=encoded_name,
                kind="Upstream",
                timeout_seconds=timeout_seconds if timeout_seconds is not None else self._timeout_seconds,
            )
        except _UpstreamError as exc:
            return exc.as_response()

        # Rewrite Location/self-URLs to the browser-visible prefix so a redirect from
        # the pod stays inside this hop's ``/proxy/<name>/`` rather than escaping to
        # the upstream bind address.
        response_headers: dict[str, str] = {}
        for k, v in upstream_resp.headers.items():
            lk = k.lower()
            if lk in _HOP_BY_HOP:
                continue
            if lk in _LOCATION_HEADERS:
                v = _rewrite_location(v, upstream_base=base, proxy_prefix=effective_prefix)
            response_headers[k] = v

        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_resp.aclose),
        )


# Resolves a federation peer id to its controller base URL (scheme://host[:port]),
# or None when the peer is unknown / unreachable.
PeerAddressResolver = Callable[[str], str | None]
# Mints a short-lived aud="federation" bearer identifying this controller to a peer.
FederationTokenMinter = Callable[[], str]


class FederatedEndpointProxy:
    """Forwards a /proxy request for a federated endpoint to the peer that owns it.

    The endpoint lives on a child cluster; this controller cannot reach the child's
    pod, so it forwards to the child controller's own path-style ``/proxy/<name>``
    and lets that hop reach the pod. The child authorizes the forward by this
    controller's short-lived ``aud="federation"`` bearer (the only cross-cluster
    credential); the browser's ``Cookie`` / ``Authorization`` are dropped so no
    parent session or user token leaks across the boundary.

    The parent's browser-facing prefix (``/proxy/<name>``, ``/proxy/t/<tok>/<name>``)
    is sent as ``X-Forwarded-Prefix`` so the child rewrites the pod's redirects and
    self-URLs to it directly; the child's response — including ``Location`` — is then
    relayed verbatim.
    """

    def __init__(
        self,
        peer_address: PeerAddressResolver,
        mint_token: FederationTokenMinter,
        *,
        timeout_seconds: float = PROXY_TIMEOUT_SECONDS,
    ) -> None:
        self._peer_address = peer_address
        self._mint_token = mint_token
        self._timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(
            timeout=_build_timeout(timeout_seconds), follow_redirects=False, limits=_HTTPX_LIMITS
        )

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient. Idempotent."""
        await self._client.aclose()

    async def dispatch(
        self,
        request: Request,
        *,
        peer_id: str,
        encoded_name: str,
        sub_path: str,
        proxy_prefix: str,
        timeout_seconds: float | None = None,
    ) -> Response:
        """Forward ``request`` to the peer controller's ``/proxy/<encoded_name>``.

        ``proxy_prefix`` is this hop's browser-facing prefix, forwarded so the child
        (and pod) build URLs the browser can follow back through this controller.

        ``timeout_seconds`` is the endpoint's override (carried on the mirror row) so
        this parent hop waits as long as the child hop will; ``None`` uses the default.
        """
        base = self._peer_address(peer_id)
        if base is None:
            logger.warning("Proxy %s %s -> peer %r unavailable", request.method, request.url.path, peer_id)
            return JSONResponse({"error": f"Peer '{peer_id}' unavailable"}, status_code=502)

        upstream_url = f"{base.rstrip('/')}/proxy/{encoded_name}/{sub_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        logger.info("Federated proxy %s %s -> %s (%s)", request.method, request.url.path, upstream_url, peer_id)

        # _HOP_BY_HOP drops the browser's Cookie/Authorization; the child gets only
        # this controller's federation bearer.
        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        forward_headers.update(_build_forwarded_headers(request, proxy_prefix=proxy_prefix))
        forward_headers["authorization"] = f"Bearer {self._mint_token()}"

        try:
            upstream_resp = await _send_upstream(
                self._client,
                request,
                upstream_url,
                forward_headers,
                name=peer_id,
                kind="Peer",
                timeout_seconds=timeout_seconds if timeout_seconds is not None else self._timeout_seconds,
            )
        except _UpstreamError as exc:
            return exc.as_response()

        # The child already rewrote Location/self-URLs to this hop's prefix (sent as
        # X-Forwarded-Prefix), so relay its response headers verbatim minus hop-by-hop.
        response_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP}
        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_resp.aclose),
        )
