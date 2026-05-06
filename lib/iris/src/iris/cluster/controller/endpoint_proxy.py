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
both directions with no size cap; the only backstop is
:data:`PROXY_TIMEOUT_SECONDS`.

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

logger = logging.getLogger(__name__)

# Resolves an endpoint wire name (e.g. ``/system/log-server``) to an
# upstream address (``host:port`` or ``http(s)://host:port``), or None if
# unknown. Decoupled from the storage layer so the proxy doesn't need to
# know whether the address came from the SQL endpoint store, the
# controller's in-memory ``system_endpoints`` map, or anywhere else.
EndpointResolver = Callable[[str], str | None]

PROXY_ROUTE = "/proxy/{endpoint_name:str}/{sub_path:path}"
PROXY_TIMEOUT_SECONDS: float = 30.0

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
_LOCATION_HEADERS: tuple[str, ...] = ("location", "content-location")

# Bound the connection pool explicitly so httpx default drift cannot silently
# change resource usage on the controller.
_HTTPX_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)


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
            timeout=timeout_seconds,
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
    ) -> Response:
        """Forward ``request`` to ``encoded_name`` and stream the response back.

        ``encoded_name`` uses ``.`` for path separators (resolved with both
        slash-prefixed and bare forms). ``sub_path`` is the upstream path
        with no leading slash. ``proxy_prefix`` is prepended to rewritten
        ``Location`` / ``Content-Location`` values and forwarded as
        ``X-Forwarded-Prefix``; pass ``""`` when the public URL already
        roots the upstream (subdomain style).
        """
        # Iris wire-format names start with '/'. Try the slash-prefixed form
        # first (the common case for task-registered endpoints), then the bare
        # form for endpoints registered without a leading slash.
        slashed = encoded_name.replace(".", "/")
        address = self._resolve(f"/{slashed}") or self._resolve(slashed)
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

        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        forward_headers.update(_build_forwarded_headers(request, proxy_prefix=proxy_prefix))

        upstream_req = self._client.build_request(
            request.method,
            upstream_url,
            headers=forward_headers,
            content=request.stream(),
        )
        try:
            upstream_resp = await self._client.send(upstream_req, stream=True)
        except (httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            logger.warning("Proxy timeout for %s: %s", encoded_name, exc)
            return JSONResponse(
                {"error": f"Upstream timeout after {self._timeout_seconds:g}s"},
                status_code=504,
            )
        except httpx.HTTPError as exc:
            logger.warning("Proxy upstream error for %s: %s", encoded_name, exc)
            return JSONResponse(
                {"error": f"Upstream error: {exc!r}"},
                status_code=502,
            )

        response_headers: dict[str, str] = {}
        for k, v in upstream_resp.headers.items():
            lk = k.lower()
            if lk in _HOP_BY_HOP:
                continue
            if lk in _LOCATION_HEADERS:
                v = _rewrite_location(v, upstream_base=base, proxy_prefix=proxy_prefix)
            response_headers[k] = v

        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_resp.aclose),
        )
