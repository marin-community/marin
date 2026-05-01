# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic HTTP reverse proxy for registered task endpoints.

External clients hit ``/proxy/<endpoint_name>/<sub_path>`` on the controller
dashboard; the proxy resolves the endpoint name via a caller-supplied
``resolve: (name) -> address | None`` callable (with ``.`` -> ``/``
substitution on the URL-encoded name) and forwards method, path, query
string, and filtered headers to the upstream's ``address``. Bodies are
streamed in both directions with no size cap; the only backstop is
:data:`PROXY_TIMEOUT_SECONDS`.

Hop-by-hop headers, ``Cookie`` / ``Set-Cookie``, and ``Authorization`` are
stripped (in both directions for cookies; client -> upstream for
``Authorization``). Forwarding the controller's session JWT to an arbitrary
upstream would be a credential leak, and dashboards that maintain their own
cookie state would shadow the controller session — both are intentionally
prevented here.

``Location`` and ``Content-Location`` response headers are rewritten so 3xx
redirects (and any other absolute-URL hints) keep the browser inside the
proxy instead of escaping to the upstream's bind address. Without this,
upstreams that emit absolute self-URLs (Starlette canonical-slash redirects,
``/`` -> ``/login`` flows, ...) would navigate the user out of
``iris-dev.oa.dev/proxy/<name>/`` straight to the upstream IP.

Route pattern::

    <ANY-METHOD> /proxy/{endpoint_name:str}/{sub_path:path}
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

    async def handle(self, request: Request) -> Response:
        """Handle one proxied request.

        On success returns a :class:`StreamingResponse` whose body is the
        upstream's body streamed chunk-by-chunk. On failure returns a
        :class:`JSONResponse` with the error contract documented in the
        spec. Never raises.
        """
        url_name = request.path_params["endpoint_name"]
        sub_path = request.path_params["sub_path"]
        # Iris wire-format names start with '/'. Try the slash-prefixed form
        # first (the common case for task-registered endpoints), then the bare
        # form for endpoints registered without a leading slash.
        slashed = url_name.replace(".", "/")
        address = self._resolve(f"/{slashed}") or self._resolve(slashed)
        if address is None:
            return JSONResponse(
                {"error": f"No endpoint '{url_name}'"},
                status_code=404,
            )

        base = address if "://" in address else f"http://{address}"
        upstream_url = f"{base}/{sub_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}

        upstream_req = self._client.build_request(
            request.method,
            upstream_url,
            headers=forward_headers,
            content=request.stream(),
        )
        try:
            upstream_resp = await self._client.send(upstream_req, stream=True)
        except (httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            logger.warning("Proxy timeout for %s: %s", url_name, exc)
            return JSONResponse(
                {"error": f"Upstream timeout after {self._timeout_seconds:g}s"},
                status_code=504,
            )
        except httpx.HTTPError as exc:
            logger.warning("Proxy upstream error for %s: %s", url_name, exc)
            return JSONResponse(
                {"error": f"Upstream error: {exc!r}"},
                status_code=502,
            )

        proxy_prefix = f"/proxy/{url_name}"
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
