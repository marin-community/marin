# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic HTTP reverse proxy for registered task endpoints.

External clients hit ``/proxy/<endpoint_name>/<sub_path>`` on the controller
dashboard; the proxy resolves the endpoint via :class:`EndpointStore` (with
``.`` -> ``/`` substitution on the URL-encoded name) and forwards method,
path, query string, and filtered headers to the upstream's ``address``.
Bodies are streamed in both directions with no size cap; the only backstop
is :data:`PROXY_TIMEOUT_SECONDS`.

Hop-by-hop headers, ``Cookie`` / ``Set-Cookie``, and ``Authorization`` are
stripped (in both directions for cookies; client -> upstream for
``Authorization``). Forwarding the controller's session JWT to an arbitrary
upstream would be a credential leak, and dashboards that maintain their own
cookie state would shadow the controller session — both are intentionally
prevented here.

Route pattern::

    <ANY-METHOD> /proxy/{endpoint_name:str}/{sub_path:path}
"""

import logging

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from iris.cluster.controller.stores import ControllerStore

logger = logging.getLogger(__name__)

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

# Bound the connection pool explicitly so httpx default drift cannot silently
# change resource usage on the controller.
_HTTPX_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)


class EndpointProxy:
    """Forwards arbitrary HTTP requests to a registered endpoint.

    The proxy resolves the endpoint name (with ``.`` -> ``/`` substitution)
    against ``ControllerStore.endpoints``, then forwards request method,
    path suffix, query string, and filtered headers to the upstream's
    ``address``. Bodies are streamed in both directions with no size cap.
    Hop-by-hop headers, ``Cookie`` / ``Set-Cookie``, and ``Authorization``
    are stripped (see :data:`_HOP_BY_HOP`).

    Lifecycle: construct once on dashboard startup; await :meth:`close` on
    shutdown to drain the underlying httpx connection pool. The proxy is
    safe for concurrent use across requests.
    """

    def __init__(self, store: ControllerStore, *, timeout_seconds: float = PROXY_TIMEOUT_SECONDS) -> None:
        self._store = store
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
        row = self._store.endpoints.resolve(f"/{slashed}") or self._store.endpoints.resolve(slashed)
        if row is None:
            return JSONResponse(
                {"error": f"No endpoint '{url_name}'"},
                status_code=404,
            )

        base = row.address if "://" in row.address else f"http://{row.address}"
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

        response_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP}
        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_resp.aclose),
        )
