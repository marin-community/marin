# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloud Run reverse proxy for the Iris controller, protected by GCP IAP.

All requests are forwarded to the controller VM discovered via GCE labels.
Authentication is enforced by Cloud Run's native IAP integration before
requests reach this container, so the proxy trusts the IAP-supplied
``X-Goog-Authenticated-User-Email`` header for audit logging.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from discovery import get_controller_url
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Headers that should not be forwarded to the controller.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        # IAP headers — informational only, not needed upstream.
        "x-goog-iap-jwt-assertion",
        "x-goog-authenticated-user-email",
        "x-goog-authenticated-user-id",
    }
)

_client: httpx.AsyncClient | None = None
_client_base_url: str | None = None


@asynccontextmanager
async def _lifespan(app: Starlette):
    yield
    if _client is not None:
        await _client.aclose()


async def _get_client() -> httpx.AsyncClient:
    """Return an httpx client pointed at the current controller URL.

    Recreates the client when the discovered URL changes (e.g. controller VM
    replaced).
    """
    global _client, _client_base_url

    url = get_controller_url()
    if _client is None or url != _client_base_url:
        if _client is not None:
            await _client.aclose()
        _client = httpx.AsyncClient(base_url=url, timeout=120.0)
        _client_base_url = url
    return _client


def _build_upstream_headers(request: Request, iap_email: str | None) -> dict[str, str]:
    """Build the header dict to forward to the controller."""
    headers: dict[str, str] = {}

    for key, value in request.headers.items():
        if key.lower() not in _HOP_BY_HOP:
            headers[key] = value

    if iap_email:
        headers["x-forwarded-user"] = iap_email

    return headers


async def _proxy(request: Request) -> Response:
    """Forward any request to the controller."""
    # IAP-supplied identity header. Format: "accounts.google.com:user@example.com".
    iap_email_raw = request.headers.get("x-goog-authenticated-user-email")
    iap_email = iap_email_raw.split(":", 1)[-1] if iap_email_raw else None

    client = await _get_client()
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    upstream_headers = _build_upstream_headers(request, iap_email)
    body = await request.body()

    upstream_resp = await client.request(
        method=request.method,
        url=path,
        headers=upstream_headers,
        content=body,
    )

    # Forward the response, stripping hop-by-hop and encoding headers.
    # httpx already decompresses the body, so content-encoding and
    # content-length from the upstream are stale.
    _strip_resp = {"transfer-encoding", "connection", "content-encoding", "content-length"}
    resp_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in _strip_resp}
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
    )


async def _health(request: Request) -> JSONResponse:
    """Health check — pings the controller to verify connectivity."""
    try:
        client = await _get_client()
        resp = await client.get("/health", timeout=5.0)
        controller_ok = resp.status_code == 200
    except Exception:
        controller_ok = False

    return JSONResponse({"status": "ok", "controller_reachable": controller_ok})


routes = [
    Route("/health", _health, methods=["GET"]),
    Route("/{path:path}", _proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
]

app = Starlette(routes=routes, lifespan=_lifespan)
