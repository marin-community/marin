# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloud Run reverse proxy for the Iris controller.

All requests are forwarded to the controller VM discovered via GCE labels.
When IAP is enabled, the proxy validates the ``X-Goog-IAP-JWT-Assertion``
header for defense-in-depth.  When ``REQUIRE_IAP=false`` (the default),
IAP validation is skipped and all requests are forwarded without auth —
useful for initial testing before the LB + IAP stack is configured.
"""

import logging
import os
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from discovery import get_controller_url

logger = logging.getLogger(__name__)

REQUIRE_IAP = os.environ.get("REQUIRE_IAP", "false").lower() in ("true", "1", "yes")

_IRIS_TOKEN_HEADER = "x-iris-token"

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


def _get_proxy_token() -> str | None:
    """Read the proxy's own API key from the environment (set via Secret Manager)."""
    secret_name = os.environ.get("PROXY_TOKEN_SECRET")
    if not secret_name:
        return None
    try:
        from google.cloud import secretmanager

        sm = secretmanager.SecretManagerServiceClient()
        project = os.environ.get("GCP_PROJECT", "hai-gcp-models")
        name = f"projects/{project}/secrets/{secret_name}/versions/latest"
        response = sm.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8").strip()
    except Exception:
        logger.exception("Failed to read proxy token from Secret Manager")
        return None


_proxy_token: str | None = None


@asynccontextmanager
async def _lifespan(app: Starlette):
    global _proxy_token
    _proxy_token = _get_proxy_token()
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

    # If the CLI sent a controller JWT via X-Iris-Token, promote it to
    # Authorization for the controller.  Otherwise keep the original
    # Authorization header (e.g. browser session cookie flow where IAP uses
    # cookie-based auth and Authorization is the controller JWT).
    iris_token = request.headers.get(_IRIS_TOKEN_HEADER)
    if iris_token:
        headers["authorization"] = f"Bearer {iris_token}"
        headers.pop(_IRIS_TOKEN_HEADER, None)

    if iap_email:
        headers["x-forwarded-user"] = iap_email

    return headers


async def _proxy(request: Request) -> Response:
    """Forward any request to the controller."""
    iap_email: str | None = None

    if REQUIRE_IAP:
        from iap import IapValidationError, validate_iap_jwt

        try:
            iap_email = validate_iap_jwt(dict(request.headers))
        except IapValidationError as exc:
            logger.warning("IAP validation failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=401)

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

    # Forward the response, stripping hop-by-hop headers.
    resp_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in {"transfer-encoding", "connection"}
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
    )


async def _health(request: Request) -> JSONResponse:
    """Health check — does not require IAP."""
    # Optionally ping the controller to verify connectivity.
    if _proxy_token:
        try:
            client = await _get_client()
            resp = await client.get("/health", timeout=5.0)
            controller_ok = resp.status_code == 200
        except Exception:
            controller_ok = False
    else:
        controller_ok = None  # unknown, no token configured

    return JSONResponse(
        {
            "status": "ok",
            "controller_reachable": controller_ok,
        }
    )


routes = [
    Route("/health", _health, methods=["GET"]),
    Route("/{path:path}", _proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
]

app = Starlette(routes=routes, lifespan=_lifespan)
