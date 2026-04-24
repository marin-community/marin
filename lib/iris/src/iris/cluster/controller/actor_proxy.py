# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor proxy for forwarding ActorService RPCs to actors within the cluster.

External clients send actor calls to the controller; the proxy resolves the
target endpoint from the controller's DB and forwards the raw request to the
actor server on the worker VM. All ActorService methods are proxied
transparently (raw byte forwarding, no deserialization).

Route pattern::

    POST /iris.actor.ActorService/{method}
    X-Iris-Actor-Endpoint: /user/job/actor-name
"""

import logging

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from iris.cluster.controller.stores import ControllerStore

logger = logging.getLogger(__name__)

# Header used by ProxyResolver to tell the proxy which endpoint to forward to.
# Duplicated from iris.actor.resolver.ACTOR_ENDPOINT_HEADER to avoid a
# cluster → actor import dependency.
ACTOR_ENDPOINT_HEADER = "x-iris-actor-endpoint"

PROXY_ROUTE = "/iris.actor.ActorService/{method}"
PROXY_TIMEOUT_SECONDS = 60.0

# Headers that should not be forwarded to upstream (hop-by-hop or routing-specific).
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        ACTOR_ENDPOINT_HEADER,
    }
)


class ActorProxy:
    """Forwards ActorService RPCs to actors resolved from the endpoint store."""

    def __init__(self, store: ControllerStore):
        self._store = store
        self._client = httpx.AsyncClient(timeout=PROXY_TIMEOUT_SECONDS)

    async def close(self) -> None:
        await self._client.aclose()

    async def handle(self, request: Request) -> Response:
        """Proxy an ActorService RPC to the resolved actor endpoint."""
        method = request.path_params["method"]
        endpoint_name = request.headers.get(ACTOR_ENDPOINT_HEADER)
        if not endpoint_name:
            return JSONResponse(
                {"error": f"Missing {ACTOR_ENDPOINT_HEADER} header"},
                status_code=400,
            )

        address = self._resolve_endpoint(endpoint_name)
        if address is None:
            return JSONResponse(
                {"error": f"No endpoint found for '{endpoint_name}'"},
                status_code=404,
            )

        base = address if "://" in address else f"http://{address}"
        upstream_url = f"{base}/iris.actor.ActorService/{method}"
        body = await request.body()
        forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}

        try:
            upstream_resp = await self._client.post(
                upstream_url,
                content=body,
                headers=forward_headers,
            )
        except httpx.HTTPError as exc:
            logger.warning("Proxy upstream error for %s: %s", endpoint_name, exc)
            return JSONResponse(
                {"error": f"Upstream error: {exc}"},
                status_code=502,
            )

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type"),
        )

    def _resolve_endpoint(self, name: str) -> str | None:
        """Resolve an endpoint name to an address via the in-memory store."""
        row = self._store.endpoints.resolve(name)
        if row is None:
            return None
        return row.address
