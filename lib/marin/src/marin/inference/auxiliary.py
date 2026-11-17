# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct HTTP forwarding for inference metadata and tokenization."""

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from marin.inference.http_proxy import forwardable_request_headers, forwardable_response_headers
from marin.inference.types import InferenceRequest

logger = logging.getLogger(__name__)

AUXILIARY_ROUTES = (
    ("GET", "/v1/models"),
    ("GET", "/metrics"),
    ("POST", "/tokenize"),
    ("POST", "/detokenize"),
)


def is_auxiliary_request(method: str, path: str) -> bool:
    return (method, path) in AUXILIARY_ROUTES


async def forward_auxiliary_request(client: httpx.AsyncClient, base_url: str, request: InferenceRequest) -> Response:
    """Forward a buffered HTTP request without decoding its response body."""
    url = base_url.rstrip("/") + request.path
    if request.query_string:
        url += "?" + request.query_string
    async with client.stream(
        request.method,
        url,
        content=request.payload,
        headers=forwardable_request_headers(dict(request.headers)),
    ) as response:
        payload = b"".join([chunk async for chunk in response.aiter_raw()])
        return Response(
            payload, status_code=response.status_code, headers=forwardable_response_headers(response.headers)
        )


async def auxiliary_response(
    client: httpx.AsyncClient, addresses: Sequence[str], request: InferenceRequest, timeout_seconds: float
) -> Response:
    """Try ready endpoints in random order with an equal time budget for each."""
    response = JSONResponse({"error": "no ready auxiliary endpoint"}, status_code=503)
    for address in random.sample(addresses, len(addresses)):
        try:
            async with asyncio.timeout(timeout_seconds / len(addresses)):
                response = await forward_auxiliary_request(client, address, request)
        except (httpx.TransportError, TimeoutError) as error:
            logger.warning("Auxiliary request failed at %s: %s", address, error)
            timed_out = isinstance(error, (httpx.TimeoutException, TimeoutError))
            response = JSONResponse(
                {"error": f"auxiliary request failed: {type(error).__name__}: {error}"},
                status_code=504 if timed_out else 503,
            )
            continue
        if response.status_code not in (502, 503, 504):
            return response
    return response


def build_auxiliary_app(*, upstream_base_url: str, request_timeout_seconds: float) -> Starlette:
    """Expose only model-info, metrics, and tokenization routes to a local backend."""
    client = httpx.AsyncClient(timeout=request_timeout_seconds)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with client:
            yield

    async def forward(request: Request) -> Response:
        forwarded = InferenceRequest(
            request_id="auxiliary",
            method=request.method,
            path=request.url.path,
            payload=await request.body(),
            query_string=request.url.query,
            headers=tuple(request.headers.items()),
        )
        try:
            return await forward_auxiliary_request(client, upstream_base_url, forwarded)
        except httpx.TimeoutException as error:
            return JSONResponse({"error": f"auxiliary upstream timed out: {error}"}, status_code=504)
        except httpx.RequestError as error:
            return JSONResponse({"error": f"auxiliary upstream unavailable: {error}"}, status_code=503)

    return Starlette(
        routes=[Route(path, forward, methods=[method]) for method, path in AUXILIARY_ROUTES],
        lifespan=lifespan,
    )
