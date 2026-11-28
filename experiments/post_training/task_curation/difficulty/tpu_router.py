# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream each solver request to a qualified direct TPU slice."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import httpx
from marin.inference.http_proxy import forwardable_request_headers, forwardable_response_headers
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


def pool_router(
    names: Sequence[str] | Callable[[], Sequence[str]],
    resolve: Callable[[str], str],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    if not callable(names) and not names:
        raise ValueError("A solver router needs qualified slice endpoints")
    pending: dict[str, int] = {}
    client = httpx.AsyncClient(timeout=1800, transport=transport, limits=httpx.Limits(max_connections=256))

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with client:
            yield

    async def forward(request: Request):
        # One event loop owns this counter; increment before awaiting discovery or I/O.
        available = tuple(names() if callable(names) else names)
        if not available:
            return JSONResponse({"error": "qualified TPU slice unavailable"}, status_code=503)
        selected = min(available, key=lambda name: pending.get(name, 0))
        pending[selected] = pending.get(selected, 0) + 1
        upstream = None
        try:
            address = await asyncio.to_thread(resolve, selected)
            url = address.rstrip("/") + request.url.path
            if request.url.query:
                url += "?" + request.url.query
            body = await request.body()
            upstream = await client.send(
                client.build_request(
                    request.method, url, content=body, headers=forwardable_request_headers(request.headers)
                ),
                stream=True,
            )
        except Exception:
            pending[selected] -= 1
            if upstream is not None:
                await upstream.aclose()
            # Never replay a possibly accepted POST; the existing runtime owns retry accounting.
            return JSONResponse({"error": "qualified TPU slice unavailable"}, status_code=503)

        async def chunks():
            try:
                async for block in upstream.aiter_raw():
                    yield block
            finally:
                await upstream.aclose()
                pending[selected] -= 1

        return StreamingResponse(
            chunks(), status_code=upstream.status_code, headers=forwardable_response_headers(upstream.headers)
        )

    return Starlette(routes=[Route("/{path:path}", forward, methods=["GET", "POST"])], lifespan=lifespan)
