# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser dashboard and OpenAI-compatible reverse proxy for a quick-serve job.

The dashboard is a single self-contained Vue page served at ``/``. It and every
``/v1/*`` request resolve through the Iris controller proxy's
``/proxy/<encoded-name>/`` prefix, so all browser fetches use relative URLs
(``new URL(path, location.href)``) — the proxy does not rewrite HTML bodies, so an
absolute path like ``/v1/chat/completions`` would escape the prefix.

``/v1/*`` requests are reverse-proxied to whichever serving backend runs on the
slice (see :mod:`marin.inference.serving_backend`) with the response streamed back
verbatim, so server-sent-event token streaming works end to end.
"""

import dataclasses
import importlib.resources
import logging
import socket
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import httpx
import uvicorn
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Request headers that must not be forwarded to the upstream backend server; httpx
# recomputes Host/Content-Length, and the rest are hop-by-hop per RFC 7230.
_REQUEST_DROP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
# Response headers dropped so the framing matches the re-chunked StreamingResponse.
# Content-Encoding is preserved because aiter_raw() yields the undecoded body.
_RESPONSE_DROP_HEADERS = frozenset({"content-length", "connection", "keep-alive", "transfer-encoding"})


@dataclass(frozen=True)
class ServingInfo:
    """Static serving metadata surfaced at ``/info`` and rendered by the dashboard."""

    model: str
    backend: str
    tensor_parallel_size: int
    max_model_len: int | None
    dtype: str
    has_chat_template: bool
    tpu_type: str
    endpoint: str


def build_dashboard_app(
    *,
    upstream_base_url: str,
    model_id: str,
    info: ServingInfo,
    request_timeout_seconds: float = 600.0,
) -> Starlette:
    """Build the Starlette app fronting a local serving backend.

    Args:
        upstream_base_url: Root URL of the backend's OpenAI server (without ``/v1``).
        model_id: The model id the backend reports; surfaced to the dashboard.
        info: Static serving metadata returned from ``/info``.
        request_timeout_seconds: Per-request timeout for upstream proxying.
    """
    state: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        state["client"] = httpx.AsyncClient(
            base_url=upstream_base_url,
            timeout=httpx.Timeout(request_timeout_seconds, connect=10.0),
        )
        try:
            yield
        finally:
            await state.pop("client").aclose()

    async def index(_request: Request) -> Response:
        return HTMLResponse(DASHBOARD_HTML)

    async def serving_info(_request: Request) -> Response:
        return JSONResponse(dataclasses.asdict(info))

    async def health(_request: Request) -> Response:
        client = state["client"]
        try:
            response = await client.get("/health")
            ready = response.status_code == 200
        except httpx.HTTPError:
            ready = False
        return JSONResponse(
            {"status": "ok" if ready else "loading", "model": model_id},
            status_code=200 if ready else 503,
        )

    async def proxy(request: Request) -> Response:
        client = state["client"]
        body = await request.body()
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_DROP_HEADERS}
        upstream_request = client.build_request(
            request.method,
            request.url.path,
            params=dict(request.query_params),
            content=body,
            headers=fwd_headers,
        )
        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"upstream request failed: {exc}"}, status_code=502)

        resp_headers = {k: v for k, v in upstream_response.headers.items() if k.lower() not in _RESPONSE_DROP_HEADERS}

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            body_iter(),
            status_code=upstream_response.status_code,
            headers=resp_headers,
            media_type=upstream_response.headers.get("content-type"),
        )

    return Starlette(
        routes=[
            Route("/", index),
            Route("/dashboard", index),
            Route("/info", serving_info),
            Route("/health", health),
            Route("/v1/{path:path}", proxy, methods=["GET", "POST", "OPTIONS"]),
        ],
        lifespan=lifespan,
    )


def bind_serving_socket(host: str, port: int) -> socket.socket:
    """Bind a listening socket up front so the port is claimed before serving.

    Iris allocates the task's named port from a range (30000-40000) that overlaps
    the OS ephemeral range, so any ephemeral socket the task later opens — notably
    vLLM's many internal sockets — can squat the port we need. Binding here, before the
    backend starts, removes the port from the ephemeral pool and reserves it for us.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    return sock


@dataclass(frozen=True)
class BackgroundServer:
    """A uvicorn server running on a daemon thread, for as long as the thread is up."""

    thread: threading.Thread

    def is_alive(self) -> bool:
        return self.thread.is_alive()


@contextmanager
def serve_app_background(
    app: Starlette,
    sock: socket.socket,
    *,
    name: str = "quick-serve-dashboard",
    start_timeout_seconds: float = 30.0,
) -> Iterator[BackgroundServer]:
    """Run ``app`` under uvicorn on an already-bound ``sock`` in a daemon thread.

    The caller owns the listening socket (see :func:`bind_serving_socket`) so it can
    be claimed before any competing socket in the process can take the port.
    """
    host, port = sock.getsockname()[:2]
    config = uvicorn.Config(app, log_level="info", log_config=None, workers=1)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name=name, daemon=True)
    logger.info("Starting %s on %s:%d", name, host, port)
    thread.start()
    started = ExponentialBackoff(initial=0.02, maximum=1, jitter=0).wait_until(
        lambda: server.started or not thread.is_alive(),
        timeout=Duration.from_seconds(start_timeout_seconds),
    )
    if not started or not server.started:
        server.should_exit = True
        thread.join()
        raise RuntimeError(f"{name} failed to start")
    try:
        yield BackgroundServer(thread=thread)
    finally:
        logger.info("Stopping %s on %s:%d", name, host, port)
        server.should_exit = True
        thread.join()


# Single-file Vue 3 dashboard served at /, read from a sibling .html file. Vue is
# loaded from a CDN by the browser directly (not through the Iris proxy); all
# same-origin fetches stay relative so they resolve under the controller proxy prefix.
DASHBOARD_HTML = (importlib.resources.files(__package__) / "quick_serve_dashboard.html").read_text(encoding="utf-8")
