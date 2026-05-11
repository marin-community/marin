# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ASGI app wiring for the finelog log + stats server."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from connectrpc.compression.gzip import GzipCompression
from connectrpc.compression.zstd import ZstdCompression
from connectrpc.interceptor import Interceptor
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from finelog.rpc.finelog_stats_connect import StatsServiceWSGIApplication
from finelog.rpc.logging_connect import LogServiceWSGIApplication
from finelog.server.interceptors import ConcurrencyLimitInterceptor
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl

logger = logging.getLogger(__name__)

# Up four parents from server/asgi.py reaches lib/finelog/.
_VUE_DIST_DIR = Path(__file__).parent.parent.parent.parent / "dashboard" / "dist"
_DOCKER_VUE_DIST_DIR = Path("/app/dashboard/dist")


def _vue_dist_dir() -> Path | None:
    for candidate in (_VUE_DIST_DIR, _DOCKER_VUE_DIST_DIR):
        if candidate.is_dir() and (candidate / "index.html").is_file():
            return candidate
    return None


_NOT_BUILT_HTML = (
    "<!doctype html><html><body>"
    "<h1>Dashboard not built</h1>"
    "<p>Run <code>npm run build</code> in <code>lib/finelog/dashboard</code>.</p>"
    "</body></html>"
)

# When fronted by a reverse proxy at a sub-path the proxy sends
# X-Forwarded-Prefix; we rewrite <base href="/"> so vue-router/asset URLs
# resolve under that prefix instead of the origin root.
_BASE_HREF_PLACEHOLDER = b'<base href="/"'


def _index_html_with_base(raw: bytes, prefix: str) -> bytes:
    if not prefix or prefix == "/":
        return raw
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    replacement = f'<base href="{prefix}"'.encode()
    return raw.replace(_BASE_HREF_PLACEHOLDER, replacement, 1)


# Cap on concurrent read RPCs. Both FetchLogs and Query fan out into
# DuckDB scans across hundreds of MB of parquet; unbounded parallelism
# evicts the page cache and wedges the process. Tune alongside the
# working-set caps in duckdb_store.py.
_MAX_CONCURRENT_FETCH_LOGS = 4
_MAX_CONCURRENT_QUERY = 4

# zstd first so clients that advertise both pick it: the negotiator walks the
# client's Accept-Encoding in order, so listing zstd first only matters via
# the client side, but we keep it first here for symmetry/readability.
# Memray on prod showed gzip.compress accounting for ~66% of allocated bytes.
_DEFAULT_COMPRESSIONS = (ZstdCompression(), GzipCompression())

# CRON(2026-05-12) -- remove this legacy workaround as all workers & clients
# will be updated by this point.
# Pre-#5212 (b212f0015) the LogService proto package was iris.logging, so old
# worker images push to /iris.logging.LogService/*. Wire format is identical
# (same field numbers across PushLogsRequest/LogEntry/etc.), so we rewrite
# the path before routing.
_LEGACY_PATH_PREFIX = "/iris.logging.LogService/"
_CURRENT_PATH_PREFIX = "/finelog.logging.LogService/"


class _LegacyIrisLoggingPathMiddleware:
    """Rewrite legacy iris.logging.LogService URLs onto the current path."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith(_LEGACY_PATH_PREFIX):
            tail = scope["path"][len(_LEGACY_PATH_PREFIX) :]
            new_path = _CURRENT_PATH_PREFIX + tail
            scope = {**scope, "path": new_path, "raw_path": new_path.encode()}
        await self._app(scope, receive, send)


def build_log_server_asgi(
    service: LogServiceImpl,
    *,
    interceptors: Iterable[Interceptor] = (),
    max_concurrent_fetch_logs: int = _MAX_CONCURRENT_FETCH_LOGS,
    max_concurrent_query: int = _MAX_CONCURRENT_QUERY,
    stats_service: StatsServiceImpl | None = None,
) -> Starlette:
    """Build the ASGI app that serves LogService and (optionally) StatsService.

    A ``ConcurrencyLimitInterceptor`` is appended to each service's chain
    to cap parallel ``FetchLogs`` and ``Query`` RPCs.
    """
    log_chain: list[Interceptor] = list(interceptors)
    log_chain.append(ConcurrencyLimitInterceptor({"FetchLogs": max_concurrent_fetch_logs}))
    log_wsgi_app = LogServiceWSGIApplication(
        service=service, interceptors=tuple(log_chain), compressions=_DEFAULT_COMPRESSIONS
    )

    async def _health(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    routes = [
        Route("/health", _health),
        Mount(log_wsgi_app.path, app=WSGIMiddleware(log_wsgi_app)),
    ]
    if stats_service is not None:
        stats_chain: list[Interceptor] = list(interceptors)
        stats_chain.append(ConcurrencyLimitInterceptor({"Query": max_concurrent_query}))
        stats_wsgi_app = StatsServiceWSGIApplication(
            service=stats_service, interceptors=tuple(stats_chain), compressions=_DEFAULT_COMPRESSIONS
        )
        routes.append(Mount(stats_wsgi_app.path, app=WSGIMiddleware(stats_wsgi_app)))

    # SPA shell at "/" and any unknown path so Vue Router can take over
    # client-side. Static assets under dist/static/* are content-hashed.
    # Resolved at request time so the server boots cleanly without a built dist.
    dist = _vue_dist_dir()
    if dist is not None:
        routes.append(Mount("/static", app=StaticFiles(directory=dist / "static"), name="static"))
        favicon = dist / "favicon.ico"
        if favicon.is_file():
            routes.append(Route("/favicon.ico", lambda _r: FileResponse(favicon)))

        async def _spa_index(request: Request) -> Response:
            prefix = request.headers.get("x-forwarded-prefix", "")
            html = _index_html_with_base((dist / "index.html").read_bytes(), prefix)
            return Response(html, media_type="text/html")

        routes.append(Route("/", _spa_index))
        routes.append(Route("/{rest:path}", _spa_index))
    else:
        logger.info("Dashboard dist not found; serving placeholder at /")

        async def _placeholder(_: Request) -> Response:
            return Response(_NOT_BUILT_HTML, media_type="text/html")

        routes.append(Route("/", _placeholder))

    return Starlette(routes=routes, middleware=[Middleware(_LegacyIrisLoggingPathMiddleware)])
