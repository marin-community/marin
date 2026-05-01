# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ASGI app wiring for the finelog log + stats server."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

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

# Vue dashboard build output. Path from this file (server/asgi.py) up to
# lib/finelog/ is four parent directories, then down into dashboard/dist.
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

# The dashboard template ships with `<base href="/">` so URLs resolve against
# the origin root by default. When fronted by a reverse proxy at a sub-path
# (e.g. https://foo.bar/proxy/log-server/), the proxy sets X-Forwarded-Prefix
# and we rewrite the base href so fetch()/vue-router/asset URLs all resolve
# under that prefix instead.
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


# Cap on concurrent FetchLogs RPCs. Each read can fan out into DuckDB scans
# across hundreds of MB of parquet; allowing unbounded parallelism evicts the
# page cache and wedges the process. Tune alongside the working-set caps in
# duckdb_store.py.
_MAX_CONCURRENT_FETCH_LOGS = 4

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
    stats_service: StatsServiceImpl | None = None,
) -> Starlette:
    """Build the ASGI app that serves both the LogService and StatsService.

    A ``ConcurrencyLimitInterceptor`` is appended to ``interceptors`` to cap
    parallel ``FetchLogs`` RPCs. Callers override ``max_concurrent_fetch_logs``
    in tests that want to exercise the cap deterministically.

    ``stats_service`` is optional for tests that don't exercise the stats
    surface; production wiring (``run_log_server``) always supplies one
    backed by the same ``LogStore`` as ``service``.
    """
    chain: list[Interceptor] = list(interceptors)
    chain.append(ConcurrencyLimitInterceptor({"FetchLogs": max_concurrent_fetch_logs}))
    log_wsgi_app = LogServiceWSGIApplication(service=service, interceptors=tuple(chain))

    async def _health(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    routes = [
        Route("/health", _health),
        Mount(log_wsgi_app.path, app=WSGIMiddleware(log_wsgi_app)),
    ]
    if stats_service is not None:
        stats_wsgi_app = StatsServiceWSGIApplication(service=stats_service, interceptors=tuple(interceptors))
        routes.append(Mount(stats_wsgi_app.path, app=WSGIMiddleware(stats_wsgi_app)))

    # Dashboard. dist/static/* is content-hashed (cache-friendly); dist/index.html
    # is the SPA shell served for "/" and any unknown path so Vue Router can
    # take over client-side. Resolved at request time so the server still boots
    # cleanly when the dist hasn't been built yet (tests, fresh checkout).
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
