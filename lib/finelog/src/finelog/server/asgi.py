# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ASGI app wiring for the finelog log + stats server."""

from __future__ import annotations

from collections.abc import Iterable

from connectrpc.interceptor import Interceptor
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from finelog.rpc.logging_connect import LogServiceWSGIApplication
from finelog.rpc.finelog_stats_connect import StatsServiceWSGIApplication
from finelog.server.interceptors import ConcurrencyLimitInterceptor
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl

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
    return Starlette(routes=routes, middleware=[Middleware(_LegacyIrisLoggingPathMiddleware)])
