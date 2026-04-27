# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ASGI app wiring for the finelog log server."""

from __future__ import annotations

from collections.abc import Iterable

from connectrpc.interceptor import Interceptor
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

from finelog.rpc.logging_connect import LogServiceWSGIApplication
from finelog.server.interceptors import ConcurrencyLimitInterceptor
from finelog.server.service import LogServiceImpl

# Cap on concurrent FetchLogs RPCs. Each read can fan out into DuckDB scans
# across hundreds of MB of parquet; allowing unbounded parallelism evicts the
# page cache and wedges the process. Tune alongside the working-set caps in
# duckdb_store.py.
_MAX_CONCURRENT_FETCH_LOGS = 4


def build_log_server_asgi(
    service: LogServiceImpl,
    *,
    interceptors: Iterable[Interceptor] = (),
    max_concurrent_fetch_logs: int = _MAX_CONCURRENT_FETCH_LOGS,
) -> Starlette:
    """Build the ASGI app that serves the LogService RPC endpoints.

    A ``ConcurrencyLimitInterceptor`` is appended to ``interceptors`` to cap
    parallel ``FetchLogs`` RPCs. Callers override ``max_concurrent_fetch_logs``
    in tests that want to exercise the cap deterministically.
    """
    chain: list[Interceptor] = list(interceptors)
    chain.append(ConcurrencyLimitInterceptor({"FetchLogs": max_concurrent_fetch_logs}))
    log_wsgi_app = LogServiceWSGIApplication(service=service, interceptors=tuple(chain))
    routes = [Mount(log_wsgi_app.path, app=WSGIMiddleware(log_wsgi_app))]
    return Starlette(routes=routes)
