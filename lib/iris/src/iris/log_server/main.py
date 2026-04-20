# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone log server process.

Hosts LogServiceImpl on a dedicated port, separate from the controller.
Deployed as a standalone container (see ``lib/iris/Dockerfile`` ``log-server``
stage), or hosted in-thread by the Controller for local/test mode via
``build_log_server_asgi``.

Usage:
    python -m iris.log_server.main --port 10001 --log-dir /var/cache/iris/logs --remote-log-dir gs://bucket/logs
"""

import argparse
import logging
import os
import signal
import sys
from collections.abc import Iterable
from pathlib import Path

import uvicorn
from connectrpc.interceptor import Interceptor
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

from iris.log_server.server import LogServiceImpl
from iris.rpc.auth import AuthInterceptor, JwtTokenManager, NullAuthInterceptor
from iris.rpc.interceptors import SLOW_RPC_THRESHOLD_MS, ConcurrencyLimitInterceptor, RequestTimingInterceptor
from iris.rpc.logging_connect import LogServiceWSGIApplication
from iris.rpc.stats import RpcStatsCollector
from iris.rpc.stats_connect import StatsServiceWSGIApplication
from iris.rpc.stats_service import RpcStatsService

# Cap on concurrent FetchLogs RPCs. Each read can fan out into DuckDB scans
# across hundreds of MB of parquet; allowing unbounded parallelism evicts the
# page cache and wedges the process. Tune alongside the working-set caps in
# duckdb_store.py.
_MAX_CONCURRENT_FETCH_LOGS = 4

logger = logging.getLogger("iris.log_server")


# Env var used by the subprocess entrypoint to receive the controller's JWT
# signing key. Passed out-of-band (env, not argv) so it does not leak via ps.
JWT_KEY_ENV_VAR = "IRIS_LOG_SERVER_JWT_KEY"

# Env var signalling that the dashboard is running in strict-auth mode so
# the log server should reject unauthenticated pushes/fetches instead of
# falling back to anonymous.
AUTH_STRICT_ENV_VAR = "IRIS_LOG_SERVER_AUTH_STRICT"


def build_log_server_asgi(
    service: LogServiceImpl,
    *,
    interceptors: Iterable[Interceptor] = (),
    max_concurrent_fetch_logs: int = _MAX_CONCURRENT_FETCH_LOGS,
    stats_collector: RpcStatsCollector | None = None,
) -> Starlette:
    """Build the ASGI app that serves the LogService RPC endpoints.

    A ``ConcurrencyLimitInterceptor`` is appended to ``interceptors`` to cap
    parallel ``FetchLogs`` RPCs. Callers override ``max_concurrent_fetch_logs``
    in tests that want to exercise the cap deterministically.

    When ``stats_collector`` is supplied a ``RequestTimingInterceptor`` is
    inserted into the chain to feed it, and an ``iris.stats.StatsService``
    endpoint is mounted alongside the LogService so the snapshot can be
    fetched directly from the log-server process (no proxy yet — operators
    curl ``/iris.stats.StatsService/GetRpcStats`` against the log-server
    port).
    """
    chain: list[Interceptor] = list(interceptors)
    if stats_collector is not None:
        chain.append(RequestTimingInterceptor(collector=stats_collector))
    chain.append(ConcurrencyLimitInterceptor({"FetchLogs": max_concurrent_fetch_logs}))
    log_wsgi_app = LogServiceWSGIApplication(service=service, interceptors=tuple(chain))
    routes = [Mount(log_wsgi_app.path, app=WSGIMiddleware(log_wsgi_app))]
    if stats_collector is not None:
        # Reuse the caller's auth chain (sans timing) so the stats endpoint
        # is gated the same way as LogService but doesn't pollute its own
        # numbers with self-reads.
        stats_wsgi_app = StatsServiceWSGIApplication(
            service=RpcStatsService(stats_collector),
            interceptors=tuple(interceptors),
        )
        routes.append(Mount(stats_wsgi_app.path, app=WSGIMiddleware(stats_wsgi_app)))
    return Starlette(routes=routes)


def _build_auth_interceptors(signing_key: str | None, strict: bool) -> tuple[Interceptor, ...]:
    """Select the auth interceptor chain based on controller auth posture.

    - ``strict=True``: require and verify a bearer JWT on every request.
      Matches a dashboard running with a real auth provider.
    - ``strict=False, signing_key set``: verify when present, allow
      anonymous when absent. Matches the dashboard's null-auth path while
      still validating worker tokens.
    - no signing key: anonymous-admin for everything (purely local tests).

    The verifier has no DB handle, so revocation is not enforced at the log
    server — revoked workers stop pushing within one JWT TTL cycle.
    """
    if not signing_key:
        return (NullAuthInterceptor(),)
    verifier = JwtTokenManager(signing_key)
    if strict:
        return (AuthInterceptor(verifier=verifier),)
    return (NullAuthInterceptor(verifier=verifier),)


def run_log_server(
    *,
    port: int,
    log_dir: Path,
    remote_log_dir: str,
    signing_key: str | None = None,
    strict_auth: bool = False,
) -> None:
    """Start a standalone log server, block until SIGTERM/SIGINT."""
    log_dir.mkdir(parents=True, exist_ok=True)

    service = LogServiceImpl(log_dir=log_dir, remote_log_dir=remote_log_dir)
    interceptors = _build_auth_interceptors(signing_key, strict=strict_auth)
    stats_collector = RpcStatsCollector(slow_threshold_ms=SLOW_RPC_THRESHOLD_MS)
    app = build_log_server_asgi(service, interceptors=interceptors, stats_collector=stats_collector)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        log_config=None,
        timeout_keep_alive=120,
    )
    server = uvicorn.Server(config)

    def _shutdown(_signum, _frame):
        logger.info("Log server shutting down")
        server.should_exit = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Log server starting on port %d (log_dir=%s)", port, log_dir)
    server.run()

    service.close()
    logger.info("Log server stopped")


def main() -> None:
    from rigging.log_setup import configure_logging

    parser = argparse.ArgumentParser(description="Iris log server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--log-dir", type=Path, required=True, help="Local log storage directory")
    parser.add_argument("--remote-log-dir", type=str, required=True, help="Remote log storage URI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    configure_logging(level=getattr(logging, args.log_level))
    signing_key = os.environ.get(JWT_KEY_ENV_VAR)
    strict_auth = bool(os.environ.get(AUTH_STRICT_ENV_VAR))
    run_log_server(
        port=args.port,
        log_dir=args.log_dir,
        remote_log_dir=args.remote_log_dir,
        signing_key=signing_key,
        strict_auth=strict_auth,
    )


if __name__ == "__main__":
    sys.exit(main())
