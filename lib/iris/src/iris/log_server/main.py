# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone log server process.

Hosts LogServiceImpl on a dedicated port, separate from the controller.
Started as a subprocess by the controller's main() entry point in
production; the Controller also spawns this app in-thread for local/test
mode via ``build_log_server_asgi``.

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
from iris.rpc.auth import AuthInterceptor, NullAuthInterceptor
from iris.rpc.interceptors import ConcurrencyLimitInterceptor
from iris.rpc.logging_connect import LogServiceWSGIApplication

# Cap on concurrent FetchLogs RPCs. Each read can fan out into DuckDB scans
# across hundreds of MB of parquet; allowing unbounded parallelism evicts the
# page cache and wedges the process. Tune alongside the working-set caps in
# duckdb_store.py.
_MAX_CONCURRENT_FETCH_LOGS = 4

logger = logging.getLogger(__name__)


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
) -> Starlette:
    """Build the ASGI app that serves the LogService RPC endpoints.

    A ``ConcurrencyLimitInterceptor`` is appended to ``interceptors`` to cap
    parallel ``FetchLogs`` RPCs. Callers override ``max_concurrent_fetch_logs``
    in tests that want to exercise the cap deterministically.
    """
    full_interceptors = (
        *tuple(interceptors),
        ConcurrencyLimitInterceptor({"FetchLogs": max_concurrent_fetch_logs}),
    )
    wsgi_app = LogServiceWSGIApplication(service=service, interceptors=full_interceptors)
    return Starlette(routes=[Mount(wsgi_app.path, app=WSGIMiddleware(wsgi_app))])


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
    # Local import: JwtTokenManager lives in controller.auth to stay close
    # to the DB-backed token issuance path. The log server only needs the
    # verifier half.
    from iris.cluster.controller.auth import JwtTokenManager

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
    app = build_log_server_asgi(service, interceptors=interceptors)

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
