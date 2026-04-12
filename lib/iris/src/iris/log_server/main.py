# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone log server process.

Hosts LogServiceImpl on a dedicated port, separate from the controller.
Started as a subprocess by the controller's main() entry point.

Usage:
    python -m iris.log_server.main --port 10001 --log-dir /var/cache/iris/logs --remote-log-dir gs://bucket/logs
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

from iris.log_server.server import LogServiceImpl
from iris.rpc.logging_connect import LogServiceWSGIApplication

logger = logging.getLogger(__name__)


def run_log_server(*, port: int, log_dir: Path, remote_log_dir: str) -> None:
    """Start a standalone log server, block until SIGTERM/SIGINT."""
    log_dir.mkdir(parents=True, exist_ok=True)

    service = LogServiceImpl(log_dir=log_dir, remote_log_dir=remote_log_dir)
    wsgi_app = LogServiceWSGIApplication(service=service)

    app = Starlette(routes=[Mount(wsgi_app.path, app=WSGIMiddleware(wsgi_app))])

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
    run_log_server(port=args.port, log_dir=args.log_dir, remote_log_dir=args.remote_log_dir)


if __name__ == "__main__":
    main()
