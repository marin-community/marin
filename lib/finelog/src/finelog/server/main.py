# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone finelog log server.

Hosts ``LogServiceImpl`` on a dedicated port. No auth, no stats, no JWT —
finelog ships pure log ingest + query. Wires only the
``ConcurrencyLimitInterceptor`` for ``FetchLogs``.

Usage:
    python -m finelog.server.main --port 10001 --log-dir /var/cache/finelog/logs --remote-log-dir gs://bucket/logs
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import uvicorn
from rigging.log_setup import configure_logging

from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl

logger = logging.getLogger("finelog.server")


def run_log_server(
    *,
    port: int,
    log_dir: Path,
    remote_log_dir: str,
) -> None:
    """Start a standalone log server, block until SIGTERM/SIGINT."""
    log_dir.mkdir(parents=True, exist_ok=True)

    service = LogServiceImpl(log_dir=log_dir, remote_log_dir=remote_log_dir)
    app = build_log_server_asgi(service)

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
    parser = argparse.ArgumentParser(description="Finelog log server")
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
    run_log_server(
        port=args.port,
        log_dir=args.log_dir,
        remote_log_dir=args.remote_log_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
