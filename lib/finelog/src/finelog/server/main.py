# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone finelog log server.

Hosts ``LogServiceImpl`` on a dedicated port. No auth, no stats, no JWT —
finelog ships pure log ingest + query. Wires only the
``ConcurrencyLimitInterceptor`` for ``FetchLogs``.

All flags also read from environment variables so the Docker image can be
configured without overriding ``CMD``.

Usage:
    python -m finelog.server.main --port 10001 --log-dir /var/cache/finelog --remote-log-dir gs://bucket/logs
    FINELOG_PORT=20001 python -m finelog.server.main
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import click
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


@click.command(context_settings={"show_default": True})
@click.option("--port", type=int, default=10001, envvar="FINELOG_PORT", help="Port to bind.")
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=Path("/var/cache/finelog"),
    envvar="FINELOG_LOG_DIR",
    help="Local log storage directory.",
)
@click.option(
    "--remote-log-dir",
    type=str,
    default="",
    envvar="FINELOG_REMOTE_DIR",
    help="Remote log storage URI (e.g. gs://bucket/path); empty disables offload.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    envvar="FINELOG_LOG_LEVEL",
    help="Log level.",
)
def main(port: int, log_dir: Path, remote_log_dir: str, log_level: str) -> None:
    configure_logging(level=getattr(logging, log_level))
    run_log_server(port=port, log_dir=log_dir, remote_log_dir=remote_log_dir)


if __name__ == "__main__":
    sys.exit(main())
