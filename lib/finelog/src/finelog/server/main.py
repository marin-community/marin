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
import threading
from pathlib import Path

import click
import pyarrow as pa
import uvicorn
from rigging.log_setup import configure_logging

from finelog.server.asgi import build_log_server_asgi
from finelog.server.service import LogServiceImpl
from finelog.server.stats_service import StatsServiceImpl
from finelog.store.layout_migration import migrate_to_namespaced_layout

logger = logging.getLogger("finelog.server")

# Period for the pool/RSS diagnostics line.
_POOL_DIAGNOSTICS_INTERVAL_SEC = 60.0


def _read_proc_self_status_kb(field: str) -> int:
    """Return the value of a ``/proc/self/status`` field in KiB (Linux-only).

    Returns 0 when the field is missing — caller treats 0 as "unknown".
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _emit_pool_diagnostics(service: LogServiceImpl) -> None:
    """Log a snapshot of the arrow pool, process RSS, and per-namespace ram.

    Lets us tell apart the two leak hypotheses we care about:
      * RSS tracks ``pool.bytes_allocated`` → live pyarrow buffers (e.g.
        IPC retention, ARROW-7305 territory).
      * RSS >> ``pool.bytes_allocated`` and grows monotonically →
        allocator arena retention (ARROW-6910 territory).
    """
    pool = pa.default_memory_pool()
    summary = service.log_store.memory_summary()
    rss_kb = _read_proc_self_status_kb("VmRSS")
    vmsize_kb = _read_proc_self_status_kb("VmSize")
    logger.info(
        "pool_diag backend=%s pool_bytes=%d pool_max=%d rss_kb=%d vmsize_kb=%d " "namespaces=%d ram_bytes=%d chunks=%d",
        pool.backend_name,
        pool.bytes_allocated(),
        pool.max_memory(),
        rss_kb,
        vmsize_kb,
        summary["namespaces"],
        summary["ram_bytes"],
        summary["chunks"],
    )


def _start_pool_diagnostics(service: LogServiceImpl, stop_event: threading.Event) -> threading.Thread:
    """Spawn a daemon thread that calls ``_emit_pool_diagnostics`` periodically."""

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                _emit_pool_diagnostics(service)
            except Exception:
                logger.warning("pool_diag failed", exc_info=True)
            stop_event.wait(_POOL_DIAGNOSTICS_INTERVAL_SEC)

    thread = threading.Thread(target=_loop, name="finelog-pool-diag", daemon=True)
    thread.start()
    return thread


def run_log_server(
    *,
    port: int,
    log_dir: Path,
    remote_log_dir: str,
) -> None:
    """Start a standalone log server, block until SIGTERM/SIGINT."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Synchronous migration before the store is constructed and before
    # uvicorn binds the listening socket. A multi-minute walk just delays the
    # accept(); clients see "connection refused" or block on connect — same
    # observable behavior as a slow boot.
    migrate_to_namespaced_layout(log_dir)

    service = LogServiceImpl(log_dir=log_dir, remote_log_dir=remote_log_dir)
    stats_service = StatsServiceImpl(log_store=service.log_store)
    app = build_log_server_asgi(service, stats_service=stats_service)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        log_config=None,
        timeout_keep_alive=120,
    )
    server = uvicorn.Server(config)

    diag_stop = threading.Event()

    def _shutdown(_signum, _frame):
        logger.info("Log server shutting down")
        server.should_exit = True
        diag_stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _start_pool_diagnostics(service, diag_stop)

    logger.info("Log server starting on port %d (log_dir=%s)", port, log_dir)
    server.run()

    diag_stop.set()
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
    help="Remote log storage URI (e.g. gs://bucket/path); empty disables remote copy.",
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
