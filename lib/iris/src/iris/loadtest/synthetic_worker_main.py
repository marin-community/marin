# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entrypoint for a single synthetic worker.

The parent harness spawns one of these per worker via ``python -m
iris.loadtest.synthetic_worker_main`` so each uvicorn server runs in its own
OS process — giving real GIL/CPU isolation from the Controller under test.

Protocol with the parent:
  1. Parent spawns with ``stdout=PIPE`` and ``start_new_session=True``.
  2. Child binds a free ephemeral port, starts uvicorn, waits for
     ``server.started``, then prints exactly one line:

         READY port=<int>\\n

     and flushes stdout. This is the parent's rendezvous signal.
  3. Child runs uvicorn on the main thread + the lifecycle advance loop on a
     daemon thread. On SIGTERM/SIGINT (or SIGKILL from the parent on abrupt
     preemption) it exits — 0 for SIGTERM, whatever the kernel gives on SIGKILL.

Registration against the controller DB stays in the parent; the child never
opens the controller sqlite.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

from iris.cluster.types import WorkerId
from iris.rpc.worker_connect import WorkerServiceWSGIApplication

from iris.loadtest.synthetic_worker import (
    LifecycleDelays,
    _SyntheticTaskProvider,
    _WorkerServiceAdapter,
)

logger = logging.getLogger(__name__)


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Bind-and-release to discover an ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic worker subprocess")
    p.add_argument("--worker-id", required=True)
    p.add_argument("--slice-id", required=True)
    p.add_argument("--scale-group", required=True)
    p.add_argument("--zone", required=True)
    p.add_argument("--device-variant", required=True)
    p.add_argument("--tpu-worker-id", type=int, default=0)
    p.add_argument("--building-seconds", type=float, required=True)
    p.add_argument("--running-seconds", type=float, required=True)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    delays = LifecycleDelays(
        building_seconds=args.building_seconds,
        running_seconds=args.running_seconds,
    )
    provider = _SyntheticTaskProvider(WorkerId(args.worker_id), delays)
    adapter = _WorkerServiceAdapter(provider)

    port = _find_free_port(args.host)
    rpc_wsgi_app = WorkerServiceWSGIApplication(service=adapter)
    rpc_app = WSGIMiddleware(rpc_wsgi_app)
    app = Starlette(routes=[Mount(rpc_wsgi_app.path, app=rpc_app)])

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=port,
            log_level="error",
            log_config=None,
            timeout_keep_alive=30,
            loop="asyncio",
            lifespan="off",
        )
    )

    stop = threading.Event()

    def _handle_signal(_signum: int, _frame) -> None:
        stop.set()
        server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _wait_ready_then_announce() -> None:
        # Poll server.started — uvicorn sets it once the socket is bound and
        # the accept loop is live. Then emit the single READY line the parent
        # is blocked on.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not getattr(server, "started", False):
            if stop.is_set():
                return
            time.sleep(0.01)
        sys.stdout.write(f"READY port={port}\n")
        sys.stdout.flush()

    def _run_lifecycle() -> None:
        cadence = max(0.1, min(delays.building_seconds, delays.running_seconds) / 4.0)
        while not stop.is_set():
            try:
                provider.advance(delays)
            except Exception:
                logger.exception("synthetic worker %s lifecycle advance failed", args.worker_id)
            if stop.wait(cadence):
                return

    threading.Thread(target=_wait_ready_then_announce, name="synth-ready", daemon=True).start()
    threading.Thread(target=_run_lifecycle, name="synth-life", daemon=True).start()

    # Runs in the main thread; returns when should_exit is set.
    server.run()
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
