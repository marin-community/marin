# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for the silo broker: one long-lived Iris job.

    python -m silo.broker.main --endpoint-name /muchanem/silo-broker [--state-url s3://.../broker-state.json]

Required environment:
    SILO_API_TOKEN     workers present this (the DAYTONA_API_KEY analogue)
    SILO_HOST_SECRET   shared with hosts only
Optional:
    SILO_BROKER_STATE_URL   where snapshots and the host registry persist (same as --state-url)
    SILO_HEARTBEAT_SECONDS, SILO_HEARTBEAT_TIMEOUT_SECONDS, SILO_HOST_SUSPECT_AFTER_SECONDS,
    SILO_HOST_DEAD_AFTER_SECONDS, SILO_HOST_MAX_INFLIGHT_CREATES, SILO_HOST_QUARANTINE_*
                            liveness and placement safety (silo.broker.core.BrokerConfig)
    SILO_BROKER_CONTROL_THREADS, SILO_BROKER_HOST_IO_THREADS, SILO_BROKER_CREATE_THREADS
                            thread-pool sizes (silo.broker.server.PoolSizes)

The broker registers itself in the Iris endpoint registry. Hosts find it by that
name, and workers can re-find it by that name through the Iris controller's
proxy (``/whoami``) if the broker restarts somewhere else -- so a restart does
not strand every worker holding the old address.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading

import anyio
import uvicorn
from iris.client.client import iris_ctx

from silo.broker.core import Broker, BrokerConfig
from silo.broker.server import PoolSizes, build_app, host_client_factory
from silo.broker.state import StatePersister, open_state_store, restore

logger = logging.getLogger("silo.broker")

MAINTENANCE_SECONDS = 5.0


def _maintain(broker: Broker, persister: StatePersister | None, stop: threading.Event) -> None:
    """Liveness transitions and state saves, off every request path."""
    save_failing = False
    while not stop.wait(MAINTENANCE_SECONDS):
        try:
            broker.sweep()
        except Exception:
            logger.exception("host health sweep failed")
        if persister is None:
            continue
        try:
            persister.save_if_changed()
            if save_failing:
                logger.warning("broker state saved again")
            save_failing = False
        except Exception:
            # Loud every time: a broker that cannot persist loses its snapshots on restart.
            logger.exception("could NOT save broker state; a restart now would lose snapshots")
            save_failing = True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint-name", required=True, help="absolute Iris endpoint name, e.g. /me/silo-broker")
    parser.add_argument("--port", type=int, default=0, help="0 = kernel-assigned (hosts share node networking)")
    parser.add_argument(
        "--state-url",
        default=os.environ.get("SILO_BROKER_STATE_URL"),
        help="persist snapshots + host registry here (local path or fsspec URL); unset = in memory only",
    )
    args = parser.parse_args(argv)
    if not args.endpoint_name.startswith("/"):
        parser.error("--endpoint-name must be absolute so it is not re-prefixed by the job namespace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    api_token = os.environ["SILO_API_TOKEN"]
    host_secret = os.environ["SILO_HOST_SECRET"]

    config, warnings = BrokerConfig.from_env()
    for warning in warnings:
        logger.warning("config: %s", warning)
    pool_sizes = PoolSizes.from_env()
    logger.info("broker config: %s", config.describe())
    logger.info("broker thread pools: %s", pool_sizes)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    advertise = os.environ.get("IRIS_ADVERTISE_HOST") or socket.gethostbyname(socket.gethostname())
    self_url = f"http://{advertise}:{sock.getsockname()[1]}"

    broker = Broker(host_secret=host_secret, host_client_factory=host_client_factory(host_secret), config=config)
    # Every start may be a restart: until hosts have re-reported their sandboxes,
    # an unknown sandbox id is "retry", not "not found".
    broker.begin_warmup(config.suspect_after_seconds)

    persister = None
    store = open_state_store(args.state_url)
    if store is None:
        logger.warning("no --state-url / SILO_BROKER_STATE_URL: snapshots live in memory only and a restart loses them")
    else:
        writable, _ = restore(broker, store)
        persister = StatePersister(broker, store, writable=writable)
        logger.info("broker state url=%s writable=%s", store.url, writable)

    stop = threading.Event()
    threading.Thread(target=_maintain, args=(broker, persister, stop), name="silo-maintain", daemon=True).start()

    app = build_app(
        broker, api_token=api_token, host_secret=host_secret, self_url=lambda: self_url, pool_sizes=pool_sizes
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="none"))

    try:
        with iris_ctx().registry.registered(args.endpoint_name, self_url, {"service": "silo-broker"}):
            logger.info("broker serving at %s, registered as %s", self_url, args.endpoint_name)
            anyio.run(server.serve, [sock])
    finally:
        stop.set()
        if persister is not None:
            try:
                persister.save_if_changed()
            except Exception:
                logger.exception("final broker state save failed")


if __name__ == "__main__":
    main()
