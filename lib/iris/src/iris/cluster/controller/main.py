# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Click-based CLI for the Iris controller daemon."""

import logging
import signal
import sys
import threading
from pathlib import Path

import click

from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the controller.

    Sets up a consistent log format with timestamps and component names
    that makes it easy to grep and analyze logs.
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicate logs
    root.handlers.clear()
    root.addHandler(handler)


@click.group()
def cli():
    """Iris Controller - Cluster control plane."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option(
    "--bundle-prefix", default=None, help="URI prefix for job bundles (e.g., gs://bucket/path or file:///path)"
)
@click.option("--scheduler-interval", default=0.5, type=float, help="Scheduler loop interval (seconds)")
@click.option("--worker-timeout", default=60.0, type=float, help="Worker heartbeat timeout (seconds)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config for autoscaling")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def serve(
    host: str,
    port: int,
    bundle_prefix: str | None,
    scheduler_interval: float,
    worker_timeout: float,
    config_file: str | None,
    log_level: str,
):
    """Start the Iris controller service.

    When --config is provided, the controller runs an integrated autoscaler
    that provisions/terminates VM slices based on pending task demand.
    """
    from iris.cluster.vm.autoscaler import Autoscaler
    from iris.cluster.vm.config import create_autoscaler_from_config, load_config

    configure_logging(level=getattr(logging, log_level))

    logger.info("Initializing Iris controller")

    # Load cluster config first to extract bundle_prefix if not provided via CLI
    autoscaler: Autoscaler | None = None
    if config_file:
        logger.info("Loading cluster config from %s", config_file)
        try:
            cluster_config = load_config(Path(config_file))
            logger.info("Cluster config loaded: %d scale groups defined", len(cluster_config.scale_groups))
        except Exception as e:
            logger.exception("Failed to load cluster config from %s", config_file)
            raise click.ClickException(f"Failed to load cluster config: {e}") from e

        # Extract bundle_prefix from config if not provided via CLI
        if bundle_prefix is None and cluster_config.controller_vm.bundle_prefix:
            bundle_prefix = cluster_config.controller_vm.bundle_prefix
            logger.info("Using bundle_prefix from config: %s", bundle_prefix)

        try:
            autoscaler = create_autoscaler_from_config(cluster_config)
            logger.info("Autoscaler created with %d scale groups", len(autoscaler.groups))
        except Exception as e:
            logger.exception("Failed to create autoscaler from config")
            raise click.ClickException(f"Failed to create autoscaler: {e}") from e

        try:
            autoscaler.reconcile()
            logger.info("Autoscaler initial reconcile completed")
        except Exception as e:
            logger.exception("Autoscaler initial reconcile failed")
            raise click.ClickException(f"Autoscaler reconcile failed: {e}") from e
    else:
        logger.info("No cluster config provided, autoscaler disabled")

    logger.info("Configuration: host=%s port=%d bundle_prefix=%s", host, port, bundle_prefix)
    logger.info("Configuration: scheduler_interval=%.2fs worker_timeout=%.2fs", scheduler_interval, worker_timeout)

    config = ControllerConfig(
        host=host,
        port=port,
        bundle_prefix=bundle_prefix,
        scheduler_interval_seconds=scheduler_interval,
        worker_timeout_seconds=worker_timeout,
    )

    try:
        controller = Controller(
            config=config,
            worker_stub_factory=RpcWorkerStubFactory(),
            autoscaler=autoscaler,
        )
        logger.info("Controller instance created")
    except Exception as e:
        logger.exception("Failed to create controller")
        raise click.ClickException(f"Failed to create controller: {e}") from e

    try:
        controller.start()
        logger.info("Controller started successfully on %s:%d", host, port)
    except Exception as e:
        logger.exception("Failed to start controller")
        raise click.ClickException(f"Failed to start controller: {e}") from e

    logger.info("Controller is ready to accept connections")

    stop_event = threading.Event()

    def handle_shutdown(_signum, _frame):
        logger.info("Shutdown signal received, stopping controller...")
        controller.stop()
        logger.info("Controller stopped")
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    stop_event.wait()


if __name__ == "__main__":
    cli()
