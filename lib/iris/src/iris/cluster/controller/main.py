# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-based CLI for the Iris controller daemon."""

import logging
import signal
import threading
from pathlib import Path

import click

from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.controller.state import HEARTBEAT_FAILURE_THRESHOLD
from iris.logging import configure_logging
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


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
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config for autoscaling")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def serve(
    host: str,
    port: int,
    bundle_prefix: str | None,
    scheduler_interval: float,
    config_file: str | None,
    log_level: str,
):
    """Start the Iris controller service.

    When --config is provided, the controller runs an integrated autoscaler
    that provisions/terminates VM slices based on pending task demand.
    """
    from iris.cluster.controller.autoscaler import Autoscaler
    from iris.cluster.config import load_config, create_autoscaler
    from iris.cluster.platform.factory import create_platform

    configure_logging(level=getattr(logging, log_level))

    logger.info("Initializing Iris controller")

    # Load cluster config first to extract bundle_prefix if not provided via CLI
    autoscaler: Autoscaler | None = None
    cluster_config = None
    if config_file:
        logger.info("Loading cluster config from %s", config_file)
        try:
            cluster_config = load_config(Path(config_file))
            logger.info("Cluster config loaded: %d scale groups defined", len(cluster_config.scale_groups))
        except Exception as e:
            logger.exception("Failed to load cluster config from %s", config_file)
            raise click.ClickException(f"Failed to load cluster config: {e}") from e

        # Extract bundle_prefix from config if not provided via CLI
        if bundle_prefix is None and cluster_config.controller.bundle_prefix:
            bundle_prefix = cluster_config.controller.bundle_prefix
            logger.info("Using bundle_prefix from config: %s", bundle_prefix)

        try:
            platform = create_platform(
                platform_config=cluster_config.platform,
                ssh_config=cluster_config.defaults.ssh,
            )
            logger.info("Platform created")

            # Pass cluster_config through to platform.create_slice() for bootstrap.
            # If no docker_image is configured, pass None to disable bootstrap.
            bootstrap_cluster_config = cluster_config if cluster_config.defaults.bootstrap.docker_image else None

            autoscaler = create_autoscaler(
                platform=platform,
                autoscaler_config=cluster_config.defaults.autoscaler,
                scale_groups=cluster_config.scale_groups,
                label_prefix=cluster_config.platform.label_prefix or "iris",
                cluster_config=bootstrap_cluster_config,
            )
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

    heartbeat_failure_threshold = (
        cluster_config.controller.heartbeat_failure_threshold if cluster_config else HEARTBEAT_FAILURE_THRESHOLD
    )

    logger.info("Configuration: host=%s port=%d bundle_prefix=%s", host, port, bundle_prefix)
    logger.info("Configuration: scheduler_interval=%.2fs", scheduler_interval)

    config = ControllerConfig(
        host=host,
        port=port,
        bundle_prefix=bundle_prefix,
        scheduler_interval=Duration.from_seconds(scheduler_interval),
        heartbeat_failure_threshold=heartbeat_failure_threshold,
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
