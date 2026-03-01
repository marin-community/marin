# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-based CLI for the Iris controller daemon."""

import logging
import os
import signal
import threading
from pathlib import Path

import click

from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.controller.state import HEARTBEAT_FAILURE_THRESHOLD
from iris.logging import configure_logging
from iris.marin_fs import marin_temp_bucket
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


def default_bundle_prefix() -> str:
    """Return a region-local temp bucket path for bundle storage.

    Uses the same marin_temp_bucket API that log_prefix uses, with a 7-day TTL
    since bundles are ephemeral and regenerated on each job submission.
    """
    return marin_temp_bucket(ttl_days=7, prefix="iris/bundles")


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
    from iris.rpc import config_pb2

    configure_logging(level=getattr(logging, log_level))

    logger.info("Initializing Iris controller (git_hash=%s)", os.environ.get("IRIS_GIT_HASH", "unknown"))

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
        if bundle_prefix is None and cluster_config.storage.bundle_prefix:
            bundle_prefix = cluster_config.storage.bundle_prefix
            logger.info("Using bundle_prefix from config: %s", bundle_prefix)

        try:
            platform = create_platform(
                platform_config=cluster_config.platform,
                ssh_config=cluster_config.defaults.ssh,
            )
            logger.info("Platform created")

            base_worker_config = None
            if cluster_config.defaults.worker.docker_image:
                base_worker_config = config_pb2.WorkerConfig()
                base_worker_config.CopyFrom(cluster_config.defaults.worker)
                if not base_worker_config.controller_address:
                    base_worker_config.controller_address = platform.discover_controller(cluster_config.controller)
                if cluster_config.storage.log_prefix:
                    base_worker_config.log_prefix = cluster_config.storage.log_prefix
                base_worker_config.platform.CopyFrom(cluster_config.platform)

            autoscaler = create_autoscaler(
                platform=platform,
                autoscaler_config=cluster_config.defaults.autoscaler,
                scale_groups=cluster_config.scale_groups,
                label_prefix=cluster_config.platform.label_prefix or "iris",
                base_worker_config=base_worker_config,
            )
            logger.info("Autoscaler created with %d scale groups", len(autoscaler.groups))
        except Exception as e:
            logger.exception("Failed to create autoscaler from config")
            raise click.ClickException(f"Failed to create autoscaler: {e}") from e
    else:
        logger.info("No cluster config provided, autoscaler disabled")

    heartbeat_failure_threshold = (
        cluster_config.controller.heartbeat_failure_threshold if cluster_config else HEARTBEAT_FAILURE_THRESHOLD
    )

    if bundle_prefix is None:
        bundle_prefix = default_bundle_prefix()
        logger.info("Using auto-detected bundle_prefix: %s", bundle_prefix)

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
