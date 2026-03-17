# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-based CLI for the Iris controller daemon."""

import logging
import os
import signal
import threading
from pathlib import Path

import click

from iris.cluster.controller.auth import ControllerAuth, create_controller_auth
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.controller.transitions import HEARTBEAT_FAILURE_THRESHOLD
from iris.logging import configure_logging
from iris.rpc import config_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


LOCAL_STATE_DIR_DEFAULT = Path("/var/cache/iris/controller")
HOURLY_CHECKPOINT_SECONDS = 3600.0


@click.group()
def cli():
    """Iris Controller - Cluster control plane."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option("--scheduler-interval", default=0.5, type=float, help="Scheduler loop interval (seconds)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config (required)")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option(
    "--checkpoint-path",
    default=None,
    help="Restore from this specific checkpoint DB copy instead of latest.sqlite3",
)
@click.option(
    "--checkpoint-interval",
    default=None,
    type=float,
    help="Periodic checkpoint interval in seconds (default: no periodic checkpointing)",
)
def serve(
    host: str,
    port: int,
    scheduler_interval: float,
    config_file: str | None,
    log_level: str,
    checkpoint_path: str | None,
    checkpoint_interval: float | None,
):
    """Start the Iris controller service.

    When --config is provided, the controller runs an integrated autoscaler
    that provisions/terminates VM slices based on pending task demand.
    """
    from iris.cluster.controller.autoscaler import Autoscaler
    from iris.cluster.controller.checkpoint import download_checkpoint_to_local
    from iris.cluster.controller.db import ControllerDB
    from iris.cluster.config import load_config, create_autoscaler
    from iris.cluster.platform.factory import create_platform

    configure_logging(level=getattr(logging, log_level))

    logger.info("Initializing Iris controller (git_hash=%s)", os.environ.get("IRIS_GIT_HASH", "unknown"))

    if not config_file:
        raise click.ClickException("--config is required. Provide a cluster config YAML with storage.remote_state_dir.")

    # --- Load cluster config ---
    logger.info("Loading cluster config from %s", config_file)
    try:
        cluster_config = load_config(Path(config_file))
        logger.info("Cluster config loaded: %d scale groups defined", len(cluster_config.scale_groups))
    except Exception as e:
        logger.exception("Failed to load cluster config from %s", config_file)
        raise click.ClickException(f"Failed to load cluster config: {e}") from e

    remote_state_dir = cluster_config.storage.remote_state_dir
    if not remote_state_dir:
        raise click.ClickException(
            "storage.remote_state_dir is required in the cluster config. "
            "Example: storage: { remote_state_dir: 'gs://my-bucket/iris/state' }"
        )
    logger.info("Using remote_state_dir from config: %s", remote_state_dir)

    local_state_dir = (
        Path(cluster_config.storage.local_state_dir)
        if cluster_config.storage.local_state_dir
        else LOCAL_STATE_DIR_DEFAULT
    )

    heartbeat_failure_threshold = cluster_config.controller.heartbeat_failure_threshold or HEARTBEAT_FAILURE_THRESHOLD

    # --- Restore or reuse local DB ---
    local_state_dir.mkdir(parents=True, exist_ok=True)
    db_path = local_state_dir / "controller.sqlite3"
    if db_path.exists():
        logger.info("Local DB exists at %s, skipping remote restore", db_path)
    else:
        restored = download_checkpoint_to_local(remote_state_dir, db_path, checkpoint_path)
        if checkpoint_path and not restored:
            raise click.ClickException(f"Checkpoint DB not found: {checkpoint_path}")

    db = ControllerDB(db_path=db_path)

    # --- Create autoscaler (needs db) ---
    autoscaler: Autoscaler | None = None
    base_worker_config = None
    try:
        platform = create_platform(
            platform_config=cluster_config.platform,
            ssh_config=cluster_config.defaults.ssh,
        )
        logger.info("Platform created")

        if cluster_config.defaults.worker.docker_image:
            base_worker_config = config_pb2.WorkerConfig()
            base_worker_config.CopyFrom(cluster_config.defaults.worker)
            if not base_worker_config.controller_address:
                base_worker_config.controller_address = platform.discover_controller(cluster_config.controller)
            base_worker_config.platform.CopyFrom(cluster_config.platform)

        autoscaler = create_autoscaler(
            platform=platform,
            autoscaler_config=cluster_config.defaults.autoscaler,
            scale_groups=cluster_config.scale_groups,
            label_prefix=cluster_config.platform.label_prefix or "iris",
            base_worker_config=base_worker_config,
            db=db,
        )
        logger.info("Autoscaler created with %d scale groups", len(autoscaler.groups))

        # Restore autoscaler state (tracked slices/workers/backoff) from the DB
        # so restarted controllers don't lose cloud resource tracking and
        # scale up duplicates.
        autoscaler.restore_from_db(db, platform)
        logger.info("Autoscaler state restored from DB")
    except Exception as e:
        logger.exception("Failed to create autoscaler from config")
        raise click.ClickException(f"Failed to create autoscaler: {e}") from e

    # Workers need the resolved remote_state_dir to upload task artifacts (profiles).
    if base_worker_config is not None:
        base_worker_config.storage_prefix = remote_state_dir

    if checkpoint_interval is None:
        checkpoint_interval = HOURLY_CHECKPOINT_SECONDS
        logger.info("Defaulting to hourly checkpointing")

    logger.info("Configuration: host=%s port=%d remote_state_dir=%s", host, port, remote_state_dir)
    logger.info("Configuration: scheduler_interval=%.2fs", scheduler_interval)

    auth = create_controller_auth(cluster_config.auth, db=db) if cluster_config else ControllerAuth()
    if auth.worker_token and base_worker_config is not None:
        base_worker_config.auth_token = auth.worker_token

    config = ControllerConfig(
        host=host,
        port=port,
        remote_state_dir=remote_state_dir,
        scheduler_interval=Duration.from_seconds(scheduler_interval),
        heartbeat_failure_threshold=heartbeat_failure_threshold,
        checkpoint_interval=Duration.from_seconds(checkpoint_interval) if checkpoint_interval else None,
        local_state_dir=local_state_dir,
        auth_verifier=auth.verifier,
        auth_provider=auth.provider,
        auth=auth,
    )

    try:
        controller = Controller(
            config=config,
            worker_stub_factory=RpcWorkerStubFactory(),
            autoscaler=autoscaler,
            db=db,
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
        logger.info("Shutdown signal received, writing final checkpoint...")
        try:
            path, result = controller.begin_checkpoint()
            logger.info(
                "Final checkpoint written: %s (jobs=%d tasks=%d workers=%d)",
                path,
                result.job_count,
                result.task_count,
                result.worker_count,
            )
        except Exception:
            logger.exception("Final checkpoint on shutdown failed")
        logger.info("Stopping controller...")
        controller.stop()
        logger.info("Controller stopped")
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    stop_event.wait()


if __name__ == "__main__":
    cli()
