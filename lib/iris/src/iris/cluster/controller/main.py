# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI for the Iris controller daemon.

The core logic lives in ``run_controller_serve`` so it can be called from both
the standalone ``python -m iris.cluster.controller.main serve`` entrypoint
(used by Dockerfile / GCP bootstrap / k8s) and the ``iris cluster controller
serve`` subcommand in the main CLI.
"""

import logging
import os
import signal
import tempfile
import threading
from pathlib import Path

import click

from iris.cluster.controller.auth import ControllerAuth, create_controller_auth
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.transitions import HEARTBEAT_FAILURE_THRESHOLD
from iris.rpc import config_pb2
from rigging.timing import Duration

logger = logging.getLogger(__name__)


LOCAL_STATE_DIR_DEFAULT = Path("/var/cache/iris/controller")
HOURLY_CHECKPOINT_SECONDS = 3600.0


def run_controller_serve(
    cluster_config: config_pb2.IrisClusterConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 10000,
    checkpoint_path: str | None = None,
    checkpoint_interval: float | None = None,
    dry_run: bool = False,
) -> None:
    """Start the Iris controller, block until SIGTERM/SIGINT.

    This is the shared implementation used by both the standalone daemon
    entrypoint and the ``iris cluster controller serve`` CLI command.
    """
    from iris.cluster.config import create_autoscaler, make_provider
    from iris.cluster.controller.autoscaler import Autoscaler
    from iris.cluster.controller.checkpoint import download_checkpoint_to_local
    from iris.cluster.controller.db import ControllerDB
    from iris.cluster.providers.factory import create_provider_bundle
    from iris.cluster.providers.k8s.tasks import K8sTaskProvider

    logger.info("Initializing Iris controller (git_hash=%s)", os.environ.get("IRIS_GIT_HASH", "unknown"))

    remote_state_dir = cluster_config.storage.remote_state_dir
    if not remote_state_dir:
        raise ValueError(
            "storage.remote_state_dir is required in the cluster config. "
            "Example: storage: { remote_state_dir: 'gs://my-bucket/iris/state' }"
        )
    logger.info("Using remote_state_dir from config: %s", remote_state_dir)

    if dry_run:
        _dry_run_tmpdir = tempfile.mkdtemp(prefix="iris-dry-run-")
        local_state_dir = Path(_dry_run_tmpdir)
        logger.info("Dry-run mode: using temporary local state dir %s", local_state_dir)
    elif cluster_config.storage.local_state_dir:
        local_state_dir = Path(cluster_config.storage.local_state_dir)
    else:
        local_state_dir = LOCAL_STATE_DIR_DEFAULT

    heartbeat_failure_threshold = cluster_config.controller.heartbeat_failure_threshold or HEARTBEAT_FAILURE_THRESHOLD

    # --- Restore or reuse local DB ---
    local_state_dir.mkdir(parents=True, exist_ok=True)
    db_dir = local_state_dir / "db"
    db_path = db_dir / ControllerDB.DB_FILENAME
    auth_db_path = db_dir / ControllerDB.AUTH_DB_FILENAME
    if db_path.exists() and auth_db_path.exists():
        logger.info("Local DB exists at %s, skipping remote restore", db_dir)
    else:
        if db_path.exists() and not auth_db_path.exists():
            logger.warning(
                "Main DB exists at %s but auth DB is missing — fetching from remote",
                db_path,
            )
        restored = download_checkpoint_to_local(remote_state_dir, db_dir, checkpoint_dir=checkpoint_path)
        if checkpoint_path and not restored:
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    db = ControllerDB(db_dir=db_dir)

    # --- Create provider ---
    provider = make_provider(cluster_config)
    logger.info("Provider created: %s", type(provider).__name__)

    # --- Create autoscaler (only for WorkerProvider; KubernetesProvider manages its own pods) ---
    # In dry-run mode the autoscaler is fully gated anyway, and creating the
    # provider bundle requires platform credentials (GCP SSH keys etc.) that
    # are unavailable on a local dev machine.
    autoscaler: Autoscaler | None = None
    base_worker_config = None
    if dry_run:
        logger.info("Dry-run mode: skipping autoscaler and provider bundle creation")
    elif not isinstance(provider, K8sTaskProvider):
        bundle = create_provider_bundle(
            platform_config=cluster_config.platform,
            cluster_config=cluster_config,
            ssh_config=cluster_config.defaults.ssh,
        )
        workers = bundle.workers
        logger.info("Provider bundle created")

        if cluster_config.defaults.worker.docker_image:
            base_worker_config = config_pb2.WorkerConfig()
            base_worker_config.CopyFrom(cluster_config.defaults.worker)
            if not base_worker_config.controller_address:
                base_worker_config.controller_address = bundle.controller.discover_controller(cluster_config.controller)
            base_worker_config.platform.CopyFrom(cluster_config.platform)

        autoscaler = create_autoscaler(
            platform=workers,
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
        autoscaler.restore_from_db(db, workers)
        logger.info("Autoscaler state restored from DB")

    # Workers need the resolved remote_state_dir to upload task artifacts (profiles).
    if base_worker_config is not None:
        base_worker_config.storage_prefix = remote_state_dir

    if checkpoint_interval is None:
        checkpoint_interval = HOURLY_CHECKPOINT_SECONDS
        logger.info("Defaulting to hourly checkpointing")

    logger.info("Configuration: host=%s port=%d remote_state_dir=%s", host, port, remote_state_dir)

    auth = create_controller_auth(cluster_config.auth, db=db) if cluster_config else ControllerAuth()
    if auth.worker_token and base_worker_config is not None:
        base_worker_config.auth_token = auth.worker_token

    config = ControllerConfig(
        host=host,
        port=port,
        remote_state_dir=remote_state_dir,
        heartbeat_failure_threshold=heartbeat_failure_threshold,
        checkpoint_interval=Duration.from_seconds(checkpoint_interval) if checkpoint_interval else None,
        local_state_dir=local_state_dir,
        auth_verifier=auth.verifier,
        auth_provider=auth.provider,
        auth=auth,
        dry_run=dry_run,
    )

    controller = Controller(
        config=config,
        provider=provider,
        autoscaler=autoscaler,
        db=db,
    )
    logger.info("Controller instance created")

    controller.start()
    logger.info("Controller started successfully on %s:%d", host, port)
    logger.info("Controller is ready to accept connections")

    stop_event = threading.Event()

    def handle_shutdown(_signum, _frame):
        # Second signal force-exits immediately.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Write a final checkpoint then exit. Do NOT call controller.stop()
        # here — its shutdown path runs autoscaler.shutdown() which terminates
        # every worker VM in the cluster. On a controller restart, workers must
        # survive; the new controller picks them up from the checkpoint. Even on
        # a full cluster teardown, `iris cluster stop` handles VM cleanup via
        # stop_all(), so the SIGTERM handler never needs to delete VMs itself.
        logger.info("Shutdown signal received")
        if not config.dry_run:
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
        logger.info("Controller exiting")
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    stop_event.wait()


# ---------------------------------------------------------------------------
# Standalone CLI — used by Dockerfile, GCP bootstrap, and k8s entrypoints
# (python -m iris.cluster.controller.main serve)
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Iris Controller - Cluster control plane."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option("--config", "config_file", type=click.Path(exists=True), required=True, help="Cluster config YAML")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option(
    "--checkpoint-path",
    default=None,
    help="Restore from this specific checkpoint directory (e.g. gs://bucket/.../controller-state/1234567890)",
)
@click.option(
    "--checkpoint-interval",
    default=None,
    type=float,
    help="Periodic checkpoint interval in seconds (default: no periodic checkpointing)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Start in dry-run mode: compute scheduling but suppress all side effects",
)
def serve(
    host: str,
    port: int,
    config_file: str,
    log_level: str,
    checkpoint_path: str | None,
    checkpoint_interval: float | None,
    dry_run: bool,
):
    """Start the Iris controller service."""
    from iris.cluster.config import load_config
    from rigging.log_setup import configure_logging

    configure_logging(level=getattr(logging, log_level))

    cluster_config = load_config(Path(config_file))
    logger.info("Cluster config loaded from %s: %d scale groups", config_file, len(cluster_config.scale_groups))

    run_controller_serve(
        cluster_config,
        host=host,
        port=port,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    cli()
