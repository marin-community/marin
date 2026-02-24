# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-based CLI for the Iris worker daemon."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import click

from iris.cluster.config import load_config
from iris.cluster.platform.bootstrap import zone_to_multi_region
from iris.cluster.platform.factory import create_platform
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.worker.env_probe import detect_gcp_zone
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.logging import configure_logging


def _load_task_default_env() -> dict[str, str]:
    """Load default task environment injected by bootstrap."""
    raw = os.environ.get("IRIS_TASK_DEFAULT_ENV_JSON", "")
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("IRIS_TASK_DEFAULT_ENV_JSON must decode to a dictionary")
    return {str(k): str(v) for k, v in parsed.items()}


def _configure_docker_ar_auth() -> None:
    """Configure Docker to authenticate with Artifact Registry for the worker's continent.

    On GCP, task containers are pulled by the Docker CLI inside the worker
    container (via the host docker socket). The host's bootstrap configured
    gcloud auth for its own docker config, but the worker container has its own
    config. This runs ``gcloud auth configure-docker`` inside the worker
    container so that ``docker create`` can pull AR images for tasks.
    """
    zone = detect_gcp_zone()
    if not zone:
        return

    multi_region = zone_to_multi_region(zone)
    if not multi_region:
        return

    ar_host = f"{multi_region}-docker.pkg.dev"
    logger = logging.getLogger(__name__)
    logger.info("Configuring Docker auth for %s", ar_host)
    result = subprocess.run(
        ["gcloud", "auth", "configure-docker", ar_host, "-q"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        logger.warning("gcloud auth configure-docker failed: %s", result.stderr)
    else:
        logger.info("Docker AR auth configured for %s", ar_host)


def _load_worker_attributes() -> dict[str, str]:
    """Parse IRIS_WORKER_ATTRIBUTES JSON into a map."""
    raw = os.environ.get("IRIS_WORKER_ATTRIBUTES", "")
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("IRIS_WORKER_ATTRIBUTES must decode to a dictionary")
    return {str(k): str(v) for k, v in parsed.items()}


@click.group()
def cli():
    """Iris Worker - Job execution daemon."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--cache-dir", required=True, help="Cache directory (must be a host-visible path for Docker mounts)")
@click.option("--port-range", default="30000-40000", help="Port range for job ports (start-end)")
@click.option("--worker-id", default=None, help="Worker ID (auto-generated if not provided)")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    required=False,
    help="Cluster config for platform-based controller discovery",
)
@click.option("--controller-address", default=None, help="Controller address host:port (overrides --config discovery)")
@click.option(
    "--runtime",
    type=click.Choice(["docker", "kubernetes"]),
    default="docker",
    help=("Container runtime backend " "(docker for GCP/Manual, " "kubernetes for Pod-per-task execution on CoreWeave)"),
)
def serve(
    host: str,
    port: int,
    cache_dir: str,
    port_range: str,
    worker_id: str | None,
    config_file: str | None,
    controller_address: str | None,
    runtime: str,
):
    """Start the Iris worker service."""
    configure_logging(level=logging.INFO)
    logging.getLogger(__name__).info("Iris worker starting (git_hash=%s)", os.environ.get("IRIS_GIT_HASH", "unknown"))

    log_prefix = None
    default_task_image = None
    gcp_project = ""
    cluster_config = None
    if config_file:
        cluster_config = load_config(Path(config_file))
        log_prefix = cluster_config.storage.log_prefix or None
        default_task_image = cluster_config.defaults.default_task_image or None
        if cluster_config.platform.HasField("gcp"):
            gcp_project = cluster_config.platform.gcp.project_id

    if controller_address:
        resolved_controller_address = f"http://{controller_address}"
    elif cluster_config:
        platform = create_platform(
            platform_config=cluster_config.platform,
            ssh_config=cluster_config.defaults.ssh,
        )
        resolved_controller_address = f"http://{platform.discover_controller(cluster_config.controller)}"
    else:
        raise click.ClickException("Either --controller-address or --config must be provided")

    port_start, port_end = map(int, port_range.split("-"))

    if runtime == "kubernetes":
        container_runtime = KubernetesRuntime()
    else:
        _configure_docker_ar_auth()
        container_runtime = DockerRuntime()

    config = WorkerConfig(
        host=host,
        port=port,
        cache_dir=Path(cache_dir).expanduser(),
        port_range=(port_start, port_end),
        controller_address=resolved_controller_address,
        worker_id=worker_id,
        worker_attributes=_load_worker_attributes(),
        default_task_env=_load_task_default_env(),
        default_task_image=default_task_image,
        gcp_project=gcp_project,
        log_prefix=log_prefix,
    )

    worker = Worker(config, container_runtime=container_runtime)

    click.echo(f"Starting Iris worker on {host}:{port}")
    click.echo(f"  Cache dir: {config.cache_dir}")
    click.echo(f"  Controller: {resolved_controller_address}")
    click.echo(f"  Runtime: {runtime}")
    worker.start()
    worker.wait()  # Block until worker is stopped


@cli.command()
@click.option("--cache-dir", required=True, help="Cache directory")
def cleanup(cache_dir: str):
    """Clean up cached bundles, venvs, and images."""
    cache_path = Path(cache_dir).expanduser()
    if cache_path.exists():
        shutil.rmtree(cache_path)
        click.echo(f"Removed cache directory: {cache_path}")
    else:
        click.echo(f"Cache directory does not exist: {cache_path}")


if __name__ == "__main__":
    cli()
