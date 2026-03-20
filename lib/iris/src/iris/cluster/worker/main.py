# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-based CLI for the Iris worker daemon."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import click
from google.protobuf.json_format import ParseDict

from iris.cluster.platform.factory import create_platform
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.worker.env_probe import detect_gcp_zone
from iris.cluster.worker.worker import Worker, worker_config_from_proto
from iris.logging import configure_logging
from iris.rpc import config_pb2


def _configure_docker_ar_auth(ar_host: str) -> None:
    """Configure Docker to authenticate with the given Artifact Registry host."""
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


@click.group()
def cli():
    """Iris Worker - Job execution daemon."""
    pass


@cli.command()
<<<<<<< Updated upstream
@click.option("--worker-config", type=click.Path(exists=True), required=True, help="Path to WorkerConfig JSON file")
def serve(worker_config: str):
||||||| constructed merge base
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--cache-dir", required=True, help="Cache directory (must be a host-visible path for Docker mounts)")
@click.option("--port-range", default="30000-40000", help="Port range for job ports (start-end)")
@click.option(
    "--controller-address", default=None, help="Controller URL for auto-registration (e.g., http://controller:8080)"
)
@click.option("--worker-id", default=None, help="Worker ID (auto-generated if not provided)")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Cluster config for platform-based controller discovery",
)
def serve(
    host: str,
    port: int,
    cache_dir: str,
    port_range: str,
    controller_address: str | None,
    worker_id: str | None,
    config_file: str | None,
):
=======
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--cache-dir", required=True, help="Cache directory (must be a host-visible path for Docker mounts)")
@click.option("--port-range", default="30000-40000", help="Port range for job ports (start-end)")
@click.option("--worker-id", default=None, help="Worker ID (auto-generated if not provided)")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    required=True,
    help="Cluster config for platform-based controller discovery",
)
def serve(
    host: str,
    port: int,
    cache_dir: str,
    port_range: str,
    worker_id: str | None,
    config_file: str,
):
>>>>>>> Stashed changes
    """Start the Iris worker service."""
    configure_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Iris worker starting (git_hash=%s)", os.environ.get("IRIS_GIT_HASH", "unknown"))

<<<<<<< Updated upstream
    with open(worker_config) as f:
        wc_proto = ParseDict(json.load(f), config_pb2.WorkerConfig())

    platform = create_platform(platform_config=wc_proto.platform, ssh_config=config_pb2.SshConfig())
    zone = detect_gcp_zone()

    def resolve_image(image: str) -> str:
        return platform.resolve_image(image, zone=zone)

    if wc_proto.default_task_image:
        resolved = resolve_image(wc_proto.default_task_image)
        if resolved != wc_proto.default_task_image and "-docker.pkg.dev/" in resolved:
            _configure_docker_ar_auth(resolved.split("/")[0])

    config = worker_config_from_proto(wc_proto, resolve_image=resolve_image)

    container_runtime = DockerRuntime(cache_dir=config.cache_dir)
||||||| constructed merge base
    if config_file and not controller_address:
        from iris.cluster.config import load_config
        from iris.cluster.platform.factory import create_platform

        cluster_config = load_config(Path(config_file))
        platform = create_platform(
            platform_config=cluster_config.platform,
            ssh_config=cluster_config.defaults.ssh,
        )
        controller_address = f"http://{platform.discover_controller(cluster_config.controller)}"

    port_start, port_end = map(int, port_range.split("-"))

    config = WorkerConfig(
        host=host,
        port=port,
        cache_dir=Path(cache_dir).expanduser(),
        port_range=(port_start, port_end),
        controller_address=controller_address,
        worker_id=worker_id,
    )
=======
    from iris.cluster.config import load_config
    from iris.cluster.platform.factory import create_platform

    cluster_config = load_config(Path(config_file))
    platform = create_platform(
        platform_config=cluster_config.platform,
        ssh_config=cluster_config.defaults.ssh,
    )
    controller_address = f"http://{platform.discover_controller(cluster_config.controller)}"

    port_start, port_end = map(int, port_range.split("-"))

    config = WorkerConfig(
        host=host,
        port=port,
        cache_dir=Path(cache_dir).expanduser(),
        port_range=(port_start, port_end),
        controller_address=controller_address,
        worker_id=worker_id,
    )
>>>>>>> Stashed changes

    worker = Worker(config, container_runtime=container_runtime)

    click.echo(f"Starting Iris worker on {config.host}:{config.port}")
    click.echo(f"  Cache dir: {config.cache_dir}")
<<<<<<< Updated upstream
    click.echo(f"  Controller: {config.controller_address}")
    click.echo("  Runtime: docker")
||||||| constructed merge base
    if controller_address:
        click.echo(f"  Controller: {controller_address}")
=======
    click.echo(f"  Controller: {controller_address}")
>>>>>>> Stashed changes
    worker.start()
    worker.wait()


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
