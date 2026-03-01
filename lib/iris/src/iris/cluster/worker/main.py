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
from google.protobuf.json_format import ParseDict

from iris.cluster.platform.factory import create_platform
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.kubernetes import KubernetesRuntime
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
@click.option("--worker-config", type=click.Path(exists=True), required=True, help="Path to WorkerConfig JSON file")
def serve(worker_config: str):
    """Start the Iris worker service."""
    configure_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Iris worker starting (git_hash=%s)", os.environ.get("IRIS_GIT_HASH", "unknown"))

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

    if wc_proto.runtime == "kubernetes":
        container_runtime = KubernetesRuntime()
    else:
        container_runtime = DockerRuntime()

    worker = Worker(config, container_runtime=container_runtime)

    click.echo(f"Starting Iris worker on {config.host}:{config.port}")
    click.echo(f"  Cache dir: {config.cache_dir}")
    click.echo(f"  Controller: {config.controller_address}")
    click.echo(f"  Runtime: {wc_proto.runtime or 'docker'}")
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
