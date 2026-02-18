# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-based CLI for the Iris worker daemon."""

import logging
import os
import shutil
from pathlib import Path

import click

from iris.cluster.config import load_config
from iris.cluster.platform.factory import create_platform
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.logging import configure_logging


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
    "--iris-worker-prefix",
    "iris_worker_prefix",
    required=True,
    help="Storage prefix for worker logs (e.g., gs://marin-tmp-us-central2/ttl=30d/iris-logs)",
)
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
    iris_worker_prefix: str,
    config_file: str,
):
    """Start the Iris worker service."""
    # Set IRIS_WORKER_PREFIX environment variable (required for log persistence)
    os.environ["IRIS_WORKER_PREFIX"] = iris_worker_prefix

    configure_logging(level=logging.INFO)

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
        log_prefix=iris_worker_prefix,
    )

    worker = Worker(config)

    click.echo(f"Starting Iris worker on {host}:{port}")
    click.echo(f"  Cache dir: {config.cache_dir}")
    click.echo(f"  Controller: {controller_address}")
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
