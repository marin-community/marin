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

"""Click-based CLI for the Iris worker daemon."""

import logging
import shutil
from pathlib import Path

import click

from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.logging import configure_logging


@click.group()
def cli():
    """Iris Worker - Job execution daemon."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--cache-dir", default="~/.cache/iris-worker", help="Cache directory")
@click.option(
    "--registry",
    default="localhost:5000",
    help="Docker registry for built images (optional for autoscaler-managed workers)",
)
@click.option("--port-range", default="30000-40000", help="Port range for job ports (start-end)")
@click.option(
    "--controller-address", default=None, help="Controller URL for auto-registration (e.g., http://controller:8080)"
)
@click.option("--worker-id", default=None, help="Worker ID (auto-generated if not provided)")
def serve(
    host: str,
    port: int,
    cache_dir: str,
    registry: str,
    port_range: str,
    controller_address: str | None,
    worker_id: str | None,
):
    """Start the Iris worker service."""
    configure_logging(level=logging.INFO)

    port_start, port_end = map(int, port_range.split("-"))

    config = WorkerConfig(
        host=host,
        port=port,
        cache_dir=Path(cache_dir).expanduser(),
        registry=registry,
        port_range=(port_start, port_end),
        controller_address=controller_address,
        worker_id=worker_id,
    )

    worker = Worker(config)

    click.echo(f"Starting Iris worker on {host}:{port}")
    click.echo(f"  Registry: {registry}")
    click.echo(f"  Cache dir: {config.cache_dir}")
    if controller_address:
        click.echo(f"  Controller: {controller_address}")
    worker.start()
    worker.wait()  # Block until worker is stopped


@cli.command()
@click.option("--cache-dir", default="~/.cache/iris-worker", help="Cache directory")
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
