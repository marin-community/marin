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

"""Click-based CLI for the Fluster worker daemon.

Provides two commands:
- serve: Start the worker service
- cleanup: Remove cached bundles, venvs, and images
"""

import shutil
from pathlib import Path

import click

from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import ImageCache, VenvCache
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.docker import DockerRuntime
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl


@click.group()
def cli():
    """Fluster Worker - Job execution daemon."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--cache-dir", default="~/.cache/fluster-worker", help="Cache directory")
@click.option("--registry", required=True, help="Docker registry for built images")
@click.option("--max-concurrent-jobs", default=10, type=int, help="Max concurrent jobs")
@click.option("--port-range", default="30000-40000", help="Port range for job ports (start-end)")
@click.option("--max-bundles", default=100, type=int, help="Max cached bundles")
@click.option("--max-images", default=50, type=int, help="Max cached Docker images")
def serve(
    host: str,
    port: int,
    cache_dir: str,
    registry: str,
    max_concurrent_jobs: int,
    port_range: str,
    max_bundles: int,
    max_images: int,
):
    """Start the Fluster worker service."""
    cache_path = Path(cache_dir).expanduser()

    port_start, port_end = map(int, port_range.split("-"))

    # Initialize components
    bundle_cache = BundleCache(cache_path, max_bundles=max_bundles)
    venv_cache = VenvCache()
    image_cache = ImageCache(cache_path, registry=registry, max_images=max_images)
    runtime = DockerRuntime()
    port_allocator = PortAllocator((port_start, port_end))

    manager = JobManager(
        bundle_cache=bundle_cache,
        venv_cache=venv_cache,
        image_cache=image_cache,
        runtime=runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    service = WorkerServiceImpl(manager)
    dashboard = WorkerDashboard(service, host, port)

    click.echo(f"Starting Fluster worker on {host}:{port}")
    click.echo(f"  Registry: {registry}")
    click.echo(f"  Cache dir: {cache_path}")
    click.echo(f"  Max concurrent jobs: {max_concurrent_jobs}")
    dashboard.run()


@cli.command()
@click.option("--cache-dir", default="~/.cache/fluster-worker", help="Cache directory")
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
