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

"""Click-based CLI for the Iris controller daemon."""

import signal
import threading
from pathlib import Path

import click

from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory


@click.group()
def cli():
    """Iris Controller - Cluster control plane."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option("--bundle-dir", default="/var/cache/iris/bundles", help="Directory for job bundles")
@click.option("--scheduler-interval", default=0.5, type=float, help="Scheduler loop interval (seconds)")
@click.option("--worker-timeout", default=60.0, type=float, help="Worker heartbeat timeout (seconds)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config for autoscaling")
def serve(
    host: str,
    port: int,
    bundle_dir: str,
    scheduler_interval: float,
    worker_timeout: float,
    config_file: str | None,
):
    """Start the Iris controller service.

    When --config is provided, the controller runs an integrated autoscaler
    that provisions/terminates VM slices based on pending task demand.
    """
    from iris.cluster.vm.autoscaler import Autoscaler
    from iris.cluster.vm.config import create_autoscaler_from_config, load_config

    config = ControllerConfig(
        host=host,
        port=port,
        bundle_dir=Path(bundle_dir),
        scheduler_interval_seconds=scheduler_interval,
        worker_timeout_seconds=worker_timeout,
    )

    autoscaler: Autoscaler | None = None
    if config_file:
        click.echo(f"Loading cluster config from {config_file}...")
        cluster_config = load_config(Path(config_file))
        autoscaler = create_autoscaler_from_config(cluster_config)
        autoscaler.reconcile()
        click.echo(f"Autoscaler initialized with {len(autoscaler.groups)} scale group(s)")

    controller = Controller(
        config=config,
        worker_stub_factory=RpcWorkerStubFactory(),
        autoscaler=autoscaler,
    )

    click.echo(f"Starting Iris controller on {host}:{port}")
    click.echo(f"  Bundle dir: {config.bundle_dir}")
    click.echo(f"  Scheduler interval: {scheduler_interval}s")
    click.echo(f"  Worker timeout: {worker_timeout}s")
    if autoscaler:
        click.echo("  Autoscaler: enabled")

    controller.start()

    stop_event = threading.Event()

    def handle_shutdown(_signum, _frame):
        click.echo("\nShutting down controller...")
        controller.stop()
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    stop_event.wait()


if __name__ == "__main__":
    cli()
