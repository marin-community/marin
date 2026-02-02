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

"""Controller VM management subcommands."""

import signal
import threading

import click

from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.vm.config import create_autoscaler_from_config
from iris.cluster.vm.controller import create_controller
from iris.time_utils import Duration

from iris.cli._helpers import (
    build_and_push_image,
    extract_controller_image_params,
    handle_error,
)


@click.group()
@click.pass_context
def controller(ctx):
    """Controller VM management."""
    pass


@controller.command("start")
@click.pass_context
def controller_start(ctx):
    """Boot controller GCE VM and wait for health.

    Automatically builds and pushes the controller image before starting.
    """
    config = ctx.obj["config"]

    params = extract_controller_image_params(config)
    if not params:
        raise click.ClickException(
            "Cannot extract controller image params from config. "
            "config.controller_vm.image must be a valid Artifact Registry tag.\n"
            "Expected format: REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:VERSION\n"
            f"Got: {config.controller_vm.image or 'None'}"
        )

    build_and_push_image(params)
    click.echo()

    ctrl = create_controller(config)

    click.echo("Starting controller...")
    try:
        address = ctrl.start()
        click.echo(f"Controller started successfully at {address}")
    except Exception as e:
        handle_error(ctx, "Failed to start controller", e)


@controller.command("stop")
@click.pass_context
def controller_stop(ctx):
    """Stop controller VM."""
    config = ctx.obj["config"]
    ctrl = create_controller(config)

    click.echo("Stopping controller...")
    try:
        ctrl.stop()
        click.echo("Controller stopped successfully")
    except Exception as e:
        handle_error(ctx, "Failed to stop controller", e)


@controller.command("restart")
@click.pass_context
def controller_restart(ctx):
    """Stop then start controller."""
    config = ctx.obj["config"]
    ctrl = create_controller(config)

    click.echo("Restarting controller...")
    try:
        address = ctrl.restart()
        click.echo(f"Controller restarted successfully at {address}")
    except Exception as e:
        handle_error(ctx, "Failed to restart controller", e)


@controller.command("reload")
@click.pass_context
def controller_reload(ctx):
    """Reload controller by re-running bootstrap on existing VM.

    Automatically builds and pushes the controller image, then SSHs into
    the existing VM and re-runs the bootstrap script to pull the latest
    image and restart the container.
    """
    config = ctx.obj["config"]

    params = extract_controller_image_params(config)
    if not params:
        raise click.ClickException(
            "Cannot extract controller image params from config. "
            "config.controller_vm.image must be a valid Artifact Registry tag.\n"
            "Expected format: REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:VERSION\n"
            f"Got: {config.controller_vm.image or 'None'}"
        )

    build_and_push_image(params)
    click.echo()

    ctrl = create_controller(config)

    click.echo("Reloading controller...")
    try:
        address = ctrl.reload()
        click.echo(f"Controller reloaded successfully at {address}")
    except Exception as e:
        handle_error(ctx, "Failed to reload controller", e)


@controller.command("status")
@click.pass_context
def controller_status(ctx):
    """Show controller status."""
    config = ctx.obj["config"]
    ctrl = create_controller(config)

    try:
        status = ctrl.status()
        click.echo("\nController Status:")
        click.echo(f"  Running: {status.running}")
        click.echo(f"  Healthy: {status.healthy}")
        click.echo(f"  Address: {status.address or 'N/A'}")
        if status.vm_name:
            click.echo(f"  VM Name: {status.vm_name}")
    except Exception as e:
        handle_error(ctx, "Failed to get controller status", e)


@controller.command("run-local")
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option(
    "--bundle-prefix",
    default=None,
    help="URI prefix for job bundles (e.g., gs://bucket/path or file:///path). Required.",
)
@click.option("--scheduler-interval", default=0.5, type=float, help="Scheduler loop interval (seconds)")
@click.option("--worker-timeout", default=60.0, type=float, help="Worker heartbeat timeout (seconds)")
@click.pass_context
def controller_run_local(
    ctx,
    host: str,
    port: int,
    bundle_prefix: str | None,
    scheduler_interval: float,
    worker_timeout: float,
):
    """Run controller locally for development.

    Examples:
        uv run iris cluster --config=examples/demo.yaml controller run-local --bundle-prefix gs://my-bucket/iris/bundles
    """
    cluster_config = ctx.obj["config"]

    effective_bundle_prefix = bundle_prefix or (
        cluster_config.controller_vm.bundle_prefix if cluster_config.controller_vm else None
    )

    if not effective_bundle_prefix:
        click.echo(
            "Error: bundle_prefix is required. Set via --bundle-prefix or controller_vm.bundle_prefix in config.\n"
            "Example: --bundle-prefix gs://my-bucket/iris/bundles",
            err=True,
        )
        raise SystemExit(1)

    controller_config = ControllerConfig(
        host=host,
        port=port,
        bundle_prefix=effective_bundle_prefix,
        scheduler_interval_seconds=scheduler_interval,
        worker_timeout=Duration.from_seconds(worker_timeout),
    )

    autoscaler = None
    if cluster_config.scale_groups:
        autoscaler = create_autoscaler_from_config(cluster_config)
        autoscaler.reconcile()
        click.echo(f"Autoscaler initialized with {len(autoscaler.groups)} scale group(s)")

    ctrl = Controller(
        config=controller_config,
        worker_stub_factory=RpcWorkerStubFactory(),
        autoscaler=autoscaler,
    )

    click.echo(f"Starting Iris controller on {host}:{port}")
    click.echo(f"  Bundle prefix: {effective_bundle_prefix}")
    click.echo(f"  Scheduler interval: {scheduler_interval}s")
    click.echo(f"  Worker timeout: {worker_timeout}s")
    if autoscaler:
        click.echo("  Autoscaler: enabled")

    ctrl.start()
    click.echo("Controller started. Press Ctrl+C to stop.\n")

    stop_event = threading.Event()

    def handle_sigint(signum, frame):
        click.echo("\nShutting down controller...")
        ctrl.stop()
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    stop_event.wait()
    click.echo("Controller stopped")
