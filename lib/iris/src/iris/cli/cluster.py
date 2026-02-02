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

"""Cluster lifecycle commands: start, stop, restart, reload, status, init, dashboard."""

import signal
import threading
from pathlib import Path

import click

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.cluster.vm.config import (
    create_autoscaler_from_config,
    create_manual_autoscaler,
    load_config,
)
from iris.cluster.vm.controller import create_controller
from iris.cluster.vm.debug import controller_tunnel, discover_controller_vm
from iris.rpc import cluster_pb2
from iris.rpc.proto_utils import vm_state_name

from iris.cli._helpers import (
    build_and_push_image,
    build_cluster_images,
    extract_controller_image_params,
    format_status_table,
    format_timestamp,
    handle_error,
    terminate_slices_parallel,
)

# Import sub-groups so they can be registered on the cluster group below.
from iris.cli.autoscaler import autoscaler  # noqa: E402
from iris.cli.controller import controller  # noqa: E402
from iris.cli.debug import debug  # noqa: E402
from iris.cli.slice import slice_group  # noqa: E402
from iris.cli.vm import vm  # noqa: E402


@click.group()
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config file")
@click.pass_context
def cluster(ctx, config_file: str | None):
    """Cluster management commands."""
    parent_obj = ctx.obj or {}
    ctx.ensure_object(dict)
    ctx.obj.update(parent_obj)
    if config_file:
        ctx.obj["config"] = load_config(config_file)
        ctx.obj["config_file"] = config_file


# Register sub-groups
cluster.add_command(controller)
cluster.add_command(autoscaler)
cluster.add_command(slice_group)
cluster.add_command(vm)
cluster.add_command(debug)


# ─── Cluster-level commands ──────────────────────────────────────────────────


@cluster.command("start")
@click.option("--local", is_flag=True, help="Run locally without GCP")
@click.pass_context
def cluster_start(ctx, local: bool):
    """Start controller VM and wait for health.

    Automatically builds and pushes both worker and controller images, then
    boots the controller GCE VM which runs the autoscaler internally.

    Use --local to run entirely on the local machine without GCP.
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    if local:
        config = make_local_config(config)

    manager = ClusterManager(config)

    if not manager.is_local:
        build_cluster_images(config)

    click.echo("Starting controller...")
    try:
        address = manager.start()
        click.echo(f"Controller started at {address}")
        click.echo("\nController is running with integrated autoscaler.")
        click.echo("Use 'iris cluster --config=... status' to check cluster state.")
    except Exception as e:
        click.echo(f"Failed to start controller: {e}", err=True)
        raise SystemExit(1) from e


@cluster.command("stop")
@click.pass_context
def cluster_stop(ctx):
    """Stop controller and terminate all slices.

    Terminates slices in parallel for faster cleanup.
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    ctrl = create_controller(config)
    click.echo("Stopping controller...")
    try:
        ctrl.stop()
        click.echo("Controller stopped")
    except Exception as e:
        click.echo(f"Warning: Failed to stop controller: {e}", err=True)

    click.echo("Discovering existing slices...")
    autoscaler_obj = create_autoscaler_from_config(config)
    autoscaler_obj.reconcile()

    slice_ids = [vm_group.slice_id for group in autoscaler_obj.groups.values() for vm_group in group.vm_groups()]

    if not slice_ids:
        click.echo("No slices to terminate")
        return

    terminate_slices_parallel(autoscaler_obj, slice_ids)
    click.echo("Cluster stopped")


@cluster.command("restart")
@click.pass_context
def cluster_restart(ctx):
    """Restart cluster by stopping then starting."""
    ctx.invoke(cluster_stop)
    click.echo("")
    ctx.invoke(cluster_start)


@cluster.command("reload")
@click.option("--no-build", is_flag=True, help="Skip image building (use existing images)")
@click.option("--validate", is_flag=True, help="Submit a health check after reload")
@click.pass_context
def cluster_reload(ctx, no_build: bool, validate: bool):
    """Reload cluster by rebuilding images and reloading the controller.

    Faster than a full restart: rebuilds Docker images and reloads the controller VM.
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    if not no_build:
        build_cluster_images(config)

    manager = ClusterManager(config)

    click.echo("Reloading controller (workers will be re-bootstrapped automatically)...")
    try:
        address = manager.reload()
        click.echo(f"Controller reloaded at {address}")
        click.echo("Workers will be re-bootstrapped with latest image on next controller reconcile.")
    except Exception as e:
        click.echo(f"Failed to reload cluster: {e}", err=True)
        raise SystemExit(1) from e

    if validate:
        click.echo("\nValidating cluster health...")
        try:
            _validate_cluster_health(config)
            click.echo("Cluster validation passed.")
        except Exception as e:
            click.echo(f"Cluster validation failed: {e}", err=True)
            raise SystemExit(1) from e


@cluster.command("status")
@click.pass_context
def cluster_status_cmd(ctx):
    """Show cluster status including controller and autoscaler."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    ctrl = create_controller(config)
    try:
        ctrl_status = ctrl.status()
        click.echo("Controller Status:")
        click.echo(f"  Running: {ctrl_status.running}")
        click.echo(f"  Healthy: {ctrl_status.healthy}")
        click.echo(f"  Address: {ctrl_status.address or 'N/A'}")
        if ctrl_status.vm_name:
            click.echo(f"  VM Name: {ctrl_status.vm_name}")
    except Exception as e:
        click.echo(f"Failed to get controller status: {e}", err=True)
        click.echo("")

    try:
        autoscaler_obj = create_autoscaler_from_config(config)
        autoscaler_obj.reconcile()

        status = autoscaler_obj.get_status()
        click.echo("\nAutoscaler Status:")
        if not status.groups:
            click.echo("  No scale groups configured")
        else:
            click.echo(format_status_table(status))
    except Exception as e:
        click.echo(f"Failed to get autoscaler status: {e}", err=True)


@cluster.command("init")
@click.argument("hosts", nargs=-1, required=True)
@click.option("--controller-address", required=True, help="Controller address for workers (e.g., http://10.0.0.1:10000)")
@click.option("--docker-image", required=True, help="Docker image for worker container")
@click.option("--ssh-user", default="root", help="SSH user for connecting to hosts")
@click.option("--ssh-key", default=None, type=click.Path(exists=True), help="SSH private key file")
@click.option("--worker-port", default=10001, type=int, help="Port for worker service")
def cluster_init(
    hosts: tuple[str, ...],
    controller_address: str,
    docker_image: str,
    ssh_user: str,
    ssh_key: str | None,
    worker_port: int,
):
    """Initialize workers on specified hosts.

    Creates a ManualVmManager and bootstraps workers on the given hosts.
    """
    click.echo(f"Initializing {len(hosts)} hosts...")
    for host in hosts:
        click.echo(f"  {host}")

    autoscaler_obj = create_manual_autoscaler(
        hosts=list(hosts),
        controller_address=controller_address,
        docker_image=docker_image,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        worker_port=worker_port,
    )

    scale_group_name = "manual"
    group = autoscaler_obj.get_group(scale_group_name)
    if not group:
        click.echo(f"Error: Scale group '{scale_group_name}' not found", err=True)
        raise SystemExit(1)

    slice_obj = group.scale_up()
    slice_id = slice_obj.slice_id

    click.echo(f"\nCreated slice: {slice_id}")
    click.echo("Bootstrap is running in background threads.")
    click.echo("\nMonitor progress with:")
    click.echo("  Watch slices: iris cluster vm status --config ...")

    click.echo("\nInitial status:")
    for vm_item in slice_obj.vms():
        state = vm_state_name(vm_item.info.state)
        click.echo(f"  {vm_item.info.vm_id}: {state} ({vm_item.info.address})")


@cluster.command("dashboard")
@click.option("--port", default=10000, type=int, help="Local port for tunnel")
@click.pass_context
def cluster_dashboard(ctx, port: int):
    """Open SSH tunnel to controller and print dashboard URL.

    Discovers the controller VM, establishes an SSH port-forward tunnel, and
    prints the local dashboard URL.  Blocks until Ctrl+C.

    Examples:
        iris cluster --config=examples/eu-west4.yaml dashboard
        iris cluster --config=examples/eu-west4.yaml dashboard --port 8080
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    zone = config.zone
    project = config.project_id
    label_prefix = config.label_prefix or "iris"

    if not zone or not project:
        click.echo("Error: Config must specify zone and project_id", err=True)
        raise SystemExit(1)

    click.echo(f"Discovering controller in {zone}...")

    vm_name = discover_controller_vm(zone, project, label_prefix)
    if not vm_name:
        click.echo(f"No controller VM found in zone {zone}", err=True)
        raise SystemExit(1)

    click.echo(f"Found controller: {vm_name}")
    click.echo(f"Establishing SSH tunnel (localhost:{port} -> {vm_name}:10000)...")

    stop = threading.Event()

    def on_signal(sig, frame):
        click.echo("\nClosing tunnel...")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    with controller_tunnel(zone, project, local_port=port, label_prefix=label_prefix) as url:
        click.echo(f"\nDashboard:      {url}")
        click.echo(f"Controller RPC: {url}")
        click.echo("\nPress Ctrl+C to close tunnel.")
        stop.wait()


# ─── Internal helpers ────────────────────────────────────────────────────────


def _validate_cluster_health(config) -> None:
    """Validate cluster by submitting a test job and waiting for it to complete."""
    zone = config.zone
    project = config.project_id
    label_prefix = config.label_prefix or "iris"

    with controller_tunnel(zone, project, label_prefix=label_prefix) as tunnel_url:
        click.echo(f"  Connected to controller at {tunnel_url}")
        client = IrisClient.remote(tunnel_url, workspace=Path.cwd())

        def _validate_hello():
            print("Reload validation job OK")
            return 42

        click.echo("  Submitting validation job...")
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_validate_hello),
            name="reload-validate",
            resources=ResourceSpec(cpu=1),
        )
        click.echo(f"  Job submitted: {job.job_id}")
        click.echo("  Waiting for job (workers may need to scale up)...")
        status = job.wait(timeout=600, raise_on_failure=True)
        click.echo(f"  Job completed: {cluster_pb2.JobState.Name(status.state)}")
