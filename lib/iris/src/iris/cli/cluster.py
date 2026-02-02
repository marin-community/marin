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

"""Cluster management CLI commands.

All cluster subcommands live here: lifecycle (start/stop/restart/reload/status/init),
controller VM management, autoscaler status, slice CRUD, VM operations, and the
dashboard tunnel.
"""

import concurrent.futures
import json
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from connectrpc.errors import ConnectError

from iris.client import IrisClient
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.cluster.vm.config import (
    create_autoscaler_from_config,
    create_manual_autoscaler,
    load_config,
)
from iris.cluster.vm.controller import create_controller
from iris.cluster.vm.debug import controller_tunnel, discover_controller_vm
from iris.cluster.vm.vm_platform import compute_slice_state_counts, slice_all_ready, slice_any_failed
from iris.rpc import cluster_connect, cluster_pb2, vm_pb2
from iris.rpc.proto_utils import format_accelerator_display, vm_state_name
from iris.time_utils import Duration, Timestamp

# Import debug sub-group — defined in its own module because it's large and self-contained.
from iris.cli.debug import debug


# =============================================================================
# Helpers (shared across subcommands in this module)
# =============================================================================


def _handle_error(ctx: click.Context, message: str, error: Exception) -> None:
    """Print error message, optionally show traceback, then exit."""
    click.echo(f"{message}: {error}", err=True)
    if ctx.obj and ctx.obj.get("traceback"):
        traceback.print_exc()
    raise SystemExit(1)


def _format_timestamp(ms: int) -> str:
    if ms == 0:
        return "-"
    return Timestamp.from_ms(ms).as_formatted_date()


def _format_status_table(status: vm_pb2.AutoscalerStatus) -> str:
    header = f"{'Scale Group':<18} {'Booting':>8} {'Initializing':>12} {'Ready':>6} {'Failed':>7} {'Demand':>7}"
    lines = [header]
    for group in status.groups:
        counts = compute_slice_state_counts(group.slices)
        line = (
            f"{group.name:<18} "
            f"{counts.get('booting', 0):>8} "
            f"{counts.get('initializing', 0):>12} "
            f"{counts.get('ready', 0):>6} "
            f"{counts.get('failed', 0):>7} "
            f"{group.current_demand:>7}"
        )
        lines.append(line)
    return "\n".join(lines)


def _get_autoscaler_status(controller_url: str) -> vm_pb2.AutoscalerStatus:
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
    return client.get_autoscaler_status(request).status


def _get_vm_logs(controller_url: str, vm_id: str, tail: int) -> tuple[str, str, int]:
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetVmLogsRequest(vm_id=vm_id, tail=tail)
    response = client.get_vm_logs(request)
    return response.logs, response.vm_id, response.state


def load_autoscaler(config_file: str, dry_run: bool = False):
    """Load Autoscaler from config file and discover existing slices.

    Returns tuple of (autoscaler, config).
    """
    config = load_config(Path(config_file))
    autoscaler_obj = create_autoscaler_from_config(config, dry_run=dry_run)
    autoscaler_obj.reconcile()
    return autoscaler_obj, config


def _require_config(ctx: click.Context) -> str:
    config_file = ctx.obj.get("config_file")
    if not config_file:
        click.echo("Error: --config is required for this command", err=True)
        raise SystemExit(1)
    return config_file


def _parse_artifact_registry_tag(image_tag: str) -> tuple[str, str, str, str] | None:
    """Parse ``REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:VERSION``."""
    if "-docker.pkg.dev/" not in image_tag:
        return None
    parts = image_tag.split("/")
    if len(parts) < 4:
        return None
    registry = parts[0]
    if not registry.endswith("-docker.pkg.dev"):
        return None
    region = registry.replace("-docker.pkg.dev", "")
    project = parts[1]
    image_and_version = parts[3]
    if ":" in image_and_version:
        image_name, version = image_and_version.split(":", 1)
    else:
        image_name = image_and_version
        version = "latest"
    return region, project, image_name, version


@dataclass
class _ImageBuildParams:
    image_type: Literal["worker", "controller"]
    region: str
    project: str
    image_name: str
    version: str

    @property
    def local_tag(self) -> str:
        return f"{self.image_name}:{self.version}"


def _extract_image_params(image_tag: str, image_type: Literal["worker", "controller"]) -> _ImageBuildParams | None:
    parsed = _parse_artifact_registry_tag(image_tag)
    if not parsed:
        return None
    region, project, image_name, version = parsed
    return _ImageBuildParams(image_type=image_type, region=region, project=project, image_name=image_name, version=version)


def _build_and_push_image(params: _ImageBuildParams) -> None:
    from iris.build import build_image, push_to_registries

    click.echo(f"Building {params.image_type} image: {params.local_tag}")
    click.echo(f"  Region: {params.region}")
    click.echo(f"  Project: {params.project}")
    click.echo()
    build_image(
        image_type=params.image_type, tag=params.local_tag, push=False,
        dockerfile=None, context=None, platform="linux/amd64", region=(), project=params.project,
    )
    click.echo()
    push_to_registries(
        source_tag=params.local_tag, regions=(params.region,),
        project=params.project, image_name=params.image_name, version=params.version,
    )


def _build_cluster_images(config) -> None:
    for tag, typ in [(config.bootstrap.docker_image, "worker"), (config.controller_vm.image, "controller")]:
        if tag:
            params = _extract_image_params(tag, typ)
            if params:
                _build_and_push_image(params)
                click.echo()


def _terminate_slices_parallel(autoscaler_obj, slice_ids: tuple[str, ...] | list[str]) -> None:
    def terminate_one(sid: str) -> tuple[str, Exception | None]:
        try:
            for group in autoscaler_obj.groups.values():
                for vm_group in group.vm_groups():
                    if vm_group.slice_id == sid:
                        group.scale_down(sid)
                        return sid, None
            return sid, ValueError(f"Slice {sid} not found")
        except Exception as e:
            return sid, e

    click.echo(f"Terminating {len(slice_ids)} slice(s) in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slice_ids)) as executor:
        futures = {executor.submit(terminate_one, sid): sid for sid in slice_ids}
        for future in concurrent.futures.as_completed(futures):
            sid, error = future.result()
            if error:
                click.echo(f"Failed to terminate {sid}: {error}", err=True)
            else:
                click.echo(f"Terminated: {sid}")


def _wait_for_slice(slice_obj, poll_interval: float = 5.0) -> bool:
    """Poll until slice reaches terminal state. Returns True if all VMs ready."""
    last_states: dict[str, str] = {}
    slice_id = slice_obj.slice_id
    click.echo(f"Waiting for slice {slice_id} to reach READY state...")
    while True:
        status = slice_obj.status()
        for vm_item in slice_obj.vms():
            state = vm_state_name(vm_item.info.state)
            if last_states.get(vm_item.info.vm_id) != state:
                click.echo(f"  {vm_item.info.vm_id}: {state}")
                last_states[vm_item.info.vm_id] = state
        if status.all_ready:
            click.echo(f"Slice {slice_id} is READY ({len(list(slice_obj.vms()))} VMs)")
            return True
        if status.any_failed:
            click.echo(f"Slice {slice_id} has FAILED VMs:", err=True)
            for vm_item in slice_obj.vms():
                if vm_item.info.state == vm_pb2.VM_STATE_FAILED:
                    click.echo(f"  {vm_item.info.vm_id}: {vm_item.info.init_error}", err=True)
            return False
        time.sleep(poll_interval)


# =============================================================================
# Top-level cluster group
# =============================================================================


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


# =============================================================================
# Cluster lifecycle commands
# =============================================================================


@cluster.command("start")
@click.option("--local", is_flag=True, help="Run locally without GCP")
@click.pass_context
def cluster_start(ctx, local: bool):
    """Start controller VM and wait for health.

    Builds and pushes images, then boots the controller GCE VM.
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
        _build_cluster_images(config)
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
    """Stop controller and terminate all slices."""
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
    slice_ids = [vg.slice_id for g in autoscaler_obj.groups.values() for vg in g.vm_groups()]
    if not slice_ids:
        click.echo("No slices to terminate")
        return
    _terminate_slices_parallel(autoscaler_obj, slice_ids)
    click.echo("Cluster stopped")


@cluster.command("restart")
@click.pass_context
def cluster_restart(ctx):
    """Restart cluster by stopping then starting."""
    ctx.invoke(cluster_stop)
    click.echo("")
    ctx.invoke(cluster_start)


@cluster.command("reload")
@click.option("--no-build", is_flag=True, help="Skip image building")
@click.option("--validate", is_flag=True, help="Submit a health check after reload")
@click.pass_context
def cluster_reload(ctx, no_build: bool, validate: bool):
    """Rebuild images and reload the controller (faster than full restart)."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)
    if not no_build:
        _build_cluster_images(config)
    manager = ClusterManager(config)
    click.echo("Reloading controller (workers will be re-bootstrapped automatically)...")
    try:
        address = manager.reload()
        click.echo(f"Controller reloaded at {address}")
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
            click.echo(_format_status_table(status))
    except Exception as e:
        click.echo(f"Failed to get autoscaler status: {e}", err=True)


@cluster.command("init")
@click.argument("hosts", nargs=-1, required=True)
@click.option("--controller-address", required=True, help="Controller address (e.g., http://10.0.0.1:10000)")
@click.option("--docker-image", required=True, help="Docker image for worker container")
@click.option("--ssh-user", default="root", help="SSH user")
@click.option("--ssh-key", default=None, type=click.Path(exists=True), help="SSH private key file")
@click.option("--worker-port", default=10001, type=int, help="Port for worker service")
def cluster_init(hosts, controller_address, docker_image, ssh_user, ssh_key, worker_port):
    """Initialize workers on specified hosts."""
    click.echo(f"Initializing {len(hosts)} hosts...")
    for host in hosts:
        click.echo(f"  {host}")
    autoscaler_obj = create_manual_autoscaler(
        hosts=list(hosts), controller_address=controller_address, docker_image=docker_image,
        ssh_user=ssh_user, ssh_key=ssh_key, worker_port=worker_port,
    )
    group = autoscaler_obj.get_group("manual")
    if not group:
        click.echo("Error: Scale group 'manual' not found", err=True)
        raise SystemExit(1)
    slice_obj = group.scale_up()
    click.echo(f"\nCreated slice: {slice_obj.slice_id}")
    click.echo("Bootstrap is running in background threads.")
    click.echo("\nInitial status:")
    for vm_item in slice_obj.vms():
        click.echo(f"  {vm_item.info.vm_id}: {vm_state_name(vm_item.info.state)} ({vm_item.info.address})")


@cluster.command("dashboard")
@click.option("--port", default=10000, type=int, help="Local port for tunnel")
@click.pass_context
def cluster_dashboard(ctx, port: int):
    """Open SSH tunnel to controller and print dashboard URL.

    Discovers the controller VM, establishes an SSH port-forward tunnel, and
    prints the local dashboard URL. Blocks until Ctrl+C.
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


# =============================================================================
# Controller subcommands
# =============================================================================


@cluster.group()
@click.pass_context
def controller(ctx):
    """Controller VM management."""
    pass


@controller.command("start")
@click.pass_context
def controller_start(ctx):
    """Boot controller GCE VM and wait for health."""
    config = ctx.obj["config"]
    params = _extract_image_params(config.controller_vm.image, "controller") if config.controller_vm.image else None
    if not params:
        raise click.ClickException(
            "Cannot extract controller image params. "
            f"config.controller_vm.image must be a valid Artifact Registry tag. Got: {config.controller_vm.image or 'None'}"
        )
    _build_and_push_image(params)
    click.echo()
    ctrl = create_controller(config)
    click.echo("Starting controller...")
    try:
        address = ctrl.start()
        click.echo(f"Controller started successfully at {address}")
    except Exception as e:
        _handle_error(ctx, "Failed to start controller", e)


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
        _handle_error(ctx, "Failed to stop controller", e)


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
        _handle_error(ctx, "Failed to restart controller", e)


@controller.command("reload")
@click.pass_context
def controller_reload(ctx):
    """Reload controller by re-running bootstrap on existing VM."""
    config = ctx.obj["config"]
    params = _extract_image_params(config.controller_vm.image, "controller") if config.controller_vm.image else None
    if not params:
        raise click.ClickException(
            "Cannot extract controller image params. "
            f"config.controller_vm.image must be a valid Artifact Registry tag. Got: {config.controller_vm.image or 'None'}"
        )
    _build_and_push_image(params)
    click.echo()
    ctrl = create_controller(config)
    click.echo("Reloading controller...")
    try:
        address = ctrl.reload()
        click.echo(f"Controller reloaded successfully at {address}")
    except Exception as e:
        _handle_error(ctx, "Failed to reload controller", e)


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
        _handle_error(ctx, "Failed to get controller status", e)


@controller.command("run-local")
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option("--bundle-prefix", default=None, help="URI prefix for job bundles. Required.")
@click.option("--scheduler-interval", default=0.5, type=float, help="Scheduler loop interval (seconds)")
@click.option("--worker-timeout", default=60.0, type=float, help="Worker heartbeat timeout (seconds)")
@click.pass_context
def controller_run_local(ctx, host, port, bundle_prefix, scheduler_interval, worker_timeout):
    """Run controller locally for development."""
    cluster_config = ctx.obj["config"]
    effective_bundle_prefix = bundle_prefix or (
        cluster_config.controller_vm.bundle_prefix if cluster_config.controller_vm else None
    )
    if not effective_bundle_prefix:
        click.echo(
            "Error: bundle_prefix is required. Set via --bundle-prefix or controller_vm.bundle_prefix in config.",
            err=True,
        )
        raise SystemExit(1)
    controller_config = ControllerConfig(
        host=host, port=port, bundle_prefix=effective_bundle_prefix,
        scheduler_interval_seconds=scheduler_interval, worker_timeout=Duration.from_seconds(worker_timeout),
    )
    autoscaler = None
    if cluster_config.scale_groups:
        autoscaler = create_autoscaler_from_config(cluster_config)
        autoscaler.reconcile()
        click.echo(f"Autoscaler initialized with {len(autoscaler.groups)} scale group(s)")
    ctrl = Controller(config=controller_config, worker_stub_factory=RpcWorkerStubFactory(), autoscaler=autoscaler)
    click.echo(f"Starting Iris controller on {host}:{port}")
    click.echo(f"  Bundle prefix: {effective_bundle_prefix}")
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


# =============================================================================
# Autoscaler subcommands
# =============================================================================


@cluster.group()
def autoscaler():
    """Autoscaler commands."""


@autoscaler.command("status")
@click.option("--controller-url", required=True, help="Controller URL (e.g., http://localhost:10000)")
@click.pass_context
def autoscaler_status_cmd(ctx, controller_url: str):
    """Show autoscaler status and scaling decisions."""
    try:
        status = _get_autoscaler_status(controller_url)
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None

    click.echo("Autoscaler Status")
    click.echo("=================")
    click.echo(f"Last evaluation: {_format_timestamp(status.last_evaluation_ms)}")

    if status.current_demand:
        click.echo("\nCurrent Demand:")
        for group_name, demand in status.current_demand.items():
            click.echo(f"  {group_name}: {demand} pending tasks")

    if not status.groups:
        click.echo("\nNo scale groups configured")
        return

    click.echo("\nScale Groups:")
    for group in status.groups:
        counts = compute_slice_state_counts(group.slices)
        total = sum(counts.values())
        click.echo(f"\n  {group.name}:")
        click.echo(f"    Type: {format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)}")
        click.echo(f"    Min/Max slices: {group.config.min_slices}/{group.config.max_slices}")
        click.echo(
            f"    Current slices: {total} "
            f"(booting: {counts.get('booting', 0)}, initializing: {counts.get('initializing', 0)}, "
            f"ready: {counts.get('ready', 0)}, failed: {counts.get('failed', 0)})"
        )
        click.echo(f"    Demand: current={group.current_demand}, peak={group.peak_demand}")
        if group.last_scale_up_ms > 0:
            click.echo(f"    Last scale up: {_format_timestamp(group.last_scale_up_ms)}")
        if group.last_scale_down_ms > 0:
            click.echo(f"    Last scale down: {_format_timestamp(group.last_scale_down_ms)}")
        if group.consecutive_failures > 0:
            click.echo(f"    Consecutive failures: {group.consecutive_failures}")
            if group.backoff_until_ms > 0:
                click.echo(f"    Backoff until: {_format_timestamp(group.backoff_until_ms)}")


# =============================================================================
# Slice subcommands
# =============================================================================


@cluster.group("slice")
def slice_group():
    """Slice management commands."""
    pass


@slice_group.command("create")
@click.option("--scale-group", required=True, help="Scale group name")
@click.option("--tag", "tags", multiple=True, help="Tags in KEY=VALUE format")
@click.option("--dry-run", is_flag=True, help="Log commands without executing them")
@click.option("--wait/--no-wait", default=True, help="Wait for slice to reach READY (default: wait)")
@click.pass_context
def slice_create(ctx, scale_group, tags, dry_run, wait):
    """Create a new slice in the specified scale group."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file, dry_run=dry_run)
    group = autoscaler_obj.get_group(scale_group)
    if not group:
        available = ", ".join(autoscaler_obj.groups.keys()) or "(none)"
        click.echo(f"Error: Unknown scale group '{scale_group}'. Available: {available}", err=True)
        raise SystemExit(1)
    tag_dict: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            click.echo(f"Error: Invalid tag format '{tag}', expected KEY=VALUE", err=True)
            raise SystemExit(1)
        k, v = tag.split("=", 1)
        tag_dict[k] = v
    try:
        slice_obj = group.scale_up(tags=tag_dict or None)
        click.echo(f"Created slice: {slice_obj.slice_id}")
        if wait:
            if not _wait_for_slice(slice_obj):
                raise SystemExit(1)
        else:
            click.echo("VMs are now booting. Use 'iris cluster slice list --config ...' to monitor progress.")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None


@slice_group.command("terminate")
@click.argument("slice_ids", nargs=-1, required=False)
@click.option("--all", "terminate_all", is_flag=True, help="Terminate all slices")
@click.option("--scale-group", default=None, help="Only terminate slices in this scale group (with --all)")
@click.pass_context
def slice_terminate(ctx, slice_ids, terminate_all, scale_group):
    """Terminate one or more slices and all their VMs."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)
    if terminate_all:
        slice_ids = tuple(
            vg.slice_id for name, g in autoscaler_obj.groups.items()
            if not scale_group or name == scale_group
            for vg in g.vm_groups()
        )
        if not slice_ids:
            click.echo("No slices found to terminate")
            return
    elif not slice_ids:
        click.echo("Error: Either provide SLICE_IDS or use --all", err=True)
        raise SystemExit(1)
    _terminate_slices_parallel(autoscaler_obj, slice_ids)


@slice_group.command("list")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def slice_list(ctx, scale_group, as_json):
    """List all slices."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)
    all_slices = []
    for name, group in autoscaler_obj.groups.items():
        if scale_group and name != scale_group:
            continue
        for vm_group in group.vm_groups():
            status = vm_group.status()
            all_slices.append((name, vm_group, status))
    if as_json:
        output = [
            {
                "slice_id": vg.slice_id, "scale_group": gn, "all_ready": st.all_ready,
                "any_failed": st.any_failed, "vm_count": len(list(vg.vms())), "created_at_ms": vg.created_at_ms,
            }
            for gn, vg, st in all_slices
        ]
        click.echo(json.dumps(output, indent=2))
    else:
        if not all_slices:
            click.echo("No slices found")
            return
        click.echo(f"{'SLICE ID':<40} {'SCALE GROUP':<20} {'STATUS':<10} {'VMs'}")
        click.echo("-" * 80)
        for gn, vg, st in all_slices:
            status_str = "READY" if st.all_ready else ("FAILED" if st.any_failed else "PENDING")
            click.echo(f"{vg.slice_id:<40} {gn:<20} {status_str:<10} {len(list(vg.vms()))}")


@slice_group.command("get")
@click.argument("slice_id")
@click.pass_context
def slice_get(ctx, slice_id):
    """Get detailed info for a specific slice."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)
    for group_name, group in autoscaler_obj.groups.items():
        for vm_group in group.vm_groups():
            if vm_group.slice_id == slice_id:
                status = vm_group.status()
                status_str = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
                vms = list(vm_group.vms())
                click.echo(f"Slice: {vm_group.slice_id}")
                click.echo(f"Scale Group: {group_name}")
                click.echo(f"Status: {status_str}")
                click.echo(f"Created: {_format_timestamp(vm_group.created_at_ms)}")
                click.echo(f"VMs: {len(vms)}")
                click.echo()
                for vm_item in vms:
                    state = vm_state_name(vm_item.info.state)
                    click.echo(f"  {vm_item.info.vm_id}:")
                    click.echo(f"    State: {state}")
                    click.echo(f"    Address: {vm_item.info.address or '(none)'}")
                    if vm_item.info.zone:
                        click.echo(f"    Zone: {vm_item.info.zone}")
                    if vm_item.info.worker_id:
                        click.echo(f"    Worker ID: {vm_item.info.worker_id}")
                    if vm_item.info.init_phase:
                        click.echo(f"    Init Phase: {vm_item.info.init_phase}")
                    if vm_item.info.init_error:
                        click.echo(f"    Error: {vm_item.info.init_error}")
                return
    click.echo(f"Error: Slice {slice_id} not found", err=True)
    raise SystemExit(1)


# =============================================================================
# VM subcommands
# =============================================================================


@cluster.group()
@click.option("--controller-url", help="Controller URL for RPC mode (e.g., http://localhost:10000)")
@click.pass_context
def vm(ctx, controller_url: str | None):
    """VM management commands."""
    ctx.obj["controller_url"] = controller_url


@vm.command("status")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.pass_context
def vm_status(ctx, scale_group):
    """Show VM and slice status."""
    controller_url = ctx.obj.get("controller_url")
    config_file = ctx.obj.get("config_file")
    if controller_url:
        _vm_status_via_controller(controller_url, scale_group)
    elif config_file:
        _vm_status_via_autoscaler(config_file, scale_group)
    else:
        click.echo("Error: Either --controller-url or cluster --config is required", err=True)
        raise SystemExit(1)


def _vm_status_via_controller(controller_url: str, scale_group: str | None):
    try:
        as_status = _get_autoscaler_status(controller_url)
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None
    if not as_status.groups:
        click.echo("No scale groups configured")
        return
    for group in as_status.groups:
        if scale_group and group.name != scale_group:
            continue
        counts = compute_slice_state_counts(group.slices)
        total = sum(counts.values())
        click.echo(f"\nScale Group: {group.name}")
        click.echo(f"  Accelerator: {format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)}")
        click.echo(f"  Slices: {counts.get('ready', 0)}/{total} ready")
        click.echo(f"    Booting: {counts.get('booting', 0)}")
        click.echo(f"    Initializing: {counts.get('initializing', 0)}")
        click.echo(f"    Failed: {counts.get('failed', 0)}")
        click.echo(f"  Demand: {group.current_demand} (peak: {group.peak_demand})")
        if group.backoff_until_ms > 0:
            click.echo(f"  Backoff until: {_format_timestamp(group.backoff_until_ms)}")
            click.echo(f"  Consecutive failures: {group.consecutive_failures}")
        if group.slices:
            click.echo("  Slices:")
            for si in group.slices:
                ss = "READY" if slice_all_ready(si) else ("FAILED" if slice_any_failed(si) else "PENDING")
                click.echo(f"    {si.slice_id}: {ss}")
                for vi in si.vms:
                    click.echo(f"      {vi.vm_id}: {vm_state_name(vi.state)} ({vi.address})")
                    if vi.init_error:
                        click.echo(f"        Error: {vi.init_error}")
    click.echo(f"\nLast evaluation: {_format_timestamp(as_status.last_evaluation_ms)}")


def _vm_status_via_autoscaler(config_file: str, scale_group: str | None):
    autoscaler_obj, _ = load_autoscaler(config_file)
    groups = autoscaler_obj.groups
    if scale_group:
        groups = {k: v for k, v in groups.items() if k == scale_group}
    if not groups:
        click.echo("No scale groups found")
        return
    for name, group in groups.items():
        vm_groups = group.vm_groups()
        click.echo(f"\nScale Group: {name}")
        click.echo(f"  Accelerator: {format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)}")
        click.echo(f"  Slices: {len(vm_groups)}")
        for vg in vm_groups:
            status = vg.status()
            ss = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
            click.echo(f"\n    Slice: {vg.slice_id} [{ss}]")
            click.echo(f"      Created: {_format_timestamp(vg.created_at_ms)}")
            for vi in vg.vms():
                click.echo(f"        {vi.info.vm_id}: {vm_state_name(vi.info.state)} ({vi.info.address or 'no address'})")
                if vi.info.init_error:
                    click.echo(f"          Error: {vi.info.init_error}")


@vm.command("logs")
@click.argument("vm_id")
@click.option("--tail", type=int, default=0, help="Show last N lines (0 = all)")
@click.pass_context
def vm_logs(ctx, vm_id, tail):
    """Show VM initialization logs."""
    controller_url = ctx.obj.get("controller_url")
    config_file = ctx.obj.get("config_file")
    if controller_url:
        try:
            log_content, returned_vm_id, state = _get_vm_logs(controller_url, vm_id, tail)
        except ConnectError as e:
            from connectrpc.code import Code

            if e.code == Code.NOT_FOUND:
                click.echo(f"VM not found: {vm_id}", err=True)
            else:
                click.echo(f"Error fetching logs: {e}", err=True)
            raise SystemExit(1) from None
        except Exception as e:
            click.echo(f"Error connecting to controller: {e}", err=True)
            raise SystemExit(1) from None
        click.echo(f"VM: {returned_vm_id}")
        click.echo(f"State: {vm_state_name(state)}")
        click.echo("---")
        click.echo(log_content if log_content else "(no logs available)")
    elif config_file:
        autoscaler_obj, _ = load_autoscaler(config_file)
        for group in autoscaler_obj.groups.values():
            for vg in group.vm_groups():
                for vi in vg.vms():
                    if vi.info.vm_id == vm_id:
                        log_content = vi.init_log(tail if tail > 0 else None)
                        click.echo(f"VM: {vm_id}")
                        click.echo(f"State: {vm_state_name(vi.info.state)}")
                        click.echo("---")
                        click.echo(log_content if log_content else "(no logs available)")
                        return
        click.echo(f"Error: VM {vm_id} not found", err=True)
        raise SystemExit(1)
    else:
        click.echo("Error: Either --controller-url or cluster --config is required", err=True)
        raise SystemExit(1)


@vm.command("get")
@click.argument("vm_id")
@click.pass_context
def vm_get(ctx, vm_id):
    """Get detailed info for a specific VM."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)
    for group_name, group in autoscaler_obj.groups.items():
        for vg in group.vm_groups():
            for vi in vg.vms():
                if vi.info.vm_id == vm_id:
                    click.echo(f"VM: {vi.info.vm_id}")
                    click.echo(f"Slice: {vi.info.slice_id}")
                    click.echo(f"Scale Group: {group_name}")
                    click.echo(f"State: {vm_state_name(vi.info.state)}")
                    click.echo(f"Address: {vi.info.address or '(none)'}")
                    if vi.info.zone:
                        click.echo(f"Zone: {vi.info.zone}")
                    click.echo(f"Created: {_format_timestamp(vi.info.created_at_ms)}")
                    click.echo(f"State Changed: {_format_timestamp(vi.info.state_changed_at_ms)}")
                    if vi.info.worker_id:
                        click.echo(f"Worker ID: {vi.info.worker_id}")
                        click.echo(f"Worker Healthy: {vi.info.worker_healthy}")
                    if vi.info.init_phase:
                        click.echo(f"Init Phase: {vi.info.init_phase}")
                    if vi.info.init_error:
                        click.echo(f"Init Error: {vi.info.init_error}")
                    return
    click.echo(f"Error: VM {vm_id} not found", err=True)
    raise SystemExit(1)


# =============================================================================
# Register debug sub-group
# =============================================================================

cluster.add_command(debug)


# =============================================================================
# Internal helpers
# =============================================================================


def _validate_cluster_health(config) -> None:
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
        job = client.submit(entrypoint=Entrypoint.from_callable(_validate_hello), name="reload-validate", resources=ResourceSpec(cpu=1))
        click.echo(f"  Job submitted: {job.job_id}")
        click.echo("  Waiting for job (workers may need to scale up)...")
        status = job.wait(timeout=600, raise_on_failure=True)
        click.echo(f"  Job completed: {cluster_pb2.JobState.Name(status.state)}")
