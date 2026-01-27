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

"""Top-level Iris CLI entry point.

This module provides the main CLI interface for Iris cluster management,
organized into subcommand groups for cluster operations and image builds.
"""

import concurrent.futures
import datetime
import json
import logging
import signal
import threading
import time
import traceback
from pathlib import Path

import click
from connectrpc.errors import ConnectError

from iris.build import build_image, push_to_registries
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.vm.config import (
    create_autoscaler_from_config,
    create_manual_autoscaler,
    load_config,
)
from iris.cluster.vm.controller import create_controller
from iris.cluster.vm.vm_platform import compute_slice_state_counts, slice_all_ready, slice_any_failed
from iris.rpc import cluster_connect, cluster_pb2, vm_pb2
from iris.rpc.proto_utils import vm_state_name
from iris.rpc_cli import ServiceCommands

# =============================================================================
# Helper Functions
# =============================================================================


def handle_error(ctx: click.Context, message: str, error: Exception) -> None:
    """Handle an error, optionally showing traceback based on --traceback flag."""
    click.echo(f"{message}: {error}", err=True)
    if ctx.obj and ctx.obj.get("traceback"):
        traceback.print_exc()
    raise SystemExit(1)


def _format_timestamp(ms: int) -> str:
    """Format millisecond timestamp as human-readable string."""
    if ms == 0:
        return "-"
    dt = datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_status_table(status: vm_pb2.AutoscalerStatus) -> str:
    """Format autoscaler status as a table showing slice states."""
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
    """Fetch autoscaler status from the controller via Connect RPC."""
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
    response = client.get_autoscaler_status(request)
    return response.status


def _get_vm_logs(controller_url: str, vm_id: str, tail: int) -> tuple[str, str, int]:
    """Fetch VM logs from the controller via Connect RPC.

    Returns:
        Tuple of (logs, vm_id, state)
    """
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetVmLogsRequest(vm_id=vm_id, tail=tail)
    response = client.get_vm_logs(request)
    return response.logs, response.vm_id, response.state


def _load_autoscaler(config_file: str, dry_run: bool = False):
    """Load Autoscaler from config file and discover existing slices.

    Returns tuple of (autoscaler, config) so caller can access config settings.
    """
    config = load_config(Path(config_file))
    autoscaler = create_autoscaler_from_config(config, dry_run=dry_run)
    autoscaler.reconcile()
    return autoscaler, config


def _require_config(ctx: click.Context) -> str:
    """Get config file from context, raising error if not provided."""
    config_file = ctx.obj.get("config_file")
    if not config_file:
        click.echo("Error: --config is required for this command", err=True)
        raise SystemExit(1)
    return config_file


def _parse_artifact_registry_tag(image_tag: str) -> tuple[str, str, str, str] | None:
    """Parse a GCP Artifact Registry image tag into components.

    Args:
        image_tag: Full image tag like "europe-west4-docker.pkg.dev/project/repo/image:version"

    Returns:
        Tuple of (region, project, image_name, version) or None if not a valid AR tag.
    """
    if "-docker.pkg.dev/" not in image_tag:
        return None

    # Parse: region-docker.pkg.dev/project/repo/image:version
    parts = image_tag.split("/")
    if len(parts) < 4:
        return None

    # Extract region from "region-docker.pkg.dev"
    registry = parts[0]
    if not registry.endswith("-docker.pkg.dev"):
        return None
    region = registry.replace("-docker.pkg.dev", "")

    project = parts[1]
    # repo is parts[2] (e.g., "marin")
    image_and_version = parts[3]

    if ":" in image_and_version:
        image_name, version = image_and_version.split(":", 1)
    else:
        image_name = image_and_version
        version = "latest"

    return region, project, image_name, version


def _build_and_push_controller_image(config) -> None:
    """Build and push the controller image specified in config.

    Extracts the image tag from config.controller_vm.gcp.image, builds the
    controller image locally, and pushes it to the appropriate registry.

    Args:
        config: IrisClusterConfig proto with controller_vm.gcp.image set.

    Raises:
        click.ClickException: If image tag is invalid or build fails.
    """
    controller_vm = config.controller_vm
    if controller_vm.WhichOneof("controller") != "gcp":
        raise click.ClickException("--build only supported for GCP controller (controller_vm.gcp)")

    image_tag = controller_vm.gcp.image
    if not image_tag:
        raise click.ClickException("controller_vm.gcp.image not set in config")

    parsed = _parse_artifact_registry_tag(image_tag)
    if not parsed:
        raise click.ClickException(
            f"Cannot parse image tag: {image_tag}\n" "Expected format: REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:VERSION"
        )

    region, project, image_name, version = parsed
    local_tag = f"{image_name}:{version}"

    click.echo(f"Building controller image: {local_tag}")
    click.echo(f"  Region: {region}")
    click.echo(f"  Project: {project}")
    click.echo()

    # Build the image
    build_image(
        image_type="controller",
        tag=local_tag,
        push=False,
        dockerfile=None,
        context=None,
        platform="linux/amd64",
        region=(),
        project=project,
    )

    # Push to registry
    click.echo()
    push_to_registries(
        source_tag=local_tag,
        regions=(region,),
        project=project,
        image_name=image_name,
        version=version,
    )


def _wait_for_slice_obj(slice_obj, poll_interval: float = 5.0) -> bool:
    """Poll until slice reaches terminal state (all READY or any FAILED).

    Returns True if all VMs are ready, False if any failed.
    """
    last_states: dict[str, str] = {}
    slice_id = slice_obj.slice_id

    click.echo(f"Waiting for slice {slice_id} to reach READY state...")

    while True:
        status = slice_obj.status()

        # Log state changes
        for vm in slice_obj.vms():
            state = vm_state_name(vm.info.state)
            if last_states.get(vm.info.vm_id) != state:
                click.echo(f"  {vm.info.vm_id}: {state}")
                last_states[vm.info.vm_id] = state

        if status.all_ready:
            vm_count = len(list(slice_obj.vms()))
            click.echo(f"Slice {slice_id} is READY ({vm_count} VMs)")
            return True

        if status.any_failed:
            click.echo(f"Slice {slice_id} has FAILED VMs:", err=True)
            for vm in slice_obj.vms():
                if vm.info.state == vm_pb2.VM_STATE_FAILED:
                    click.echo(f"  {vm.info.vm_id}: {vm.info.init_error}", err=True)
            return False

        time.sleep(poll_interval)


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--traceback", "show_traceback", is_flag=True, help="Show full stack traces on errors")
@click.pass_context
def iris(ctx, verbose: bool, show_traceback: bool):
    """Iris cluster management."""
    ctx.ensure_object(dict)
    ctx.obj["traceback"] = show_traceback

    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# =============================================================================
# Service RPC Commands (dynamically generated from protobuf)
# =============================================================================

# Add service commands as top-level groups for direct RPC access:
#   iris controller-rpc list-jobs --url http://localhost:10000
#   iris worker-rpc heartbeat --url http://localhost:10001
#   iris actor-rpc invoke --url http://localhost:10002
iris.add_command(ServiceCommands("controller", name="controller-rpc", help="Controller service RPC methods"))
iris.add_command(ServiceCommands("worker", name="worker-rpc", help="Worker service RPC methods"))
iris.add_command(ServiceCommands("actor", name="actor-rpc", help="Actor service RPC methods"))


# =============================================================================
# Cluster Commands
# =============================================================================


@iris.group()
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


@cluster.group()
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

    _build_and_push_controller_image(config)
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
    image and restart the container. Faster than a full restart.
    """
    config = ctx.obj["config"]

    _build_and_push_controller_image(config)
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
@click.option("--bundle-dir", default="/var/cache/iris/bundles", help="Directory for job bundles")
@click.option("--scheduler-interval", default=0.5, type=float, help="Scheduler loop interval (seconds)")
@click.option("--worker-timeout", default=60.0, type=float, help="Worker heartbeat timeout (seconds)")
@click.pass_context
def controller_run_local(
    ctx,
    host: str,
    port: int,
    bundle_dir: str,
    scheduler_interval: float,
    worker_timeout: float,
):
    """Run controller locally for development.

    Runs the controller process locally (no GCE VM) with integrated autoscaler.
    The autoscaler provisions/terminates VMs based on pending task demand.

    Examples:
        # Run local controller with autoscaler
        uv run iris cluster --config=examples/demo.yaml controller run-local

        # Custom port
        uv run iris cluster --config=examples/demo.yaml controller run-local --port 8080
    """
    cluster_config = ctx.obj["config"]

    controller_config = ControllerConfig(
        host=host,
        port=port,
        bundle_dir=Path(bundle_dir),
        scheduler_interval_seconds=scheduler_interval,
        worker_timeout_seconds=worker_timeout,
    )

    autoscaler = None
    if cluster_config.scale_groups:
        autoscaler = create_autoscaler_from_config(cluster_config)
        autoscaler.reconcile()
        click.echo(f"Autoscaler initialized with {len(autoscaler.groups)} scale group(s)")

    controller = Controller(
        config=controller_config,
        worker_stub_factory=RpcWorkerStubFactory(),
        autoscaler=autoscaler,
    )

    click.echo(f"Starting Iris controller on {host}:{port}")
    click.echo(f"  Bundle dir: {controller_config.bundle_dir}")
    click.echo(f"  Scheduler interval: {scheduler_interval}s")
    click.echo(f"  Worker timeout: {worker_timeout}s")
    if autoscaler:
        click.echo("  Autoscaler: enabled")

    controller.start()
    click.echo("Controller started. Press Ctrl+C to stop.\n")

    stop_event = threading.Event()

    def handle_sigint(signum, frame):
        click.echo("\nShutting down controller...")
        controller.stop()
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    stop_event.wait()
    click.echo("Controller stopped")


# =============================================================================
# Autoscaler Commands
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
        click.echo(f"    Type: {group.config.accelerator_type}")
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
# Slice Commands
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
def slice_create(ctx, scale_group: str, tags: tuple[str, ...], dry_run: bool, wait: bool):
    """Create a new slice in the specified scale group."""
    config_file = _require_config(ctx)
    autoscaler_obj, _config = _load_autoscaler(config_file, dry_run=dry_run)

    # Validate scale group
    group = autoscaler_obj.get_group(scale_group)
    if not group:
        available = ", ".join(autoscaler_obj.groups.keys()) or "(none)"
        click.echo(f"Error: Unknown scale group '{scale_group}'. Available: {available}", err=True)
        raise SystemExit(1)

    # Parse tags
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
            success = _wait_for_slice_obj(slice_obj)
            if not success:
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
def slice_terminate(ctx, slice_ids: tuple[str, ...], terminate_all: bool, scale_group: str | None):
    """Terminate one or more slices and all their VMs.

    Slices are terminated in parallel for faster cleanup.

    Examples:
        iris cluster --config=... slice terminate iris-tpu_v5e_16-123456
        iris cluster --config=... slice terminate slice1 slice2 slice3
        iris cluster --config=... slice terminate --all
        iris cluster --config=... slice terminate --all --scale-group tpu_v5e_16
    """
    config_file = _require_config(ctx)
    autoscaler_obj, _ = _load_autoscaler(config_file)

    # Collect slice IDs to terminate
    if terminate_all:
        slice_ids = tuple(
            vm_group.slice_id
            for name, group in autoscaler_obj.groups.items()
            if not scale_group or name == scale_group
            for vm_group in group.vm_groups()
        )
        if not slice_ids:
            click.echo("No slices found to terminate")
            return
    elif not slice_ids:
        click.echo("Error: Either provide SLICE_IDS or use --all", err=True)
        raise SystemExit(1)

    def terminate_one(slice_id: str) -> tuple[str, Exception | None]:
        try:
            # Find and terminate slice across all groups
            for group in autoscaler_obj.groups.values():
                for vm_group in group.vm_groups():
                    if vm_group.slice_id == slice_id:
                        group.scale_down(slice_id)
                        return slice_id, None
            return slice_id, ValueError(f"Slice {slice_id} not found")
        except Exception as e:
            return slice_id, e

    click.echo(f"Terminating {len(slice_ids)} slice(s) in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slice_ids)) as executor:
        futures = {executor.submit(terminate_one, sid): sid for sid in slice_ids}
        for future in concurrent.futures.as_completed(futures):
            slice_id, error = future.result()
            if error:
                click.echo(f"Failed to terminate {slice_id}: {error}", err=True)
            else:
                click.echo(f"Terminated: {slice_id}")


@slice_group.command("list")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def slice_list(ctx, scale_group: str | None, as_json: bool):
    """List all slices."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = _load_autoscaler(config_file)

    # Collect slices from all groups
    all_slices = []
    for name, group in autoscaler_obj.groups.items():
        if scale_group and name != scale_group:
            continue
        for vm_group in group.vm_groups():
            status = vm_group.status()
            all_slices.append((name, vm_group, status))

    if as_json:
        output = []
        for group_name, vm_group, status in all_slices:
            output.append(
                {
                    "slice_id": vm_group.slice_id,
                    "scale_group": group_name,
                    "all_ready": status.all_ready,
                    "any_failed": status.any_failed,
                    "vm_count": len(list(vm_group.vms())),
                    "created_at_ms": vm_group.created_at_ms,
                }
            )
        click.echo(json.dumps(output, indent=2))
    else:
        if not all_slices:
            click.echo("No slices found")
            return

        click.echo(f"{'SLICE ID':<40} {'SCALE GROUP':<20} {'STATUS':<10} {'VMs'}")
        click.echo("-" * 80)
        for group_name, vm_group, status in all_slices:
            status_str = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
            vm_count = len(list(vm_group.vms()))
            click.echo(f"{vm_group.slice_id:<40} {group_name:<20} {status_str:<10} {vm_count}")


@slice_group.command("get")
@click.argument("slice_id")
@click.pass_context
def slice_get(ctx, slice_id: str):
    """Get detailed info for a specific slice."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = _load_autoscaler(config_file)

    # Find slice across all groups
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

                for vm in vms:
                    state = vm_state_name(vm.info.state)
                    click.echo(f"  {vm.info.vm_id}:")
                    click.echo(f"    State: {state}")
                    click.echo(f"    Address: {vm.info.address or '(none)'}")
                    if vm.info.zone:
                        click.echo(f"    Zone: {vm.info.zone}")
                    if vm.info.worker_id:
                        click.echo(f"    Worker ID: {vm.info.worker_id}")
                    if vm.info.init_phase:
                        click.echo(f"    Init Phase: {vm.info.init_phase}")
                    if vm.info.init_error:
                        click.echo(f"    Error: {vm.info.init_error}")
                return

    click.echo(f"Error: Slice {slice_id} not found", err=True)
    raise SystemExit(1)


# =============================================================================
# VM Commands
# =============================================================================


@cluster.group()
@click.option("--controller-url", help="Controller URL for RPC mode (e.g., http://localhost:10000)")
@click.pass_context
def vm(ctx, controller_url: str | None):
    """VM management commands.

    Uses --config from parent cluster group, or --controller-url for direct RPC access.
    """
    ctx.obj["controller_url"] = controller_url


@vm.command("status")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.pass_context
def vm_status(ctx, scale_group: str | None):
    """Show VM and slice status."""
    controller_url = ctx.obj.get("controller_url")
    config_file = ctx.obj.get("config_file")

    if controller_url:
        _status_via_controller(controller_url, scale_group)
    elif config_file:
        _status_via_autoscaler(config_file, scale_group)
    else:
        click.echo("Error: Either --controller-url or cluster --config is required", err=True)
        raise SystemExit(1)


def _status_via_controller(controller_url: str, scale_group: str | None):
    """Show status via controller RPC."""
    try:
        autoscaler_status = _get_autoscaler_status(controller_url)
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None

    if not autoscaler_status.groups:
        click.echo("No scale groups configured")
        return

    for group in autoscaler_status.groups:
        if scale_group and group.name != scale_group:
            continue

        counts = compute_slice_state_counts(group.slices)
        total = sum(counts.values())
        click.echo(f"\nScale Group: {group.name}")
        click.echo(f"  Accelerator: {group.config.accelerator_type}")
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
            for slice_info in group.slices:
                status_str = (
                    "READY" if slice_all_ready(slice_info) else ("FAILED" if slice_any_failed(slice_info) else "PENDING")
                )
                click.echo(f"    {slice_info.slice_id}: {status_str}")
                for vm_info in slice_info.vms:
                    state = vm_state_name(vm_info.state)
                    click.echo(f"      {vm_info.vm_id}: {state} ({vm_info.address})")
                    if vm_info.init_error:
                        click.echo(f"        Error: {vm_info.init_error}")

    click.echo(f"\nLast evaluation: {_format_timestamp(autoscaler_status.last_evaluation_ms)}")


def _status_via_autoscaler(config_file: str, scale_group: str | None):
    """Show status via direct Autoscaler."""
    autoscaler_obj, _ = _load_autoscaler(config_file)

    groups = autoscaler_obj.groups
    if scale_group:
        groups = {k: v for k, v in groups.items() if k == scale_group}

    if not groups:
        click.echo("No scale groups found")
        return

    for name, group in groups.items():
        vm_groups = group.vm_groups()
        click.echo(f"\nScale Group: {name}")
        click.echo(f"  Accelerator: {group.config.accelerator_type}")
        click.echo(f"  Slices: {len(vm_groups)}")

        for vm_group in vm_groups:
            status = vm_group.status()
            status_str = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
            click.echo(f"\n    Slice: {vm_group.slice_id} [{status_str}]")
            click.echo(f"      Created: {_format_timestamp(vm_group.created_at_ms)}")

            for vm in vm_group.vms():
                state = vm_state_name(vm.info.state)
                addr = vm.info.address or "no address"
                click.echo(f"        {vm.info.vm_id}: {state} ({addr})")
                if vm.info.init_error:
                    click.echo(f"          Error: {vm.info.init_error}")


@vm.command("logs")
@click.argument("vm_id")
@click.option("--tail", type=int, default=0, help="Show last N lines (0 = all)")
@click.pass_context
def vm_logs(ctx, vm_id: str, tail: int):
    """Show VM initialization logs."""
    controller_url = ctx.obj.get("controller_url")
    config_file = ctx.obj.get("config_file")

    if controller_url:
        _logs_via_controller(controller_url, vm_id, tail)
    elif config_file:
        _logs_via_autoscaler(config_file, vm_id, tail)
    else:
        click.echo("Error: Either --controller-url or cluster --config is required", err=True)
        raise SystemExit(1)


def _logs_via_controller(controller_url: str, vm_id: str, tail: int):
    """Show logs via controller RPC."""
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
    if log_content:
        click.echo(log_content)
    else:
        click.echo("(no logs available)")


def _logs_via_autoscaler(config_file: str, vm_id: str, tail: int):
    """Show logs via direct Autoscaler."""
    autoscaler_obj, _ = _load_autoscaler(config_file)

    # Find VM across all groups
    for group in autoscaler_obj.groups.values():
        for vm_group in group.vm_groups():
            for vm in vm_group.vms():
                if vm.info.vm_id == vm_id:
                    log_content = vm.init_log(tail if tail > 0 else None)
                    click.echo(f"VM: {vm_id}")
                    click.echo(f"State: {vm_state_name(vm.info.state)}")
                    click.echo("---")
                    if log_content:
                        click.echo(log_content)
                    else:
                        click.echo("(no logs available)")
                    return

    click.echo(f"Error: VM {vm_id} not found", err=True)
    raise SystemExit(1)


@vm.command("get")
@click.argument("vm_id")
@click.pass_context
def vm_get(ctx, vm_id: str):
    """Get detailed info for a specific VM."""
    config_file = _require_config(ctx)
    autoscaler_obj, _ = _load_autoscaler(config_file)

    # Find VM across all groups
    for group_name, group in autoscaler_obj.groups.items():
        for vm_group in group.vm_groups():
            for vm in vm_group.vms():
                if vm.info.vm_id == vm_id:
                    state = vm_state_name(vm.info.state)
                    click.echo(f"VM: {vm.info.vm_id}")
                    click.echo(f"Slice: {vm.info.slice_id}")
                    click.echo(f"Scale Group: {group_name}")
                    click.echo(f"State: {state}")
                    click.echo(f"Address: {vm.info.address or '(none)'}")
                    if vm.info.zone:
                        click.echo(f"Zone: {vm.info.zone}")
                    click.echo(f"Created: {_format_timestamp(vm.info.created_at_ms)}")
                    click.echo(f"State Changed: {_format_timestamp(vm.info.state_changed_at_ms)}")

                    if vm.info.worker_id:
                        click.echo(f"Worker ID: {vm.info.worker_id}")
                        click.echo(f"Worker Healthy: {vm.info.worker_healthy}")

                    if vm.info.init_phase:
                        click.echo(f"Init Phase: {vm.info.init_phase}")
                    if vm.info.init_error:
                        click.echo(f"Init Error: {vm.info.init_error}")
                    return

    click.echo(f"Error: VM {vm_id} not found", err=True)
    raise SystemExit(1)


# =============================================================================
# Cluster Init Command
# =============================================================================


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
    This is the quick path for host initialization without a config file.

    Example:
        iris cluster init 10.0.0.1 10.0.0.2 \\
            --controller-address http://10.0.0.100:10000 --docker-image gcr.io/project/worker:v1
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

    # Create a slice to trigger bootstrap
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

    # Show initial status
    click.echo("\nInitial status:")
    for vm in slice_obj.vms():
        state = vm_state_name(vm.info.state)
        click.echo(f"  {vm.info.vm_id}: {state} ({vm.info.address})")


# =============================================================================
# Main Cluster Commands (start/stop/restart/status)
# =============================================================================


@cluster.command("start")
@click.pass_context
def cluster_start(ctx):
    """Start controller VM and wait for health.

    Automatically builds and pushes the controller image, then boots the
    controller GCE VM which runs the autoscaler internally.
    The autoscaler provisions/terminates worker VMs based on task demand.
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    _build_and_push_controller_image(config)
    click.echo()

    ctrl = create_controller(config)
    click.echo("Starting controller...")
    try:
        address = ctrl.start()
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

    Examples:
        uv run iris cluster --config=examples/demo.yaml stop
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    # Stop controller first
    ctrl = create_controller(config)
    click.echo("Stopping controller...")
    try:
        ctrl.stop()
        click.echo("Controller stopped")
    except Exception as e:
        click.echo(f"Warning: Failed to stop controller: {e}", err=True)

    # Create autoscaler and reconcile
    click.echo("Discovering existing slices...")
    autoscaler_obj = create_autoscaler_from_config(config)
    autoscaler_obj.reconcile()

    # Collect all slice IDs
    slice_ids = [vm_group.slice_id for group in autoscaler_obj.groups.values() for vm_group in group.vm_groups()]

    if not slice_ids:
        click.echo("No slices to terminate")
        return

    def terminate_one(slice_id: str) -> tuple[str, Exception | None]:
        try:
            # Find and terminate slice across all groups
            for group in autoscaler_obj.groups.values():
                for vm_group in group.vm_groups():
                    if vm_group.slice_id == slice_id:
                        group.scale_down(slice_id)
                        return slice_id, None
            return slice_id, ValueError(f"Slice {slice_id} not found")
        except Exception as e:
            return slice_id, e

    click.echo(f"Terminating {len(slice_ids)} slice(s) in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slice_ids)) as executor:
        futures = {executor.submit(terminate_one, sid): sid for sid in slice_ids}
        for future in concurrent.futures.as_completed(futures):
            slice_id, error = future.result()
            if error:
                click.echo(f"Failed to terminate {slice_id}: {error}", err=True)
            else:
                click.echo(f"Terminated: {slice_id}")

    click.echo("Cluster stopped")


@cluster.command("restart")
@click.pass_context
def cluster_restart(ctx):
    """Restart cluster by stopping then starting.

    This is equivalent to running:
        iris cluster --config=... stop
        iris cluster --config=... start

    Examples:
        uv run iris cluster --config=examples/demo.yaml restart
    """
    ctx.invoke(cluster_stop)
    click.echo("")
    ctx.invoke(cluster_start)


@cluster.command("status")
@click.pass_context
def cluster_status_cmd(ctx):
    """Show cluster status including controller and autoscaler.

    Examples:
        uv run iris cluster --config=examples/demo.yaml status
    """
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    # Get controller status
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

    # Get autoscaler status
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


# =============================================================================
# Build Commands
# =============================================================================


@iris.group()
def build():
    """Image build commands."""


@build.command("worker-image")
@click.option("--tag", "-t", default="iris-worker:latest", help="Image tag (e.g., gcr.io/project/iris-worker:v1)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option(
    "--dockerfile",
    type=click.Path(exists=True),
    help="Custom Dockerfile path (defaults to Dockerfile.worker in iris root)",
)
@click.option("--context", type=click.Path(exists=True), help="Build context directory (defaults to iris root)")
@click.option("--platform", default="linux/amd64", help="Target platform (e.g., linux/amd64, linux/arm64)")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to (can be repeated)")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_worker_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris worker.

    This builds a Docker image containing the Iris worker service that can be
    deployed on VMs managed by the autoscaler. Run from the iris directory.

    Examples:
        # Build with default tag (from iris directory)
        uv run iris build worker-image

        # Build and push to GCP Artifact Registry
        uv run iris build worker-image -t iris-worker:v1 --push --region us-central1

        # Build for multiple regions
        uv run iris build worker-image -t iris-worker:v1 \\
            --push --region us-central1 --region europe-west4
    """
    build_image("worker", tag, push, dockerfile, context, platform, region, project)


@build.command("controller-image")
@click.option(
    "--tag", "-t", default="iris-controller:latest", help="Image tag (e.g., gcr.io/project/iris-controller:v1)"
)
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option(
    "--dockerfile",
    type=click.Path(exists=True),
    help="Custom Dockerfile path (defaults to Dockerfile.controller in iris root)",
)
@click.option("--context", type=click.Path(exists=True), help="Build context directory (defaults to iris root)")
@click.option("--platform", default="linux/amd64", help="Target platform (e.g., linux/amd64, linux/arm64)")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to (can be repeated)")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_controller_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris controller.

    This builds a Docker image containing the Iris controller service that manages
    the cluster lifecycle. Run from the iris directory.

    Examples:
        # Build with default tag (from iris directory)
        uv run iris build controller-image

        # Build and push to GCP Artifact Registry
        uv run iris build controller-image -t iris-controller:v1 --push --region us-central1

        # Build for multiple regions
        uv run iris build controller-image -t iris-controller:v1 \\
            --push --region us-central1 --region europe-west4
    """
    build_image("controller", tag, push, dockerfile, context, platform, region, project)


@build.command("push")
@click.argument("source_tag")
@click.option("--region", "-r", multiple=True, required=True, help="GCP Artifact Registry region (can be repeated)")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
@click.option("--image-name", default="iris-worker", help="Image name in registry")
@click.option("--version", default="latest", help="Version tag")
def build_push(source_tag: str, region: tuple[str, ...], project: str, image_name: str, version: str):
    """Push a local Docker image to GCP Artifact Registry.

    Examples:
        # Push to single region
        iris build push iris-worker:v1 -r us-central1

        # Push to multiple regions
        iris build push iris-worker:v1 -r us-central1 -r europe-west4 -r asia-northeast1
    """
    push_to_registries(
        source_tag,
        region,
        project,
        image_name=image_name,
        version=version,
    )


if __name__ == "__main__":
    iris()
