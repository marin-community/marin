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

All cluster subcommands live here: lifecycle (start/stop/restart/reload/status),
controller VM management, VM operations via controller RPC, and the dashboard tunnel.
"""

import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from connectrpc.errors import ConnectError

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.platform.cluster_manager import ClusterManager
from iris.config import make_local_config
from iris.cluster.controller.slice_lifecycle import compute_slice_state_counts, slice_all_ready, slice_any_failed
from iris.rpc import cluster_connect, cluster_pb2, vm_pb2
from iris.rpc.proto_utils import format_accelerator_display, vm_state_name
from iris.time_utils import Timestamp

from iris.cli.build import _build_image, _push_to_registries
from iris.cli.debug import debug

# =============================================================================
# Helpers
# =============================================================================


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
    return _ImageBuildParams(
        image_type=image_type, region=region, project=project, image_name=image_name, version=version
    )


def _build_and_push_image(params: _ImageBuildParams) -> None:
    click.echo(f"Building {params.image_type} image: {params.local_tag}")
    click.echo(f"  Region: {params.region}")
    click.echo(f"  Project: {params.project}")
    click.echo()
    _build_image(
        image_type=params.image_type,
        tag=params.local_tag,
        push=False,
        dockerfile=None,
        context=None,
        platform="linux/amd64",
        region=(),
        project=params.project,
    )
    click.echo()
    _push_to_registries(
        source_tag=params.local_tag,
        regions=(params.region,),
        project=params.project,
        image_name=params.image_name,
        version=params.version,
    )


def _build_cluster_images(config) -> None:
    for tag, typ in [(config.defaults.bootstrap.docker_image, "worker"), (config.controller.image, "controller")]:
        if tag:
            params = _extract_image_params(tag, typ)
            if params:
                _build_and_push_image(params)
                click.echo()


# =============================================================================
# Top-level cluster group
# =============================================================================


@click.group()
@click.pass_context
def cluster(ctx):
    """Cluster management commands."""
    parent_obj = ctx.obj or {}
    ctx.ensure_object(dict)
    ctx.obj.update(parent_obj)


# =============================================================================
# Cluster lifecycle commands
# =============================================================================


@cluster.command("start")
@click.option("--local", is_flag=True, help="Create a local cluster for testing that mimics the original config")
@click.pass_context
def cluster_start(ctx, local: bool):
    """Start controller VM and wait for health.

    Builds and pushes images, then boots the controller GCE VM.
    Use --local to create a local cluster for testing that mimics the original config.
    """
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster start")
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
@click.option("--zone", help="Only stop slices in a specific zone")
@click.pass_context
def cluster_stop(ctx, zone: str | None):
    """Stop controller and terminate all slices."""
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster stop")
    manager = ClusterManager(config)
    click.echo("Stopping controller and terminating slices...")
    result = manager.stop_cluster(zone=zone)

    if not result.discovered:
        click.echo("No slices to terminate")
        click.echo("Cluster stopped")
        return

    total = sum(len(ids) for per_zone in result.discovered.values() for ids in per_zone.values())
    click.echo(f"Discovered {total} slice(s)")

    for name, per_zone in result.discovered.items():
        for zone_name, slice_ids in per_zone.items():
            zone_label = zone_name or "default"
            click.echo(f"  Group '{name}' (zone={zone_label}): {len(slice_ids)} slice(s)")

            failures = result.failed.get(name, {}).get(zone_name, {})
            for slice_id in slice_ids:
                if slice_id in failures:
                    click.echo(f"Failed to terminate {slice_id}: {failures[slice_id]}", err=True)
                else:
                    click.echo(f"Terminated: {slice_id}")
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
        raise click.ClickException("--config is required for cluster reload")
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
        controller_url = ctx.obj.get("controller_url")
        if not controller_url:
            click.echo("Controller URL not available in context", err=True)
            raise SystemExit(1)
        try:
            _validate_cluster_health(controller_url)
            click.echo("Cluster validation passed.")
        except Exception as e:
            click.echo(f"Cluster validation failed: {e}", err=True)
            raise SystemExit(1) from e


@cluster.command("status")
@click.pass_context
def cluster_status_cmd(ctx):
    """Show cluster status including controller and autoscaler."""
    controller_url = ctx.obj.get("controller_url")
    if not controller_url:
        raise click.ClickException("--config or --controller-url required")

    click.echo("Checking controller status...")
    try:
        as_status = _get_autoscaler_status(controller_url)
        click.echo("Controller Status:")
        click.echo("  Running: True")
        click.echo("  Healthy: True")
        click.echo(f"  Address: {controller_url}")
        click.echo("\nAutoscaler Status:")
        if not as_status.groups:
            click.echo("  No scale groups configured")
        else:
            click.echo(_format_status_table(as_status))
    except Exception as e:
        click.echo("Controller Status:")
        click.echo(f"  Running: False (RPC failed: {e})")
        click.echo(f"  Address: {controller_url}")


@cluster.command("dashboard")
@click.pass_context
def cluster_dashboard(ctx):
    """Print dashboard URL and keep tunnel open.

    Uses the tunnel established by the iris group. Blocks until Ctrl+C.
    """
    controller_url = ctx.obj.get("controller_url")
    if not controller_url:
        raise click.ClickException("--config or --controller-url required on iris group")

    stop = threading.Event()

    def on_signal(sig, frame):
        click.echo("\nClosing tunnel...")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    click.echo(f"\nDashboard:      {controller_url}")
    click.echo(f"Controller RPC: {controller_url}")
    click.echo("\nPress Ctrl+C to close tunnel.")
    stop.wait()


# =============================================================================
# VM subcommands (always via controller RPC)
# =============================================================================


@cluster.group()
@click.pass_context
def vm(ctx):
    """VM management commands (via controller RPC)."""
    pass


@vm.command("status")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.pass_context
def vm_status(ctx, scale_group):
    """Show VM and slice status from the controller."""
    controller_url = ctx.obj.get("controller_url")
    if not controller_url:
        raise click.ClickException("Either --controller-url or --config is required")
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
        accel_display = format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)
        click.echo(f"  Accelerator: {accel_display}")
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


@vm.command("logs")
@click.argument("vm_id")
@click.option("--tail", type=int, default=0, help="Show last N lines (0 = all)")
@click.pass_context
def vm_logs(ctx, vm_id, tail):
    """Show VM initialization logs."""
    controller_url = ctx.obj.get("controller_url")
    if not controller_url:
        raise click.ClickException("Either --controller-url or --config is required")
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


# =============================================================================
# Register debug sub-group
# =============================================================================

cluster.add_command(debug)


# =============================================================================
# Internal helpers
# =============================================================================


def _validate_cluster_health(controller_url: str) -> None:
    click.echo(f"  Connected to controller at {controller_url}")
    client = IrisClient.remote(controller_url, workspace=Path.cwd())

    def _validate_hello():
        print("Reload validation job OK")
        return 42

    click.echo("  Submitting validation job...")
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_validate_hello), name="reload-validate", resources=ResourceSpec(cpu=1)
    )
    click.echo(f"  Job submitted: {job.job_id}")
    click.echo("  Waiting for job (workers may need to scale up)...")
    status = job.wait(timeout=600, raise_on_failure=True)
    click.echo(f"  Job completed: {cluster_pb2.JobState.Name(status.state)}")
