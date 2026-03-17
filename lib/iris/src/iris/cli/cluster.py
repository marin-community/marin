# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster management CLI commands.

All cluster subcommands live here: lifecycle (start/stop/restart/status),
controller VM management, VM operations via controller RPC, and the dashboard tunnel.
"""

import signal
import threading

import click
from connectrpc.errors import ConnectError

from iris.cli.build import (
    build_image,
    find_marin_root,
    get_git_sha,
    push_to_ghcr,
)
from iris.cli.main import require_controller_url
from iris.cluster.config import IrisConfig, make_local_config
from iris.rpc import cluster_connect, cluster_pb2, vm_pb2
from iris.rpc.proto_utils import format_accelerator_display, vm_state_name
from iris.time_utils import Timestamp

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
        counts = dict(group.slice_state_counts)
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


def _get_worker_status(controller_url: str, worker_id: str) -> cluster_pb2.Controller.GetWorkerStatusResponse:
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetWorkerStatusRequest(id=worker_id)
    return client.get_worker_status(request)


def _parse_ghcr_tag(image_tag: str) -> tuple[str, str, str] | None:
    """Parse ``ghcr.io/ORG/IMAGE:VERSION``. Returns (org, image_name, version) or None."""
    if not image_tag.startswith("ghcr.io/"):
        return None
    parts = image_tag.removeprefix("ghcr.io/").split("/")
    if len(parts) < 2:
        return None
    org = parts[0]
    image_and_version = parts[1]
    if ":" in image_and_version:
        image_name, version = image_and_version.split(":", 1)
    else:
        image_name = image_and_version
        version = "latest"
    return org, image_name, version


def _build_and_push_for_tag(image_tag: str, image_type: str, verbose: bool = False) -> None:
    """Build and push a single image to GHCR, parsing org/name/version from the tag."""
    ghcr_parsed = _parse_ghcr_tag(image_tag)
    if not ghcr_parsed:
        raise click.ClickException(f"Unrecognized image tag format (expected ghcr.io/...): {image_tag}")

    org, image_name, version = ghcr_parsed
    local_tag = f"{image_name}:{version}"
    click.echo(f"Building {image_type} image: {local_tag}")
    click.echo(f"  Registry: ghcr.io/{org}")
    click.echo()
    build_image(
        image_type=image_type,
        tag=local_tag,
        push=False,
        context=None,
        platform="linux/amd64",
        ghcr_org=org,
        verbose=verbose,
    )
    click.echo()
    push_to_ghcr(local_tag, ghcr_org=org, image_name=image_name, version=version, verbose=verbose)
    click.echo()


def _build_and_push_task_image(task_tag: str, verbose: bool = False) -> None:
    """Build and push the task image to GHCR.

    The task image uses the ``task`` target in the unified Dockerfile and needs the
    marin repo root as build context, so it can't use _build_and_push_for_tag directly.
    """
    marin_root = str(find_marin_root())

    ghcr_parsed = _parse_ghcr_tag(task_tag)
    if not ghcr_parsed:
        raise click.ClickException(f"Unrecognized image tag format (expected ghcr.io/...): {task_tag}")

    org, image_name, version = ghcr_parsed
    local_tag = f"{image_name}:{version}"
    click.echo(f"Building task image: {local_tag}")
    click.echo(f"  Registry: ghcr.io/{org}")
    click.echo()
    build_image(
        image_type="task",
        tag=local_tag,
        push=False,
        context=marin_root,
        platform="linux/amd64",
        ghcr_org=org,
        verbose=verbose,
    )
    click.echo()
    push_to_ghcr(local_tag, ghcr_org=org, image_name=image_name, version=version, verbose=verbose)
    click.echo()


def _build_cluster_images(config, verbose: bool = False) -> dict[str, str]:
    built: dict[str, str] = {}

    for tag, typ in [(config.defaults.worker.docker_image, "worker"), (config.controller.image, "controller")]:
        if tag:
            _build_and_push_for_tag(tag, typ, verbose=verbose)
            built[typ] = tag

    task_tag = config.defaults.worker.default_task_image
    if task_tag:
        _build_and_push_task_image(task_tag, verbose=verbose)
        built["task"] = task_tag

    return built


def _pin_latest_images(config) -> dict[str, str]:
    """Pin :latest image tags to the current git SHA in memory only."""

    def _pin_tag(tag: str | None, git_sha: str) -> str | None:
        if not tag:
            return tag
        if tag.endswith(":latest"):
            return f"{tag.removesuffix(':latest')}:{git_sha}"
        return tag

    tags = {
        "controller": config.controller.image,
        "worker": config.defaults.worker.docker_image,
        "task": config.defaults.worker.default_task_image,
    }
    needs_pin = any(tag.endswith(":latest") for tag in tags.values() if tag)
    if not needs_pin:
        return {k: v for k, v in tags.items() if v}

    git_sha = get_git_sha()
    pinned = {name: _pin_tag(tag, git_sha) for name, tag in tags.items()}

    if pinned["controller"]:
        config.controller.image = pinned["controller"]
    if pinned["worker"]:
        config.defaults.worker.docker_image = pinned["worker"]
    if pinned["task"]:
        config.defaults.worker.default_task_image = pinned["task"]

    click.echo("Pinning :latest image tags to git SHA for this run:")
    for name, tag in pinned.items():
        if tag:
            click.echo(f"  {name}: {tag}")

    return {k: v for k, v in pinned.items() if v}


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
    """Start controller and wait for health.

    Each platform handles its own controller lifecycle:
    - GCP: builds images, creates GCE VM, SSHes in, bootstraps
    - CoreWeave: kubectl apply ConfigMap + NodePool + Deployment + Service
    - Local: starts in-process controller

    Use --local to create a local cluster for testing that mimics the original config.
    """
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster start")
    if local:
        config = make_local_config(config)
    is_local = config.controller.WhichOneof("controller") == "local"
    if not is_local:
        _pin_latest_images(config)
        verbose = ctx.obj.get("verbose", False)
        built = _build_cluster_images(config, verbose=verbose)
        if built:
            click.echo("Built image tags:")
            for name, tag in built.items():
                click.echo(f"  {name}: {tag}")
    click.echo("Starting controller...")
    try:
        if is_local:
            from iris.cluster.local_cluster import LocalCluster

            cluster = LocalCluster(config)
            address = cluster.start()
            click.echo(f"Controller started at {address}")
            token = cluster.auto_login_token
            if token:
                click.echo(f"Dashboard: {address}?session_token={token}")
            else:
                click.echo(f"Dashboard: {address}")
            click.echo("\nController is running with integrated autoscaler.")
            click.echo("Press Ctrl+C to stop.")
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, lambda *_: cluster.close())
                signal.signal(signal.SIGTERM, lambda *_: cluster.close())
            cluster.wait()
        else:
            iris_config = IrisConfig(config)
            platform = iris_config.platform()
            address = platform.start_controller(config)
            click.echo(f"Controller started at {address}")
            click.echo("\nController is running with integrated autoscaler.")
            click.echo("Use 'iris --config=... cluster status' to check cluster state.")
    except Exception as e:
        click.echo(f"Failed to start controller: {e}", err=True)
        raise SystemExit(1) from e


@cluster.command("stop")
@click.option("--dry-run/--no-dry-run", default=False, help="Show what would be deleted without deleting")
@click.option("--label", "label_override", default=None, help="Label prefix override (default from config or 'iris')")
@click.pass_context
def cluster_stop(ctx, dry_run: bool, label_override: str | None):
    """Stop controller and terminate all slices."""
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster stop")

    if dry_run:
        click.echo("Scanning for resources (dry-run)...")
    else:
        click.echo("Stopping cluster (controller + all slices)...")

    try:
        iris_config = IrisConfig(config)
        platform = iris_config.platform()
        try:
            names = platform.stop_all(config, dry_run=dry_run, label_prefix=label_override)
        finally:
            platform.shutdown()
    except Exception as e:
        click.echo(f"Failed to stop cluster: {e}", err=True)
        raise SystemExit(1) from e

    if dry_run:
        if not names:
            click.echo("Nothing to clean up.")
        else:
            click.echo(f"Would delete {len(names)} resource(s):")
            for n in names:
                click.echo(f"  - {n}")
    else:
        click.echo("Cluster stopped")


@cluster.command("restart")
@click.pass_context
def cluster_restart(ctx):
    """Restart cluster by stopping then starting."""
    ctx.invoke(cluster_stop)
    click.echo("")
    ctx.invoke(cluster_start)


@cluster.command("status")
@click.pass_context
def cluster_status_cmd(ctx):
    """Show cluster status including controller and autoscaler."""
    controller_url = require_controller_url(ctx)
    click.echo("Checking controller status...")
    try:
        client = cluster_connect.ControllerServiceClientSync(controller_url)
        proc = client.get_process_status(cluster_pb2.GetProcessStatusRequest()).process_info
        workers = client.list_workers(cluster_pb2.Controller.ListWorkersRequest()).workers
        as_status = client.get_autoscaler_status(cluster_pb2.Controller.GetAutoscalerStatusRequest()).status
        healthy = sum(1 for w in workers if w.healthy)
        click.echo("Controller Status:")
        click.echo("  Running: True")
        click.echo("  Healthy: True")
        click.echo(f"  Address: {controller_url}")
        click.echo(f"  Git Hash: {proc.git_hash}")
        click.echo(f"  Workers: {healthy}/{len(workers)} healthy")
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
    controller_url = require_controller_url(ctx)
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


@cluster.command("dashboard-proxy")
@click.option("--port", default=8080, type=int, help="Local port to serve the dashboard on")
@click.pass_context
def cluster_dashboard_proxy(ctx, port: int):
    """Start a local dashboard that proxies RPC calls to the remote controller.

    Serves the Vue dashboard UI locally and forwards all Connect RPC requests
    to the upstream controller. Useful for viewing a remote controller without
    SSH tunneling. Rebuilds dashboard assets on each run.
    """
    import uvicorn

    from iris.cli.build import _ensure_dashboard_dist
    from iris.cluster.controller.dashboard import ProxyControllerDashboard

    # Rebuild dashboard assets so the proxy always serves the latest UI.
    _ensure_dashboard_dist()

    controller_url = require_controller_url(ctx)
    dashboard = ProxyControllerDashboard(upstream_url=controller_url, port=port)
    click.echo(f"Proxying to controller at {controller_url}")
    click.echo(f"Dashboard: http://localhost:{port}")
    uvicorn.run(dashboard.app, host="127.0.0.1", port=port, log_level="info")


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
    controller_url = require_controller_url(ctx)
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
        counts = dict(group.slice_state_counts)
        total = sum(counts.values())
        click.echo(f"\nScale Group: {group.name}")
        accel_display = format_accelerator_display(
            group.config.resources.device_type, group.config.resources.device_variant
        )
        click.echo(f"  Accelerator: {accel_display}")
        click.echo(f"  Slices: {counts.get('ready', 0)}/{total} ready")
        click.echo(f"    Booting: {counts.get('booting', 0)}")
        click.echo(f"    Initializing: {counts.get('initializing', 0)}")
        click.echo(f"    Failed: {counts.get('failed', 0)}")
        click.echo(f"  Demand: {group.current_demand} (peak: {group.peak_demand})")
        backoff_ms = Timestamp.from_proto(group.backoff_until).epoch_ms()
        if backoff_ms > 0:
            click.echo(f"  Backoff until: {_format_timestamp(backoff_ms)}")
            click.echo(f"  Consecutive failures: {group.consecutive_failures}")
        if group.slices:
            click.echo("  Slices:")
            for si in group.slices:
                all_ready = bool(si.vms) and all(vm.state == vm_pb2.VM_STATE_READY for vm in si.vms)
                any_failed = any(vm.state in (vm_pb2.VM_STATE_FAILED, vm_pb2.VM_STATE_PREEMPTED) for vm in si.vms)
                ss = "READY" if all_ready else ("FAILED" if any_failed else "PENDING")
                click.echo(f"    {si.slice_id}: {ss}")
                for vi in si.vms:
                    click.echo(f"      {vi.vm_id}: {vm_state_name(vi.state)} ({vi.address})")
                    if vi.init_error:
                        click.echo(f"        Error: {vi.init_error}")
    last_eval_ms = Timestamp.from_proto(as_status.last_evaluation).epoch_ms()
    click.echo(f"\nLast evaluation: {_format_timestamp(last_eval_ms)}")


@vm.command("logs")
@click.argument("vm_id")
@click.pass_context
def vm_logs(ctx, vm_id):
    """Show VM initialization logs."""
    controller_url = require_controller_url(ctx)
    try:
        resp = _get_worker_status(controller_url, vm_id)
    except ConnectError as e:
        from connectrpc.code import Code

        if e.code == Code.NOT_FOUND:
            click.echo(f"Worker not found: {vm_id}", err=True)
        else:
            click.echo(f"Error fetching status: {e}", err=True)
        raise SystemExit(1) from None
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None
    if resp.vm and resp.vm.vm_id:
        click.echo(f"VM: {resp.vm.vm_id}")
        click.echo(f"State: {vm_state_name(resp.vm.state)}")
    if resp.worker and resp.worker.worker_id:
        click.echo(f"Worker: {resp.worker.worker_id}")
        click.echo(f"Healthy: {resp.worker.healthy}")
    click.echo("---")
    click.echo(resp.bootstrap_logs if resp.bootstrap_logs else "(no bootstrap logs available)")


# =============================================================================
# Controller subcommands (RPC-based controller operations)
# =============================================================================


@cluster.group()
@click.pass_context
def controller(ctx):
    """Controller management commands."""
    pass


@controller.command("checkpoint")
@click.option("--stop", is_flag=True, default=False, help="Stop the controller after taking a checkpoint")
@click.pass_context
def controller_checkpoint(ctx, stop: bool):
    """Take a checkpoint of the controller state.

    Calls BeginCheckpoint on the running controller, which pauses scheduling
    briefly and writes a consistent checkpoint DB copy.
    """
    controller_url = require_controller_url(ctx)
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    try:
        resp = client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    except Exception as e:
        click.echo(f"Checkpoint failed: {e}", err=True)
        raise SystemExit(1) from e

    click.echo(f"Checkpoint DB written: {resp.checkpoint_path}")
    click.echo(f"  Jobs:    {resp.job_count}")
    click.echo(f"  Tasks:   {resp.task_count}")
    click.echo(f"  Workers: {resp.worker_count}")

    if stop:
        click.echo("Stopping controller...")
        config = ctx.obj.get("config")
        if not config:
            click.echo("--stop requires --config", err=True)
            raise SystemExit(1)
        from iris.cluster.config import IrisConfig

        iris_config = IrisConfig(config)
        platform = iris_config.platform()
        try:
            platform.stop_controller(config)
            click.echo("Controller stopped.")
        except Exception as e:
            click.echo(f"Failed to stop controller: {e}", err=True)
            raise SystemExit(1) from e


@controller.command("restart")
@click.pass_context
def controller_restart(ctx):
    """Restart controller with state preservation (remote platforms only).

    Takes a checkpoint, builds fresh images, stops the controller, and starts
    a new one. The new controller auto-restores from the checkpoint.
    Workers on separate VMs survive the restart.
    """
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required")

    is_local = config.controller.WhichOneof("controller") == "local"
    if is_local:
        raise click.ClickException(
            "controller restart is not supported for local clusters. "
            "Stop and restart the 'iris cluster start --local' process instead."
        )

    controller_url = require_controller_url(ctx)

    # Checkpoint
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    try:
        resp = client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    except Exception as e:
        click.echo(f"Checkpoint failed: {e}", err=True)
        raise SystemExit(1) from e
    finally:
        client.close()
    click.echo(f"Checkpoint: {resp.checkpoint_path} ({resp.job_count} jobs, {resp.worker_count} workers)")

    # Build fresh images so the new controller VM gets the latest code
    _pin_latest_images(config)
    verbose = ctx.obj.get("verbose", False)
    built = _build_cluster_images(config, verbose=verbose)
    if built:
        click.echo("Built image tags:")
        for name, tag in built.items():
            click.echo(f"  {name}: {tag}")

    # Restart controller in-place (re-runs bootstrap on existing VM)
    iris_config = IrisConfig(config)
    platform = iris_config.platform()
    try:
        address = platform.restart_controller(config)
    except Exception as e:
        click.echo(f"Failed to restart controller: {e}", err=True)
        raise SystemExit(1) from e
    click.echo(f"Controller restarted at {address}")
