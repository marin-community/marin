# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster management CLI commands.

All cluster subcommands live here: lifecycle (start/stop/restart/status),
controller VM management, VM operations via controller RPC, and the dashboard tunnel.
"""

import signal
import threading
import time
from pathlib import Path

import click
from connectrpc.errors import ConnectError

from iris.cli.build import (
    build_image,
    find_marin_root,
    get_git_sha,
)
from iris.cli.main import IRIS_CLUSTER_CONFIG_DIRS, require_controller_url, rpc_client
from iris.cluster.config import IrisConfig, clear_remote_state, make_local_config
from iris.rpc import config_pb2
from iris.rpc import vm_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc.proto_utils import format_accelerator_display, vm_state_name
from iris.time_proto import timestamp_from_proto
from rigging.config_discovery import list_cluster_configs
from rigging.timing import Duration, ExponentialBackoff, Timestamp

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
    with rpc_client(controller_url) as client:
        request = controller_pb2.Controller.GetAutoscalerStatusRequest()
        return client.get_autoscaler_status(request).status


def _get_worker_status(controller_url: str, worker_id: str) -> controller_pb2.Controller.GetWorkerStatusResponse:
    with rpc_client(controller_url) as client:
        request = controller_pb2.Controller.GetWorkerStatusRequest(id=worker_id)
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
        push=True,
        context=None,
        platform="linux/amd64",
        ghcr_org=org,
        verbose=verbose,
    )
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
        push=True,
        context=marin_root,
        platform="linux/amd64",
        ghcr_org=org,
        verbose=verbose,
    )
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


@cluster.command("list")
def cluster_list():
    """List available cluster configurations."""
    configs = list_cluster_configs(dirs=IRIS_CLUSTER_CONFIG_DIRS)
    if not configs:
        click.echo("No cluster configurations found.")
        return
    click.echo("Available clusters:")
    for name, path in sorted(configs.items()):
        click.echo(f"  {name:30s} {path}")


@cluster.command("start")
@click.option("--local", is_flag=True, help="Create a local cluster for testing that mimics the original config")
@click.option(
    "--fresh", is_flag=True, default=False, help="Start with an empty database, ignoring any remote checkpoint"
)
@click.pass_context
def cluster_start(ctx, local: bool, fresh: bool):
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
            from iris.cluster.providers.local.cluster import LocalCluster

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
            bundle = iris_config.provider_bundle()
            address = bundle.controller.start_controller(config, fresh=fresh)
            click.echo(f"Controller started at {address}")
            click.echo("\nController is running with integrated autoscaler.")
            click.echo("Use 'iris --config=... cluster status' to check cluster state.")
    except Exception as e:
        click.echo(f"Failed to start controller: {e}", err=True)
        raise SystemExit(1) from e


@cluster.command("start-smoke")
@click.option("--label-prefix", required=True, help="Label prefix to isolate GCP resources")
@click.option("--url-file", required=True, type=click.Path(), help="Write tunnel URL to this file when ready")
@click.option("--wait-for-workers", "min_workers", type=int, default=1, help="Min healthy workers before writing URL")
@click.option("--worker-timeout", type=int, default=600, help="Seconds to wait for workers")
@click.option("--clear-state/--no-clear-state", default=True, help="Wipe remote state before starting")
@click.pass_context
def cluster_start_smoke(ctx, label_prefix, url_file, min_workers, worker_timeout, clear_state):
    """Boot a smoke-test cluster, open tunnel, write URL to file, and block until killed.

    Designed for CI: run in background, poll for url_file, then pass URL to pytest.
    SIGINT/SIGTERM cleanly close the tunnel.
    """
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for start-smoke")

    config.platform.label_prefix = label_prefix

    # Set ephemeral state dir via marin_temp_bucket, which resolves
    # region-appropriate storage from MARIN_PREFIX.
    from rigging.filesystem import marin_temp_bucket

    config.storage.remote_state_dir = marin_temp_bucket(ttl_days=7, prefix=f"iris/state/{label_prefix}")

    _pin_latest_images(config)
    verbose = ctx.obj.get("verbose", False)
    _build_cluster_images(config, verbose=verbose)

    iris_config = IrisConfig(config)
    bundle = iris_config.provider_bundle()

    try:
        bundle.controller.stop_all(config)
    except Exception:
        click.echo("No existing cluster to stop, continuing")

    if clear_state:
        remote_state_dir = config.storage.remote_state_dir
        if remote_state_dir:
            click.echo(f"Clearing remote state: {remote_state_dir}")
            clear_remote_state(remote_state_dir)

    click.echo("Starting controller...")
    address = bundle.controller.start_controller(config)
    click.echo(f"Controller at {address}")

    stop_event = threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    try:
        with bundle.controller.tunnel(address) as url:
            click.echo(f"Tunnel ready: {url}")

            with rpc_client(url) as client:
                deadline = time.monotonic() + worker_timeout
                healthy_count = 0
                while time.monotonic() < deadline:
                    workers = client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
                    healthy = [w for w in workers if w.healthy]
                    healthy_count = len(healthy)
                    if healthy_count >= min_workers:
                        break
                    time.sleep(2)
                else:
                    raise click.ClickException(
                        f"Only {healthy_count} of {min_workers} workers healthy after {worker_timeout}s"
                    )

            click.echo(f"{healthy_count} workers ready, writing URL to {url_file}")
            Path(url_file).write_text(url)

            stop_event.wait()
    finally:
        click.echo("Shutting down (tunnel closed)")


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
        bundle = iris_config.provider_bundle()
        try:
            names = bundle.controller.stop_all(config, dry_run=dry_run, label_prefix=label_override)
        finally:
            bundle.controller.shutdown()
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


@cluster.command("create-slice")
@click.option("--scale-group", "scale_group_name", required=True, help="Scale group whose template to use")
@click.pass_context
def cluster_create_slice(ctx, scale_group_name: str):
    """Create an operator-managed slice bound to the running controller.

    Allocates a slice using the named scale group's template, tags it with
    ``iris-{prefix}-manual=true``, and bootstraps workers so they connect to
    the controller. The autoscaler ignores manual slices: they don't count
    toward demand, won't be scaled down on idle, and survive
    ``iris cluster stop``. Remove with ``iris cluster delete-slice``.
    """
    from iris.cluster.controller.autoscaler.scaling_group import (
        build_worker_config_for_group,
        prepare_slice_config,
    )
    from iris.cluster.providers.types import Labels

    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster create-slice")
    if config.controller.WhichOneof("controller") == "local":
        raise click.ClickException("create-slice is not supported for local clusters")

    sg_config = config.scale_groups.get(scale_group_name)
    if sg_config is None:
        available = ", ".join(sorted(config.scale_groups.keys())) or "(none)"
        raise click.ClickException(f"Unknown scale group '{scale_group_name}'. Available: {available}")

    # Verify the controller is reachable before creating the slice. The
    # returned URL may be a tunnel endpoint that's only reachable from the CLI
    # host; workers need the cluster-internal address instead, resolved below.
    require_controller_url(ctx)
    iris_config = IrisConfig(config)
    bundle = ctx.obj.get("provider_bundle") or iris_config.provider_bundle()

    # Resolve the address workers will connect to. Prefer an explicit value in
    # defaults.worker.controller_address, then discover it via the provider
    # (e.g., GCE label lookup). Never pass the CLI-local tunnel URL here.
    worker_controller_address = iris_config.controller_address()
    if not worker_controller_address:
        worker_controller_address = bundle.controller.discover_controller(config.controller)

    label_prefix = config.platform.label_prefix or "iris"
    labels = Labels(label_prefix)

    slice_config = prepare_slice_config(sg_config.slice_template, sg_config, label_prefix)
    slice_config.labels[labels.iris_manual] = "true"

    base_worker_config = config_pb2.WorkerConfig()
    base_worker_config.CopyFrom(config.defaults.worker)
    if not base_worker_config.controller_address:
        base_worker_config.controller_address = worker_controller_address
    base_worker_config.platform.CopyFrom(config.platform)
    if config.storage.remote_state_dir:
        base_worker_config.storage_prefix = config.storage.remote_state_dir

    worker_config = build_worker_config_for_group(base_worker_config, sg_config)

    click.echo(f"Creating manual slice from scale group '{scale_group_name}'...")
    try:
        handle = bundle.workers.create_slice(slice_config, worker_config=worker_config)
    except Exception as e:
        click.echo(f"Failed to create slice: {e}", err=True)
        raise SystemExit(1) from e

    click.echo(f"Created manual slice: {handle.slice_id}")
    click.echo(f"  Scale group: {handle.scale_group or scale_group_name}")
    click.echo(f"  Zone:        {handle.zone}")
    click.echo("Workers will register with the controller as they bootstrap.")
    click.echo(f"Terminate with: iris cluster delete-slice {handle.slice_id}")


@cluster.command("delete-slice")
@click.argument("slice_id")
@click.pass_context
def cluster_delete_slice(ctx, slice_id: str):
    """Terminate an operator-managed slice created via ``create-slice``.

    Only slices tagged ``iris-{prefix}-manual=true`` are eligible —
    autoscaler-managed slices must go through the autoscaler.
    """
    from iris.cluster.providers.types import Labels

    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster delete-slice")
    if config.controller.WhichOneof("controller") == "local":
        raise click.ClickException("delete-slice is not supported for local clusters")

    iris_config = IrisConfig(config)
    bundle = ctx.obj.get("provider_bundle") or iris_config.provider_bundle()

    label_prefix = config.platform.label_prefix or "iris"
    labels = Labels(label_prefix)

    manual_slices = bundle.workers.list_slices(zones=[], labels={labels.iris_manual: "true"})
    match = next((s for s in manual_slices if s.slice_id == slice_id), None)
    if match is None:
        raise click.ClickException(
            f"No manual slice found with id '{slice_id}'. "
            "List manual slices with the controller dashboard or "
            "`iris cluster status` to find the correct id."
        )

    click.echo(f"Terminating manual slice {slice_id}...")
    try:
        match.terminate()
    except Exception as e:
        click.echo(f"Failed to terminate slice: {e}", err=True)
        raise SystemExit(1) from e
    click.echo("Terminated.")


@cluster.command("status")
@click.pass_context
def cluster_status_cmd(ctx):
    """Show cluster status including controller and autoscaler."""
    controller_url = require_controller_url(ctx)
    click.echo("Checking controller status...")
    try:
        with rpc_client(controller_url) as client:
            proc = client.get_process_status(job_pb2.GetProcessStatusRequest()).process_info
            workers = client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
            as_status = client.get_autoscaler_status(controller_pb2.Controller.GetAutoscalerStatusRequest()).status
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
@click.option("--port", default=8080, type=int, help="Local port for the RPC proxy")
@click.pass_context
def cluster_dashboard_proxy(ctx, port: int):
    """Start a local dev dashboard that proxies RPC calls to the remote controller.

    Runs the rsbuild dev server (with hot module replacement) for the Vue
    frontend alongside a Python proxy that forwards Connect RPC requests to
    the upstream controller. Open the rsbuild URL (printed on startup) in
    your browser.
    """
    import signal
    import subprocess

    import uvicorn

    from iris.cluster.controller.dashboard import ProxyControllerDashboard
    from iris.cluster.dashboard_common import VUE_DIST_DIR

    controller_url = require_controller_url(ctx)
    dashboard = ProxyControllerDashboard(upstream_url=controller_url, port=port)
    click.echo(f"Proxying to controller at {controller_url}")

    dashboard_dir = VUE_DIST_DIR.parent
    click.echo(f"Starting rsbuild dev server in {dashboard_dir}")
    click.echo("Installing npm dependencies...")
    subprocess.run(["npm", "ci"], cwd=dashboard_dir, check=True)

    dev_proc = subprocess.Popen(["npm", "run", "dev"], cwd=dashboard_dir)

    def _cleanup(signum, frame):
        dev_proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    click.echo(f"RPC proxy: http://localhost:{port}")
    click.echo("Rsbuild dev server will print its URL above (usually http://localhost:3000)")
    try:
        uvicorn.run(dashboard.app, host="127.0.0.1", port=port, log_level="info")
    finally:
        dev_proc.terminate()
        dev_proc.wait()


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
        backoff_ms = timestamp_from_proto(group.backoff_until).epoch_ms()
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
    last_eval_ms = timestamp_from_proto(as_status.last_evaluation).epoch_ms()
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


@controller.command("serve")
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=10000, type=int, help="Bind port")
@click.option(
    "--checkpoint-path",
    default=None,
    help="Restore from this specific checkpoint directory (e.g. gs://bucket/.../controller-state/1234567890)",
)
@click.option(
    "--checkpoint-interval",
    default=None,
    type=float,
    help="Periodic checkpoint interval in seconds (default: hourly)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Start in dry-run mode: compute scheduling but suppress all side effects",
)
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Start with an empty database, ignoring any remote checkpoint",
)
@click.option(
    "--state-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Override the local state dir (default: /var/cache/iris/controller, or /tmp/dry-run/{today} in dry-run)",
)
@click.pass_context
def controller_serve(ctx, host, port, checkpoint_path, checkpoint_interval, dry_run, fresh, state_dir):
    """Start a local controller process.

    Loads the cluster config, restores from checkpoint, and runs the full
    scheduling loop. Use --dry-run to suppress all side effects (no task
    dispatch, no VM changes, no checkpoint writes) while still serving the
    dashboard and RPC for inspection.

    In --dry-run, the local state dir defaults to ``/tmp/dry-run/{today}``.
    Pass ``--state-dir /tmp/...`` to resume from an existing local state dir
    without re-downloading.

    Example (dry-run with checkpoint restore)::

        iris --config=cluster.yaml cluster controller serve --dry-run \\
            --checkpoint-path gs://bucket/controller-state/1234567890
    """
    from iris.cluster.controller.main import run_controller_serve

    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for controller serve")

    run_controller_serve(
        config,
        host=host,
        port=port,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        dry_run=dry_run,
        fresh=fresh,
        state_dir=state_dir,
    )


@controller.command("checkpoint")
@click.option("--stop", is_flag=True, default=False, help="Stop the controller after taking a checkpoint")
@click.pass_context
def controller_checkpoint(ctx, stop: bool):
    """Take a checkpoint of the controller state.

    Calls BeginCheckpoint on the running controller, which pauses scheduling
    briefly and writes a consistent checkpoint DB copy.
    """
    controller_url = require_controller_url(ctx)
    with rpc_client(controller_url) as client:
        try:
            resp = client.begin_checkpoint(controller_pb2.Controller.BeginCheckpointRequest(), timeout_ms=60_000)
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
        bundle = iris_config.provider_bundle()
        try:
            bundle.controller.stop_controller(config)
            click.echo("Controller stopped.")
        except Exception as e:
            click.echo(f"Failed to stop controller: {e}", err=True)
            raise SystemExit(1) from e


@controller.command("restart")
@click.option(
    "--skip-checkpoint",
    is_flag=True,
    default=False,
    help="Skip the pre-restart checkpoint (use if checkpoint is timing out).",
)
@click.option(
    "--checkpoint-timeout", type=int, default=300, show_default=True, help="Checkpoint RPC timeout in seconds."
)
@click.pass_context
def controller_restart(ctx, skip_checkpoint: bool, checkpoint_timeout: int):
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

    iris_config = IrisConfig(config)
    bundle = iris_config.provider_bundle()

    # Try to discover existing controller for checkpoint + restart.
    # If none exists, fall back to a fresh start (idempotent).
    try:
        controller_url = require_controller_url(ctx)
    except (RuntimeError, click.ClickException):
        click.echo("No existing controller found. Starting fresh...")
        _pin_latest_images(config)
        verbose = ctx.obj.get("verbose", False)
        built = _build_cluster_images(config, verbose=verbose)
        if built:
            click.echo("Built image tags:")
            for name, tag in built.items():
                click.echo(f"  {name}: {tag}")
        try:
            address = bundle.controller.start_controller(config)
        except Exception as e:
            click.echo(f"Failed to start controller: {e}", err=True)
            raise SystemExit(1) from e
        click.echo(f"Controller started at {address}")
        return

    # Checkpoint
    if skip_checkpoint:
        click.echo("Skipping pre-restart checkpoint.")
    else:
        click.echo(f"Taking checkpoint (timeout {checkpoint_timeout}s)...")
        with rpc_client(controller_url) as client:
            try:
                resp = client.begin_checkpoint(
                    controller_pb2.Controller.BeginCheckpointRequest(),
                    timeout_ms=checkpoint_timeout * 1000,
                )
            except Exception as e:
                click.echo(f"Checkpoint failed: {e}", err=True)
                raise SystemExit(1) from e
        click.echo(f"Checkpoint: {resp.checkpoint_path} ({resp.job_count} jobs, {resp.worker_count} workers)")

    # Build fresh images so the new controller VM gets the latest code
    _pin_latest_images(config)
    verbose = ctx.obj.get("verbose", False)
    built = _build_cluster_images(config, verbose=verbose)
    if built:
        click.echo("Built image tags:")
        for name, tag in built.items():
            click.echo(f"  {name}: {tag}")

    try:
        address = bundle.controller.restart_controller(config)
    except Exception as e:
        click.echo(f"Failed to restart controller: {e}", err=True)
        raise SystemExit(1) from e
    click.echo(f"Controller restarted at {address}")


@controller.command("worker-restart")
@click.option("--worker-id", multiple=True, help="Worker(s) to restart (repeatable; default: all)")
@click.option("--timeout", type=int, default=120, help="Max seconds to wait per worker to become healthy")
@click.option("--max-batch", type=int, default=64, help="Maximum workers to restart concurrently")
@click.option(
    "--observation-window",
    type=int,
    default=60,
    help="Seconds to observe restarted workers for failures before advancing",
)
@click.pass_context
def worker_restart(
    ctx,
    worker_id: tuple[str, ...],
    timeout: int,
    max_batch: int,
    observation_window: int,
):
    """Rolling restart of workers with adaptive batch sizing.

    Restarts workers in progressively larger batches (1, 2, 4, ... up to
    --max-batch). After each batch, waits for workers to become healthy, then
    observes them for --observation-window seconds to catch post-restart
    failures. Aborts immediately if any worker fails to come back healthy or
    develops failures during observation.

    Running Docker containers are preserved and adopted by the new worker
    process, so tasks are not disrupted.
    """
    controller_url = require_controller_url(ctx)

    with rpc_client(controller_url) as client:
        workers_resp = client.list_workers(controller_pb2.Controller.ListWorkersRequest())
        all_workers = workers_resp.workers

        if worker_id:
            requested = set(worker_id)
            workers = [w for w in all_workers if w.worker_id in requested]
            missing = requested - {w.worker_id for w in workers}
            if missing:
                click.echo(f"Workers not found: {', '.join(sorted(missing))}", err=True)
                raise SystemExit(1)
        else:
            workers = list(all_workers)

        if not workers:
            click.echo("No workers to restart")
            return

        worker_ids = [w.worker_id for w in workers]
        total = len(worker_ids)
        click.echo(
            f"Restarting {total} worker(s) "
            f"(timeout={timeout}s, observation={observation_window}s, max_batch={max_batch})"
        )

        succeeded = 0
        batch_size = 1
        offset = 0

        while offset < total:
            batch = worker_ids[offset : offset + batch_size]
            click.echo(f"\n--- Batch of {len(batch)} (workers {offset + 1}-{offset + len(batch)} of {total}) ---")

            # Issue restart RPCs for the batch
            for wid in batch:
                click.echo(f"  Restarting {wid}...")
                resp = client.restart_worker(
                    controller_pb2.Controller.RestartWorkerRequest(worker_id=wid),
                    timeout_ms=timeout * 1000,
                )
                if not resp.accepted:
                    click.echo(f"  ABORT: restart rejected for {wid}: {resp.error}", err=True)
                    _print_summary(succeeded, total - succeeded, offset)
                    raise SystemExit(1)

            # Wait for all workers in the batch to become healthy
            click.echo(f"  Waiting for {len(batch)} worker(s) to become healthy...")
            unhealthy = _wait_for_workers_healthy(client, set(batch), timeout)
            if unhealthy:
                click.echo(
                    f"  ABORT: workers did not become healthy within {timeout}s: " f"{', '.join(sorted(unhealthy))}",
                    err=True,
                )
                _print_summary(succeeded, total - succeeded, offset)
                raise SystemExit(1)

            click.echo(f"  All {len(batch)} worker(s) healthy. Observing for {observation_window}s...")
            time.sleep(observation_window)

            # Re-check health after observation window
            failed_workers = _check_worker_health(client, set(batch))
            if failed_workers:
                click.echo(
                    f"  ABORT: workers developed failures during observation: "
                    f"{', '.join(f'{wid} ({msg})' for wid, msg in sorted(failed_workers))}",
                    err=True,
                )
                _print_summary(succeeded, total - succeeded, offset)
                raise SystemExit(1)

            succeeded += len(batch)
            offset += len(batch)
            click.echo(f"  Batch OK ({succeeded}/{total} complete)")

            # Double batch size for next round, capped at max_batch
            batch_size = min(batch_size * 2, max_batch)

    click.echo(f"\nDone: {succeeded}/{total} workers restarted successfully")


def _wait_for_workers_healthy(client, worker_ids: set[str], timeout: int) -> set[str]:
    """Poll until all workers in the set are healthy. Returns IDs that failed to become healthy."""
    remaining = set(worker_ids)
    backoff = ExponentialBackoff(initial=5.0, maximum=5.0, jitter=0.0)

    def _all_healthy() -> bool:
        try:
            resp = client.list_workers(controller_pb2.Controller.ListWorkersRequest())
            for w in resp.workers:
                if w.worker_id in remaining and w.healthy:
                    remaining.discard(w.worker_id)
        except Exception:
            pass
        return len(remaining) == 0

    backoff.wait_until(_all_healthy, timeout=Duration.from_seconds(timeout))
    return remaining


def _check_worker_health(client, worker_ids: set[str]) -> list[tuple[str, str]]:
    """Check that all workers are still healthy. Returns list of (worker_id, problem) for failures."""
    failures: list[tuple[str, str]] = []
    try:
        resp = client.list_workers(controller_pb2.Controller.ListWorkersRequest())
        by_id = {w.worker_id: w for w in resp.workers}
        for wid in worker_ids:
            w = by_id.get(wid)
            if w is None:
                failures.append((wid, "disappeared"))
            elif not w.healthy:
                failures.append((wid, w.status_message or f"{w.consecutive_failures} consecutive failures"))
    except Exception as e:
        failures.append(("(rpc)", str(e)))
    return failures


def _print_summary(succeeded: int, remaining: int, offset: int):
    click.echo(f"\nSummary: {succeeded} succeeded, {remaining} remaining (aborted at worker {offset + 1})")
