# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster management CLI commands.

All cluster subcommands live here: lifecycle (start/stop/restart/status),
controller VM management, VM operations via controller RPC, and the dashboard tunnel.
"""

import json
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import click
import uvicorn
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.deploy.cli import down_cmd, logs_cmd, restart_cmd, status_cmd, up_cmd
from rigging.config_discovery import list_cluster_configs
from rigging.filesystem import marin_temp_bucket
from rigging.timing import Duration, ExponentialBackoff, Timestamp
from rigging.token_authority import generate_ed25519_keypair

from iris.cli.build import (
    build_image,
    find_marin_root,
    get_git_sha,
)
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, require_controller_url, rpc_client_for_ctx
from iris.cluster.composer import provider_bundle
from iris.cluster.config import clear_remote_state, make_local_config
from iris.cluster.controller.autoscaler.scaling_group import (
    _zone_from_template,
    build_worker_config_for_group,
    prepare_slice_config,
)
from iris.cluster.controller.dashboard import ProxyControllerDashboard
from iris.cluster.controller.main import run_controller_serve
from iris.cluster.dashboard_common import VUE_DIST_DIR
from iris.cluster.inject_env import with_injected_task_env
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.platforms.gcp.worker_bootstrap import build_worker_bootstrap_script
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.platforms.types import Labels
from iris.cluster.provenance import provenance_from_proto
from iris.rpc import controller_pb2, job_pb2, query_pb2, vm_pb2
from iris.rpc.proto_display import format_accelerator_display, vm_state_name
from iris.time_proto import timestamp_from_proto


@dataclass(frozen=True)
class WorkerBootstrapRow:
    """Per-worker fields needed to rebuild the bootstrap script CLI-side."""

    worker_id: str
    slice_id: str
    scale_group: str
    zone: str


def _fetch_worker_bootstrap_rows(client, worker_ids: list[str]) -> dict[str, WorkerBootstrapRow]:
    """Fetch slice_id/scale_group/zone for a set of workers.

    Queries the controller's workers table via ExecuteRawQuery so the CLI can
    drive the bootstrap directly without an autoscaler refresh cycle. Zone is
    only populated for GCE workers; TPU workers leave it empty and the caller
    resolves the zone from the scale-group's slice template.
    """
    if not worker_ids:
        return {}
    quoted = ", ".join("'" + wid.replace("'", "''") + "'" for wid in worker_ids)
    sql = f"SELECT worker_id, slice_id, scale_group, md_gce_zone FROM workers WHERE worker_id IN ({quoted})"
    response = client.execute_raw_query(query_pb2.RawQueryRequest(sql=sql))
    column_index = {col.name: i for i, col in enumerate(response.columns)}
    rows: dict[str, WorkerBootstrapRow] = {}
    for raw in response.rows:
        decoded = json.loads(raw)
        row = WorkerBootstrapRow(
            worker_id=decoded[column_index["worker_id"]],
            slice_id=decoded[column_index["slice_id"]],
            scale_group=decoded[column_index["scale_group"]],
            zone=decoded[column_index["md_gce_zone"]],
        )
        rows[row.worker_id] = row
    return rows


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


def _get_autoscaler_status(ctx: click.Context, controller_url: str) -> vm_pb2.AutoscalerStatus:
    with rpc_client_for_ctx(ctx, url=controller_url) as client:
        request = controller_pb2.Controller.GetAutoscalerStatusRequest()
        return client.get_autoscaler_status(request).status


def _get_worker_status(
    ctx: click.Context, controller_url: str, worker_id: str
) -> controller_pb2.Controller.GetWorkerStatusResponse:
    with rpc_client_for_ctx(ctx, url=controller_url) as client:
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


def _build_and_push_image(image_tag: str, image_type: str, git_sha: str, verbose: bool = False) -> None:
    """Build and push a single image to GHCR, parsing org/name/version from the tag.

    The task image uses the ``task`` target in the unified Dockerfile and needs the
    marin repo root as build context; other targets default to the iris root.
    """
    ghcr_parsed = _parse_ghcr_tag(image_tag)
    if not ghcr_parsed:
        raise click.ClickException(f"Unrecognized image tag format (expected ghcr.io/...): {image_tag}")

    org, image_name, version = ghcr_parsed
    local_tag = f"{image_name}:{version}"
    context = str(find_marin_root()) if image_type == "task" else None
    click.echo(f"Building {image_type} image: {local_tag}")
    click.echo(f"  Registry: ghcr.io/{org}")
    click.echo()
    build_image(
        image_type=image_type,
        tag=local_tag,
        push=True,
        context=context,
        platform="linux/amd64,linux/arm64",
        git_sha=git_sha,
        ghcr_org=org,
        verbose=verbose,
    )
    click.echo()


def _build_cluster_images(config, git_sha: str, verbose: bool = False) -> dict[str, str]:
    built: dict[str, str] = {}

    for tag, typ in [(config.defaults.worker.docker_image, "worker"), (config.controller.image, "controller")]:
        if tag:
            _build_and_push_image(tag, typ, git_sha, verbose=verbose)
            built[typ] = tag

    task_tag = config.defaults.worker.default_task_image
    if task_tag:
        _build_and_push_image(task_tag, "task", git_sha, verbose=verbose)
        built["task"] = task_tag

    return built


def _pin_latest_images(config, git_sha: str) -> dict[str, str]:
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


def _upload_private_key_to_secret_manager(resource: str, private_pem: str, accessor: str | None) -> str:
    """Add `private_pem` as a new version of the Secret Manager secret `resource`.

    `resource` is `projects/<project>/secrets/<name>`. Creates the secret container
    first when the caller has permission; otherwise adds a version to a container an
    admin pre-created. When `accessor` is set, also grants it
    roles/secretmanager.secretAccessor on the secret (e.g. the controller service
    account, so it can read the key at startup). Returns the pinned
    `gcp-secret://…/versions/<n>` reference to put in auth.signing_key.
    """
    try:
        from google.cloud import secretmanager  # noqa: PLC0415  # optional dep
    except ImportError as exc:
        raise click.ClickException(
            "storing a key in Secret Manager needs the optional dependency; install marin-rigging[secrets]"
        ) from exc
    from google.api_core.exceptions import AlreadyExists, PermissionDenied  # noqa: PLC0415  # optional dep

    project, _, name = resource.removeprefix("projects/").partition("/secrets/")
    if not project or not name:
        raise click.ClickException(f"--gcp-secret must be projects/<p>/secrets/<name>, got {resource!r}")

    client = secretmanager.SecretManagerServiceClient()
    try:
        client.create_secret(
            request={
                "parent": f"projects/{project}",
                "secret_id": name,
                "secret": {"replication": {"automatic": {}}},
            }
        )
        click.echo(f"Created Secret Manager secret {name}.")
    except AlreadyExists:
        pass
    except PermissionDenied:
        click.echo(f"No permission to create {name}; adding a version to the existing secret.")

    version = client.add_secret_version(request={"parent": resource, "payload": {"data": private_pem.encode("utf-8")}})

    if accessor is not None:
        member = accessor if ":" in accessor else f"serviceAccount:{accessor}"
        role = "roles/secretmanager.secretAccessor"
        try:
            policy = client.get_iam_policy(request={"resource": resource})
            binding = next((b for b in policy.bindings if b.role == role), None)
            if binding is not None and member in binding.members:
                click.echo(f"{member} already has {role} on {name}.")
            else:
                if binding is None:
                    policy.bindings.add(role=role, members=[member])
                else:
                    binding.members.append(member)
                client.set_iam_policy(request={"resource": resource, "policy": policy})
                click.echo(f"Granted {role} on {name} to {member}.")
        except PermissionDenied:
            click.echo(f"No permission to set IAM on {name}; grant {role} to {member} manually.")

    return f"gcp-secret://{version.name}"


@cluster.command("init-keys")
@click.option(
    "--out-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write the PRIVATE key PEM to this file (mode 0600). Reference it from auth.signing_key as file:<abs-path>.",
)
@click.option(
    "--gcp-secret",
    "gcp_secret",
    default=None,
    metavar="projects/<p>/secrets/<name>",
    help="Store the PRIVATE key as a new version of this Secret Manager secret "
    "(creating the secret when you have permission). Reference the pinned version from auth.signing_key.",
)
@click.option(
    "--accessor",
    default=None,
    metavar="[serviceAccount:]<email>",
    help="With --gcp-secret, also grant this principal roles/secretmanager.secretAccessor on the "
    "secret (e.g. the controller service account, so it can read the key at startup).",
)
def cluster_init_keys(out_file: Path | None, gcp_secret: str | None, accessor: str | None):
    """Generate a per-cluster Ed25519 signing keypair for controller auth.

    Writes the PRIVATE key to a SecretSpec destination (a file, and/or a new
    Secret Manager version) and prints the PUBLIC key + kid for the trust config.
    The private half is a crown-jewel secret: never commit it, never inline it into
    a cluster config — reference it from auth.signing_key via file: / gcp-secret://.
    The public key is inline-safe (JWKS, peer/finelog allowlists, previous_public_keys
    during a rotation overlap).
    """
    if accessor is not None and gcp_secret is None:
        raise click.UsageError("--accessor only applies with --gcp-secret")

    keypair = generate_ed25519_keypair()

    if out_file is not None:
        out_file.write_text(keypair.private_pem)
        out_file.chmod(0o600)
        click.echo(f"Wrote PRIVATE key to {out_file} (mode 0600).")
        click.echo(f"  Reference it as: auth.signing_key: file:{out_file.resolve()}")

    if gcp_secret is not None:
        reference = _upload_private_key_to_secret_manager(gcp_secret, keypair.private_pem, accessor)
        click.echo(f"Stored PRIVATE key in Secret Manager ({gcp_secret}).")
        click.echo(f"  Reference it as: auth.signing_key: {reference}")

    if out_file is None and gcp_secret is None:
        click.echo("# PRIVATE KEY (store securely; do NOT commit or inline into a cluster config):")
        click.echo(keypair.private_pem.rstrip("\n"))

    click.echo(f"\nkid: {keypair.kid}")
    click.echo("# PUBLIC KEY (inline-safe — trust config / peer & finelog allowlists / previous_public_keys):")
    click.echo(keypair.public_pem.rstrip("\n"))


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
    is_local = config.controller.controller_kind() == "local"
    if not is_local:
        git_sha = get_git_sha()
        _pin_latest_images(config, git_sha)
        verbose = ctx.obj.get("verbose", False)
        built = _build_cluster_images(config, git_sha, verbose=verbose)
        if built:
            click.echo("Built image tags:")
            for name, tag in built.items():
                click.echo(f"  {name}: {tag}")
    click.echo("Starting controller...")
    try:
        if is_local:
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
            bundle = provider_bundle(config)
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
    config.storage.remote_state_dir = marin_temp_bucket(ttl_days=7, prefix=f"iris/state/{label_prefix}")

    git_sha = get_git_sha()
    _pin_latest_images(config, git_sha)
    verbose = ctx.obj.get("verbose", False)
    _build_cluster_images(config, git_sha, verbose=verbose)

    bundle = provider_bundle(config)

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

            with rpc_client_for_ctx(ctx, url=url) as client:
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
        bundle = provider_bundle(config)
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


# =============================================================================
# Log server (finelog) shim
# =============================================================================


@cluster.group("log-server")
def log_server() -> None:
    """Manage the log server referenced by this cluster's finelog.config."""


def _require_log_server_config(ctx: click.Context) -> str:
    cfg = ctx.obj.get("config")
    if cfg is None:
        raise click.ClickException("--config is required for cluster log-server commands")
    if not cfg.finelog.config:
        raise click.ClickException(
            "cluster does not declare finelog.config; " "set it or manage the log server via `finelog deploy` directly"
        )
    return cfg.finelog.config


@log_server.command("up")
@click.option(
    "--build/--no-build",
    "build",
    default=True,
    show_default=True,
    help="Build and push the finelog image before provisioning. Pass --no-build to use the registry's existing :latest.",
)
@click.pass_context
def log_server_up(ctx: click.Context, build: bool) -> None:
    """Provision/refresh the cluster's finelog deployment (idempotent)."""
    name = _require_log_server_config(ctx)
    ctx.invoke(up_cmd, name=name, build=build)


@log_server.command("down")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation; for k8s also deletes the PVC.")
@click.pass_context
def log_server_down(ctx: click.Context, yes: bool) -> None:
    """Tear down the cluster's finelog deployment."""
    name = _require_log_server_config(ctx)
    ctx.invoke(down_cmd, name=name, yes=yes)


@log_server.command("restart")
@click.option(
    "--build/--no-build",
    "build",
    default=True,
    show_default=True,
    help="Build and push the finelog image before restarting. Pass --no-build to reuse the registry's :latest.",
)
@click.pass_context
def log_server_restart(ctx: click.Context, build: bool) -> None:
    """Restart the cluster's finelog deployment."""
    name = _require_log_server_config(ctx)
    ctx.invoke(restart_cmd, name=name, build=build)


@log_server.command("status")
@click.pass_context
def log_server_status(ctx: click.Context) -> None:
    """Show the cluster's finelog deployment status."""
    name = _require_log_server_config(ctx)
    ctx.invoke(status_cmd, name=name)


@log_server.command("logs")
@click.option("--tail", type=int, default=200, show_default=True)
@click.option("-f", "--follow", is_flag=True, help="Stream logs")
@click.pass_context
def log_server_logs(ctx: click.Context, tail: int, follow: bool) -> None:
    """Tail the cluster's finelog deployment logs."""
    name = _require_log_server_config(ctx)
    ctx.invoke(logs_cmd, name=name, tail=tail, follow=follow)


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
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster create-slice")
    if config.controller.controller_kind() == "local":
        raise click.ClickException("create-slice is not supported for local clusters")

    sg_config = config.scale_groups.get(scale_group_name)
    if sg_config is None:
        available = ", ".join(sorted(config.scale_groups.keys())) or "(none)"
        raise click.ClickException(f"Unknown scale group '{scale_group_name}'. Available: {available}")

    # Verify the controller is reachable before creating the slice. The
    # returned URL may be a tunnel endpoint that's only reachable from the CLI
    # host; workers need the cluster-internal address instead, resolved below.
    require_controller_url(ctx)
    bundle = ctx.obj.get("provider_bundle") or provider_bundle(config)

    # Resolve the address workers will connect to. Prefer an explicit value in
    # defaults.worker.controller_address, then discover it via the provider
    # (e.g., GCE label lookup). Never pass the CLI-local tunnel URL here.
    worker_controller_address = config.controller_address()
    if not worker_controller_address:
        worker_controller_address = bundle.controller.discover_controller(config.controller)

    label_prefix = config.platform.label_prefix or "iris"
    labels = Labels(label_prefix)

    slice_config = prepare_slice_config(sg_config.slice_template, sg_config, label_prefix)
    slice_config.labels[labels.iris_manual] = "true"

    # Fold operator-injected env (defaults.inject_env) into task_env so manually
    # created slices match autoscaler-provisioned workers.
    base_worker_config = with_injected_task_env(config).defaults.worker.model_copy(deep=True)
    if not base_worker_config.controller_address:
        base_worker_config.controller_address = worker_controller_address
    base_worker_config.platform = config.platform.model_copy(deep=True)
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
    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for cluster delete-slice")
    if config.controller.controller_kind() == "local":
        raise click.ClickException("delete-slice is not supported for local clusters")

    bundle = ctx.obj.get("provider_bundle") or provider_bundle(config)

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
        with rpc_client_for_ctx(ctx, url=controller_url) as client:
            proc = client.get_process_status(job_pb2.GetProcessStatusRequest()).process_info
            workers = client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
            as_status = client.get_autoscaler_status(controller_pb2.Controller.GetAutoscalerStatusRequest()).status
        healthy = sum(1 for w in workers if w.healthy)
        click.echo("Controller Status:")
        click.echo("  Running: True")
        click.echo("  Healthy: True")
        click.echo(f"  Address: {controller_url}")
        click.echo(f"  Version: {provenance_from_proto(proc.provenance)}")
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
        as_status = _get_autoscaler_status(ctx, controller_url)
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
        accel_display = format_accelerator_display(group.device_type, group.device_variant)
        click.echo(f"  Accelerator: {accel_display}")
        click.echo(f"  Slices: {counts.get('ready', 0)}/{total} ready")
        click.echo(f"    Booting: {counts.get('booting', 0)}")
        click.echo(f"    Initializing: {counts.get('initializing', 0)}")
        click.echo(f"    Failed: {counts.get('failed', 0)}")
        click.echo(f"  Demand: {group.current_demand} (peak: {group.peak_demand})")
        # Availability / churn status. ``consecutive_failures`` in the proto now
        # carries the windowed failure count from BackoffDetector — see
        # ``ScalingGroup.to_status``.
        if group.availability_status and group.availability_status != "available":
            reason = f" - {group.availability_reason}" if group.availability_reason else ""
            click.echo(f"  Availability: {group.availability_status}{reason}")
            blocked_ms = timestamp_from_proto(group.blocked_until).epoch_ms()
            if blocked_ms > 0:
                click.echo(f"  Blocked until: {_format_timestamp(blocked_ms)}")
        if group.consecutive_failures > 0:
            click.echo(f"  Recent failures: {group.consecutive_failures}")
        if group.slices:
            click.echo("  Slices:")
            for si in group.slices:
                click.echo(f"    {si.slice_id}: {si.state.upper()}")
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
        resp = _get_worker_status(ctx, controller_url, vm_id)
    except ConnectError as e:
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
    with rpc_client_for_ctx(ctx, url=controller_url) as client:
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
        bundle = provider_bundle(config)
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

    is_local = config.controller.controller_kind() == "local"
    if is_local:
        raise click.ClickException(
            "controller restart is not supported for local clusters. "
            "Stop and restart the 'iris cluster start --local' process instead."
        )

    bundle = provider_bundle(config)

    # Try to discover existing controller for checkpoint + restart.
    # If none exists, fall back to a fresh start (idempotent).
    try:
        controller_url = require_controller_url(ctx)
    except (RuntimeError, click.ClickException):
        click.echo("No existing controller found. Starting fresh...")
        git_sha = get_git_sha()
        _pin_latest_images(config, git_sha)
        verbose = ctx.obj.get("verbose", False)
        built = _build_cluster_images(config, git_sha, verbose=verbose)
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
        with rpc_client_for_ctx(ctx, url=controller_url) as client:
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
    git_sha = get_git_sha()
    _pin_latest_images(config, git_sha)
    verbose = ctx.obj.get("verbose", False)
    built = _build_cluster_images(config, git_sha, verbose=verbose)
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
@click.option("--min-batch", type=int, default=1, help="Minimum workers to restart concurrently")
@click.option("--max-batch", type=int, default=64, help="Maximum workers to restart concurrently")
@click.option(
    "--observation-window",
    type=int,
    default=30,
    help="Seconds to observe restarted workers for failures before advancing",
)
@click.option(
    "--max-failures",
    type=int,
    default=4,
    help=(
        "Abort rollout if cumulative per-worker failures exceed this budget. "
        "Covers preemptions (which surface as restart errors or workers "
        "disappearing from the controller table), bootstrap errors, and workers "
        "that don't come back healthy in time."
    ),
)
@click.option(
    "--skip-current-hash/--no-skip-current-hash",
    default=False,
    help=(
        "Skip workers whose reported git_hash matches the CLI's current working-tree hash "
        "(worker-restart bakes this into the freshly-built worker image). Useful for "
        "resuming a partial rollout without redoing already-restarted workers."
    ),
)
@click.pass_context
def worker_restart(
    ctx,
    worker_id: tuple[str, ...],
    timeout: int,
    min_batch: int,
    max_batch: int,
    observation_window: int,
    max_failures: int,
    skip_current_hash: bool,
):
    """Rolling restart of workers with adaptive batch sizing.

    Restarts workers in progressively larger batches (1, 2, 4, ... up to
    --max-batch). After each batch, waits for workers to become healthy, then
    observes them for --observation-window seconds to catch post-restart
    failures. Every per-worker failure (bootstrap error, fail-to-become-healthy,
    unhealthy after observation, vanished from the controller) counts toward
    --max-failures; the rollout aborts only once cumulative failures exceed
    the budget. This absorbs ordinary preemption noise without special-casing.

    The CLI drives the SSH bootstrap directly per worker — no controller-side
    proxy — so bootstrap stdout/stderr streams to the operator's terminal.

    Running Docker containers are preserved and adopted by the new worker
    process, so tasks are not disrupted.
    """
    controller_url = require_controller_url(ctx)

    config = ctx.obj.get("config")
    if not config:
        raise click.ClickException("--config is required for worker-restart")
    bundle = ctx.obj.get("provider_bundle") or provider_bundle(config)
    if not isinstance(bundle.workers, GcpWorkerProvider):
        raise click.ClickException("worker-restart is only supported on GCP clusters")

    # Build and push fresh worker/task images so the bootstrap pulls the operator's
    # current tree, not whatever ``:latest`` happens to resolve to. Without this,
    # repeated worker-restarts pick up whichever SHA was last published by
    # ``controller restart`` (or by autoscaler-fresh VMs racing the registry),
    # producing a fleet split across multiple git_hashes and making
    # ``--skip-current-hash`` a no-op.
    git_sha = get_git_sha()
    _pin_latest_images(config, git_sha)
    verbose = ctx.obj.get("verbose", False)
    if config.defaults.worker.docker_image:
        _build_and_push_image(config.defaults.worker.docker_image, "worker", git_sha, verbose=verbose)
    if config.defaults.worker.default_task_image:
        _build_and_push_image(config.defaults.worker.default_task_image, "task", git_sha, verbose=verbose)

    # Resolve the controller address workers will reconnect to (matches cluster_create_slice).
    worker_controller_address = config.controller_address()
    if not worker_controller_address:
        worker_controller_address = bundle.controller.discover_controller(config.controller)

    # Fold operator-injected env (defaults.inject_env) into task_env so restarted
    # workers match autoscaler-provisioned ones.
    base_worker_config = with_injected_task_env(config).defaults.worker.model_copy(deep=True)
    if not base_worker_config.controller_address:
        base_worker_config.controller_address = worker_controller_address
    base_worker_config.platform = config.platform.model_copy(deep=True)
    if config.storage.remote_state_dir:
        base_worker_config.storage_prefix = config.storage.remote_state_dir

    scale_groups = dict(config.scale_groups)

    with rpc_client_for_ctx(ctx, url=controller_url) as client:
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

        if skip_current_hash:
            target_hash = get_git_sha()
            if not target_hash or target_hash == "unknown":
                click.echo(f"--skip-current-hash requires a known git_hash (got: {target_hash!r})", err=True)
                raise SystemExit(1)
            before = len(workers)
            workers = [w for w in workers if w.metadata.provenance.tree_hash != target_hash]
            skipped = before - len(workers)
            click.echo(f"Skipping {skipped}/{before} worker(s) already at local git_hash {target_hash}")

        if not workers:
            click.echo("No workers to restart")
            return

        worker_ids = [w.worker_id for w in workers]
        total = len(worker_ids)
        click.echo(
            f"Restarting {total} worker(s) "
            f"(timeout={timeout}s, observation={observation_window}s, "
            f"max_batch={max_batch}, max_failures={max_failures})"
        )

        rows_by_id = _fetch_worker_bootstrap_rows(client, worker_ids)

        succeeded = 0
        failures = 0
        batch_size = min_batch
        offset = 0

        def _record_failure(wid: str, reason: str) -> None:
            nonlocal failures
            failures += 1
            click.echo(f"  FAILED ({failures}/{max_failures}): {wid}: {reason}", err=True)
            if failures > max_failures:
                click.echo(
                    f"  ABORT: failure budget exceeded ({failures} > {max_failures})",
                    err=True,
                )
                _print_summary(succeeded, failures, total - succeeded - failures, offset)
                raise SystemExit(1)

        missing_rows = [wid for wid in worker_ids if wid not in rows_by_id]
        for wid in sorted(missing_rows):
            _record_failure(wid, "missing from controller workers table (likely preempted)")
        worker_ids = [wid for wid in worker_ids if wid in rows_by_id]

        while offset < total:
            batch = worker_ids[offset : offset + batch_size]
            if not batch:
                break
            click.echo(f"\n--- Batch of {len(batch)} (workers {offset + 1}-{offset + len(batch)} of {total}) ---")

            # Drive each worker's bootstrap in parallel: build the bootstrap script
            # locally, then SSH into the VM directly. Bootstrap stdout/stderr streams
            # via the handle's on_line callback to the CLI logger.
            for wid in batch:
                click.echo(f"  Restarting {wid}...")

            def _restart(wid: str) -> tuple[str, str | None, str | None]:
                row = rows_by_id[wid]
                if not row.slice_id:
                    return wid, None, "no slice_id (unmanaged worker)"
                sg_config = scale_groups.get(row.scale_group)
                if sg_config is None:
                    return wid, None, f"unknown scale group {row.scale_group!r}"
                # TPU workers leave md_gce_zone empty; fall back to the scale
                # group's slice-template zone.
                zone = row.zone or _zone_from_template(sg_config.slice_template)
                if not zone:
                    return wid, None, "missing zone metadata"
                wc = build_worker_config_for_group(base_worker_config, sg_config)
                wc.worker_id = wid
                wc.slice_id = row.slice_id
                script = build_worker_bootstrap_script(wc)
                try:
                    handle = bundle.workers.worker_handle(worker_id=wid, slice_id=row.slice_id, zone=zone)
                    handle.restart_worker(script)
                except Exception as e:
                    return wid, None, str(e)
                return wid, "ok", None

            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                results = list(pool.map(_restart, batch))

            healthy_targets: set[str] = set()
            for wid, status, error in results:
                if status == "ok":
                    healthy_targets.add(wid)
                else:
                    _record_failure(wid, f"restart failed: {error}")

            if healthy_targets:
                click.echo(f"  Waiting for {len(healthy_targets)} worker(s) to become healthy...")
                unhealthy = _wait_for_workers_healthy(client, set(healthy_targets), timeout)
                for wid in sorted(unhealthy):
                    _record_failure(wid, f"did not become healthy within {timeout}s")
                healthy_targets -= unhealthy

            if healthy_targets:
                click.echo(f"  Observing {len(healthy_targets)} worker(s) for {observation_window}s...")
                time.sleep(observation_window)
                for wid, msg in _check_worker_health(client, set(healthy_targets)):
                    healthy_targets.discard(wid)
                    _record_failure(wid, f"unhealthy after observation: {msg}")

            succeeded += len(healthy_targets)
            offset += len(batch)
            click.echo(f"  Batch done: {succeeded}/{total} healthy, {failures}/{max_failures} failures")

            # Double batch size for next round, capped at max_batch
            batch_size = min(batch_size * 2, max_batch)

    click.echo(f"\nDone: {succeeded}/{total} workers restarted ({failures} failures within budget)")


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


def _print_summary(succeeded: int, failures: int, remaining: int, offset: int):
    click.echo(
        f"\nSummary: {succeeded} succeeded, {failures} failed, {remaining} remaining "
        f"(aborted at worker {offset + 1})"
    )
