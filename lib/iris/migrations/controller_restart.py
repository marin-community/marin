#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate a full Iris controller restart with state migration.

Handles the complete restart dance:

  1. Take a checkpoint via the controller's BeginCheckpoint RPC
  2. Migrate the bundle store from legacy SQLite to flat-file fsspec storage
  3. Download remote log segments to seed the local LogStore
  4. Restart the controller (build images, stop, start)
  5. Verify the new controller is healthy

The current controller already writes checkpoints to the canonical
``remote_state_dir/controller-state/`` path, so no GCS copy is needed.

Usage:

  # Full restart with all migrations
  uv run python lib/iris/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml

  # Dry run — show plan without making changes
  uv run python lib/iris/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --dry-run

  # Checkpoint + migrate only (no restart)
  uv run python lib/iris/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --no-restart

  # Skip bundle migration (already done)
  uv run python lib/iris/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --skip-bundles

  # Use a local bundles.sqlite3 instead of SCP
  uv run python lib/iris/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --local-bundle-db /tmp/bundles.sqlite3
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import fsspec.core

# This directory is not a Python package (it lives outside src/iris/), so add
# it to sys.path so sibling scripts can be imported directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_store import BundleMigrationResult, discover_controller_vm, migrate_bundle_store  # noqa: E402
from log_store import LogMigrationResult, migrate_log_store  # noqa: E402

logger = logging.getLogger("iris.migrations.controller_restart")


def _run(
    *args: str,
    timeout: float = 120,
    check: bool = True,
    capture: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = list(args)
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, check=check, cwd=cwd)


def _gcloud_ssh(vm: str, zone: str, project: str, command: str, timeout: float = 30) -> str:
    r = _run(
        "gcloud",
        "compute",
        "ssh",
        vm,
        f"--zone={zone}",
        f"--project={project}",
        f"--command={command}",
        timeout=timeout,
    )
    return r.stdout.strip()


def _gcs_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def get_controller_git_hash(vm: str, zone: str, project: str) -> str:
    """Read the IRIS_GIT_HASH env var from the running controller container."""
    output = _gcloud_ssh(
        vm,
        zone,
        project,
        "sudo docker inspect iris-controller --format='{{range .Config.Env}}{{println .}}{{end}}'",
    )
    for line in output.splitlines():
        if line.startswith("IRIS_GIT_HASH="):
            return line.split("=", 1)[1]
    return "unknown"


def trigger_checkpoint(vm: str, zone: str, project: str, port: int) -> dict:
    """Call BeginCheckpoint RPC on the running controller. Returns the JSON response."""
    output = _gcloud_ssh(
        vm,
        zone,
        project,
        f"curl -sf http://localhost:{port}/iris.cluster.ControllerService/BeginCheckpoint "
        f"-H 'Content-Type: application/json' -d '{{}}'",
        timeout=60,
    )
    return json.loads(output)


def verify_controller_health(vm: str, zone: str, project: str, port: int, timeout_sec: float = 120) -> bool:
    """Poll the controller until it responds to health checks after restart."""
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            output = _gcloud_ssh(
                vm,
                zone,
                project,
                f"curl -sf http://localhost:{port}/iris.cluster.ControllerService/GetStatus "
                f"-H 'Content-Type: application/json' -d '{{}}'",
                timeout=10,
            )
            status = json.loads(output)
            job_count = status.get("jobCount", 0)
            worker_count = status.get("workerCount", 0)
            logger.info("Controller healthy: %s jobs, %s workers", job_count, worker_count)
            return True
        except Exception:
            if attempt <= 3:
                logger.info("Waiting for controller to come up (attempt %d)...", attempt)
            time.sleep(5)

    logger.error("Controller did not become healthy within %ds", timeout_sec)
    return False


def iris_restart(config_path: str) -> None:
    """Run the Iris CLI controller restart (builds images, stops, starts)."""
    _run(
        "uv",
        "run",
        "iris",
        "--config",
        config_path,
        "cluster",
        "controller",
        "restart",
        timeout=600,
        capture=False,
    )


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Iris cluster config YAML")
@click.option("--restart/--no-restart", default=True, help="Whether to restart the controller after migrations")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would happen without making changes")
@click.option("--skip-bundles", is_flag=True, default=False, help="Skip bundle store migration")
@click.option("--skip-logs", is_flag=True, default=False, help="Skip log store migration")
@click.option("--skip-verify", is_flag=True, default=False, help="Skip post-migration verification")
@click.option(
    "--local-bundle-db",
    "local_bundle_db",
    type=click.Path(),
    default=None,
    help="Use local bundles.sqlite3 instead of SCP from controller",
)
@click.option("--max-log-segments", default=0, type=int, help="Max log segments to download (0 = all)")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
@click.option("--port", default=10000, type=int, help="Controller port")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    restart: bool,
    dry_run: bool,
    skip_bundles: bool,
    skip_logs: bool,
    skip_verify: bool,
    local_bundle_db: str | None,
    max_log_segments: int,
    project: str,
    port: int,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    from iris.cluster.config import load_config

    cluster_config = load_config(Path(config_path))
    remote_state_dir = cluster_config.storage.remote_state_dir
    label_prefix = cluster_config.platform.label_prefix or "iris"

    if not remote_state_dir:
        raise click.ClickException("storage.remote_state_dir is required in the cluster config")

    local_state_dir = (
        Path(cluster_config.storage.local_state_dir)
        if cluster_config.storage.local_state_dir
        else Path("/var/cache/iris/controller")
    )
    canonical_prefix = f"{remote_state_dir.rstrip('/')}/controller-state"
    bundle_storage_dir = f"{remote_state_dir.rstrip('/')}/bundles"
    remote_log_dir = f"{remote_state_dir.rstrip('/')}/logs"
    local_log_dir = local_state_dir / "logs"

    # =========================================================================
    # Step 0: Discover controller VM
    # =========================================================================
    click.echo("=" * 60)
    click.echo("Step 0: Discover controller VM")
    click.echo("=" * 60)

    vm_info = discover_controller_vm(project, label_prefix)
    if vm_info is None:
        raise click.ClickException(f"No controller VM found with label iris-{label_prefix}-controller=true")

    vm_name, zone = vm_info
    click.echo(f"  VM: {vm_name} ({zone})")

    git_hash = "unknown"
    if not dry_run:
        git_hash = get_controller_git_hash(vm_name, zone, project)
    click.echo(f"  Git hash: {git_hash}")

    # =========================================================================
    # Step 1: Take checkpoint
    # =========================================================================
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 1: Take checkpoint")
    click.echo("=" * 60)

    if dry_run:
        click.echo("  [dry-run] Would call BeginCheckpoint RPC")
        ckpt_response: dict = {"checkpointPath": "(dry-run)", "jobCount": "?", "taskCount": "?", "workerCount": "?"}
    else:
        click.echo("  Triggering checkpoint...")
        ckpt_response = trigger_checkpoint(vm_name, zone, project, port)

    ckpt_path = ckpt_response.get("checkpointPath", "?")
    job_count = ckpt_response.get("jobCount", 0)
    task_count = ckpt_response.get("taskCount", 0)
    worker_count = ckpt_response.get("workerCount", 0)
    click.echo(f"  Checkpoint: {ckpt_path}")
    click.echo(f"  Contents: {job_count} jobs, {task_count} tasks, {worker_count} workers")

    # Verify checkpoint is at the canonical location.
    if not dry_run:
        latest = f"{canonical_prefix}/latest.sqlite3"
        if _gcs_exists(latest):
            click.echo(f"  Verified: {latest} exists")
        else:
            click.echo(f"  WARNING: {latest} not found — checkpoint may not have uploaded yet")

    # =========================================================================
    # Step 2: Migrate bundle store
    # =========================================================================
    bundle_result: BundleMigrationResult | None = None
    if not skip_bundles:
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 2: Migrate bundle store (SQLite -> fsspec)")
        click.echo("=" * 60)

        db_path = Path(local_bundle_db) if local_bundle_db else None
        bundle_result = migrate_bundle_store(
            storage_dir=bundle_storage_dir,
            db_path=db_path,
            vm_name=vm_name if db_path is None else None,
            zone=zone if db_path is None else None,
            project=project,
            dry_run=dry_run,
            skip_verify=skip_verify,
        )
        click.echo(f"  Bundles: {bundle_result.total_bundles} ({bundle_result.total_bytes / 1024 / 1024:.1f} MB)")
        click.echo(f"  New writes: {bundle_result.new_writes}")
        if bundle_result.verification_failures:
            click.echo(f"  VERIFICATION FAILED: {len(bundle_result.verification_failures)} bundles")
            raise click.ClickException("Bundle verification failed — aborting restart")
    else:
        click.echo()
        click.echo("Step 2: Skipped (--skip-bundles)")

    # =========================================================================
    # Step 3: Migrate log store
    # =========================================================================
    log_result: LogMigrationResult | None = None
    if not skip_logs:
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 3: Migrate log store (download remote segments)")
        click.echo("=" * 60)

        log_result = migrate_log_store(
            remote_log_dir=remote_log_dir,
            local_log_dir=local_log_dir,
            dry_run=dry_run,
            max_segments=max_log_segments,
            skip_verify=skip_verify,
        )
        click.echo(f"  Remote segments: {log_result.remote_segments_found}")
        click.echo(f"  Downloaded: {log_result.segments_downloaded}")
        click.echo(f"  Skipped: {log_result.segments_skipped}")
        if not log_result.verification_ok:
            click.echo("  VERIFICATION FAILED")
            raise click.ClickException("Log store verification failed — aborting restart")
    else:
        click.echo()
        click.echo("Step 3: Skipped (--skip-logs)")

    # =========================================================================
    # Step 4: Restart controller
    # =========================================================================
    if restart:
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 4: Restart controller")
        click.echo("=" * 60)

        if dry_run:
            click.echo("  [dry-run] Would run: iris cluster controller restart")
        else:
            click.echo("  Restarting controller (build images, stop, start)...")
            click.echo("  Workers on separate VMs will survive.")
            iris_restart(config_path)
            click.echo("  Controller restart initiated.")

        # =====================================================================
        # Step 5: Verify health
        # =====================================================================
        if not dry_run and not skip_verify:
            click.echo()
            click.echo("=" * 60)
            click.echo("Step 5: Verify controller health")
            click.echo("=" * 60)

            click.echo("  Waiting for controller to become healthy...")
            healthy = verify_controller_health(vm_name, zone, project, port)
            if healthy:
                click.echo("  Controller is healthy!")
            else:
                click.echo("  WARNING: Controller did not respond within timeout")
                click.echo("  Check the controller logs for errors:")
                click.echo(f"    gcloud compute ssh {vm_name} --zone={zone} -- sudo docker logs iris-controller")
    else:
        click.echo()
        click.echo("Step 4: Skipped (--no-restart)")
        click.echo()
        click.echo("Migrations complete. To restart the controller:")
        click.echo(f"  uv run iris --config {config_path} cluster controller restart")

    # =========================================================================
    # Final report
    # =========================================================================
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    click.echo()
    click.echo("=" * 60)
    click.echo("Controller Restart Report")
    click.echo("=" * 60)
    click.echo(f"  Timestamp:   {now}")
    click.echo(f"  Controller:  {vm_name} ({zone})")
    click.echo(f"  Git hash:    {git_hash}")
    click.echo(f"  Checkpoint:  {ckpt_path}")
    click.echo(f"  State dir:   {remote_state_dir}")

    if bundle_result:
        click.echo(f"  Bundles:     {bundle_result.total_bundles} migrated, {bundle_result.new_writes} new")

    if log_result:
        click.echo(
            f"  Logs:        {log_result.segments_downloaded} segments downloaded "
            f"({log_result.total_bytes_downloaded / 1024 / 1024:.1f} MB)"
        )

    if restart and not dry_run:
        click.echo(f"  Restore from: {canonical_prefix}/latest.sqlite3")

    click.echo("=" * 60)


if __name__ == "__main__":
    main()
