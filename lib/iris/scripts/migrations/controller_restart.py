#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate a full Iris controller restart with state migration.

Handles the complete restart dance:

  1. Take a checkpoint via BeginCheckpoint RPC
  2. Migrate bundle store: SQLite → fsspec flat files
  3. Migrate log store: SQLite → Parquet (SCP back to controller)
  4. Restart the controller
  5. Verify the new controller is healthy

The controller already writes checkpoints to the canonical
``remote_state_dir/controller-state/`` path, so no GCS copy is needed.

Each step is idempotent and can be skipped individually. The sub-migrations
are invoked as separate processes so they can also be run standalone.

Usage:
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml

  # Dry run
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --dry-run

  # Checkpoint + migrate only (no restart)
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --no-restart

  # Skip bundle migration (already done)
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --skip-bundles

  # Skip log migration
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --skip-logs
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import click
import fsspec.core

from iris.cluster.config import load_config
from iris.scripts.migrations.gcloud_helpers import discover_controller_vm, gcloud_ssh, run_cmd

logger = logging.getLogger("controller-restart")

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run_script(script_name: str, *args: str, timeout: float = 600) -> None:
    """Run a sibling migration script as a subprocess."""
    script_path = SCRIPTS_DIR / script_name
    cmd = ["uv", "run", "python", str(script_path), *args]
    click.echo(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, timeout=timeout)


def _gcs_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def trigger_checkpoint(vm: str, zone: str, project: str, port: int) -> dict:
    """Call BeginCheckpoint RPC on the running controller."""
    output = gcloud_ssh(
        vm,
        zone,
        project,
        f"curl -sf http://localhost:{port}/iris.cluster.ControllerService/BeginCheckpoint "
        f"-H 'Content-Type: application/json' -d '{{}}'",
        timeout=60,
    )
    return json.loads(output)


def verify_controller_health(vm: str, zone: str, project: str, port: int, timeout_sec: float = 120) -> bool:
    """Poll the controller until it responds to health checks."""
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            output = gcloud_ssh(
                vm,
                zone,
                project,
                f"curl -sf http://localhost:{port}/iris.cluster.ControllerService/GetProcessStatus "
                f"-H 'Content-Type: application/json' -d '{{}}'",
                timeout=10,
            )
            status = json.loads(output)
            click.echo(f"  Healthy: process up, pid={status.get('pid', '?')}")
            return True
        except Exception:
            if attempt <= 3:
                click.echo(f"  Waiting for controller (attempt {attempt})...")
            time.sleep(5)

    return False


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--restart/--no-restart", default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--skip-bundles", is_flag=True, default=False)
@click.option("--skip-logs", is_flag=True, default=False)
@click.option("--skip-verify", is_flag=True, default=False)
@click.option("--local-bundle-db", type=click.Path(), default=None)
@click.option("--local-log-db", type=click.Path(), default=None)
@click.option("--max-log-rows", default=0, type=int, help="Max log rows to migrate (0 = all)")
@click.option("--project", default="hai-gcp-models")
@click.option("--port", default=10000, type=int)
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    restart: bool,
    dry_run: bool,
    skip_bundles: bool,
    skip_logs: bool,
    skip_verify: bool,
    local_bundle_db: str | None,
    local_log_db: str | None,
    max_log_rows: int,
    project: str,
    port: int,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    cluster_config = load_config(Path(config_path))
    remote_state_dir = cluster_config.storage.remote_state_dir
    label_prefix = cluster_config.platform.label_prefix or "iris"

    if not remote_state_dir:
        raise click.ClickException("storage.remote_state_dir is required")

    canonical_prefix = f"{remote_state_dir.rstrip('/')}/controller-state"

    # =========================================================================
    # Step 0: Discover controller
    # =========================================================================
    click.echo("=" * 60)
    click.echo("Step 0: Discover controller VM")
    click.echo("=" * 60)

    vm_info = discover_controller_vm(project, label_prefix)
    if vm_info is None:
        raise click.ClickException(f"No controller VM found with label iris-{label_prefix}-controller=true")
    vm_name, zone = vm_info
    click.echo(f"  VM: {vm_name} ({zone})")

    # =========================================================================
    # Step 1: Checkpoint
    # =========================================================================
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 1: Take checkpoint")
    click.echo("=" * 60)

    if dry_run:
        click.echo("  [dry-run] Would call BeginCheckpoint RPC")
    else:
        resp = trigger_checkpoint(vm_name, zone, project, port)
        click.echo(f"  Checkpoint: {resp.get('checkpointPath', '?')}")
        click.echo(
            f"  Jobs: {resp.get('jobCount', 0)}, Tasks: {resp.get('taskCount', 0)}, "
            f"Workers: {resp.get('workerCount', 0)}"
        )

        latest = f"{canonical_prefix}/latest.sqlite3"
        if _gcs_exists(latest):
            click.echo(f"  Verified: {latest} exists")
        else:
            click.echo(f"  WARNING: {latest} not found yet")

    # =========================================================================
    # Step 2: Migrate bundle store
    # =========================================================================
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 2: Migrate bundle store")
    click.echo("=" * 60)

    if skip_bundles:
        click.echo("  Skipped (--skip-bundles)")
    else:
        bundle_args = ["--config", config_path, "--project", project]
        if local_bundle_db:
            bundle_args += ["--local-db", local_bundle_db]
        if dry_run:
            bundle_args.append("--dry-run")
        if skip_verify:
            bundle_args.append("--skip-verify")
        if verbose:
            bundle_args.append("-v")
        _run_script("bundle_store.py", *bundle_args)

    # =========================================================================
    # Step 3: Migrate log store
    # =========================================================================
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 3: Migrate log store (SQLite -> Parquet)")
    click.echo("=" * 60)

    if skip_logs:
        click.echo("  Skipped (--skip-logs)")
    else:
        log_args = ["--config", config_path, "--project", project]
        if local_log_db:
            log_args += ["--local-db", local_log_db]
        if max_log_rows > 0:
            log_args += ["--max-rows", str(max_log_rows)]
        if dry_run:
            log_args.append("--dry-run")
        if skip_verify:
            # log_store.py uses --skip-upload (no verification step)
            log_args.append("--skip-upload")
        if verbose:
            log_args.append("-v")
        _run_script("log_store.py", *log_args, timeout=1200)

    # =========================================================================
    # Step 4: Restart controller
    # =========================================================================
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 4: Restart controller")
    click.echo("=" * 60)

    if not restart:
        click.echo("  Skipped (--no-restart)")
        click.echo(f"  To restart: uv run iris --config {config_path} cluster controller restart")
        return

    if dry_run:
        click.echo("  [dry-run] Would run: iris cluster controller restart")
        return

    click.echo("  Workers on separate VMs survive the restart.")
    run_cmd(
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

    # =========================================================================
    # Step 5: Verify health
    # =========================================================================
    if not skip_verify:
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 5: Verify controller health")
        click.echo("=" * 60)

        click.echo("  Waiting for controller to become healthy...")
        healthy = verify_controller_health(vm_name, zone, project, port)
        if healthy:
            click.echo("  Controller is healthy!")
        else:
            click.echo("  WARNING: Controller did not respond within timeout", err=True)
            click.echo(
                f"  Check logs: gcloud compute ssh {vm_name} --zone={zone} " "-- sudo docker logs iris-controller"
            )
            sys.exit(1)

    click.echo()
    click.echo("Done.")


if __name__ == "__main__":
    main()
