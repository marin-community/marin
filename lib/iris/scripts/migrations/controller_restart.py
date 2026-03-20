#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate a full Iris controller restart.

Handles the complete restart dance:

  1. Take a checkpoint via BeginCheckpoint RPC
  2. Restart the controller
  3. Verify the new controller is healthy

The controller already writes checkpoints to the canonical
``remote_state_dir/controller-state/`` path, so no GCS copy is needed.
Schema migrations (including auth DB separation) run automatically on
controller startup.

Usage:
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml

  # Dry run
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --dry-run

  # Checkpoint only (no restart)
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --no-restart
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
from iris.cluster.controller.db import ControllerDB

logger = logging.getLogger("controller-restart")


def _run_cmd(
    *args: str, timeout: float = 120, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    logger.debug("$ %s", " ".join(args))
    return subprocess.run(list(args), capture_output=capture, text=True, timeout=timeout, check=check)


def _discover_controller_vm(project: str, label_prefix: str) -> tuple[str, str] | None:
    """Find controller VM by its Iris label. Returns (vm_name, zone) or None."""
    r = _run_cmd(
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter=labels.iris-{label_prefix}-controller=true",
        "--format=json(name,zone)",
        timeout=30,
    )
    instances = json.loads(r.stdout)
    if not instances:
        return None
    vm = instances[0]
    zone = vm["zone"].rsplit("/", 1)[-1]
    return vm["name"], zone


def _gcloud_ssh(vm: str, zone: str, project: str, command: str, timeout: float = 30) -> str:
    """Run a command on a VM via gcloud SSH and return stdout."""
    r = _run_cmd(
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


def trigger_checkpoint(vm: str, zone: str, project: str, port: int) -> dict:
    """Call BeginCheckpoint RPC on the running controller."""
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
    """Poll the controller until it responds to health checks."""
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            output = _gcloud_ssh(
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
@click.option("--skip-verify", is_flag=True, default=False)
@click.option("--project", default="hai-gcp-models")
@click.option("--port", default=10000, type=int)
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    restart: bool,
    dry_run: bool,
    skip_verify: bool,
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

    # Step 1: Discover controller
    click.echo("=" * 60)
    click.echo("Step 1: Discover controller VM")
    click.echo("=" * 60)

    vm_info = _discover_controller_vm(project, label_prefix)
    if vm_info is None:
        raise click.ClickException(f"No controller VM found with label iris-{label_prefix}-controller=true")
    vm_name, zone = vm_info
    click.echo(f"  VM: {vm_name} ({zone})")

    # Step 2: Checkpoint
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 2: Take checkpoint")
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

        checkpoint_path = resp.get("checkpointPath", "")
        if checkpoint_path and _gcs_exists(f"{checkpoint_path}/{ControllerDB.DB_FILENAME}"):
            click.echo(f"  Verified: {checkpoint_path}/{ControllerDB.DB_FILENAME} exists")
        else:
            click.echo(f"  WARNING: checkpoint DB not found at {checkpoint_path}")

    # Step 3: Restart controller
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 3: Restart controller")
    click.echo("=" * 60)

    if not restart:
        click.echo("  Skipped (--no-restart)")
        click.echo(f"  To restart: uv run iris --config {config_path} cluster controller restart")
        return

    if dry_run:
        click.echo("  [dry-run] Would run: iris cluster controller restart")
        return

    click.echo("  Workers on separate VMs survive the restart.")
    _run_cmd(
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

    # Step 4: Verify health
    if not skip_verify:
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 4: Verify controller health")
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
