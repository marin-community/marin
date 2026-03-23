#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate a full Iris controller restart.

Handles the complete restart dance:

  1. Take a checkpoint via BeginCheckpoint RPC
  2. Restart the controller
  3. Verify the new controller is healthy

Works with any Iris platform (GCP, CoreWeave, Manual) via the Platform
protocol for discovery, tunneling, and restart.

Usage:
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml

  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/coreweave.yaml

  # Dry run
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --dry-run

  # Checkpoint only (no restart)
  uv run python lib/iris/scripts/migrations/controller_restart.py \
      --config lib/iris/examples/marin.yaml --no-restart
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import click
import fsspec.core

from iris.cluster.config import IrisConfig, load_config
from iris.cluster.controller.db import ControllerDB
from iris.cluster.platform.coreweave import configure_client_s3
from iris.rpc import cluster_connect, cluster_pb2

logger = logging.getLogger("controller-restart")


def _remote_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def trigger_checkpoint(controller_url: str) -> cluster_pb2.Controller.BeginCheckpointResponse:
    """Call BeginCheckpoint RPC on the running controller."""
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    try:
        return client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest(), timeout_ms=60_000)
    finally:
        client.close()


def verify_controller_health(controller_url: str, timeout_sec: float = 120) -> bool:
    """Poll the controller until it responds to GetProcessStatus."""
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            client = cluster_connect.ControllerServiceClientSync(controller_url)
            try:
                resp = client.get_process_status(cluster_pb2.GetProcessStatusRequest(), timeout_ms=10_000)
                click.echo(f"  Healthy: process up, pid={resp.process_info.pid}")
                return True
            finally:
                client.close()
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
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    restart: bool,
    dry_run: bool,
    skip_verify: bool,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    config = load_config(Path(config_path))
    configure_client_s3(config)

    iris_config = IrisConfig(config)
    platform = iris_config.platform()

    remote_state_dir = config.storage.remote_state_dir
    if not remote_state_dir:
        raise click.ClickException("storage.remote_state_dir is required")

    # Step 1: Discover controller and establish tunnel
    click.echo("=" * 60)
    click.echo("Step 1: Connect to controller")
    click.echo("=" * 60)

    controller_address = platform.discover_controller(config.controller)
    click.echo(f"  Controller address: {controller_address}")

    with platform.tunnel(controller_address) as controller_url:
        click.echo(f"  Tunnel URL: {controller_url}")

        # Step 2: Checkpoint
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 2: Take checkpoint")
        click.echo("=" * 60)

        if dry_run:
            click.echo("  [dry-run] Would call BeginCheckpoint RPC")
        else:
            resp = trigger_checkpoint(controller_url)
            click.echo(f"  Checkpoint: {resp.checkpoint_path}")
            click.echo(f"  Jobs: {resp.job_count}, Tasks: {resp.task_count}, " f"Workers: {resp.worker_count}")

            db_path = f"{resp.checkpoint_path}/{ControllerDB.DB_FILENAME}"
            if resp.checkpoint_path and _remote_exists(db_path):
                click.echo(f"  Verified: {db_path} exists")
            else:
                click.echo(f"  WARNING: checkpoint DB not found at {db_path}")

    # Step 3: Restart controller (outside tunnel — the restart kills the pod/VM)
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

    click.echo("  Restarting via `iris cluster controller restart`...")
    subprocess.run(
        ["uv", "run", "iris", "--config", config_path, "cluster", "controller", "restart"],
        timeout=600,
        check=True,
    )

    # Step 4: Verify health (re-establish tunnel to the new controller)
    if not skip_verify:
        click.echo()
        click.echo("=" * 60)
        click.echo("Step 4: Verify controller health")
        click.echo("=" * 60)

        click.echo("  Waiting for controller to become healthy...")
        controller_address = platform.discover_controller(config.controller)
        with platform.tunnel(controller_address) as controller_url:
            healthy = verify_controller_health(controller_url)
        if healthy:
            click.echo("  Controller is healthy!")
        else:
            click.echo("  WARNING: Controller did not respond within timeout", err=True)
            sys.exit(1)

    click.echo()
    click.echo("Done.")


if __name__ == "__main__":
    main()
