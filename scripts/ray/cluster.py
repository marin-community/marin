#!/usr/bin/env python3
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

"""
Usage:
  uv run scripts/ray/cluster.py --help
  uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml start-cluster
  uv run scripts/ray/cluster.py ssh-tpu 10.128.0.42
  uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml list-jobs
  uv run scripts/ray/cluster.py update-configs
"""

from dataclasses import dataclass
import json
import logging
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click

from src.marin.cluster import cleanup, gcp, monitoring, ray
from src.marin.cluster.config import (
    RayClusterConfig,
    find_config_by_region,
    list_available_configs,
    update_cluster_configs,
)

logger = logging.getLogger(__name__)


def check_cluster_head_running(config_path: str) -> bool:
    """Check if a Ray cluster head is already running.

    Returns True if a cluster head is detected, False otherwise.
    """
    try:
        # Try to connect to the dashboard to see if cluster is running
        with ray.ray_dashboard(config_path):
            # If we can get cluster resources, the head is running
            ray.get_ray_cluster_resources()
            return True
    except Exception:
        # Any exception means we couldn't connect to a running cluster
        return False


@dataclass
class Context:
    verbose: bool = False
    config_file: Optional[str] = None
    config_obj: Optional[RayClusterConfig] = None


# Context object to pass global options between commands
@click.group()
@click.option("--config", help="Path to cluster config file")
@click.option("--cluster", help="Cluster name to connect to")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, cluster, verbose):
    """Marin cluster management CLI."""
    ctx.ensure_object(Context)
    if cluster:
        config = find_config_by_region(cluster)

    ctx.obj.config_file = config
    ctx.obj.verbose = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if config:
        ctx.obj.config_obj = RayClusterConfig.from_yaml(config)


# Cluster commands
@cli.command("start-cluster")
@click.pass_context
def start_cluster(ctx):
    """Start the specified cluster."""
    config_obj, config_path = ctx.obj.config_obj, ctx.obj.config_file
    if not config_obj or not config_path:
        print("Error: --config required for cluster commands", file=sys.stderr)
        sys.exit(1)

    # Check if cluster head is already running
    if check_cluster_head_running(config_path):
        print(f"Warning: Cluster head for {config_obj.cluster_name} appears to already be running.")
        print("This may cause conflicts or unexpected behavior.")
        print("Consider running 'status' to check the current state,")
        print("or 'stop' first if you want to restart the cluster.")
        print()

    print(f"Starting cluster {config_obj.cluster_name}...")
    subprocess.run(["ray", "up", "-y", config_path], check=True)


@cli.command("stop-cluster")
@click.pass_context
def stop_cluster(ctx):
    """Stop cluster."""
    config_obj, config_path = ctx.obj.config_obj, ctx.obj.config_file
    if not config_obj or not config_path:
        print("Error: --config required for cluster commands", file=sys.stderr)
        sys.exit(1)

    print(f"Stopping cluster {config_obj.cluster_name}...")
    subprocess.run(["ray", "down", "-y", config_path], check=True)
    print("Cluster stopped successfully!")


@cli.command("restart-cluster")
@click.pass_context
@click.option("--preserve-jobs", help="Whether to preserve jobs during restart", default=True)
def restart_cluster(ctx, preserve_jobs):
    """Restart cluster with job preservation."""
    config_obj, config_path = ctx.obj.config_obj, ctx.obj.config_file
    if not config_obj or not config_path:
        print("Error: --config required for cluster commands", file=sys.stderr)
        sys.exit(1)

    print(f"Restarting cluster {config_obj.cluster_name}...")

    if not preserve_jobs:
        print("Stopping cluster...")
        subprocess.run(["ray", "down", "-y", config_path], check=True)

        print("Starting cluster...")
        subprocess.run(["ray", "up", "-y", "--no-config-cache", config_path], check=True)
        print("Cluster restarted successfully!")
        return

    # Backup jobs
    print("Backing up jobs...")
    with tempfile.TemporaryDirectory() as backup_dir:
        with ray.ray_dashboard(config_path):
            ray.backup_jobs(config_path, backup_dir)

        # Restart cluster
        print("Stopping cluster...")
        subprocess.run(["ray", "down", "-y", config_path], check=True)

        print("Starting cluster...")
        subprocess.run(["ray", "up", "-y", "--no-config-cache", config_path], check=True)

        # Restore jobs
        print("Restoring jobs...")
        with ray.ray_dashboard(config_path):
            ray.restore_jobs(config_path, backup_dir)

    print("Cluster restarted successfully!")


@cli.command("backup-jobs")
@click.argument("backup_dir")
@click.pass_context
def cluster_backup_jobs(ctx, backup_dir):
    """Backup Ray jobs to specified directory."""
    with ray.ray_dashboard(ctx.obj.config_file):
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        ray.backup_jobs(ctx.obj.config_file, backup_dir)
        print(f"Jobs backed up successfully to {backup_dir}")


@cli.command("restore-jobs")
@click.argument("backup_dir")
@click.pass_context
def cluster_restore_jobs(ctx, backup_dir):
    """Restore Ray jobs from specified directory."""
    with ray.ray_dashboard(ctx.obj.config_file):
        ray.restore_jobs(ctx.obj.config_file, backup_dir)
        print(f"Jobs restored successfully from {backup_dir}")


@cli.command("get-status")
@click.pass_context
def get_status(ctx):
    """Get cluster status."""
    with ray.ray_dashboard(ctx.obj.config_file):
        status_result = ray.get_ray_cluster_resources()
        print(json.dumps(status_result, indent=2))


@cli.command("list-configs")
@click.pass_context
def cluster_list_configs(ctx):
    """List available cluster configurations."""
    configs = list_available_configs()
    if not configs:
        print("No cluster configurations found in infra/")
        return

    print("Available cluster configurations:")
    for config_path in configs:
        print(f"  {config_path}")


@cli.command("update-configs")
@click.pass_context
def cluster_update_configs(ctx):
    """Update all cluster configuration files from templates."""
    print("Updating cluster configuration files...")
    update_cluster_configs("infra/")
    print("Cluster configurations updated successfully!")


# SSH commands
@cli.command("ssh-tpu")
@click.argument("target")
@click.option("--project", help="GCP project ID")
@click.option("--zone", help="GCP zone")
@click.argument("extra_args", nargs=-1)
@click.pass_context
def ssh_connect(ctx, target, project, zone, extra_args):
    """SSH to TPU node by IP address."""
    project = project or gcp.get_project_id()
    zone = zone or gcp.get_default_zone()

    if not project or not zone:
        print("Error: Could not determine project or zone", file=sys.stderr)
        sys.exit(1)

    # Find TPU by IP and SSH to it
    tpu_result = gcp.find_tpu_by_ip(target, project, zone)
    if tpu_result:
        tpu_name, tpu_zone, worker_id = tpu_result
        print(f"Connecting to TPU {tpu_name} worker {worker_id} at IP {target}")
        gcp.ssh_to_tpu(tpu_name, tpu_zone, project, list(extra_args) if extra_args else None, worker_id)
    else:
        print(f"Error: No TPU found with IP {target}", file=sys.stderr)
        sys.exit(1)


@cli.command("ssh-head")
@click.argument("extra_args", nargs=-1)
@click.pass_context
def ssh_head(ctx, extra_args):
    """SSH to cluster head node using ray attach."""
    cmd_args = ["ray", "attach", ctx.obj.config_file]
    if extra_args:
        cmd_args.extend(["--", *extra_args])
    subprocess.run(cmd_args, check=True)


@cli.command("list-workers")
@click.pass_context
def list_workers(ctx):
    """List Ray workers."""
    with ray.ray_dashboard(ctx.obj.config_file):
        print(json.dumps(ray.list_workers(), indent=2))


# Job commands
@cli.command("list-jobs")
@click.pass_context
def list_jobs(ctx):
    """List Ray jobs."""
    with ray.ray_dashboard(ctx.obj.config_file):
        print(json.dumps(ray.list_jobs(), indent=2))


@cli.command("submit-job")
@click.argument("entrypoint")
@click.option("--working-dir", help="Working directory for the job")
@click.option("--runtime-env", help="Runtime environment JSON")
@click.pass_context
def submit_job(ctx, entrypoint, working_dir, runtime_env):
    """Submit a Ray job."""
    runtime_env_dict = json.loads(runtime_env) if runtime_env else None

    with ray.ray_dashboard(ctx.obj.config_file):
        job_id = ray.submit_job(entrypoint, working_dir, runtime_env_dict)
        print(f"Job submitted with ID: {job_id}")


# Clean commands
@cli.command("clean-tpu-processes")
@click.option("--tpu-type", default="v4-8", help="TPU type (default: v4-8)")
@click.pass_context
def clean_tpu_processes(ctx, tpu_type):
    """Kill straggling TPU processes."""
    cleanup.kill_tpu_processes(tpu_type)
    print(f"Submitted cleanup job for TPU type {tpu_type}")


@cli.command("clean-preempted-tpus")
@click.option("--dry-run", is_flag=True, help="Show what would be cleaned")
@click.pass_context
def clean_preempted_tpus(ctx, dry_run):
    """Clean preempted TPU nodes."""
    config_obj = ctx.obj.config_obj
    deleted = gcp.cleanup_preempted_tpus(config_obj.project_id, config_obj.zone, dry_run)
    action = "Would delete" if dry_run else "Deleted"
    print(f"{action} {len(deleted)} preempted TPUs: {deleted}")


# Top-level commands
@cli.command("add-worker")
@click.argument("tpu_type")
@click.option(
    "--capacity",
    type=click.Choice(["preemptible", "best_effort", "reserved"]),
    default="preemptible",
    help="Capacity type",
)
@click.option("--name", help="Custom TPU name")
@click.pass_context
def add_worker(ctx, tpu_type, capacity, name):
    """Add manual TPU worker to cluster."""
    config_obj = ctx.obj.config_obj
    print(f"Adding {tpu_type} worker with {capacity} capacity...")
    ray.add_manual_worker(config_obj, tpu_type, capacity, name)
    print("Worker added successfully!")


@cli.command("init-worker")
@click.argument("name")
@click.pass_context
def init_worker(ctx, name):
    """Initialize Ray on a manual TPU worker."""
    config_obj = ctx.obj.config_obj
    print(f"Initializing Ray on worker {name}...")
    ray.initialize_manual_worker(config_obj.config_file, name)
    print("Worker initialized successfully!")


@cli.command("open-dashboard")
@click.pass_context
def open_dashboard(ctx):
    """Open Ray dashboard in browser."""
    with ray.ray_dashboard(ctx.obj.config_file) as dashboard_url:
        print(f"Ray dashboard is running at: {dashboard_url}")
        print("Press Ctrl+C to stop the dashboard.")
        # Keep the context manager alive until interrupted
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Stopping Ray dashboard...")


@cli.command("monitor-cluster")
@click.option("--wandb", is_flag=True, help="Log metrics to wandb")
@click.pass_context
def monitor_cluster(ctx, wandb):
    """Monitor cluster health."""
    config_obj = ctx.obj.config_obj
    if not config_obj:
        print("Error: --config required for monitoring", file=sys.stderr)
        sys.exit(1)

    config_zones = {config_obj.region: [config_obj.zone]}
    health_data = monitoring.monitor_cluster_health(
        config_zones=config_zones, project=config_obj.project_id, log_to_wandb=wandb
    )

    summary = monitoring.get_cluster_health_summary(health_data)
    print(summary)

    if wandb:
        print("Metrics logged to wandb")


@cli.command("show-logs")
@click.option("--tail", default=100, help="Number of lines to tail")
@click.pass_context
def show_logs(ctx, tail):
    """View cluster logs."""
    log_command = f"tail -n {tail} -f /tmp/ray/session_latest/logs/monitor*"
    subprocess.run(["ray", "exec", ctx.obj.config_file, log_command], check=True)


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()


if __name__ == "__main__":
    main()
