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

"""Fray CLI for cluster job management."""

import contextlib
import datetime
import getpass
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import click
import yaml

from fray.cluster import Cluster, create_cluster
from fray.cluster.base import CpuConfig, Entrypoint, EnvironmentConfig, GpuConfig, JobRequest, ResourceConfig, TpuConfig

logger = logging.getLogger(__name__)


@click.group()
@click.option("--cluster", required=True, help="Cluster: 'local' or 'ray:region' or 'ray:/path/to/config.yaml'")
@click.pass_context
def main(ctx: click.Context, cluster: str):
    """Fray cluster job management."""
    ctx.ensure_object(dict)
    ctx.obj["cluster_spec"] = cluster


@main.command()
@click.option("--extra", type=str, default="", help="Dependency groups: 'preprocessing,tpu'")
@click.option("--cpus", type=int, default=1, help="CPUs (default: 1)")
@click.option("--memory", type=str, default="4g", help="Memory (default: 4g)")
@click.option("--disk", type=str, default="10g", help="Disk (default: 10g)")
@click.option("--tpu", type=str, help="TPU type: 'v4-8', 'v5p-16'")
@click.option("--gpu", type=str, help="GPU type: 'A100', 'H100'")
@click.option("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
@click.option("--env", "-e", multiple=True, help="Environment variables: KEY=VALUE or KEY")
@click.option("--auto-stop", is_flag=True, help="Stop job on exit")
@click.option("--no-wait", is_flag=True, help="Don't wait for completion")
@click.argument("command", nargs=-1, required=True)
@click.pass_context
def submit(ctx, extra, cpus, memory, disk, tpu, gpu, gpu_count, env, auto_stop, no_wait, command):
    """Submit a job to the cluster.

    Examples:
        fray --cluster=local submit -- python script.py --arg=value
        fray --cluster=ray:us-west2 submit --tpu=v4-8 -- python train.py
        fray --cluster=ray:us-east1 submit --extra=preprocessing -- python process.py
    """
    # Parse command
    if not command:
        click.echo("Error: No command provided", err=True)
        sys.exit(1)

    full_cmd = list(command)

    # Load env vars from .marin.yaml
    env_dict = {}
    marin_yaml = Path(".marin.yaml")
    if marin_yaml.exists():
        try:
            with open(marin_yaml, "r") as f:
                marin_cfg = yaml.safe_load(f) or {}
            if isinstance(marin_cfg.get("env"), dict):
                for k, v in marin_cfg["env"].items():
                    env_dict[str(k)] = "" if v is None else str(v)
        except Exception as e:
            logger.warning(f"Failed to parse .marin.yaml: {e}")

    for env_var in env:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env_dict[key] = value
        else:
            env_dict[env_var] = ""

    try:
        env_dict["GIT_COMMIT"] = subprocess.getoutput("git rev-parse HEAD")
    except Exception as e:
        logger.warning(f"Failed to get git commit hash: {e}")

    # Build device config
    device = CpuConfig()
    if tpu and gpu:
        click.echo("Error: Cannot specify both --tpu and --gpu", err=True)
        sys.exit(1)

    if tpu:
        tpu_chips = parse_tpu_count(tpu)
        device = TpuConfig(type=tpu, count=tpu_chips)
    elif gpu:
        device = GpuConfig(type=gpu, count=gpu_count)

    # Build environment config
    extra_groups = [g for g in extra.split(",") if g]
    if tpu and "tpu" not in extra_groups:
        extra_groups.append("tpu")

    env_config = EnvironmentConfig(
        workspace=os.getcwd(),
        extras=extra_groups,
        env_vars=env_dict,
    )

    resources = ResourceConfig(
        cpu=cpus,
        ram=memory,
        disk=disk,
        device=device,
    )

    entrypoint = Entrypoint(
        binary=full_cmd[0],
        args=full_cmd[1:],
    )

    job_req = JobRequest(
        name=generate_job_name(" ".join(full_cmd)),
        entrypoint=entrypoint,
        resources=resources,
        environment=env_config,
    )

    cluster_spec = ctx.obj["cluster_spec"]
    cluster = create_cluster(cluster_spec)

    with cluster.connect():
        submit_and_monitor(cluster, job_req, tpu, auto_stop, no_wait)


@contextlib.contextmanager
def auto_stop(cluster: Cluster, job_id: str, should_stop: bool):
    """Terminate `job_id` on exit if `should_stop` is True."""
    if not should_stop:
        yield
        return

    try:
        yield
    finally:
        click.echo(f"Auto-stopping job {job_id}...")
        try:
            cluster.terminate(job_id)
        except Exception as e:
            logger.error(f"Failed to stop: {e}")


def submit_and_monitor(cluster, job_req, is_tpu, should_stop, no_wait):
    """Submit job and handle monitoring/cleanup."""
    job_id = cluster.launch(job_req)
    with auto_stop(cluster, job_id, should_stop and not no_wait):
        click.echo(f"Job submitted: {job_id}")

        # TPU jobs: print and exit
        if is_tpu:
            click.echo("Note: TPU jobs don't support log streaming. Check dashboard.")
            return

        # Stream logs unless --no-wait
        if not no_wait:
            try:
                for line in cluster.monitor(job_id):
                    print(line, end="")
            except KeyboardInterrupt:
                click.echo("\nInterrupted. Job still running.")


@main.command("list")
@click.pass_context
def list_jobs(ctx):
    """List all jobs."""
    cluster_spec = ctx.obj["cluster_spec"]
    cluster = create_cluster(cluster_spec)

    with cluster.connect():
        jobs = cluster.list_jobs()

        if not jobs:
            click.echo("No jobs found.")
            return

        # Table header
        click.echo(f"{'JOB_ID':<40} {'NAME':<30} {'STATUS':<12} {'START':<20}")
        click.echo("-" * 102)
        for job in jobs:
            if job.start_time:
                start = datetime.datetime.fromtimestamp(job.start_time).strftime("%Y-%m-%d %H:%M:%S")
            else:
                start = "N/A"
            click.echo(f"{job.job_id:<40} {job.name:<30} {job.status:<12} {start:<20}")


@main.command()
@click.argument("job_id")
@click.pass_context
def wait(ctx, job_id):
    """Wait for job completion."""
    cluster_spec = ctx.obj["cluster_spec"]
    cluster = create_cluster(cluster_spec)

    with cluster.connect():
        click.echo(f"Waiting for {job_id}...")
        info = cluster.wait(job_id)
        click.echo(f"Job finished: {info.status}")
        if info.error_message:
            click.echo(f"Error: {info.error_message}", err=True)


@main.command()
@click.argument("job_id")
@click.pass_context
def terminate(ctx, job_id):
    """Terminate a job."""
    cluster_spec = ctx.obj["cluster_spec"]
    cluster = create_cluster(cluster_spec)

    with cluster.connect():
        cluster.terminate(job_id)
        click.echo(f"Terminated: {job_id}")


@main.command()
@click.argument("job_id")
@click.pass_context
def logs(ctx, job_id):
    """Stream logs from a job."""
    cluster_spec = ctx.obj["cluster_spec"]
    cluster = create_cluster(cluster_spec)

    with cluster.connect():
        try:
            for line in cluster.monitor(job_id):
                print(line, end="")
        except KeyboardInterrupt:
            click.echo("\nInterrupted.")


@main.command()
@click.option("--clusters", type=str, help="Comma-separated regions/configs")
def dashboard(clusters):
    """Start dashboard SSH tunnels for Ray clusters.

    Examples:
        fray dashboard --clusters=us-west2,us-east1
        fray dashboard  # Auto-discover all active clusters
    """
    from fray.cluster.ray import DashboardConfig, ray_dashboard

    cluster_configs = []
    if clusters:
        for c in clusters.split(","):
            cluster_configs.append(parse_cluster_config(f"ray:{c}"))

    cfg = DashboardConfig(cluster_configs=cluster_configs)
    click.echo("Starting dashboard SSH tunnels...")
    click.echo("Press Ctrl+C to stop.")

    try:
        with ray_dashboard(cfg) as conn:
            click.echo("\nActive dashboard tunnels:")
            for name, info in conn.clusters.items():
                ports = conn.port_mappings[name]
                click.echo(f"  {name} ({info.head_ip}):")
                click.echo(f"    Dashboard: http://localhost:{ports.dashboard_port}")
                click.echo(f"    GCS: localhost:{ports.gcs_port}")
                click.echo(f"    API: localhost:{ports.api_port}")

            import time

            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nShutting down.")


def parse_cluster_config(cluster_spec: str) -> str:
    """Parse 'ray:region' or 'ray:/path' to config path."""
    if cluster_spec.startswith("ray:/"):
        return cluster_spec[4:]  # Strip 'ray:' prefix
    elif cluster_spec.startswith("ray:"):
        region = cluster_spec[4:]
        from fray.cluster.ray.config import find_config_by_region

        return find_config_by_region(region)
    raise ValueError(f"Invalid cluster spec: {cluster_spec}")


def parse_tpu_count(tpu_type: str) -> int:
    """Parse TPU chip count from type."""
    if tpu_type in {"v4-8", "v5p-8"}:
        return 4
    match = re.search(r"-(\d+)$", tpu_type)
    if not match:
        raise ValueError(f"Cannot parse TPU type: {tpu_type}")
    chips = int(match.group(1))
    if chips > 8:
        raise ValueError("Only single TPU nodes supported")
    return chips


def generate_job_name(command: str) -> str:
    """Generate job name from command."""
    parts = command.split()
    entrypoint = parts[0] if parts else "unknown"
    if "/" in entrypoint:
        entrypoint = entrypoint.split("/")[-1]
    if "." in entrypoint:
        entrypoint = entrypoint.split(".")[0]

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"fray-{getpass.getuser()}-{entrypoint}-{timestamp}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
