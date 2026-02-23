# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
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

"""Command-line interface for marin cluster job management."""

import subprocess
import sys
import time

import click
import yaml

from fray.v1.cluster.ray import DashboardConfig, ray_dashboard
from marin.cluster.config import RayClusterConfig, find_config_by_region, resolve_infra_dir


def _resolve_cluster_config(cluster: str, infra_dir: str | None) -> str:
    """Resolve cluster identifier to a YAML config path."""
    if cluster.endswith(".yaml"):
        return cluster
    return find_config_by_region(cluster, infra_dir=resolve_infra_dir(infra_dir))


@click.group()
def main():
    """Marin cluster job management."""
    pass


@main.command()
@click.argument("script")
@click.option("--cluster", required=True, help="Cluster name or config path")
@click.option("--extra", default="", help="Dependency extras (e.g., 'cpu,tpu')")
@click.option("--infra-dir", help="Path to infra/ directory with cluster configs")
@click.option("-e", "--env", "env_vars", multiple=True, help="Environment variable KEY=VALUE")
@click.option("--no-wait", is_flag=True, help="Don't wait for job completion")
@click.option("--auto-stop", is_flag=True, help="Stop job on exit")
@click.option("--tpu", help="TPU type to reserve (e.g., 'v4-8')")
@click.option("--entrypoint-num-cpus", type=float, help="Number of CPUs to reserve for entrypoint")
@click.option("--entrypoint-num-gpus", type=float, help="Number of GPUs to reserve for entrypoint")
@click.option("--entrypoint-memory", type=int, help="Memory in bytes to reserve for entrypoint")
@click.option("--submission-id", help="Custom submission ID for the job")
@click.argument("script_args", nargs=-1)
def submit(
    script,
    cluster,
    extra,
    infra_dir,
    env_vars,
    no_wait,
    auto_stop,
    tpu,
    entrypoint_num_cpus,
    entrypoint_num_gpus,
    entrypoint_memory,
    submission_id,
    script_args,
):
    """Submit a script to Ray cluster.

    Submit SCRIPT to a Ray cluster with optional dependencies and environment variables.

    Example:

        marin submit speedrun.py --cluster=us-central2 --extra=cpu -- --epochs=10
    """
    ray_cmd = ["uv", "run", "python", "-m", "marin.run.ray_run"]

    ray_cmd.extend(["--cluster", cluster])

    if extra:
        ray_cmd.extend(["--extra", extra])

    if infra_dir:
        ray_cmd.extend(["--infra-dir", infra_dir])

    if no_wait:
        ray_cmd.append("--no_wait")

    if auto_stop:
        ray_cmd.append("--auto-stop")

    if tpu:
        ray_cmd.extend(["--tpu", tpu])

    if entrypoint_num_cpus is not None:
        ray_cmd.extend(["--entrypoint-num-cpus", str(entrypoint_num_cpus)])

    if entrypoint_num_gpus is not None:
        ray_cmd.extend(["--entrypoint-num-gpus", str(entrypoint_num_gpus)])

    if entrypoint_memory is not None:
        ray_cmd.extend(["--entrypoint-memory", str(entrypoint_memory)])

    if submission_id:
        ray_cmd.extend(["--submission-id", submission_id])

    for env_var in env_vars:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            ray_cmd.extend(["-e", key, value])
        else:
            ray_cmd.extend(["-e", env_var])

    ray_cmd.append("--")
    ray_cmd.append("python")
    ray_cmd.append(script)
    ray_cmd.extend(script_args)

    result = subprocess.run(ray_cmd)
    sys.exit(result.returncode)


@main.command("dashboard")
@click.option("--cluster", required=True, help="Cluster name or config path")
@click.option("--infra-dir", help="Path to infra/ directory with cluster configs")
@click.option("--open-browser", is_flag=True, help="Open dashboard URL in browser")
def dashboard(cluster: str, infra_dir: str | None, open_browser: bool) -> None:
    """Open an SSH tunnel to a cluster dashboard and keep it alive."""
    cluster_config = _resolve_cluster_config(cluster, infra_dir)

    with ray_dashboard(DashboardConfig.from_cluster(cluster_config)) as connection:
        cluster_name = next(iter(connection.clusters.keys()))
        ports = connection.port_mappings[cluster_name]
        dashboard_url = f"http://localhost:{ports.dashboard_port}"

        click.echo(f"Cluster: {cluster_name}")
        click.echo(f"Config: {cluster_config}")
        click.echo(f"Dashboard: {dashboard_url}")
        click.echo("Press Ctrl+C to close the dashboard tunnel.")

        if open_browser:
            import webbrowser

            webbrowser.open(dashboard_url)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\nClosing dashboard tunnel.")


@main.command("cluster-info")
@click.option("--cluster", required=True, help="Cluster name or config path")
@click.option("--infra-dir", help="Path to infra/ directory with cluster configs")
def cluster_info(cluster: str, infra_dir: str | None) -> None:
    """Print resolved cluster configuration details."""
    cluster_config = _resolve_cluster_config(cluster, infra_dir)
    resolved_infra = resolve_infra_dir(infra_dir)

    cfg = RayClusterConfig.from_yaml(cluster_config)
    with open(cluster_config, "r") as f:
        raw = yaml.safe_load(f) or {}

    click.echo(f"cluster_name: {cfg.cluster_name}")
    click.echo(f"config_file: {cluster_config}")
    click.echo(f"infra_dir: {resolved_infra}")
    click.echo(f"project_id: {cfg.project_id}")
    click.echo(f"region: {cfg.region}")
    click.echo(f"zone: {cfg.zone}")
    click.echo(f"docker_image: {cfg.docker_image}")
    click.echo(f"head_node_type: {raw.get('head_node_type', '')}")


if __name__ == "__main__":
    main()
