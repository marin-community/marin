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
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import click
import yaml

from fray.v1.cluster.ray.auth import ray_auth_secret
from fray.v1.cluster.ray.dashboard import DashboardConfig, ray_dashboard
from marin.cluster import gcp
from marin.cluster.config import (
    RayClusterConfig,
    find_config_by_region,
    list_available_configs,
    update_cluster_configs,
)

TPU_TYPE_TO_VM_IMAGE = {
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}


logger = logging.getLogger(__name__)


def _list_jobs(filters: list[str] | None = None) -> list[dict]:
    """Fetch the list of jobs using the Ray CLI."""
    cmd = ["ray", "list", "jobs", "--detail", "--format=json", "--limit=10000"]
    for f in filters or []:
        cmd.extend(["--filter", f])

    result = subprocess.check_output(cmd, text=True, timeout=60)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON output from ray list jobs: {result}")
        return []


def _select_job(jobs: list[dict], job_id: str, match: bool) -> dict | None:
    if match:
        matches = [
            job for job in jobs if job_id in (job.get("submission_id") or "") or job_id in (job.get("job_id") or "")
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda job: (job.get("start_time") or 0, job.get("submission_id") or ""),
        )

    for job in jobs:
        if job_id == job.get("submission_id") or job_id == job.get("job_id"):
            return job
    return None


def _print_job_logs(job_id: str, tail: int | None = None, grep: str | None = None) -> None:
    cmd = ["ray", "job", "logs", job_id]
    ray_proc = None
    grep_proc = None
    if tail is not None or grep is not None:
        try:
            ray_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if grep is not None:
                grep_proc = subprocess.Popen(
                    ["grep", grep],
                    stdin=ray_proc.stdout,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                ray_proc.stdout.close()
                prev_proc = grep_proc
            else:
                prev_proc = ray_proc

            if tail is not None:
                tail_proc = subprocess.Popen(
                    ["tail", f"-{tail}"],
                    stdin=prev_proc.stdout,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                prev_proc.stdout.close()
                output, _ = tail_proc.communicate(timeout=120)
            else:
                output, _ = prev_proc.communicate(timeout=120)

            print(output, end="")
        finally:
            if grep_proc is not None:
                try:
                    grep_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    grep_proc.kill()
            if ray_proc is not None:
                try:
                    ray_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ray_proc.kill()
    else:
        subprocess.run(cmd, check=False)


def _submit_job(
    entrypoint: str,
    working_dir: str | None = None,
    runtime_env: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> str:
    """Submit a job to Ray cluster and return job ID."""
    cmd = ["ray", "job", "submit"]

    if working_dir:
        cmd.extend(["--working-dir", working_dir])

    if runtime_env:
        cmd.extend(["--runtime-env-json", json.dumps(runtime_env)])

    if resources:
        for resource, amount in resources.items():
            cmd.extend([f"--{resource}", str(amount)])

    cmd.extend(["--", entrypoint])

    result = subprocess.check_output(cmd, text=True, timeout=500)
    # Extract job ID from output (usually in format "Job submitted with ID: <id>")
    output_lines = result.strip().split("\n")
    for line in output_lines:
        if "submitted with ID:" in line:
            return line.split(":")[-1].strip()

    # Fallback: return full output if we can't parse job ID
    return result.strip()


def _stop_job(job_id: str) -> None:
    """Stop a running Ray job.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.

    Args:
        job_id: The job ID or submission ID to stop
    """
    cmd = ["ray", "job", "stop", job_id]
    subprocess.check_output(cmd, text=True, timeout=60)


def _add_manual_worker(
    config: RayClusterConfig,
    tpu_type: str,
    capacity_type: str = "preemptible",
    tpu_name: str | None = None,
    version: str | None = None,
) -> None:
    """Add a manual TPU worker to the cluster.

    Args:
        config: Cluster configuration
        tpu_type: TPU type (e.g., v4-128, v5p-8)
        capacity_type: Capacity type (reserved, preemptible, best_effort)
        tpu_name: Custom TPU name (generated if None)
        version: TPU VM image version (auto-detected if None)
    """
    from levanter.infra.cli_helpers import default_run_id
    from levanter.infra.tpus import (
        setup_vm_docker,
        start_tpu_vm_queued_resources,
    )

    # Generate TPU name if not provided
    if tpu_name is None:
        tpu_name = f"ray-worker-manual-{default_run_id()}"

    # Determine TPU generation and version
    tpu_gen = tpu_type.split("-")[0]
    if version is None:
        version = TPU_TYPE_TO_VM_IMAGE.get(tpu_gen, "tpu-ubuntu2204-base")

    logger.info(f"Creating TPU with name: {tpu_name}")
    start_tpu_vm_queued_resources(
        tpu_name=tpu_name,
        tpu_type=tpu_type,
        capacity_type=capacity_type,
        version=version,
        zone=config.zone,
        node_count=1,
    )

    # Setup Docker
    setup_vm_docker(
        tpu_name=tpu_name,
        zone=config.zone,
        node_count=1,
    )

    # Setup worker entrypoint
    logger.info(f"Setting up worker on TPU: {tpu_name}")
    _initialize_manual_worker(config.config_file, tpu_name)


def _initialize_manual_worker(config_file: str, tpu_name: str) -> None:
    """Setup the worker entrypoint script and start the container.

    This script configures the worker to automatically poll for a new head_ip
    at startup. This allows manual workers to resume in the case of a cluster restart.
    """
    from levanter.infra.tpus import run_command, tpu_ssh

    with open(config_file, "r") as f:
        cluster_config = yaml.safe_load(f)

    initialization_commands = cluster_config.get("initialization_commands", [])
    setup_commands = cluster_config.get("setup_commands", []) + cluster_config.get("worker_setup_commands", [])
    worker_run_options = cluster_config["docker"]["worker_run_options"]
    zone = cluster_config["provider"]["availability_zone"]
    cluster_name = cluster_config["cluster_name"]
    docker_container_name = cluster_config["docker"]["container_name"]
    docker_image = cluster_config["docker"]["image"]
    region = cluster_config["provider"]["region"]
    bucket = f"marin-{region}"

    print(f"Initializing Ray on worker {tpu_name}...")
    print(f"Zone: {zone}")
    print(f"Cluster name: {cluster_name}")
    print(f"Container name: {docker_container_name}")
    print(f"Docker image: {docker_image}")

    setup_commands = "\n".join(setup_commands)

    entry_script_content = f"""#!/bin/bash
set -x
set -eo pipefail

export BUCKET="{bucket}"

{setup_commands}

# Entry and setup commands will automatically re-run if the container is restarted

echo 'Checking for head node IP...'
gcloud compute instances list \\
  --filter="labels.ray-node-name:ray-{cluster_name}-head AND labels.ray-node-type=head" \\
  --format="value(networkInterfaces[0].networkIP)" > /tmp/head_ip

HEAD_IP=$(cat /tmp/head_ip | head -1 | awk '{{print $1}}' || true)
if [ -z "$HEAD_IP" ]; then
  echo 'Failed to resolve head node IP' >&2
  exit 1
fi

echo "Found head node IP: $HEAD_IP"
ray start --address=${{HEAD_IP}}:6379 --block
echo "Ray worker crashed. Sleeping 10 seconds to avoid rapid restart..."
sleep 10
    """

    init_commands = "\n".join(initialization_commands)

    init_script_content = f"""#!/bin/bash
{init_commands}
"""

    with (
        tempfile.NamedTemporaryFile("w", prefix="entry", suffix=".sh", delete=False) as entry_sh,
        tempfile.NamedTemporaryFile("w", prefix="init", suffix=".sh", delete=False) as init_sh,
    ):
        entry_sh.write(entry_script_content)
        init_sh.write(init_script_content)

        entry_sh.flush()
        init_sh.flush()
        run_command(
            *[
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "scp",
                "--worker=all",
                f"--zone={zone}",
                entry_sh.name,
                init_sh.name,
                f"{tpu_name}:/tmp/",
            ]
        )
        entry_name = os.path.basename(entry_sh.name)
        init_name = os.path.basename(init_sh.name)
        tpu_ssh(
            tpu_name,
            zone,
            1,
            " && ".join(
                [
                    f"mv /tmp/{init_name} /tmp/init.sh",
                    f"mv /tmp/{entry_name} /tmp/entry.sh",
                    "chmod 755 /tmp/init.sh /tmp/entry.sh",
                    "bash /tmp/init.sh",
                    f"(docker rm -f {docker_container_name} || true)",
                ]
            ),
        )

    # Start the Docker container
    docker_command = [
        "docker",
        "run",
        "-d",
        "--net=host",
        f"--name={docker_container_name}",
        "--init",
        "--privileged",
        *worker_run_options,
        docker_image,
        "/bin/bash",
        "/tmp/entry.sh",
    ]

    logger.info(f"Starting container: {' '.join(docker_command)}")
    tpu_ssh(tpu_name, zone, 1, *docker_command)


def _check_cluster_head_running(config_path: str) -> bool:
    """Check if a Ray cluster head is already running.

    Returns True if a cluster head is detected, False otherwise.
    """
    try:
        with ray_dashboard(DashboardConfig.from_cluster(config_path)):
            return True
    except Exception:
        return False


def _download_working_directory(
    cluster_config: str, job_id: str, working_dir: str, remote_working_dir: str, local_path: str
) -> str:
    """Download the working directory for `job_id`."""
    dest_dir = os.path.join(local_path, job_id, working_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_dir = os.path.join(dest_dir, "")  # Add trailing slash for rsync

    rsync_command = ["ray", "rsync-down", cluster_config, remote_working_dir, dest_dir]
    subprocess.check_output(rsync_command, text=True, timeout=300)

    logger.info(f"Working directory for job {remote_working_dir} saved to {dest_dir}")
    return dest_dir


def _save_runtime_env_entrypoint(job_details: dict[str, Any], job_id: str, local_path: str) -> dict[str, Any]:
    """Save the runtime environment and entrypoint for the job."""
    runtime_env = job_details.get("runtime_env", {})
    runtime_env.pop("working_dir", None)  # Remove unnecessary fields
    runtime_env.pop("_ray_commit", None)
    runtime_env["entrypoint"] = job_details["entrypoint"]

    env_file = os.path.join(local_path, job_id, "runtime_env.json")
    with open(env_file, "w") as f:
        json.dump(runtime_env, f, indent=4)
    return runtime_env


def _resubmit_job(
    job_id: str,
    entrypoint: str,
    working_dir: str,
    runtime_env: dict[str, Any] | None,
    raise_errors: bool,
) -> None:
    """Resubmit the job using the working directory and runtime environment."""
    runtime_env_args = ["--runtime-env-json", json.dumps(runtime_env)] if runtime_env else []

    logger.info(f"Resubmitting job {job_id}...")
    import shlex

    job_array = [
        "ray",
        "job",
        "submit",
        "--no-wait",
        "--working-dir",
        working_dir,
        *runtime_env_args,
        "--",
        *shlex.split(entrypoint),
    ]
    job_str = " ".join(job_array)

    logger.info(f"Submitting the job: {shlex.quote(job_str)}")

    try:
        subprocess.check_output(job_array, text=True, timeout=500)
        logger.info(f"Successfully resubmitted job {job_id}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to resubmit job {job_id}: {e}")
        if raise_errors:
            raise ValueError(f"Failed to resubmit job {job_id}") from e


def _backup_jobs(cluster_config: str, local_path: str, raise_errors: bool = False) -> None:
    """Backup jobs from the given Ray cluster.

    Note: This requires RAY_ADDRESS to be set, typically via start_ray_dashboard_with_wait.
    """
    logger.info("Fetching jobs from Ray Jobs API...")

    # Clear the backup directory if it exists
    backup_dir = Path(local_path)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True)

    jobs_data = _list_jobs()
    if not jobs_data:
        logger.info("No jobs found.")
        return

    for job_details in jobs_data:
        job_id = job_details["job_id"]
        status = job_details["status"]
        if status in {"SUCCEEDED", "FAILED", "STOPPED", "PENDING"}:
            continue
        logger.info(f"Backing up job {job_id} with status {status}...")
        runtime_env = job_details["runtime_env"]
        working_dir = runtime_env["working_dir"].split("/")[-1][:-4]
        remote_working_dir = f"/tmp/ray/session_latest/runtime_resources/working_dir_files/{working_dir}/"

        try:
            _download_working_directory(cluster_config, job_id, working_dir, remote_working_dir, local_path)
            _save_runtime_env_entrypoint(job_details, job_id, local_path)
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            if raise_errors:
                raise e

    logger.info("All jobs backed up.")


def _restore_jobs(local_path: str, raise_errors: bool = False) -> None:
    """Perform the 'after' stage actions: resubmit jobs.

    Note: This requires RAY_ADDRESS to be set, typically via start_ray_dashboard_with_wait.
    """
    backup_dir = Path(local_path)
    if not backup_dir.exists():
        logger.error("No backup data found. Run _backup_jobs() first.")
        return

    for job_id_dir in backup_dir.iterdir():
        if not job_id_dir.is_dir():
            continue

        files = list(job_id_dir.iterdir())
        if len(files) != 2:
            logger.warning(f"Incomplete backup for job {job_id_dir.name}. Skipping.")
            continue
        working_dir = next((f for f in files if f.is_dir()), None)
        runtime_env_file = job_id_dir / "runtime_env.json"

        if not working_dir or not runtime_env_file.exists():
            logger.warning(f"Incomplete backup for job {job_id_dir.name}. Skipping.")
            continue

        with open(runtime_env_file, "r") as f:
            runtime_env = json.load(f)

        entrypoint = runtime_env.pop("entrypoint", None)
        if not entrypoint:
            logger.error(f"No entrypoint found for job {job_id_dir.name}. Skipping.")
            if raise_errors:
                raise ValueError(f"No entrypoint found for job {job_id_dir.name}.")
            continue

        _resubmit_job(job_id_dir.name, entrypoint, str(working_dir), runtime_env, raise_errors)

    logger.info("All jobs resubmitted.")


@dataclass
class Context:
    verbose: bool = False
    config_file: str | None = None
    config_obj: RayClusterConfig | None = None


def _maybe_add_ray_verbose(ctx: Context, cmd_args: list[str]) -> list[str]:
    """Add `-v` to Ray CLI commands when cluster.py verbose mode is enabled.

    Most Ray CLI invocations here are of the form `["ray", "<subcommand>", ...]`.
    We insert `-v` after the subcommand (e.g. `ray up -v ...`) since Ray exposes per-subcommand verbose flags.
    """
    if ctx.verbose:
        if len(cmd_args) < 2:
            return cmd_args
        return [*cmd_args[:2], "-v", *cmd_args[2:]]
    return cmd_args


# Context object to pass global options between commands
@click.group()
@click.option("--config", help="Path to Ray cluster config file (infra/marin-*.yaml)")
@click.option("--cluster", help="Cluster name to connect to")
@click.option("--verbose", is_flag=True, help="Enable verbose logging (also passes `-v` to Ray cluster commands).")
@click.pass_context
def cli(ctx, config, cluster, verbose):
    """Marin cluster management CLI."""
    ctx.ensure_object(Context)
    # Auto-load shared env from .marin.yaml if present (common workflow)
    try:
        marin_yaml = Path(".marin.yaml")
        if marin_yaml.exists():
            with open(marin_yaml, "r") as f:
                data = yaml.safe_load(f) or {}
            env = data.get("env", {}) or {}
            if isinstance(env, dict):
                for k, v in env.items():
                    if k not in os.environ:
                        os.environ[k] = "" if v is None else str(v)
                logger.debug(f"Loaded env vars from {marin_yaml}")
    except Exception as e:
        logger.warning(f"Failed to load .marin.yaml: {e}")
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

    if _check_cluster_head_running(config_path):
        print(f"Warning: Cluster head for {config_obj.cluster_name} appears to already be running.")
        print("This may cause conflicts or unexpected behavior.")
        print("Consider running 'status' to check the current state,")
        print("or 'stop' first if you want to restart the cluster.")
        print()

    print(f"Starting cluster {config_obj.cluster_name}...")
    subprocess.run(_maybe_add_ray_verbose(ctx.obj, ["ray", "up", "-y", config_path]), check=True)


@cli.command("stop-cluster")
@click.pass_context
def stop_cluster(ctx):
    """Stop cluster."""
    config_obj, config_path = ctx.obj.config_obj, ctx.obj.config_file
    if not config_obj or not config_path:
        print("Error: --config required for cluster commands", file=sys.stderr)
        sys.exit(1)

    _stop_cluster_internal(ctx.obj, config_obj, config_path)
    print("Cluster stopped successfully!")


def _stop_cluster_internal(ctx: Context, config_obj: RayClusterConfig, config_path: str):
    """Terminate a Ray cluster.

    N.B. We terminate the Ray coordinator node first to avoid restarting any new TPUs while
    shutting down. We then explicitly shut down the TPU nodes in parallel. Ray serializes this
    and often times out by default.

    Finally we call ray down to finish up any leftover resources.
    """
    print(f"Terminating coordinator node for cluster {config_obj.cluster_name}...")
    terminated_head = gcp.terminate_head_node(config_obj.cluster_name, config_obj.project_id, config_obj.zone)
    if terminated_head:
        print(f"Terminated head node: {terminated_head}")

    print(f"Terminating TPUs for cluster {config_obj.cluster_name} in zone {config_obj.zone}...")
    terminated_tpus = gcp.terminate_tpus_in_cluster(config_obj.project_id, config_obj.zone, config_obj.cluster_name)
    if terminated_tpus:
        print(f"Terminated {len(terminated_tpus)} TPUs")

    print(f"Cleaning up Ray cluster state for {config_obj.cluster_name}...")
    subprocess.run(
        _maybe_add_ray_verbose(ctx, ["ray", "down", "-y", config_path]),
        check=False,  # check=False since instances may already be gone
    )


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
    backup_dir = tempfile.TemporaryDirectory()

    if preserve_jobs:
        print("Backing up jobs...")
        try:
            with ray_dashboard(DashboardConfig.from_cluster(config_path)):
                _backup_jobs(config_path, backup_dir.name)
        except Exception as e:
            print()
            print("=" * 60)
            print(
                f"Failed to back up jobs from cluster {config_obj.cluster_name} ({e}) "
                + "(disable with --preserve-jobs=0)"
            )
            print("=" * 60)
            print("Proceed with shutdown? (y/n): ", end="")
            choice = input().strip().lower()
            if choice != "y":
                print("Aborting cluster restart.")
                return
            print("Proceeding with cluster restart without job preservation.")

    print("Stopping cluster...")
    _stop_cluster_internal(ctx.obj, config_obj, config_path)

    print("Starting cluster...")
    subprocess.run(
        _maybe_add_ray_verbose(ctx.obj, ["ray", "up", "-y", "--no-config-cache", config_path]),
        check=True,
    )

    if preserve_jobs:
        print("Restoring jobs...")
        with ray_dashboard(DashboardConfig.from_cluster(config_path)):
            _restore_jobs(str(backup_dir))

    print("Cluster restarted successfully!")


@cli.command("backup-jobs")
@click.argument("backup_dir")
@click.pass_context
def cluster_backup_jobs(ctx, backup_dir):
    """Backup Ray jobs to specified directory."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        _backup_jobs(ctx.obj.config_file, backup_dir)
        print(f"Jobs backed up successfully to {backup_dir}")


@cli.command("restore-jobs")
@click.argument("backup_dir")
@click.pass_context
def cluster_restore_jobs(ctx, backup_dir):
    """Restore Ray jobs from specified directory."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        _restore_jobs(backup_dir)
        print(f"Jobs restored successfully from {backup_dir}")


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
    cmd_args = _maybe_add_ray_verbose(ctx.obj, ["ray", "attach", ctx.obj.config_file])
    if extra_args:
        cmd_args.extend(["--", *extra_args])
    subprocess.run(cmd_args, check=True)


@cli.command("list-workers")
@click.pass_context
def list_workers(ctx):
    """List Ray workers."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        result = subprocess.check_output(
            ["ray", "list", "workers", "--format=json", f"--limit={1000}"],
            text=True,
            timeout=60,
        )
        print(json.dumps(json.loads(result), indent=2))


# Job commands
@cli.command("list-jobs")
@click.pass_context
def list_jobs(ctx):
    """List Ray jobs."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        print(json.dumps(_list_jobs(), indent=2))


@cli.command("submit-job")
@click.argument("entrypoint")
@click.option("--working-dir", help="Working directory for the job")
@click.option("--runtime-env", help="Runtime environment JSON")
@click.pass_context
def submit_job(ctx, entrypoint, working_dir, runtime_env):
    """Submit a Ray job."""
    runtime_env_dict = json.loads(runtime_env) if runtime_env else None

    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        job_id = _submit_job(entrypoint, working_dir, runtime_env_dict)
        print(f"Job submitted with ID: {job_id}")


@cli.command("stop-job")
@click.argument("job_id")
@click.pass_context
def stop_job(ctx, job_id):
    """Stop a running Ray job."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        _stop_job(job_id)
        print(f"Job {job_id} stop requested")


@cli.command("job-logs")
@click.argument("job_id")
@click.option("--follow", "-f", is_flag=True, help="Follow the logs (stream in real-time)")
@click.option("--tail", "-n", type=int, default=None, help="Show only the last N lines of logs")
@click.option("--grep", "-g", type=str, default=None, help="Filter lines containing this pattern")
@click.pass_context
def job_logs(ctx, job_id, follow, tail, grep):
    """View logs for a Ray job."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        if follow:
            subprocess.run(["ray", "job", "logs", "--follow", job_id], check=False)
            return
        _print_job_logs(job_id, tail=tail, grep=grep)


@cli.command("wait-job")
@click.argument("job_id")
@click.option("--match", is_flag=True, help="Match job_id as a substring of submission_id.")
@click.option("--poll", type=float, default=5.0, show_default=True, help="Polling interval in seconds.")
@click.option("--timeout", type=float, default=None, help="Timeout in seconds before exiting.")
@click.option("--show-logs", is_flag=True, help="Print logs after completion.")
@click.option("--tail", "-n", type=int, default=200, show_default=True, help="Show only the last N lines of logs.")
@click.option("--grep", "-g", type=str, default=None, help="Filter lines containing this pattern.")
@click.pass_context
def wait_job(ctx, job_id, match, poll, timeout, show_logs, tail, grep):
    """Wait for a Ray job to finish."""
    with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
        if job_id == "latest":
            print("Error: job_id must be an explicit submission id; 'latest' is not supported.", file=sys.stderr)
            sys.exit(1)
        start = time.time()
        last_status = None

        while True:
            job = _select_job(_list_jobs(), job_id, match=match)
            if job is None:
                print(f"Job '{job_id}' not found.", file=sys.stderr)
                sys.exit(1)

            status = job.get("status")
            message = job.get("message", "")
            if status != last_status:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] {job.get('submission_id')} status={status} {message}".rstrip())
                last_status = status

            if status in {"SUCCEEDED", "FAILED", "STOPPED"}:
                break

            if timeout is not None and (time.time() - start) > timeout:
                print("Timeout reached while waiting for job completion.", file=sys.stderr)
                sys.exit(2)

            time.sleep(poll)

        if show_logs:
            _print_job_logs(job.get("submission_id") or job_id, tail=tail, grep=grep)


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
    _add_manual_worker(config_obj, tpu_type, capacity, name)
    print("Worker added successfully!")


@cli.command("init-worker")
@click.argument("name")
@click.pass_context
def init_worker(ctx, name):
    """Initialize Ray on a manual TPU worker."""
    config_obj = ctx.obj.config_obj
    print(f"Initializing Ray on worker {name}...")
    _initialize_manual_worker(config_obj.config_file, name)
    print("Worker initialized successfully!")


@cli.command("dashboard")
@click.pass_context
def open_dashboard(ctx):
    """Open dashboard for all active Ray clusters."""
    config_obj = ctx.obj.config_obj
    if config_obj:
        with ray_dashboard(DashboardConfig.from_cluster(ctx.obj.config_file)):
            try:
                time.sleep(86400)
            except KeyboardInterrupt:
                print("\nShutting down...")
        return

    with ray_dashboard(DashboardConfig()) as conn:
        if not conn.clusters:
            print("No active clusters found")
            return

        print(f"Connected to {len(conn.clusters)} clusters:")
        for name, info in conn.clusters.items():
            ports = conn.port_mappings[name]
            direct_url = f"http://localhost:{ports.dashboard_port}"
            print(f"  {name} ({info.zone}) - {direct_url}")
            print(f"    IP: {info.external_ip} ({info.head_ip})")
            dashboard_url = f"http://localhost:{ports.dashboard_port}"
            gcs_url = f"localhost:{ports.gcs_port}"
            api_url = f"localhost:{ports.api_port}"
            print(f"    Dashboard: {dashboard_url} | GCS: {gcs_url} | API: {api_url}")
            print()

        print("\nPress Ctrl+C to stop")
        try:
            time.sleep(86400)
        except KeyboardInterrupt:
            print("\nShutting down...")


@cli.command("auth")
@click.option(
    "--secret",
    default=None,
    help="GCP Secret Manager secret containing the Ray auth token (default: RAY_AUTH_TOKEN).",
)
@click.option(
    "--copy/--no-copy",
    default=True,
    help="Copy the token to your clipboard (default: copy).",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the Ray dashboard in your browser (default: open).",
)
@click.pass_context
def auth(ctx, secret: str | None, copy: bool, open_browser: bool):
    """Open a cluster dashboard and copy the Ray auth token to clipboard.

    This is a convenience wrapper for token-auth-enabled clusters to avoid
    manually fetching/pasting tokens.
    """
    if not ctx.obj.config_obj:
        raise click.UsageError("--config/--cluster is required for auth")

    config_path = ctx.obj.config_file
    secret = ray_auth_secret(secret)

    token = subprocess.check_output(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret}"],
        text=True,
    ).strip()
    if not token:
        raise RuntimeError(f"Secret {secret} returned empty token")

    if copy:
        clipboard_commands: list[list[str]] = [
            ["pbcopy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]

        used_clipboard_command: str | None = None
        for clipboard_cmd in clipboard_commands:
            if not shutil.which(clipboard_cmd[0]):
                continue
            try:
                subprocess.run(clipboard_cmd, input=token, text=True, check=True)
            except subprocess.CalledProcessError:
                continue
            used_clipboard_command = clipboard_cmd[0]
            break

        print()
        print("Ray dashboard token auth")
        print(f"- Token source: Secret Manager `{secret}`")
        if used_clipboard_command:
            print(f"- Token copied to clipboard via `{used_clipboard_command}` (paste when prompted).")
        else:
            print(token)
            print("- Clipboard copy unavailable (install `xclip` or `xsel`); token printed above.")

    with ray_dashboard(DashboardConfig.from_cluster(config_path)) as conn:
        cluster_name = next(iter(conn.clusters.keys()))
        ports = conn.port_mappings[cluster_name]
        url = f"http://localhost:{ports.dashboard_port}"
        print()
        print(f"Dashboard: {url}")
        print()
        print("Auth steps (Ray token auth prompt):")
        print("  1) Paste the token (already in your clipboard) into the prompt.")
        print("  2) Click Submit.")
        print()
        print("If you are not prompted, you already have a `ray-authentication-token` cookie for this host.")
        print("If the token has been rotated, clear that cookie or use an incognito window.")

        if open_browser:
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", url], check=False)
                else:
                    subprocess.run(["xdg-open", url], check=False)
            except FileNotFoundError:
                pass

        print("\nPress Ctrl+C to stop")
        try:
            time.sleep(86400)
        except KeyboardInterrupt:
            print("\nShutting down...")


@cli.command("show-logs")
@click.option("--tail", default=100, help="Number of lines to tail")
@click.pass_context
def show_logs(ctx, tail):
    """View cluster logs."""
    log_command = f"tail -n {tail} -f /tmp/ray/session_latest/logs/monitor*"
    subprocess.run(
        _maybe_add_ray_verbose(ctx.obj, ["ray", "exec", ctx.obj.config_file, log_command]),
        check=True,
    )


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()


if __name__ == "__main__":
    main()
