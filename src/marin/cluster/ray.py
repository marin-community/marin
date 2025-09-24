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

"""Ray utilities for cluster management."""

import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

from .config import RayClusterConfig

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find a free port on the local machine.

    Returns:
        An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


TPU_TYPE_TO_VM_IMAGE = {
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}


class RayCommandError(RuntimeError):
    """Exception raised when a Ray command fails with detailed error information."""

    def __init__(
        self,
        command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        message: str | None = None,
    ):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

        if message is None:
            command_str = " ".join(command)
            message = f"Ray command failed: {command_str}"

        if stderr:
            message += f"\nSTDERR: {stderr}"
        if stdout:
            message += f"\nSTDOUT: {stdout}"

        super().__init__(message)


def run_ray_command(
    command: list[str],
    timeout: int = 30,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a Ray command using `subprocess`.

    Args:
        command: Command to run as a list of strings
        timeout: Timeout in seconds
        check: Whether to raise an exception on non-zero exit code
        capture_output: Whether to capture stdout/stderr
        text: Whether to return output as text
        env: Environment variables (defaults to os.environ)

    Returns:
        CompletedProcess instance

    Raises:
        RayCommandError: If the command fails and check=True
    """
    if env is None:
        env = os.environ.copy() | {"TERM": "dumb"}

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text,
            env=env,
            check=check,
            timeout=timeout,
            start_new_session=True,  # Creates new process group to avoid terminal issues
        )
        return result
    except subprocess.CalledProcessError as e:
        raise RayCommandError(
            command=command,
            returncode=e.returncode,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        ) from e


def get_head_ip_from_config(cluster_config: str) -> str:
    """Get the head node IP from cluster config using ray get_head_ip command."""
    try:
        result = subprocess.run(
            ["ray", "get_head_ip", cluster_config],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        head_ip = result.stdout.strip()
        if not head_ip:
            raise RuntimeError("Empty head IP returned")
        return head_ip
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get head IP: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout getting head IP") from None


@dataclass
class DashboardInfo:
    dashboard_port: int
    gcs_port: int
    api_port: int
    ssh_process: subprocess.Popen


def start_ssh_tunnel_to_head(cluster_config: str, head_ip: str) -> DashboardInfo:
    """Start SSH tunnel to Ray head node with port forwarding.

    Args:
        cluster_config: Path to the Ray cluster configuration file
        head_ip: IP address of the head node

    Returns:
        Tuple of (ssh_process, dashboard_port, gcs_port, api_port)

    Raises:
        RuntimeError: If SSH tunnel fails to start
    """
    # Find free ports for forwarding
    ports = set()
    while len(ports) < 3:
        ports.add(find_free_port())

    dashboard_port, gcs_port, api_port = list(ports)

    ssh_cmd = [
        "ssh",
        "-tt",
        f"-L{dashboard_port}:localhost:8265",  # Dashboard
        f"-L{gcs_port}:localhost:6379",  # Ray GCS
        f"-L{api_port}:localhost:10001",  # Ray API server
        "-i",
        os.path.expanduser("~/.ssh/marin_ray_cluster.pem"),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        f"ray@{head_ip}",
        "while true; do sleep 86400; done",
    ]

    logger.info(
        f"Starting SSH tunnel to {head_ip} with ports: "
        + f"\ndashboard={dashboard_port}"
        + f"\nray={gcs_port}"
        + f"\napi={api_port}"
        + f"\n View the dashboard at http://localhost:{dashboard_port}"
    )

    ssh_process = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "TERM": "dumb"},
    )

    # Wait for tunnel to be ready
    max_retries = 30
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # Test dashboard connection
            response = requests.get(f"http://localhost:{dashboard_port}/api/version", timeout=5)
            if response.status_code == 200:
                logger.info(f"SSH tunnel is ready - dashboard accessible on port {dashboard_port}")
                return DashboardInfo(
                    dashboard_port=dashboard_port,
                    gcs_port=gcs_port,
                    api_port=api_port,
                    ssh_process=ssh_process,
                )
        except (requests.ConnectionError, requests.Timeout):
            if attempt < max_retries - 1:
                logger.info(f"SSH tunnel not ready, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.warning(f"SSH tunnel failed to become ready after {max_retries} attempts")
                ssh_process.terminate()
                ssh_process.wait()
                raise RuntimeError(f"SSH tunnel failed to become ready after {max_retries} attempts") from None

    # Should not reach here
    ssh_process.terminate()
    ssh_process.wait()
    raise RuntimeError("Unexpected error in SSH tunnel setup")


@contextmanager
def ray_dashboard(cluster_config: str, ray_init: bool = False) -> Generator[DashboardInfo, None, None]:
    """Context manager for Ray dashboard connection using SSH tunneling.

    This establishes an SSH tunnel to the Ray head node and sets up environment variables
    so that Ray CLI commands and Python API can connect to the cluster.

    Args:
        cluster_config: Path to the Ray cluster configuration file

    Yields:
        DashboardInfo object with port information and SSH process
    """
    # Save original environment variables
    original_ray_address = os.environ.get("RAY_ADDRESS")
    original_ray_api_server_address = os.environ.get("RAY_API_SERVER_ADDRESS")
    original_ray_dashboard_address = os.environ.get("RAY_DASHBOARD_ADDRESS")

    # Get head IP and start SSH tunnel
    logger.info(f"Getting head IP for cluster config: {cluster_config}")
    head_ip = get_head_ip_from_config(cluster_config)
    dashboard_info = start_ssh_tunnel_to_head(cluster_config, head_ip)

    # Set environment variables for Ray CLI and Python API
    os.environ["RAY_ADDRESS"] = f"http://localhost:{dashboard_info.dashboard_port}"
    os.environ["RAY_API_SERVER_ADDRESS"] = f"ray://localhost:{dashboard_info.api_port}"
    os.environ["RAY_DASHBOARD_ADDRESS"] = f"http://localhost:{dashboard_info.dashboard_port}"

    try:
        if ray_init:
            import ray

            ray.init(
                address=f"ray://localhost:{dashboard_info.api_port}",
                runtime_env={"working_dir": "."},
            )
        yield dashboard_info
    finally:
        # Restore original environment variables
        if original_ray_address is not None:
            os.environ["RAY_ADDRESS"] = original_ray_address
        else:
            os.environ.pop("RAY_ADDRESS", None)

        if original_ray_api_server_address is not None:
            os.environ["RAY_API_SERVER_ADDRESS"] = original_ray_api_server_address
        else:
            os.environ.pop("RAY_API_SERVER_ADDRESS", None)

        if original_ray_dashboard_address is not None:
            os.environ["RAY_DASHBOARD_ADDRESS"] = original_ray_dashboard_address
        else:
            os.environ.pop("RAY_DASHBOARD_ADDRESS", None)

        # Clean up the SSH tunnel
        ssh_process = dashboard_info.ssh_process
        if ssh_process and ssh_process.poll() is None:
            ssh_process.terminate()
            ssh_process.wait()


def list_jobs() -> list[dict]:
    """Fetch the list of jobs using the Ray CLI.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    result = run_ray_command(["ray", "list", "jobs", "--detail", "--format=json", "--limit=10000"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def submit_job(
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

    result = run_ray_command(cmd, timeout=500, capture_output=False)
    # Extract job ID from output (usually in format "Job submitted with ID: <id>")
    output_lines = result.stdout.strip().split("\n")
    for line in output_lines:
        if "submitted with ID:" in line:
            return line.split(":")[-1].strip()

    # Fallback: return full output if we can't parse job ID
    return result.stdout.strip()


def download_working_directory(
    cluster_config: str, job_id: str, working_dir: str, remote_working_dir: str, local_path: str
) -> str:
    """Download the working directory for `job_id`."""
    dest_dir = os.path.join(local_path, job_id, working_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_dir = os.path.join(dest_dir, "")  # Add trailing slash for rsync

    rsync_command = ["ray", "rsync-down", cluster_config, remote_working_dir, dest_dir]
    run_ray_command(rsync_command)

    logger.info(f"Working directory for job {remote_working_dir} saved to {dest_dir}")
    return dest_dir


def save_runtime_env_entrypoint(job_details: dict[str, Any], job_id: str, local_path: str) -> dict[str, Any]:
    """Save the runtime environment and entrypoint for the job."""
    runtime_env = job_details.get("runtime_env", {})
    runtime_env.pop("working_dir", None)  # Remove unnecessary fields
    runtime_env.pop("_ray_commit", None)
    runtime_env["entrypoint"] = job_details["entrypoint"]

    env_file = os.path.join(local_path, job_id, "runtime_env.json")
    with open(env_file, "w") as f:
        json.dump(runtime_env, f, indent=4)
    return runtime_env


def resubmit_job(
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
        "--working-dir",
        working_dir,
        *runtime_env_args,
        "--",
        *shlex.split(entrypoint),
    ]
    job_str = " ".join(job_array)

    logger.info(f"Submitting the job: {shlex.quote(job_str)}")

    try:
        run_ray_command(job_array, timeout=500, capture_output=False)
        logger.info(f"Successfully resubmitted job {job_id}")
    except RayCommandError as e:
        logger.error(f"Failed to resubmit job {job_id}: {e}")
        if raise_errors:
            raise ValueError(f"Failed to resubmit job {job_id}") from e


#  {
#    "type": "SUBMISSION",
#    "job_id": "49000000",
#    "submission_id": "raysubmit_H9G6A2FEvtMjuLp5",
#    "driver_info": {
#      "id": "49000000",
#      "node_ip_address": "10.130.0.2",
#      "pid": "737995"
#    },
#    "status": "STOPPED",
#    "entrypoint": " python -m marin.training.training ...
#    "message": "Job was intentionally stopped.",
#    "error_type": null,
#    "start_time": null,
#    "end_time": null,
#    "metadata": null,
#    "runtime_env": null,
#    "driver_agent_http_address": null,
#    "driver_node_id": null,
#    "driver_exit_code": null
#  },


def backup_jobs(cluster_config: str, local_path: str, raise_errors: bool = False) -> None:
    """Backup jobs from the given Ray cluster.

    Note: This requires RAY_ADDRESS to be set, typically via start_ray_dashboard_with_wait.
    """
    logger.info("Fetching jobs from Ray Jobs API...")

    # Clear the backup directory if it exists
    backup_dir = Path(local_path)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True)

    jobs_data = list_jobs()
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
            download_working_directory(cluster_config, job_id, working_dir, remote_working_dir, local_path)
            save_runtime_env_entrypoint(job_details, job_id, local_path)
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            if raise_errors:
                raise e

    logger.info("All jobs backed up.")


def restore_jobs(cluster_config: str, local_path: str, raise_errors: bool = False) -> None:
    """Perform the 'after' stage actions: resubmit jobs.

    Note: This requires RAY_ADDRESS to be set, typically via start_ray_dashboard_with_wait.
    """
    backup_dir = Path(local_path)
    if not backup_dir.exists():
        logger.error("No backup data found. Run backup_jobs() first.")
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

        resubmit_job(job_id_dir.name, entrypoint, str(working_dir), runtime_env, raise_errors)

    logger.info("All jobs resubmitted.")


def list_nodes() -> list[dict[str, Any]]:
    """Get list of Ray nodes.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    result = run_ray_command(["ray", "list", "nodes", "--format=json", "--limit=10000"])
    return json.loads(result.stdout)


def list_workers(limit: int = 10000) -> list[dict[str, Any]]:
    """Get list of Ray workers.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    result = run_ray_command(["ray", "list", "workers", "--format=json", f"--limit={limit}"])
    return json.loads(result.stdout)


def list_tasks(limit: int = 1000) -> list[dict[str, Any]]:
    """Get list of Ray tasks.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    result = run_ray_command(["ray", "list", "tasks", "--format=json", f"--limit={limit}"])
    return json.loads(result.stdout)


def list_actors(limit: int = 1000) -> list[dict[str, Any]]:
    """Get list of Ray actors.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    result = run_ray_command(["ray", "list", "actors", "--format=json", f"--limit={limit}"])
    return json.loads(result.stdout)


def get_cluster_utilization() -> dict[str, Any]:
    """Get cluster resource utilization summary.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    try:
        nodes = list_nodes()
        workers = list_workers()

        total_nodes = len(nodes)
        total_workers = len(workers)

        # Calculate resource totals
        total_cpus = sum(node.get("Resources", {}).get("CPU", 0) for node in nodes)
        total_gpus = sum(node.get("Resources", {}).get("GPU", 0) for node in nodes)
        total_memory = sum(node.get("Resources", {}).get("memory", 0) for node in nodes)

        # Count TPU resources
        tpu_resources = {}
        for node in nodes:
            resources = node.get("Resources", {})
            for resource, amount in resources.items():
                if resource.startswith("TPU"):
                    tpu_resources[resource] = tpu_resources.get(resource, 0) + amount

        # Count nodes by status
        node_status = Counter()
        for node in nodes:
            status = node.get("Alive", False)
            node_status["alive" if status else "dead"] += 1

        return {
            "total_nodes": total_nodes,
            "total_workers": total_workers,
            "node_status": dict(node_status),
            "resources": {
                "CPU": total_cpus,
                "GPU": total_gpus,
                "memory": total_memory,
                "TPU": tpu_resources,
            },
            "worker_ips": list(set(worker.get("ip") for worker in workers if worker.get("ip"))),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get cluster utilization: {e}") from e


def get_ray_cluster_resources() -> dict[str, Any]:
    """Get Ray cluster resources using ray CLI.

    Note: This requires RAY_ADDRESS to be set, typically via ray_dashboard context manager.
    """
    result = run_ray_command(["ray", "status", "--format=json"])
    return json.loads(result.stdout)


def add_manual_worker(
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

    logger = logging.getLogger(__name__)

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
    initialize_manual_worker(config.config_file, tpu_name)


def initialize_manual_worker(config_file: str, tpu_name: str) -> None:
    """Setup the worker entrypoint script and start the container.

    This script configures the worker to automatically poll for a new head_ip
    at startup. This allows manual workers to resume in the case of a cluster restart.
    """
    from levanter.infra.tpus import run_command, tpu_ssh

    logger = logging.getLogger(__name__)

    cluster_config = yaml.safe_load(open(config_file, "r"))

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
