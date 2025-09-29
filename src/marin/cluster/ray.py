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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

from .config import RayClusterConfig
from .dashboard_proxy import DashboardProxy

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard connections."""

    cluster_configs: list[str] = field(default_factory=list)  # Empty = auto-discover
    ray_init: bool = False  # Initialize Ray client
    proxy_port: int = 9999  # Port for proxy dashboard when multiple clusters

    @classmethod
    def from_cluster(cls, cluster_name: str, ray_init: bool = False) -> "DashboardConfig":
        """Create config for a single cluster by name."""
        return cls(cluster_configs=[cluster_name], ray_init=ray_init)


@dataclass
class ClusterInfo:
    """Information about a Ray cluster."""

    cluster_name: str
    config_path: str
    head_ip: str  # Internal IP address (10.x.x.x)
    zone: str
    project: str


@dataclass
class DashboardConnection:
    """Manages SSH tunnel and proxy for one or more clusters."""

    clusters: dict[str, ClusterInfo]  # cluster_name -> info
    port_mappings: dict[str, tuple[int, int, int]]  # cluster_name -> (dashboard, gcs, api)
    ssh_process: subprocess.Popen
    proxy: "DashboardProxy | None" = None

    def get_dashboard_url(self, cluster_name: str | None = None) -> str:
        """Get dashboard URL for a specific cluster or the proxy URL."""
        if self.proxy and len(self.clusters) > 1:
            proxy_url = f"http://localhost:{self.proxy.proxy_port}"
            return f"{proxy_url}/{cluster_name}" if cluster_name else proxy_url

        # Single cluster or specific cluster requested
        name = cluster_name or next(iter(self.clusters.keys()))
        dashboard_port = self.port_mappings[name][0]
        return f"http://localhost:{dashboard_port}"


def find_free_port(start_port: int = 9000, max_attempts: int = 1000) -> int:
    """Find a free port on the local machine by scanning from start_port.

    Args:
        start_port: Port to start scanning from
        max_attempts: Maximum number of ports to try

    Returns:
        An available port number

    Raises:
        RuntimeError: If no free port found within max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")


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


def allocate_ports(clusters: dict[str, ClusterInfo]) -> dict[str, tuple[int, int, int]]:
    """Allocate local ports for each cluster using Ray's traditional ports.

    First cluster uses Ray defaults: 8265 (dashboard), 6379 (GCS), 10001 (API).
    Additional clusters scan upward from these starting points.
    """
    port_mappings = {}

    # Ray's traditional port assignments
    dashboard_start = 8265
    gcs_start = 6379
    api_start = 10001

    for i, cluster_name in enumerate(sorted(clusters.keys())):
        dashboard_port = find_free_port(dashboard_start + i)
        gcs_port = find_free_port(gcs_start + i)
        api_port = find_free_port(api_start + i)
        port_mappings[cluster_name] = (dashboard_port, gcs_port, api_port)

    return port_mappings


def find_config_by_cluster_name(cluster_name: str) -> str | None:
    """Find config file for a given cluster name."""
    # Look for config files in infra/ directory
    infra_dir = Path("infra")
    if not infra_dir.exists():
        return None

    for config_file in infra_dir.glob("*.yaml"):
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                if config.get("cluster_name") == cluster_name:
                    return str(config_file)
        except Exception:
            continue
    return None


def load_cluster_info(config_path: str) -> ClusterInfo:
    """Load cluster info from config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cluster_name = config["cluster_name"]
    zone = config["provider"]["availability_zone"]
    project = config["provider"].get("project_id", "")

    # Get internal IP from gcloud
    try:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--filter",
            f"labels.ray-cluster-name={cluster_name} AND labels.ray-node-type=head",
            "--format",
            "value(networkInterfaces[0].networkIP)",
            "--limit",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        head_ip = result.stdout.strip()
        if not head_ip:
            raise RuntimeError(f"No head node found for cluster {cluster_name}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to get internal IP for cluster {cluster_name}: {e}") from e

    return ClusterInfo(cluster_name=cluster_name, config_path=config_path, head_ip=head_ip, zone=zone, project=project)


def discover_active_clusters() -> dict[str, ClusterInfo]:
    """Discover all active Ray clusters across all zones.

    Uses gcloud to find all compute instances with ray-node-type=head label.
    """
    cmd = ["gcloud", "compute", "instances", "list", "--filter", "labels.ray-node-type=head", "--format", "json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        instances = json.loads(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to discover active clusters: {e}") from e

    clusters = {}
    for instance in instances:
        # Extract cluster info from instance metadata
        labels = instance.get("labels", {})
        cluster_name = labels.get("ray-cluster-name")

        if not cluster_name:
            continue

        # Get network info
        internal_ip = instance["networkInterfaces"][0]["networkIP"]
        zone = instance["zone"].split("/")[-1]
        project = instance["zone"].split("/")[6]

        # Find corresponding config file
        config_path = find_config_by_cluster_name(cluster_name)
        if not config_path:
            logger.warning(f"No config file found for cluster {cluster_name}")
            continue

        clusters[cluster_name] = ClusterInfo(
            cluster_name=cluster_name, config_path=config_path, head_ip=internal_ip, zone=zone, project=project
        )

    logger.info(f"Discovered {len(clusters)} active clusters")
    return clusters


def create_ssh_proxy_chain(
    clusters: dict[str, ClusterInfo], port_mappings: dict[str, tuple[int, int, int]]
) -> subprocess.Popen:
    """Create single SSH connection that proxies to all cluster head nodes.

    Uses the first cluster as jump host, then forwards ports to other clusters'
    internal IPs through that connection.
    """
    # Sort for deterministic ordering
    cluster_names = sorted(clusters.keys())
    first_cluster = clusters[cluster_names[0]]

    # Build SSH command with all port forwards
    ssh_cmd = [
        "ssh",
        "-tt",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-i",
        os.path.expanduser("~/.ssh/marin_ray_cluster.pem"),
    ]

    # Add port forwards for each cluster
    for cluster_name in cluster_names:
        cluster = clusters[cluster_name]
        dashboard_port, gcs_port, api_port = port_mappings[cluster_name]
        ssh_cmd.extend(
            [
                f"-L{dashboard_port}:{cluster.head_ip}:8265",
                f"-L{gcs_port}:{cluster.head_ip}:6379",
                f"-L{api_port}:{cluster.head_ip}:10001",
            ]
        )

    # Connect to first cluster's external IP
    first_external_ip = get_head_ip_from_config(first_cluster.config_path)
    ssh_cmd.extend([f"ray@{first_external_ip}", "while true; do sleep 86400; done"])

    logger.info(f"Creating SSH proxy chain through {first_cluster.cluster_name}")
    logger.info(f"Tunneling to {len(clusters)} clusters")

    return subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "TERM": "dumb"},
    )


def wait_for_tunnel(
    clusters: dict[str, ClusterInfo], port_mappings: dict[str, tuple[int, int, int]], timeout: int = 10
) -> None:
    """Wait for SSH tunnel to be ready by testing dashboard connections."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def check_dashboard(cluster_name: str, dashboard_port: int) -> tuple[str, bool]:
        """Check if a dashboard is accessible."""
        try:
            response = requests.get(f"http://localhost:{dashboard_port}/api/version", timeout=5)
            return (cluster_name, response.status_code == 200)
        except (requests.ConnectionError, requests.Timeout):
            return (cluster_name, False)

    max_retries = timeout
    retry_delay = 1

    for attempt in range(max_retries):
        # Test all dashboards concurrently
        with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
            futures = {
                executor.submit(check_dashboard, cluster_name, port_mappings[cluster_name][0]): cluster_name
                for cluster_name in clusters
            }

            results = {}
            for future in as_completed(futures):
                cluster_name, is_ready = future.result()
                results[cluster_name] = is_ready

        if all(results.values()):
            logger.info("SSH tunnel is ready - all dashboards accessible")
            return

        if attempt < max_retries - 1:
            logger.info(f"SSH tunnel not ready, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    raise RuntimeError(f"SSH tunnel failed to become ready after {max_retries} attempts")


@contextmanager
def ray_dashboard(config: DashboardConfig) -> Generator[DashboardConnection, None, None]:
    """Dashboard context manager supporting single and multiple clusters.

    Creates a single SSH connection that can tunnel to multiple Ray clusters
    using internal IP routing. For multiple clusters, starts a Flask proxy
    in a background thread.

    Args:
        config: Dashboard configuration

    Yields:
        DashboardConnection with cluster details and access methods
    """

    # Determine clusters to connect to
    if not config.cluster_configs:
        # Auto-discover active clusters across all zones
        clusters = discover_active_clusters()
        if not clusters:
            raise RuntimeError("No active Ray clusters found")
    else:
        # Load specified cluster configs
        clusters = {}
        for config_path in config.cluster_configs:
            info = load_cluster_info(config_path)
            clusters[info.cluster_name] = info

    # Allocate local ports for each cluster
    port_mappings = allocate_ports(clusters)

    # Create single SSH connection with port forwards to all clusters
    ssh_process = create_ssh_proxy_chain(clusters, port_mappings)

    # Wait for SSH tunnel to be ready
    wait_for_tunnel(clusters, port_mappings, timeout=30)

    # Create connection object
    connection = DashboardConnection(clusters=clusters, port_mappings=port_mappings, ssh_process=ssh_process)

    # Save original environment variables
    original_env = {
        "RAY_ADDRESS": os.environ.get("RAY_ADDRESS"),
        "RAY_API_SERVER_ADDRESS": os.environ.get("RAY_API_SERVER_ADDRESS"),
        "RAY_DASHBOARD_ADDRESS": os.environ.get("RAY_DASHBOARD_ADDRESS"),
    }

    # For single cluster, set Ray environment variables
    if len(clusters) == 1:
        cluster_name = next(iter(clusters.keys()))
        dashboard_port, _, api_port = port_mappings[cluster_name]

        # Set new values
        os.environ["RAY_ADDRESS"] = f"http://localhost:{dashboard_port}"
        os.environ["RAY_API_SERVER_ADDRESS"] = f"ray://localhost:{api_port}"
        os.environ["RAY_DASHBOARD_ADDRESS"] = f"http://localhost:{dashboard_port}"
    else:
        # Multiple clusters - start Flask proxy
        proxy = DashboardProxy(clusters, port_mappings, config.proxy_port)
        proxy.start()
        connection.proxy = proxy

    try:
        # Initialize Ray if requested (single cluster only)
        if config.ray_init and len(clusters) == 1:
            import ray

            cluster_name = next(iter(clusters.keys()))
            api_port = port_mappings[cluster_name][2]
            ray.init(address=f"ray://localhost:{api_port}", runtime_env={"working_dir": "."})

        yield connection

    finally:
        # Cleanup
        if connection.proxy:
            connection.proxy.stop()

        if ssh_process and ssh_process.poll() is None:
            ssh_process.terminate()
            ssh_process.wait()

        # Restore environment variables if changed
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
