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
import random
import socket
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard connections."""

    cluster_configs: list[str] = field(default_factory=list)  # Empty = auto-discover
    ray_init: bool = False  # Initialize Ray client

    @classmethod
    def from_cluster(cls, cluster_name: str, ray_init: bool = False) -> "DashboardConfig":
        """Create config for a single cluster by name."""
        return cls(cluster_configs=[cluster_name], ray_init=ray_init)


@dataclass
class ResourceUsage:
    """Resource usage information for a single resource type."""

    used: str
    total: str

    def percentage(self) -> float:
        """Calculate usage percentage, handling unit conversions."""
        try:
            # Strip common units for numeric comparison
            used_val = float(self.used.replace("TiB", "").replace("GiB", "").replace("MiB", "").replace("KiB", ""))
            total_val = float(self.total.replace("TiB", "").replace("GiB", "").replace("MiB", "").replace("KiB", ""))
            return (used_val / total_val * 100) if total_val > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0


@dataclass
class ClusterStatus:
    """Parsed Ray cluster status information."""

    active_nodes: list[str] = field(default_factory=list)
    pending_nodes: list[str] = field(default_factory=list)
    resources: dict[str, ResourceUsage] = field(default_factory=dict)

    def active_count(self) -> int:
        """Number of active nodes."""
        return len(self.active_nodes)

    def pending_count(self) -> int:
        """Number of pending nodes."""
        return len(self.pending_nodes)


@dataclass
class ClusterInfo:
    """Information about a Ray cluster."""

    cluster_name: str
    config_path: str
    head_ip: str  # Internal IP address (10.x.x.x)
    external_ip: str | None
    zone: str
    project: str


@dataclass
class RayPortMapping:
    """Local port mappings for SSH tunnel to a Ray cluster."""

    dashboard_port: int  # Ray dashboard (default 8265)
    gcs_port: int  # Ray GCS (default 6379)
    api_port: int  # Ray API server (default 10001)


@dataclass
class DashboardConnection:
    """Manages SSH tunnel and proxy for one or more clusters."""

    clusters: dict[str, ClusterInfo]  # cluster_name -> info
    port_mappings: dict[str, RayPortMapping]  # cluster_name -> port mapping
    ssh_process: subprocess.Popen


def find_free_port(start_port: int = 9000, max_attempts: int = 1000) -> int:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")


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


def allocate_ports(clusters: dict[str, ClusterInfo]) -> dict[str, RayPortMapping]:
    """Allocate local ports for each cluster using Ray's traditional ports.

    First cluster uses Ray defaults: 8265 (dashboard), 6379 (GCS), 10001 (API).
    Additional clusters scan upward from these starting points.
    """
    port_mappings = {}

    # Ray's traditional port assignments
    next_dashboard_port = 8265
    next_gcs_port = 6379
    next_api_port = 10001

    for cluster_name in sorted(clusters.keys()):
        # search from the previous port to avoid accidental reusing a port
        dashboard_port = find_free_port(next_dashboard_port)
        gcs_port = find_free_port(next_gcs_port)
        api_port = find_free_port(next_api_port)
        port_mappings[cluster_name] = RayPortMapping(dashboard_port=dashboard_port, gcs_port=gcs_port, api_port=api_port)
        next_dashboard_port = dashboard_port + 1
        next_gcs_port = gcs_port + 1
        next_api_port = api_port + 1

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

    # Get internal and external IPs from gcloud
    try:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--filter",
            f"labels.ray-cluster-name={cluster_name} AND labels.ray-node-type=head",
            "--format",
            "value(networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP)",
            "--limit",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        ips = result.stdout.strip().split("\t")
        if not ips or not ips[0]:
            raise RuntimeError(f"No head node found for cluster {cluster_name}")
        head_ip = ips[0]
        external_ip = ips[1] if len(ips) > 1 and ips[1] else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to get IPs for cluster {cluster_name}: {e}") from e

    return ClusterInfo(
        cluster_name=cluster_name,
        config_path=config_path,
        head_ip=head_ip,
        external_ip=external_ip,
        zone=zone,
        project=project,
    )


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
        access_configs = instance["networkInterfaces"][0].get("accessConfigs", [])
        external_ip = access_configs[0]["natIP"] if access_configs else None
        zone = instance["zone"].split("/")[-1]
        project = instance["zone"].split("/")[6]

        # Find corresponding config file
        config_path = find_config_by_cluster_name(cluster_name)
        if not config_path:
            logger.warning(f"No config file found for cluster {cluster_name}")
            continue

        clusters[cluster_name] = ClusterInfo(
            cluster_name=cluster_name,
            config_path=config_path,
            head_ip=internal_ip,
            external_ip=external_ip,
            zone=zone,
            project=project,
        )

    logger.info(f"Discovered {len(clusters)} active clusters")
    return clusters


def create_ssh_proxy_chain(
    clusters: dict[str, ClusterInfo], port_mappings: dict[str, RayPortMapping]
) -> subprocess.Popen:
    """Create single SSH connection that proxies to all cluster head nodes.

    Uses the first cluster as jump host, then forwards ports to other clusters'
    internal IPs through that connection.
    """
    # Sort for deterministic ordering
    cluster_names = sorted(clusters.keys())

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
        ports = port_mappings[cluster_name]
        ssh_cmd.extend(
            [
                f"-L{ports.dashboard_port}:{cluster.head_ip}:8265",
                f"-L{ports.gcs_port}:{cluster.head_ip}:6379",
                f"-L{ports.api_port}:{cluster.head_ip}:10001",
            ]
        )

    # shuffle clusters and find one we can ping
    clusters_list = list(clusters.values())

    random.shuffle(clusters_list)
    cluster_to_use = None
    if len(clusters_list) > 1:
        for cluster in clusters_list:
            if not cluster.external_ip:
                continue
            try:
                subprocess.run(
                    ["ping", "-c", "1", "-W", "2", cluster.external_ip],
                    capture_output=False,
                    text=True,
                    check=True,
                    timeout=2,
                )
                cluster_to_use = cluster
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
    else:
        cluster_to_use = clusters_list[0]

    if not cluster_to_use:
        raise RuntimeError("No reachable cluster found with external IP")

    ssh_cmd.extend([f"ray@{cluster_to_use.external_ip}", "while true; do sleep 86400; done"])
    logger.info(f"Creating SSH proxy chain through {cluster_to_use.cluster_name}")
    logger.info(f"Tunneling to {len(clusters)} clusters. Port mapping: {port_mappings}")

    return subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "TERM": "dumb"},
    )


def wait_for_tunnel(clusters: dict[str, ClusterInfo], port_mappings: dict[str, RayPortMapping]) -> None:
    """Wait for SSH tunnel to be ready by testing dashboard connections."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def check_dashboard(cluster_name: str, dashboard_port: int) -> tuple[str, bool]:
        """Check if a dashboard is accessible."""
        try:
            response = requests.get(f"http://localhost:{dashboard_port}/api/version", timeout=3)
            return (cluster_name, response.status_code == 200)
        except (requests.ConnectionError, requests.Timeout):
            return (cluster_name, False)

    max_retries = 3
    retry_delay = 1

    results = {}
    for attempt in range(max_retries):
        # Test all dashboards concurrently
        with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
            futures = {
                executor.submit(check_dashboard, cluster_name, port_mappings[cluster_name].dashboard_port): cluster_name
                for cluster_name in clusters
            }

            for future in as_completed(futures):
                cluster_name, is_ready = future.result()
                results[cluster_name] = is_ready

        if all(results.values()):
            logger.info("SSH tunnel is ready - all dashboards accessible")
            return

        if attempt < max_retries - 1:
            logger.info(f"SSH tunnel not ready, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    for cluster_name, is_ready in results.items():
        if not is_ready:
            logger.error(f"Dashboard for cluster {cluster_name} is not accessible")


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

    port_mappings = allocate_ports(clusters)
    ssh_process = create_ssh_proxy_chain(clusters, port_mappings)
    wait_for_tunnel(clusters, port_mappings)
    connection = DashboardConnection(clusters=clusters, port_mappings=port_mappings, ssh_process=ssh_process)

    # Save original environment variables
    original_env = {
        "RAY_ADDRESS": os.environ.get("RAY_ADDRESS"),
        "RAY_API_SERVER_ADDRESS": os.environ.get("RAY_API_SERVER_ADDRESS"),
        "RAY_DASHBOARD_ADDRESS": os.environ.get("RAY_DASHBOARD_ADDRESS"),
        "RAY_GCS_ADDRESS": os.environ.get("RAY_GCS_ADDRESS"),
    }

    # For single cluster, set Ray environment variables
    if len(clusters) == 1:
        cluster_name = next(iter(clusters.keys()))
        ports = port_mappings[cluster_name]

        # Set new values
        os.environ["RAY_ADDRESS"] = f"http://localhost:{ports.dashboard_port}"
        os.environ["RAY_API_SERVER_ADDRESS"] = f"ray://localhost:{ports.api_port}"
        os.environ["RAY_DASHBOARD_ADDRESS"] = f"http://localhost:{ports.dashboard_port}"
        os.environ["RAY_GCS_ADDRESS"] = f"localhost:{ports.gcs_port}"

    try:
        # Initialize Ray if requested
        if config.ray_init and len(clusters) == 1:
            import ray

            cluster_name = next(iter(clusters.keys()))
            api_port = port_mappings[cluster_name].api_port
            ray.init(address=f"ray://localhost:{api_port}", runtime_env={"working_dir": "."})

        yield connection

    finally:
        # Cleanup
        if ssh_process and ssh_process.poll() is None:
            ssh_process.terminate()
            ssh_process.wait()

        # Restore environment variables if changed
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
