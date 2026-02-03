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

"""Controller lifecycle management.

This module provides classes for managing the Iris controller lifecycle,
supporting both GCP-managed VMs and manually-deployed controllers.

GcpController creates and manages a GCE VM running the controller container.
ManualController wraps a pre-existing controller at a static address.
Both implement the same ControllerProtocol for uniform handling.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from iris.cluster.vm.config import config_to_dict
from iris.managed_thread import ThreadContainer
from iris.cluster.vm.gcp_tpu_platform import (
    controller_address_metadata_key,
    controller_metadata_key,
)
from iris.cluster.vm.ssh import (
    DirectSshConnection,
    GceSshConnection,
    SshConnection,
    run_streaming_with_retry,
)
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

# Constants
CONTROLLER_CONTAINER_NAME = "iris-controller"
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50
HEALTH_CHECK_TIMEOUT_SECONDS = 120
MAX_RETRIES = 5

# Backoff parameters for health check polling: start at 2s, cap at 10s
HEALTH_CHECK_BACKOFF_INITIAL = 2.0
HEALTH_CHECK_BACKOFF_MAX = 10.0


# ============================================================================
# Health Checking
# ============================================================================


@dataclass
class HealthCheckResult:
    """Result of a health check with diagnostic info."""

    healthy: bool
    curl_output: str = ""
    curl_error: str = ""
    container_status: str = ""
    container_logs: str = ""

    def __bool__(self) -> bool:
        return self.healthy

    def summary(self) -> str:
        if self.healthy:
            return "healthy"
        parts = []
        if self.container_status:
            parts.append(f"container={self.container_status}")
        if self.curl_error:
            parts.append(f"curl_error={self.curl_error[:50]}")
        return ", ".join(parts) if parts else "unknown failure"


def check_health(
    conn: SshConnection,
    port: int = 10001,
    container_name: str = "iris-worker",
) -> HealthCheckResult:
    """Check if worker/controller is healthy via health endpoint.

    Returns HealthCheckResult with diagnostic info on failure.
    """
    result = HealthCheckResult(healthy=False)

    try:
        logger.info("Running health check: curl -sf http://localhost:%d/health", port)
        curl_result = conn.run(f"curl -sf http://localhost:{port}/health", timeout=Duration.from_seconds(10))
        if curl_result.returncode == 0:
            result.healthy = True
            result.curl_output = curl_result.stdout.strip()
            logger.info("Health check succeeded")
            return result
        result.curl_error = curl_result.stderr.strip() or f"exit code {curl_result.returncode}"
        logger.info("Health check failed: %s", result.curl_error)
    except Exception as e:
        result.curl_error = str(e)
        logger.info("Health check exception: %s", e)

    # Gather diagnostics on failure
    try:
        status_result = conn.run(
            f"sudo docker inspect --format='{{{{.State.Status}}}}' {container_name} 2>/dev/null || echo 'not_found'",
            timeout=Duration.from_seconds(10),
        )
        result.container_status = status_result.stdout.strip()
    except Exception as e:
        result.container_status = f"error: {e}"

    if result.container_status in ("restarting", "exited", "dead", "not_found"):
        try:
            cmd = f"sudo docker logs {container_name} --tail 20 2>&1"
            logs_result = conn.run(cmd, timeout=Duration.from_seconds(15))
            if logs_result.returncode == 0 and logs_result.stdout.strip():
                result.container_logs = logs_result.stdout.strip()
        except Exception as e:
            result.container_logs = f"error fetching logs: {e}"

    return result


def wait_healthy_via_ssh(
    conn: SshConnection,
    port: int,
    timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    container_name: str = CONTROLLER_CONTAINER_NAME,
) -> bool:
    """Poll health endpoint via SSH until healthy or timeout.

    Uses SSH to run curl on the remote host, avoiding firewall issues
    that can block external HTTP access to the controller port.

    On failure, logs diagnostic info including container status and logs.
    """
    logger.info("Starting SSH-based health check (port=%d, timeout=%ds)", port, int(timeout))
    start_time = time.monotonic()
    backoff = ExponentialBackoff(
        initial=HEALTH_CHECK_BACKOFF_INITIAL,
        maximum=HEALTH_CHECK_BACKOFF_MAX,
    )

    attempt = 0
    last_result = None

    while True:
        attempt += 1
        last_result = check_health(conn, port, container_name)
        elapsed = time.monotonic() - start_time

        if last_result.healthy:
            logger.info("SSH health check succeeded after %d attempts (%.1fs)", attempt, elapsed)
            return True

        logger.info("SSH health check attempt %d failed (%.1fs): %s", attempt, elapsed, last_result.summary())

        if elapsed >= timeout:
            break

        interval = backoff.next_interval()
        remaining = timeout - elapsed
        time.sleep(min(interval, remaining))

    # Health check failed - log detailed diagnostics
    logger.error("=" * 60)
    logger.error("SSH health check FAILED after %d attempts (%.1fs)", attempt, timeout)
    logger.error("=" * 60)
    logger.error("Final status: %s", last_result.summary())
    if last_result.container_status:
        logger.error("Container status: %s", last_result.container_status)
    if last_result.curl_error:
        logger.error("Health endpoint error: %s", last_result.curl_error)

    # Try to get more detailed container logs (more lines than check_health fetches)
    try:
        logs_result = conn.run(f"sudo docker logs {container_name} --tail 50 2>&1", timeout=Duration.from_seconds(15))
        if logs_result.returncode == 0 and logs_result.stdout.strip():
            logger.error("Container logs (last 50 lines):\n%s", logs_result.stdout.strip())
        elif last_result.container_logs:
            logger.error("Container logs:\n%s", last_result.container_logs)
        else:
            logger.error("No container logs available (container may not exist)")
    except Exception:
        if last_result.container_logs:
            logger.error("Container logs:\n%s", last_result.container_logs)
        else:
            logger.error("No container logs available (container may not exist)")

    logger.error("=" * 60)
    return False


@dataclass
class ControllerStatus:
    """Status of the controller."""

    running: bool
    address: str | None
    healthy: bool
    vm_name: str | None = None


class ControllerProtocol(Protocol):
    """Protocol for controller implementations."""

    def start(self) -> str:
        """Start controller, return address. Idempotent - returns existing if healthy."""
        ...

    def stop(self) -> None:
        """Stop controller."""
        ...

    def restart(self) -> str:
        """Stop then start controller."""
        ...

    def reload(self) -> str:
        """Re-run bootstrap on existing VM without recreating it.

        Unlike restart() which deletes and recreates the VM, reload() SSHs
        into the existing VM and re-runs the bootstrap script to pull the
        latest image and restart the container.

        Raises:
            RuntimeError: If the controller VM doesn't exist or health check fails.
        """
        ...

    def discover(self) -> str | None:
        """Find existing controller address, or None."""
        ...

    def status(self) -> ControllerStatus:
        """Get controller status."""
        ...

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        """Fetch startup script logs for debugging. Returns None if unavailable."""
        ...


# Bootstrap script for controller VM - simpler than worker bootstrap
CONTROLLER_BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-controller] ================================================"
echo "[iris-controller] Starting controller bootstrap at $(date -Iseconds)"
echo "[iris-controller] ================================================"

# Write config file if provided
{config_setup}

# Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[iris-controller] [1/5] Docker not found, installing..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "[iris-controller] [1/5] Docker installation complete"
else
    echo "[iris-controller] [1/5] Docker already installed: $(docker --version)"
fi

echo "[iris-controller] [2/5] Ensuring Docker daemon is running..."
sudo systemctl start docker || true
if sudo docker info > /dev/null 2>&1; then
    echo "[iris-controller] [2/5] Docker daemon is running"
else
    echo "[iris-controller] [2/5] ERROR: Docker daemon failed to start"
    exit 1
fi

# Configure docker for GCP Artifact Registry
echo "[iris-controller] [3/5] Configuring Docker for GCP Artifact Registry..."
sudo gcloud auth configure-docker \
  europe-west4-docker.pkg.dev,us-central1-docker.pkg.dev,us-docker.pkg.dev --quiet 2>/dev/null || true
echo "[iris-controller] [3/5] Docker registry configuration complete"

echo "[iris-controller] [4/5] Pulling image: {docker_image}"
echo "[iris-controller]       This may take several minutes for large images..."
if sudo docker pull {docker_image}; then
    echo "[iris-controller] [4/5] Image pull complete"
else
    echo "[iris-controller] [4/5] ERROR: Image pull failed"
    exit 1
fi

# Stop existing controller if running
echo "[iris-controller] [5/5] Starting controller container..."
if sudo docker ps -a --format '{{{{.Names}}}}' | grep -q "^{container_name}$"; then
    echo "[iris-controller]       Stopping existing container..."
    sudo docker stop {container_name} 2>/dev/null || true
    sudo docker rm {container_name} 2>/dev/null || true
fi

# Create cache directory
sudo mkdir -p /var/cache/iris

# Start controller container with restart policy
sudo docker run -d --name {container_name} \
    --network=host \
    --restart=unless-stopped \
    -v /var/cache/iris:/var/cache/iris \
    {config_volume} \
    {docker_image} \
    python -m iris.cluster.controller.main serve \
        --host 0.0.0.0 --port {port} {config_flag}

echo "[iris-controller] [5/5] Controller container started"

# Wait for health
echo "[iris-controller] Waiting for controller to become healthy..."
for i in $(seq 1 60); do
    echo "[iris-controller] Health check attempt $i/60 at $(date -Iseconds)..."
    if curl -sf http://localhost:{port}/health > /dev/null 2>&1; then
        echo "[iris-controller] ================================================"
        echo "[iris-controller] Controller is healthy! Bootstrap complete."
        echo "[iris-controller] ================================================"
        exit 0
    fi
    # Show brief container status every 5 attempts
    if [ $((i % 5)) -eq 0 ]; then
        STATUS=$(sudo docker inspect --format='{{{{.State.Status}}}}' {container_name} 2>/dev/null || echo 'unknown')
        echo "[iris-controller] Container status: $STATUS"
    fi
    sleep 2
done

echo "[iris-controller] ================================================"
echo "[iris-controller] ERROR: Controller failed to become healthy"
echo "[iris-controller] ================================================"
echo "[iris-controller] Container logs (last 50 lines):"
sudo docker logs {container_name} --tail 50
exit 1
"""

CONFIG_SETUP_TEMPLATE = """
sudo mkdir -p /etc/iris
cat > /tmp/iris_config.yaml << 'IRIS_CONFIG_EOF'
{config_yaml}
IRIS_CONFIG_EOF
sudo mv /tmp/iris_config.yaml /etc/iris/config.yaml
echo "[iris-controller] Config written to /etc/iris/config.yaml"
"""


def _build_controller_bootstrap_script(
    docker_image: str,
    port: int,
    config_yaml: str = "",
) -> str:
    """Build bootstrap script for controller VM.

    Args:
        docker_image: Docker image to run
        port: Controller port
        config_yaml: Optional YAML config to write to /etc/iris/config.yaml
    """
    if config_yaml:
        config_setup = CONFIG_SETUP_TEMPLATE.format(config_yaml=config_yaml)
        config_volume = "-v /etc/iris/config.yaml:/etc/iris/config.yaml:ro"
        config_flag = "--config /etc/iris/config.yaml"
    else:
        config_setup = "# No config file provided"
        config_volume = ""
        config_flag = ""

    return CONTROLLER_BOOTSTRAP_SCRIPT.format(
        docker_image=docker_image,
        container_name=CONTROLLER_CONTAINER_NAME,
        port=port,
        config_setup=config_setup,
        config_volume=config_volume,
        config_flag=config_flag,
    )


def _check_health_rpc(address: str, timeout: float = 5.0, log_result: bool = False) -> bool:
    """Check controller health via RPC.

    Uses a lightweight ListJobs request to verify connectivity, avoiding raw HTTP.
    """
    try:
        client = ControllerServiceClientSync(address)
        client.list_jobs(cluster_pb2.Controller.ListJobsRequest(), timeout_ms=int(timeout * 1000))
        if log_result:
            logger.info("Health check %s: rpc_ok", address)
        return True
    except Exception as e:
        if log_result:
            logger.info("Health check %s: rpc_error=%s", address, type(e).__name__)
        return False


class GcpController:
    """Controller on GCE VM with metadata-based discovery.

    Creates and manages a standard GCE VM (not TPU) running the controller
    container. The VM is tagged with metadata for worker discovery.
    """

    def __init__(self, config: config_pb2.IrisClusterConfig):
        self.config = config
        platform = config.platform.gcp
        self.project_id = platform.project_id
        if platform.zone:
            self.zone = platform.zone
        elif platform.default_zones:
            self.zone = platform.default_zones[0]
        else:
            raise RuntimeError("platform.gcp.zone or platform.gcp.default_zones is required for controller")
        self._gcp_config = config.controller.gcp
        self._label_prefix = config.platform.label_prefix or "iris"
        self._vm_name = f"iris-controller-{self._label_prefix}"

    def _serialize_config(self) -> str:
        """Serialize cluster config to YAML for the controller VM."""
        import yaml

        return yaml.dump(config_to_dict(self.config), default_flow_style=False)

    def start(self) -> str:
        """Start controller GCE VM.

        Idempotent: returns existing controller address if healthy.
        Otherwise cleans up stale controller and creates a new one.

        Bootstrap is done via SSH with streamed logs for visibility.
        """
        logger.info("Starting controller (project=%s, zone=%s)", self.project_id, self.zone)

        existing = self.discover()
        if existing:
            logger.info("Found existing controller at %s, checking health...", existing)
            conn = GceSshConnection(
                project_id=self.project_id,
                zone=self.zone,
                vm_name=self._vm_name,
            )
            port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT
            if wait_healthy_via_ssh(conn, port):
                logger.info("Existing controller at %s is healthy", existing)
                return existing
            logger.info("Existing controller at %s is unhealthy, cleaning up", existing)
            self.stop()
        else:
            logger.info("No existing controller found, creating new VM")

        # Create VM (without startup script - we bootstrap via SSH)
        address = self._create_vm()
        port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT

        # Print SSH command for user inspection
        logger.info(
            "To inspect the VM manually:\n  gcloud compute ssh %s --project=%s --zone=%s",
            self._vm_name,
            self.project_id,
            self.zone,
        )

        # Bootstrap via SSH with streamed logs
        conn = GceSshConnection(
            project_id=self.project_id,
            zone=self.zone,
            vm_name=self._vm_name,
        )

        config_yaml = self._serialize_config()
        bootstrap_script = _build_controller_bootstrap_script(self.config.controller.image, port, config_yaml)

        def on_line(line: str) -> None:
            logger.info("[%s] %s", self._vm_name, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(bootstrap_script)}",
            max_retries=MAX_RETRIES,
            on_line=on_line,
        )

        # Final health check (logs diagnostics on failure)
        if not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {address} failed health check after bootstrap")

        self._tag_metadata(address)
        logger.info("Controller started successfully at %s", address)
        return address

    def stop(self) -> None:
        """Terminate controller GCE VM."""
        self._delete_vm()

    def restart(self) -> str:
        """Stop then start controller."""
        self.stop()
        return self.start()

    def reload(self) -> str:
        """Re-run bootstrap on existing VM without recreating it.

        SSHs into the existing controller VM and re-runs the bootstrap script
        to pull the latest image and restart the container. Faster than restart()
        since it doesn't delete/recreate the VM.
        """
        vm_name = self._find_controller_name()
        if not vm_name:
            raise RuntimeError("Controller VM not found. Use 'start' to create a new one.")

        port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT

        logger.info("Reloading controller on VM %s via SSH", vm_name)

        conn = GceSshConnection(
            project_id=self.project_id,
            zone=self.zone,
            vm_name=vm_name,
        )

        config_yaml = self._serialize_config()
        bootstrap_script = _build_controller_bootstrap_script(self.config.controller.image, port, config_yaml)

        def on_line(line: str) -> None:
            logger.info("[%s] %s", vm_name, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(bootstrap_script)}",
            max_retries=MAX_RETRIES,
            on_line=on_line,
        )

        address = self._get_vm_address()

        # Health check (logs diagnostics on failure)
        if not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {address} failed health check after reload")

        logger.info("Controller reloaded at %s", address)
        return address

    def discover(self) -> str | None:
        """Query GCP for existing controller address.

        Looks for a running VM with the controller metadata tag and returns
        its controller address from metadata. Uses prefix-scoped metadata keys.
        """
        meta_key = controller_metadata_key(self._label_prefix)
        addr_key = controller_address_metadata_key(self._label_prefix)
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={self.project_id}",
            f"--filter=metadata.items.{meta_key}=true AND status=RUNNING",
            f"--format=value(metadata.items.filter(key:{addr_key}).firstof(value))",
            "--limit=1",
        ]
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to query controller: %s", result.stderr.strip())
            return None

        address = result.stdout.strip()
        return address if address else None

    def status(self) -> ControllerStatus:
        """Get controller status from GCP."""
        address = self.discover()
        if not address:
            return ControllerStatus(running=False, address=None, healthy=False, vm_name=None)

        vm_name = self._find_controller_name()
        healthy = _check_health_rpc(address)

        return ControllerStatus(
            running=True,
            address=address,
            healthy=healthy,
            vm_name=vm_name,
        )

    def _find_controller_name(self) -> str | None:
        """Find the name of the running controller VM."""
        meta_key = controller_metadata_key(self._label_prefix)
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={self.project_id}",
            f"--filter=metadata.items.{meta_key}=true AND status=RUNNING",
            "--format=value(name)",
            "--limit=1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def _create_vm(self) -> str:
        """Create GCE VM, return controller address.

        This only creates the VM - bootstrap is done via SSH in start().
        """
        machine_type = self._gcp_config.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = self._gcp_config.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB
        port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT
        meta_key = controller_metadata_key(self._label_prefix)

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            self._vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            f"--machine-type={machine_type}",
            f"--boot-disk-size={boot_disk_size}GB",
            "--image-family=debian-12",
            "--image-project=debian-cloud",
            "--scopes=cloud-platform",
            f"--metadata={meta_key}=true",
            "--format=json",
        ]

        logger.info("Creating controller VM: %s (zone=%s, type=%s)", self._vm_name, self.zone, machine_type)
        logger.debug("Running: %s", " ".join(cmd))

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "already exists" in error_msg.lower():
                logger.info("Controller VM already exists (detected in %.1fs), getting its IP", elapsed)
                return self._get_vm_address()
            raise RuntimeError(f"Failed to create controller VM: {error_msg}")

        logger.info("VM created in %.1fs", elapsed)

        try:
            parsed = json.loads(result.stdout)
            # gcloud returns either a single object or a list
            vm_data: dict = parsed[0] if isinstance(parsed, list) else parsed
            network_interfaces = vm_data.get("networkInterfaces", [])
            if network_interfaces:
                # Use internal IP for worker communication (external IPs may be blocked by firewall)
                ip = network_interfaces[0].get("networkIP")
                if ip:
                    return f"http://{ip}:{port}"
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to parse VM creation output: %s", e)

        return self._get_vm_address()

    def _get_vm_address(self) -> str:
        """Get the internal IP address of the controller VM.

        Uses internal IP for worker communication since external IPs may be blocked
        by firewall rules. GCP's default-allow-internal rule permits internal traffic.
        """
        port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            self._vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            "--format=value(networkInterfaces[0].networkIP)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get controller VM address: {result.stderr.strip()}")

        ip = result.stdout.strip()
        if not ip:
            raise RuntimeError("Controller VM has no internal IP")

        return f"http://{ip}:{port}"

    def _delete_vm(self) -> None:
        """Delete controller GCE VM."""
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "delete",
            self._vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            "--quiet",
        ]
        logger.info("Deleting controller VM: %s", self._vm_name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip()
            if "not found" not in error.lower():
                logger.warning("Failed to delete controller VM: %s", error)

    def _tag_metadata(self, address: str) -> None:
        """Tag VM with controller address metadata for worker discovery."""
        addr_key = controller_address_metadata_key(self._label_prefix)
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "add-metadata",
            self._vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            f"--metadata={addr_key}={address}",
        ]
        logger.debug("Tagging controller VM with address: %s", address)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to tag controller VM: %s", result.stderr.strip())

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        """Fetch the startup script logs from the controller VM via serial console.

        This can be used to debug startup failures by viewing the startup script
        output. Returns None if logs cannot be fetched.

        Args:
            tail_lines: Number of lines to return from the end of the log
        """
        vm_name = self._find_controller_name()
        if not vm_name:
            logger.warning("Cannot fetch logs: controller VM not found")
            return None

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "get-serial-port-output",
            vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
        ]

        logger.info("Fetching serial console output from %s...", vm_name)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning("Failed to fetch serial console output: %s", result.stderr.strip())
            return None

        # Filter for iris-controller lines and take the last N lines
        all_lines = result.stdout.split("\n")
        iris_lines = [line for line in all_lines if "[iris-controller]" in line]

        if iris_lines:
            output_lines = iris_lines[-tail_lines:]
            logger.info("Found %d iris-controller log lines (showing last %d)", len(iris_lines), len(output_lines))
            return "\n".join(output_lines)

        # If no iris-controller lines, return the raw tail
        logger.info("No [iris-controller] lines found, returning raw serial output tail")
        return "\n".join(all_lines[-tail_lines:])


CONTROLLER_STOP_SCRIPT = """
set -e
echo "[iris-controller] Stopping controller container"
sudo docker stop {container_name} 2>/dev/null || true
sudo docker rm {container_name} 2>/dev/null || true
echo "[iris-controller] Controller stopped"
"""


class ManualController:
    """Controller on a manually-managed host via SSH bootstrap.

    SSHs into the configured host to start/stop the controller container.
    Requires controller.manual.host to be configured.
    """

    def __init__(self, config: config_pb2.IrisClusterConfig):
        self.config = config
        self._manual_config = config.controller.manual
        self._bootstrapped = False

        if not self._manual_config.host:
            raise RuntimeError("controller.manual.host is required for ManualController")

        port = self._manual_config.port or DEFAULT_CONTROLLER_PORT
        self.address = f"http://{self._manual_config.host}:{port}"

    def _serialize_config(self) -> str:
        """Serialize cluster config to YAML for the controller VM."""
        import yaml

        return yaml.dump(config_to_dict(self.config), default_flow_style=False)

    def start(self) -> str:
        """Start controller via SSH bootstrap."""
        if not self.config.controller.image:
            raise RuntimeError("controller.image required for SSH bootstrap")

        host = self._manual_config.host
        port = self._manual_config.port or DEFAULT_CONTROLLER_PORT

        logger.info("Bootstrapping controller on %s via SSH", host)

        conn = self._create_ssh_connection(host)
        config_yaml = self._serialize_config()
        bootstrap_script = _build_controller_bootstrap_script(self.config.controller.image, port, config_yaml)

        def on_line(line: str) -> None:
            logger.info("[%s] %s", host, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(bootstrap_script)}",
            max_retries=MAX_RETRIES,
            on_line=on_line,
        )

        self._bootstrapped = True

        if not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {self.address} failed health check after bootstrap")

        logger.info("Controller started at %s", self.address)
        return self.address

    def stop(self) -> None:
        """Stop controller via SSH."""
        if not self._bootstrapped:
            logger.info("Controller was not bootstrapped by us, skipping stop")
            return

        host = self._manual_config.host
        logger.info("Stopping controller on %s via SSH", host)

        conn = self._create_ssh_connection(host)
        stop_script = CONTROLLER_STOP_SCRIPT.format(container_name=CONTROLLER_CONTAINER_NAME)

        try:
            result = conn.run(f"bash -c {shlex.quote(stop_script)}", timeout=Duration.from_seconds(60))
            if result.returncode != 0:
                logger.warning("Stop script returned %d: %s", result.returncode, result.stderr)
        except Exception as e:
            logger.warning("Failed to stop controller: %s", e)

        self._bootstrapped = False

    def restart(self) -> str:
        """Stop then start controller."""
        self.stop()
        return self.start()

    def reload(self) -> str:
        """Re-run bootstrap on existing host.

        For ManualController this is the same as start() since there's no VM
        to preserve - we just re-run the bootstrap script.
        """
        return self.start()

    def discover(self) -> str | None:
        """Return address if controller is healthy via SSH check."""
        host = self._manual_config.host
        port = self._manual_config.port or DEFAULT_CONTROLLER_PORT
        conn = self._create_ssh_connection(host)
        if check_health(conn, port):
            return self.address
        return None

    def status(self) -> ControllerStatus:
        """Check health of controller via SSH."""
        host = self._manual_config.host
        port = self._manual_config.port or DEFAULT_CONTROLLER_PORT
        logger.info("Connecting to controller host %s via SSH...", host)
        conn = self._create_ssh_connection(host)
        logger.info("Checking controller health via SSH (port %d, timeout 10s)...", port)
        result = check_health(conn, port, CONTROLLER_CONTAINER_NAME)
        logger.info("Health check result: %s", "healthy" if result.healthy else result.summary())
        return ControllerStatus(
            running=result.healthy,
            address=self.address,
            healthy=result.healthy,
        )

    def _create_ssh_connection(self, host: str) -> DirectSshConnection:
        """Create SSH connection for the given host."""
        ssh = self.config.ssh
        connect_timeout = (
            Duration.from_proto(ssh.connect_timeout)
            if ssh.HasField("connect_timeout") and ssh.connect_timeout.milliseconds > 0
            else Duration.from_seconds(30)
        )
        return DirectSshConnection(
            host=host,
            user=ssh.user or "root",
            key_file=ssh.key_file or None,
            connect_timeout=connect_timeout,
        )

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        """Fetch container logs from manual controller via SSH.

        For ManualController, we fetch Docker container logs instead of
        serial console output since there's no GCP VM.
        """
        host = self._manual_config.host
        conn = self._create_ssh_connection(host)

        logger.info("Fetching container logs from %s...", host)
        try:
            cmd = f"sudo docker logs {CONTROLLER_CONTAINER_NAME} --tail {tail_lines}"
            result = conn.run(cmd, timeout=Duration.from_seconds(30))
            if result.returncode == 0:
                return result.stdout
            logger.warning("Failed to fetch container logs: %s", result.stderr)
            return None
        except Exception as e:
            logger.warning("Error fetching container logs: %s", e)
            return None


class _InProcessController(Protocol):
    """Protocol for the in-process Controller used by LocalController.

    Avoids importing iris.cluster.controller.controller at module level
    which would create a circular dependency through the autoscaler.
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...

    @property
    def url(self) -> str: ...


class LocalController:
    """In-process controller for local testing.

    Runs Controller + Autoscaler(LocalVmManagers) in the current process.
    Workers are threads, not VMs. No Docker, no GCS, no SSH.
    """

    def __init__(
        self,
        config: config_pb2.IrisClusterConfig,
        threads: ThreadContainer | None = None,
    ):
        self._config = config
        self._threads = threads
        self._controller: _InProcessController | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def start(self) -> str:
        from iris.cluster.controller.controller import (
            Controller as _InnerController,
            ControllerConfig as _InnerControllerConfig,
            RpcWorkerStubFactory,
        )
        from iris.cluster.vm.local_platform import (
            _create_local_autoscaler,
            find_free_port,
        )

        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_")
        temp = Path(self._temp_dir.name)
        bundle_dir = temp / "bundles"
        bundle_dir.mkdir()
        cache_path = temp / "cache"
        cache_path.mkdir()
        fake_bundle = temp / "fake_bundle"
        fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname='local'\n")

        port = self._config.controller.local.port or find_free_port()
        address = f"http://127.0.0.1:{port}"

        controller_threads = self._threads.create_child("controller") if self._threads else None
        autoscaler_threads = controller_threads.create_child("autoscaler") if controller_threads else None

        autoscaler = _create_local_autoscaler(
            self._config,
            address,
            cache_path,
            fake_bundle,
            threads=autoscaler_threads,
        )
        self._controller = _InnerController(
            config=_InnerControllerConfig(
                host="127.0.0.1",
                port=port,
                bundle_prefix=self._config.controller.bundle_prefix or f"file://{bundle_dir}",
            ),
            worker_stub_factory=RpcWorkerStubFactory(),
            autoscaler=autoscaler,
            threads=controller_threads,
        )
        self._controller.start()
        return self._controller.url

    def stop(self) -> None:
        if self._controller:
            self._controller.stop()
            self._controller = None
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def restart(self) -> str:
        self.stop()
        return self.start()

    def reload(self) -> str:
        return self.restart()

    def discover(self) -> str | None:
        return self._controller.url if self._controller else None

    def status(self) -> ControllerStatus:
        if self._controller:
            return ControllerStatus(
                running=True,
                address=self._controller.url,
                healthy=True,
            )
        return ControllerStatus(running=False, address="", healthy=False)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        return "(local controller — no startup logs)"


def create_controller(
    config: config_pb2.IrisClusterConfig,
    threads: ThreadContainer | None = None,
) -> ControllerProtocol:
    """Factory function to create appropriate controller type.

    Dispatches based on the controller.controller oneof field:
    - gcp: Creates GcpController for GCP-managed VMs
    - manual: Creates ManualController for SSH bootstrap to pre-existing hosts
    - local: Creates LocalController for in-process testing

    Args:
        config: Cluster configuration.
        threads: Optional parent ThreadContainer. Only used by LocalController
            to integrate in-process threads into the caller's hierarchy.
    """
    controller = config.controller
    which = controller.WhichOneof("controller")
    if which == "gcp":
        return GcpController(config)
    if which == "manual":
        return ManualController(config)
    if which == "local":
        return LocalController(config, threads=threads)
    raise ValueError("No controller config specified in controller")
