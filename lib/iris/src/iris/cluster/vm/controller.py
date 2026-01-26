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
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from iris.cluster.vm.config import config_to_dict
from iris.cluster.vm.gcp_tpu_platform import (
    CONTROLLER_ADDRESS_METADATA_KEY,
    CONTROLLER_METADATA_KEY,
)
from iris.cluster.vm.ssh import (
    DirectSshConnection,
    GceSshConnection,
    SshConnection,
    check_health,
    run_streaming_with_retry,
)
from iris.rpc import vm_pb2
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)

# Constants
CONTROLLER_CONTAINER_NAME = "iris-controller"
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50
HEALTH_CHECK_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

# Backoff parameters for health check polling: start at 2s, cap at 10s
HEALTH_CHECK_BACKOFF_INITIAL = 2.0
HEALTH_CHECK_BACKOFF_MAX = 10.0

# Backoff parameters for retry loops: start at 10s, cap at 60s
RETRY_BACKOFF_INITIAL = 10.0
RETRY_BACKOFF_MAX = 60.0


def wait_healthy_via_ssh(
    conn: SshConnection,
    port: int,
    timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> bool:
    """Poll health endpoint via SSH until healthy or timeout.

    Uses SSH to run curl on the remote host, avoiding firewall issues
    that can block external HTTP access to the controller port.
    """
    logger.info("Starting SSH-based health check (port=%d, timeout=%ds)", port, int(timeout))
    start_time = time.time()
    attempt = 0

    backoff = ExponentialBackoff(
        initial=HEALTH_CHECK_BACKOFF_INITIAL,
        maximum=HEALTH_CHECK_BACKOFF_MAX,
    )

    def check_with_logging() -> bool:
        nonlocal attempt
        attempt += 1
        result = check_health(conn, port)
        elapsed = time.time() - start_time
        if result:
            logger.info("SSH health check succeeded after %d attempts (%.1fs)", attempt, elapsed)
        else:
            logger.info("SSH health check attempt %d failed (%.1fs)", attempt, elapsed)
        return result

    success = backoff.wait_until(check_with_logging, timeout=timeout)
    if not success:
        logger.warning("SSH health check failed after %d attempts", attempt)
    return success


class RetryableError(Exception):
    """Error that should trigger a retry."""


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


def _check_health_http(address: str, timeout: float = 5.0, log_result: bool = False) -> bool:
    """Check controller health via HTTP endpoint.

    Args:
        address: Controller address (e.g. http://1.2.3.4:10000)
        timeout: Request timeout in seconds
        log_result: If True, log the result of the health check
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{address}/health")
            if log_result:
                logger.info("Health check %s: status=%d", address, response.status_code)
            return response.status_code == 200
    except httpx.HTTPError as e:
        if log_result:
            logger.info("Health check %s: error=%s", address, type(e).__name__)
        return False
    except Exception as e:
        if log_result:
            logger.info("Health check %s: unexpected error=%s", address, e)
        return False


class GcpController:
    """Controller on GCE VM with metadata-based discovery.

    Creates and manages a standard GCE VM (not TPU) running the controller
    container. The VM is tagged with metadata for worker discovery.
    """

    def __init__(self, config: vm_pb2.IrisClusterConfig):
        self.config = config
        self.project_id = config.project_id or ""
        self.zone = config.zone or "us-central1-a"
        self._gcp_config = config.controller_vm.gcp
        self._vm_name = f"iris-controller-{config.label_prefix or 'iris'}"

    def _serialize_config(self) -> str:
        """Serialize cluster config to YAML for the controller VM."""
        import yaml

        return yaml.dump(config_to_dict(self.config), default_flow_style=False)

    def start(self) -> str:
        """Start controller GCE VM with retry logic.

        Idempotent: returns existing controller address if healthy.
        Otherwise cleans up stale controller and creates a new one.
        """
        logger.info("Starting controller (project=%s, zone=%s)", self.project_id, self.zone)

        existing = self.discover()
        if existing:
            logger.info("Found existing controller at %s, checking health...", existing)
            if self._wait_healthy(existing):
                logger.info("Existing controller at %s is healthy", existing)
                return existing
            logger.info("Existing controller at %s is unhealthy, cleaning up", existing)
            self.stop()
        else:
            logger.info("No existing controller found, creating new VM")

        backoff = ExponentialBackoff(
            initial=RETRY_BACKOFF_INITIAL,
            maximum=RETRY_BACKOFF_MAX,
            factor=2.0,
        )

        for attempt in range(MAX_RETRIES):
            logger.info("Starting controller VM creation (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            try:
                address = self._create_vm()

                # Use SSH-based health check since firewall may block external port access
                port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT
                conn = GceSshConnection(
                    project_id=self.project_id,
                    zone=self.zone,
                    vm_name=self._vm_name,
                )
                if wait_healthy_via_ssh(conn, port):
                    self._tag_metadata(address)
                    logger.info("Controller started successfully at %s", address)
                    return address

                # Health check failed - try to get logs for debugging
                logger.warning("Controller health check failed after VM creation")
                logs = self.fetch_startup_logs(tail_lines=30)
                if logs:
                    logger.warning("Startup script output:\n%s", logs)
                self._delete_vm()
                raise RetryableError("Health check failed after VM creation")
            except RetryableError as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = backoff.next_interval()
                    logger.info(
                        "Retrying controller start in %.1fs (attempt %d/%d)", wait_time, attempt + 1, MAX_RETRIES
                    )
                    time.sleep(wait_time)
                else:
                    # Final failure - try to get logs one more time
                    logs = self.fetch_startup_logs(tail_lines=50)
                    if logs:
                        logger.error("Final startup script output:\n%s", logs)
                    raise RuntimeError(f"Failed to start controller after {MAX_RETRIES} attempts") from e

        raise RuntimeError("Controller start failed unexpectedly")

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
        vm_name = self._find_controller_vm_name()
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
        bootstrap_script = _build_controller_bootstrap_script(self._gcp_config.image, port, config_yaml)

        def on_line(line: str) -> None:
            logger.info("[%s] %s", vm_name, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(bootstrap_script)}",
            max_retries=MAX_RETRIES,
            on_line=on_line,
        )

        address = self._get_vm_address()

        if not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {address} failed health check after reload")

        logger.info("Controller reloaded at %s", address)
        return address

    def discover(self) -> str | None:
        """Query GCP for existing controller address.

        Looks for a running VM with the controller metadata tag and returns
        its controller address from metadata.
        """
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={self.project_id}",
            f"--filter=metadata.items.{CONTROLLER_METADATA_KEY}=true AND status=RUNNING",
            f"--format=value(metadata.items.filter(key:{CONTROLLER_ADDRESS_METADATA_KEY}).firstof(value))",
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

        vm_name = self._find_controller_vm_name()
        healthy = _check_health_http(address)

        return ControllerStatus(
            running=True,
            address=address,
            healthy=healthy,
            vm_name=vm_name,
        )

    def _find_controller_vm_name(self) -> str | None:
        """Find the name of the running controller VM."""
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={self.project_id}",
            f"--filter=metadata.items.{CONTROLLER_METADATA_KEY}=true AND status=RUNNING",
            "--format=value(name)",
            "--limit=1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def _create_vm(self) -> str:
        """Create GCE VM and run bootstrap, return controller address."""
        machine_type = self._gcp_config.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = self._gcp_config.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB
        port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT

        config_yaml = self._serialize_config()
        bootstrap_script = _build_controller_bootstrap_script(self._gcp_config.image, port, config_yaml)

        # Log bootstrap script summary for visibility
        script_lines = bootstrap_script.strip().split("\n")
        logger.info(
            "Bootstrap script prepared (%d lines). Key steps: Docker install, image pull (%s), container start",
            len(script_lines),
            self._gcp_config.image,
        )

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
            f"--metadata={CONTROLLER_METADATA_KEY}=true",
            "--metadata-from-file=startup-script=/dev/stdin",
            "--format=json",
        ]

        logger.info("Creating controller VM: %s (zone=%s, type=%s)", self._vm_name, self.zone, machine_type)
        logger.info("VM creation in progress... (this may take 30-60 seconds)")
        logger.debug("Running: %s", " ".join(cmd))

        start_time = time.time()
        result = subprocess.run(cmd, input=bootstrap_script, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "already exists" in error_msg.lower():
                logger.info("Controller VM already exists (detected in %.1fs), getting its IP", elapsed)
                return self._get_vm_address()
            logger.error("VM creation failed after %.1fs: %s", elapsed, error_msg)
            raise RetryableError(f"Failed to create controller VM: {error_msg}")

        logger.info("VM created successfully in %.1fs, extracting IP address...", elapsed)

        try:
            parsed = json.loads(result.stdout)
            # gcloud returns either a single object or a list
            vm_data: dict = parsed[0] if isinstance(parsed, list) else parsed
            network_interfaces = vm_data.get("networkInterfaces", [])
            if network_interfaces:
                access_configs = network_interfaces[0].get("accessConfigs", [])
                if access_configs:
                    ip = access_configs[0].get("natIP")
                    if ip:
                        address = f"http://{ip}:{port}"
                        logger.info("Controller VM address: %s", address)
                        logger.info(
                            "Startup script is now running on the VM. "
                            "Use 'iris cluster logs' or check serial console for progress."
                        )
                        return address
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to parse VM creation output: %s", e)

        return self._get_vm_address()

    def _get_vm_address(self) -> str:
        """Get the external IP address of the controller VM."""
        port = self._gcp_config.port or DEFAULT_CONTROLLER_PORT
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            self._vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            "--format=value(networkInterfaces[0].accessConfigs[0].natIP)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RetryableError(f"Failed to get controller VM address: {result.stderr.strip()}")

        ip = result.stdout.strip()
        if not ip:
            raise RetryableError("Controller VM has no external IP")

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

    def _wait_healthy(self, address: str, timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS) -> bool:
        """Poll health endpoint until healthy or timeout with verbose logging."""
        logger.info("Starting health check polling for %s (timeout=%ds)", address, int(timeout))
        start_time = time.time()
        attempt = 0

        backoff = ExponentialBackoff(
            initial=HEALTH_CHECK_BACKOFF_INITIAL,
            maximum=HEALTH_CHECK_BACKOFF_MAX,
        )

        def check_with_logging() -> bool:
            nonlocal attempt
            attempt += 1
            elapsed = time.time() - start_time
            result = _check_health_http(address, log_result=True)
            if result:
                logger.info("Health check succeeded after %d attempts (%.1fs elapsed)", attempt, elapsed)
            return result

        success = backoff.wait_until(check_with_logging, timeout=timeout)
        if not success:
            elapsed = time.time() - start_time
            logger.warning(
                "Health check failed after %d attempts (%.1fs elapsed, timeout=%ds)",
                attempt,
                elapsed,
                int(timeout),
            )
        return success

    def _tag_metadata(self, address: str) -> None:
        """Tag VM with controller address metadata for worker discovery."""
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "add-metadata",
            self._vm_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            f"--metadata={CONTROLLER_ADDRESS_METADATA_KEY}={address}",
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
        vm_name = self._find_controller_vm_name()
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
    Requires controller_vm.manual.host to be configured.
    """

    def __init__(self, config: vm_pb2.IrisClusterConfig):
        self.config = config
        self._manual_config = config.controller_vm.manual
        self._bootstrapped = False

        if not self._manual_config.host:
            raise RuntimeError("controller_vm.manual.host is required for ManualController")

        port = self._manual_config.port or DEFAULT_CONTROLLER_PORT
        self.address = f"http://{self._manual_config.host}:{port}"

    def _serialize_config(self) -> str:
        """Serialize cluster config to YAML for the controller VM."""
        import yaml

        return yaml.dump(config_to_dict(self.config), default_flow_style=False)

    def start(self) -> str:
        """Start controller via SSH bootstrap."""
        if not self._manual_config.image:
            raise RuntimeError("controller_vm.image required for SSH bootstrap")

        host = self._manual_config.host
        port = self._manual_config.port or DEFAULT_CONTROLLER_PORT

        logger.info("Bootstrapping controller on %s via SSH", host)

        conn = self._create_ssh_connection(host)
        config_yaml = self._serialize_config()
        bootstrap_script = _build_controller_bootstrap_script(self._manual_config.image, port, config_yaml)

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
            result = conn.run(f"bash -c {shlex.quote(stop_script)}", timeout=60)
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
        conn = self._create_ssh_connection(host)
        healthy = check_health(conn, port)
        return ControllerStatus(
            running=healthy,
            address=self.address,
            healthy=healthy,
        )

    def _create_ssh_connection(self, host: str) -> DirectSshConnection:
        """Create SSH connection for the given host."""
        return DirectSshConnection(
            host=host,
            user=self.config.ssh_user or "root",
            key_file=self.config.ssh_private_key or None,
            connect_timeout=self.config.ssh_connect_timeout_seconds or 30,
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
            result = conn.run(f"sudo docker logs {CONTROLLER_CONTAINER_NAME} --tail {tail_lines}", timeout=30)
            if result.returncode == 0:
                return result.stdout
            logger.warning("Failed to fetch container logs: %s", result.stderr)
            return None
        except Exception as e:
            logger.warning("Error fetching container logs: %s", e)
            return None


def create_controller(config: vm_pb2.IrisClusterConfig) -> ControllerProtocol:
    """Factory function to create appropriate controller type.

    Dispatches based on the controller_vm.controller oneof field:
    - gcp: Creates GcpController for GCP-managed VMs
    - manual: Creates ManualController for SSH bootstrap to pre-existing hosts
    """
    controller_vm = config.controller_vm
    which = controller_vm.WhichOneof("controller")
    if which == "gcp":
        return GcpController(config)
    if which == "manual":
        return ManualController(config)
    raise ValueError("No controller config specified in controller_vm")
