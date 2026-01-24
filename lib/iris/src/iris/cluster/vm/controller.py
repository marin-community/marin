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

from iris.cluster.vm.config import IrisClusterConfig
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


# Bootstrap script for controller VM - simpler than worker bootstrap
CONTROLLER_BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-controller] Starting controller bootstrap"

# Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[iris-controller] Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
fi

sudo systemctl start docker || true

# Configure docker for GCP Artifact Registry
sudo gcloud auth configure-docker \
  europe-west4-docker.pkg.dev,us-central1-docker.pkg.dev,us-docker.pkg.dev --quiet 2>/dev/null || true

echo "[iris-controller] Pulling image: {docker_image}"
sudo docker pull {docker_image}

# Stop existing controller if running
sudo docker stop {container_name} 2>/dev/null || true
sudo docker rm {container_name} 2>/dev/null || true

# Create cache directory
sudo mkdir -p /var/cache/iris

# Start controller container with restart policy
sudo docker run -d --name {container_name} \
    --network=host \
    --restart=unless-stopped \
    -v /var/cache/iris:/var/cache/iris \
    {docker_image}

echo "[iris-controller] Controller container started"

# Wait for health
for i in $(seq 1 60); do
    if curl -sf http://localhost:{port}/health > /dev/null 2>&1; then
        echo "[iris-controller] Controller is healthy"
        exit 0
    fi
    sleep 2
done

echo "[iris-controller] ERROR: Controller failed to become healthy"
sudo docker logs {container_name} --tail 50
exit 1
"""


def _build_controller_bootstrap_script(docker_image: str, port: int) -> str:
    """Build bootstrap script for controller VM."""
    return CONTROLLER_BOOTSTRAP_SCRIPT.format(
        docker_image=docker_image,
        container_name=CONTROLLER_CONTAINER_NAME,
        port=port,
    )


def _check_health_http(address: str, timeout: float = 5.0) -> bool:
    """Check controller health via HTTP endpoint."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{address}/health")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


class GcpController:
    """Controller on GCE VM with metadata-based discovery.

    Creates and manages a standard GCE VM (not TPU) running the controller
    container. The VM is tagged with metadata for worker discovery.
    """

    def __init__(self, config: IrisClusterConfig):
        self.config = config
        self.project_id = config.project_id or ""
        self.zone = config.zone or "us-central1-a"
        self.vm_config = config.controller_vm
        self._vm_name = f"iris-controller-{config.label_prefix}"

    def start(self) -> str:
        """Start controller GCE VM with retry logic.

        Idempotent: returns existing controller address if healthy.
        Otherwise cleans up stale controller and creates a new one.
        """
        existing = self.discover()
        if existing and self._wait_healthy(existing):
            logger.info("Existing controller at %s is healthy", existing)
            return existing

        if existing:
            logger.info("Existing controller at %s is unhealthy, cleaning up", existing)
            self.stop()

        backoff = ExponentialBackoff(
            initial=RETRY_BACKOFF_INITIAL,
            maximum=RETRY_BACKOFF_MAX,
            factor=2.0,
        )

        for attempt in range(MAX_RETRIES):
            try:
                address = self._create_vm()
                if self._wait_healthy(address):
                    self._tag_metadata(address)
                    logger.info("Controller started at %s", address)
                    return address
                logger.warning("Controller health check failed, cleaning up")
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

        port = self.vm_config.port or DEFAULT_CONTROLLER_PORT

        logger.info("Reloading controller on VM %s via SSH", vm_name)

        conn = GceSshConnection(
            project_id=self.project_id,
            zone=self.zone,
            vm_name=vm_name,
        )

        bootstrap_script = _build_controller_bootstrap_script(self.vm_config.image, port)

        def on_line(line: str) -> None:
            logger.info("[%s] %s", vm_name, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(bootstrap_script)}",
            max_retries=MAX_RETRIES,
            on_line=on_line,
        )

        address = self._get_vm_address()

        # Health check via SSH since user may not have direct port access
        backoff = ExponentialBackoff(
            initial=HEALTH_CHECK_BACKOFF_INITIAL,
            maximum=HEALTH_CHECK_BACKOFF_MAX,
        )
        if not backoff.wait_until(lambda: check_health(conn, port), timeout=HEALTH_CHECK_TIMEOUT_SECONDS):
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
        machine_type = self.vm_config.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = self.vm_config.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB
        port = self.vm_config.port or DEFAULT_CONTROLLER_PORT

        bootstrap_script = _build_controller_bootstrap_script(self.vm_config.image, port)

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

        logger.info("Creating controller VM: %s", self._vm_name)
        logger.debug("Running: %s", " ".join(cmd))

        result = subprocess.run(cmd, input=bootstrap_script, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "already exists" in error_msg.lower():
                logger.info("Controller VM already exists, getting its IP")
                return self._get_vm_address()
            raise RetryableError(f"Failed to create controller VM: {error_msg}")

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
                        return f"http://{ip}:{port}"
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to parse VM creation output: %s", e)

        return self._get_vm_address()

    def _get_vm_address(self) -> str:
        """Get the external IP address of the controller VM."""
        port = self.vm_config.port or DEFAULT_CONTROLLER_PORT
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
        """Poll health endpoint until healthy or timeout."""
        backoff = ExponentialBackoff(
            initial=HEALTH_CHECK_BACKOFF_INITIAL,
            maximum=HEALTH_CHECK_BACKOFF_MAX,
        )
        return backoff.wait_until(lambda: _check_health_http(address), timeout=timeout)

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
    Requires controller_vm.host to be configured.
    """

    def __init__(self, config: IrisClusterConfig):
        self.config = config
        self._vm_config = config.controller_vm
        self._bootstrapped = False

        if not self._vm_config.host:
            raise RuntimeError("controller_vm.host is required for ManualController")

        port = self._vm_config.port or DEFAULT_CONTROLLER_PORT
        self.address = f"http://{self._vm_config.host}:{port}"

    def start(self) -> str:
        """Start controller via SSH bootstrap."""
        if not self._vm_config.image:
            raise RuntimeError("controller_vm.image required for SSH bootstrap")

        host = self._vm_config.host
        port = self._vm_config.port or DEFAULT_CONTROLLER_PORT

        logger.info("Bootstrapping controller on %s via SSH", host)

        conn = self._create_ssh_connection(host)
        bootstrap_script = _build_controller_bootstrap_script(self._vm_config.image, port)

        def on_line(line: str) -> None:
            logger.info("[%s] %s", host, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(bootstrap_script)}",
            max_retries=MAX_RETRIES,
            on_line=on_line,
        )

        self._bootstrapped = True

        # Health check via SSH since user may not have direct port access
        if not self._wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {self.address} failed health check after bootstrap")

        logger.info("Controller started at %s", self.address)
        return self.address

    def stop(self) -> None:
        """Stop controller via SSH."""
        if not self._bootstrapped:
            logger.info("Controller was not bootstrapped by us, skipping stop")
            return

        host = self._vm_config.host
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
        host = self._vm_config.host
        port = self._vm_config.port or DEFAULT_CONTROLLER_PORT
        conn = self._create_ssh_connection(host)
        if check_health(conn, port):
            return self.address
        return None

    def status(self) -> ControllerStatus:
        """Check health of controller via SSH."""
        host = self._vm_config.host
        port = self._vm_config.port or DEFAULT_CONTROLLER_PORT
        conn = self._create_ssh_connection(host)
        healthy = check_health(conn, port)
        return ControllerStatus(
            running=healthy,
            address=self.address,
            healthy=healthy,
        )

    def _wait_healthy_via_ssh(
        self, conn: SshConnection, port: int, timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS
    ) -> bool:
        """Poll health endpoint via SSH until healthy or timeout."""
        backoff = ExponentialBackoff(
            initial=HEALTH_CHECK_BACKOFF_INITIAL,
            maximum=HEALTH_CHECK_BACKOFF_MAX,
        )
        return backoff.wait_until(lambda: check_health(conn, port), timeout=timeout)

    def _create_ssh_connection(self, host: str) -> DirectSshConnection:
        """Create SSH connection for the given host."""
        return DirectSshConnection(
            host=host,
            user=self.config.ssh_user,
            key_file=self.config.ssh_private_key,
            connect_timeout=self.config.ssh_connect_timeout_seconds,
        )


def create_controller(config: IrisClusterConfig) -> ControllerProtocol:
    """Factory function to create appropriate controller type.

    For GCP provider with controller VM enabled, creates GcpController.
    Otherwise creates ManualController using the static controller address.
    """
    if config.provider_type == "gcp" and config.controller_vm.enabled:
        return GcpController(config)
    return ManualController(config)
