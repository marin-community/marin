# Copyright The Marin Authors
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

"""Controller lifecycle as free functions using WorkerInfraProvider.

Provides start_controller() and stop_controller() — both taking a
WorkerInfraProvider instance plus a resolve_image callable.  These work
uniformly across GCP and Manual providers.  For local mode, use LocalCluster
directly (fundamentally different mechanism: in-process, no SSH/Docker).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from iris.cluster.providers.protocols import WorkerInfraProvider
from iris.cluster.providers.types import (
    Labels,
    RemoteWorkerHandle,
    StandaloneWorkerHandle,
)
from iris.cluster.providers.gcp.bootstrap import (
    build_controller_bootstrap_script_from_config,
)
from iris.cluster.providers.gcp.ssh import OS_LOGIN_METADATA
from iris.rpc import config_pb2
from rigging.timing import Deadline, Duration, ExponentialBackoff, Timer

logger = logging.getLogger(__name__)


def _identity_resolve_image(image: str, zone: str | None = None) -> str:
    return image


# Constants
CONTROLLER_CONTAINER_NAME = "iris-controller"
DEFAULT_CONTROLLER_PORT = 10000
HEALTH_CHECK_TIMEOUT_SECONDS = 120
RESTART_LOOP_THRESHOLD = 3
EARLY_TIMEOUT_SECONDS = 60

# Backoff parameters for health check polling
HEALTH_CHECK_BACKOFF_INITIAL = 2.0
HEALTH_CHECK_BACKOFF_MAX = 10.0


@dataclass
class ControllerStatus:
    """Status of the controller."""

    running: bool
    address: str | None
    healthy: bool
    vm_name: str | None = None


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
    vm: RemoteWorkerHandle,
    port: int = 10001,
    container_name: str = "iris-worker",
) -> HealthCheckResult:
    """Check if worker/controller is healthy via health endpoint.

    Returns HealthCheckResult with diagnostic info on failure.
    """
    result = HealthCheckResult(healthy=False)

    try:
        logger.info("Running health check: curl -sf http://localhost:%d/health", port)
        curl_result = vm.run_command(
            f"curl -sf http://localhost:{port}/health",
            timeout=Duration.from_seconds(10),
        )
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
        status_result = vm.run_command(
            f"sudo docker inspect --format='{{{{.State.Status}}}}' {container_name} 2>/dev/null || echo 'not_found'",
            timeout=Duration.from_seconds(10),
        )
        result.container_status = status_result.stdout.strip()
    except Exception as e:
        result.container_status = f"error: {e}"

    if result.container_status in ("restarting", "exited", "dead", "not_found"):
        try:
            cmd = f"sudo docker logs {container_name} --tail 20 2>&1"
            logs_result = vm.run_command(cmd, timeout=Duration.from_seconds(15))
            if logs_result.returncode == 0 and logs_result.stdout.strip():
                result.container_logs = logs_result.stdout.strip()
        except Exception as e:
            result.container_logs = f"error fetching logs: {e}"

    return result


def wait_healthy(
    vm: RemoteWorkerHandle,
    port: int,
    timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    container_name: str = CONTROLLER_CONTAINER_NAME,
) -> bool:
    """Poll health endpoint until healthy or timeout.

    Runs curl on the remote host via RemoteWorkerHandle.run_command(), avoiding
    firewall issues that can block external HTTP access to the controller port.

    On failure, logs diagnostic info including container status and logs.

    Detects restart loops: fails early if container is restarting repeatedly
    after 60 seconds instead of waiting for full timeout.
    """
    logger.info("Starting health check (port=%d, timeout=%ds)", port, int(timeout))
    dl = Deadline.from_seconds(timeout)
    timer = Timer()
    backoff = ExponentialBackoff(
        initial=HEALTH_CHECK_BACKOFF_INITIAL,
        maximum=HEALTH_CHECK_BACKOFF_MAX,
    )

    attempt = 0
    last_result = None
    consecutive_restarts = 0

    while True:
        attempt += 1
        last_result = check_health(vm, port, container_name)
        elapsed = timer.elapsed_seconds()

        if last_result.healthy:
            logger.info("Health check succeeded after %d attempts (%.1fs)", attempt, elapsed)
            return True

        logger.info(
            "Health check attempt %d failed (%.1fs): %s",
            attempt,
            elapsed,
            last_result.summary(),
        )

        # Detect restart loop: if container keeps restarting, fail early
        if last_result.container_status == "restarting":
            consecutive_restarts += 1
            if consecutive_restarts >= RESTART_LOOP_THRESHOLD and elapsed >= EARLY_TIMEOUT_SECONDS:
                logger.error(
                    "Container is in restart loop (restarting %d times in %.1fs). Failing early.",
                    consecutive_restarts,
                    elapsed,
                )
                break
        else:
            consecutive_restarts = 0

        if dl.expired():
            break

        interval = backoff.next_interval()
        time.sleep(min(interval, dl.remaining_seconds()))

    # Health check failed - log detailed diagnostics
    logger.error("=" * 60)
    logger.error("Health check FAILED after %d attempts (%.1fs)", attempt, elapsed)
    logger.error("=" * 60)
    logger.error("Final status: %s", last_result.summary())
    if last_result.container_status:
        logger.error("Container status: %s", last_result.container_status)
    if last_result.curl_error:
        logger.error("Health endpoint error: %s", last_result.curl_error)

    # Get detailed container logs and diagnostics
    try:
        # Full container logs (not just tail)
        logs_result = vm.run_command(f"sudo docker logs {container_name} 2>&1", timeout=Duration.from_seconds(30))
        if logs_result.returncode == 0 and logs_result.stdout.strip():
            logger.error("Container logs (full output):\n%s", logs_result.stdout.strip())
        elif last_result.container_logs:
            logger.error("Container logs:\n%s", last_result.container_logs)
        else:
            logger.error("No container logs available (container may not exist)")

        # Get container inspect for detailed status
        inspect_result = vm.run_command(
            f"sudo docker inspect {container_name} 2>&1",
            timeout=Duration.from_seconds(15),
        )
        if inspect_result.returncode == 0:
            logger.error("Container inspect output:\n%s", inspect_result.stdout.strip())

    except Exception as e:
        logger.error("Failed to fetch diagnostics: %s", e)
        if last_result.container_logs:
            logger.error("Container logs:\n%s", last_result.container_logs)
        else:
            logger.error("No container logs available (container may not exist)")

    logger.error("=" * 60)
    return False


def _controller_port(config: config_pb2.IrisClusterConfig) -> int:
    """Extract the controller port from config."""
    which = config.controller.WhichOneof("controller")
    if which == "gcp":
        return config.controller.gcp.port or DEFAULT_CONTROLLER_PORT
    if which == "manual":
        return config.controller.manual.port or DEFAULT_CONTROLLER_PORT
    return DEFAULT_CONTROLLER_PORT


def _discover_controller_vm(
    platform: WorkerInfraProvider,
    label_prefix: str,
) -> StandaloneWorkerHandle | None:
    """Find existing controller VM by labels via project-wide search.

    Passes an empty zones list so the platform searches everywhere. This is
    necessary because the controller may have been created in a zone that no
    longer matches the current config.
    """
    labels = Labels(label_prefix)
    vms = platform.list_vms(
        zones=[],
        labels={labels.iris_controller: "true"},
    )
    if len(vms) > 1:
        vm_ids = [vm.vm_id for vm in vms]
        raise RuntimeError(
            f"Multiple controller VMs found matching label "
            f"'{labels.iris_controller}=true': {vm_ids}. "
            f"Expected at most one controller VM. Remove duplicates before proceeding."
        )
    return vms[0] if vms else None


def _build_controller_vm_config(
    config: config_pb2.IrisClusterConfig,
) -> config_pb2.VmConfig:
    """Build a VmConfig for the controller VM from cluster config."""
    label_prefix = config.platform.label_prefix or "iris"
    labels = Labels(label_prefix)
    vm_config = config_pb2.VmConfig()
    vm_config.name = f"iris-controller-{label_prefix}"
    vm_config.labels[labels.iris_controller] = "true"

    which = config.controller.WhichOneof("controller")
    if which == "gcp":
        gcp_ctrl = config.controller.gcp

        zone = gcp_ctrl.zone
        if not zone:
            raise RuntimeError("controller.gcp.zone is required for GCP controller")

        vm_config.gcp.zone = zone
        vm_config.gcp.machine_type = gcp_ctrl.machine_type or "n2-standard-4"
        vm_config.gcp.boot_disk_size_gb = gcp_ctrl.boot_disk_size_gb or 100
        vm_config.gcp.service_account = gcp_ctrl.service_account
        if config.defaults.ssh.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN:
            for key, value in OS_LOGIN_METADATA.items():
                vm_config.metadata[key] = value
    elif which == "manual":
        manual_ctrl = config.controller.manual
        if not manual_ctrl.host:
            raise RuntimeError("controller.manual.host is required for manual controller")
        vm_config.manual.host = manual_ctrl.host
        ssh = config.defaults.ssh
        vm_config.manual.ssh_user = ssh.user or "root"
        if ssh.key_file:
            vm_config.manual.ssh_key_file = ssh.key_file
    else:
        raise ValueError(f"start_controller() does not support controller type: {which}. Use LocalCluster for local.")

    return vm_config


def start_controller(
    platform: WorkerInfraProvider,
    config: config_pb2.IrisClusterConfig,
    resolve_image: Callable[[str, str | None], str] | None = None,
    health_check_timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    fresh: bool = False,
) -> tuple[str, StandaloneWorkerHandle]:
    """Start or discover existing controller. Returns (address, vm_handle).

    1. Try to discover an existing healthy controller
    2. If found and healthy, return it (unless ``fresh`` is set)
    3. Otherwise, create a new VM and bootstrap it

    For GCP: creates a GCE instance, SSHs in, bootstraps the controller container.
    For Manual: allocates a host, SSHs in, bootstraps.

    If ``fresh`` is True, any existing controller VM is terminated and
    recreated so the new container starts from an empty database. The
    ``--fresh`` flag is also threaded into the bootstrap script so the
    controller skips the remote checkpoint restore on startup.
    """
    _resolve_image = resolve_image or _identity_resolve_image
    label_prefix = config.platform.label_prefix or "iris"
    port = _controller_port(config)

    # Check for existing controller
    existing_vm = _discover_controller_vm(platform, label_prefix)
    if existing_vm:
        if fresh:
            logger.info(
                "Found existing controller VM %s, terminating for --fresh start",
                existing_vm.vm_id,
            )
            existing_vm.terminate(wait=True)
        else:
            logger.info("Found existing controller VM %s, checking health...", existing_vm.vm_id)
            if wait_healthy(existing_vm, port, timeout=health_check_timeout):
                address = f"http://{existing_vm.internal_address}:{port}"
                logger.info("Existing controller at %s is healthy", address)
                return address, existing_vm
            logger.info("Existing controller is unhealthy, terminating and recreating")
            existing_vm.terminate(wait=True)

    # Create new controller VM
    vm_config = _build_controller_vm_config(config)
    logger.info("Creating controller VM: %s", vm_config.name)
    vm = platform.create_vm(vm_config)

    # Wait for connection
    if not vm.wait_for_connection(timeout=Duration.from_seconds(300)):
        vm.terminate(wait=True)
        raise RuntimeError(f"Controller VM {vm_config.name} did not become reachable within 300s")

    # Bootstrap
    bootstrap_script = build_controller_bootstrap_script_from_config(
        config,
        resolve_image=_resolve_image,
        fresh=fresh,
    )
    vm.bootstrap(bootstrap_script)

    # Health check
    address = f"http://{vm.internal_address}:{port}"
    if not wait_healthy(vm, port, timeout=health_check_timeout):
        raise RuntimeError(f"Controller at {address} failed health check after bootstrap")

    # Tag for discovery: label for list_vms(), metadata for address
    labels = Labels(label_prefix)
    vm.set_labels({labels.iris_controller: "true"})
    vm.set_metadata({labels.iris_controller_address: address})

    logger.info("Controller started at %s", address)
    return address, vm


def restart_controller(
    platform: WorkerInfraProvider,
    config: config_pb2.IrisClusterConfig,
    resolve_image: Callable[[str, str | None], str] | None = None,
    health_check_timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> tuple[str, StandaloneWorkerHandle]:
    """Restart controller container in-place on existing VM.

    Re-runs the bootstrap script on the existing controller VM, which stops the
    running container, pulls the latest image, and starts a new container.
    Much faster than a full stop+start cycle since it skips VM creation.
    """
    _resolve_image = resolve_image or _identity_resolve_image
    label_prefix = config.platform.label_prefix or "iris"
    port = _controller_port(config)

    vm = _discover_controller_vm(platform, label_prefix)
    if vm is None:
        raise RuntimeError("No existing controller VM found. Use 'iris cluster start' to create one first.")

    logger.info("Restarting controller container in-place on VM %s", vm.vm_id)

    bootstrap_script = build_controller_bootstrap_script_from_config(config, resolve_image=_resolve_image)
    vm.bootstrap(bootstrap_script)

    address = f"http://{vm.internal_address}:{port}"
    if not wait_healthy(vm, port, timeout=health_check_timeout):
        raise RuntimeError(f"Controller at {address} failed health check after restart")

    logger.info("Controller container restarted at %s", address)
    return address, vm


def stop_controller(platform: WorkerInfraProvider, config: config_pb2.IrisClusterConfig) -> None:
    """Find and terminate the controller VM.

    GCE instance deletion is synchronous (no --async), so the VM is fully gone
    when this returns. This prevents the dying controller from writing stale
    checkpoints after remote state is cleared.
    """
    label_prefix = config.platform.label_prefix or "iris"
    vm = _discover_controller_vm(platform, label_prefix)
    if vm:
        logger.info("Stopping controller VM %s", vm.vm_id)
        vm.terminate(wait=True)
    else:
        logger.info("No controller VM found to stop")
