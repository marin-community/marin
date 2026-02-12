# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Controller lifecycle as free functions using Platform.

Provides start_controller(), stop_controller(), reload_controller() — all
taking a Platform instance. These work uniformly across GCP and Manual
platforms. For local mode, use LocalController directly (fundamentally
different mechanism: in-process, no SSH/Docker).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import cast

from iris.cluster.platform.base import Platform, StandaloneVmHandle, VmHandle
from iris.cluster.platform.bootstrap import build_controller_bootstrap_script_from_config
from iris.rpc import config_pb2
from iris.time_utils import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

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
    vm: VmHandle,
    port: int = 10001,
    container_name: str = "iris-worker",
) -> HealthCheckResult:
    """Check if worker/controller is healthy via health endpoint.

    Returns HealthCheckResult with diagnostic info on failure.
    """
    result = HealthCheckResult(healthy=False)

    try:
        logger.info("Running health check: curl -sf http://localhost:%d/health", port)
        curl_result = vm.run_command(f"curl -sf http://localhost:{port}/health", timeout=Duration.from_seconds(10))
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
    vm: VmHandle,
    port: int,
    timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    container_name: str = CONTROLLER_CONTAINER_NAME,
) -> bool:
    """Poll health endpoint until healthy or timeout.

    Runs curl on the remote host via VmHandle.run_command(), avoiding firewall
    issues that can block external HTTP access to the controller port.

    On failure, logs diagnostic info including container status and logs.

    Detects restart loops: fails early if container is restarting repeatedly
    after 60 seconds instead of waiting for full timeout.
    """
    logger.info("Starting health check (port=%d, timeout=%ds)", port, int(timeout))
    start_time = time.monotonic()
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
        elapsed = time.monotonic() - start_time

        if last_result.healthy:
            logger.info("Health check succeeded after %d attempts (%.1fs)", attempt, elapsed)
            return True

        logger.info("Health check attempt %d failed (%.1fs): %s", attempt, elapsed, last_result.summary())

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

        if elapsed >= timeout:
            break

        interval = backoff.next_interval()
        remaining = timeout - elapsed
        time.sleep(min(interval, remaining))

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


def _controller_zones(config: config_pb2.IrisClusterConfig) -> list[str]:
    """Extract zones to search for controller VM."""
    which = config.controller.WhichOneof("controller")
    if which == "gcp":
        zone = config.controller.gcp.zone
        if zone:
            return [zone]
    # Manual/Local don't need zones for list_vms
    return ["local"]


def _controller_label_key(label_prefix: str) -> str:
    return f"{label_prefix}-controller"


def _controller_address_metadata_key(label_prefix: str) -> str:
    return f"{label_prefix}-controller-address"


def _discover_controller_vm(
    platform: Platform,
    zones: list[str],
    label_prefix: str,
) -> StandaloneVmHandle | None:
    """Find existing controller VM by labels.

    list_vms returns VmHandle, but controller VMs were created by create_vm()
    and support terminate/set_labels. We cast to StandaloneVmHandle since we
    know the controller VM supports these operations.
    """
    vms = platform.list_vms(
        zones=zones,
        labels={_controller_label_key(label_prefix): "true"},
    )
    if len(vms) > 1:
        vm_ids = [vm.vm_id for vm in vms]
        raise RuntimeError(
            f"Multiple controller VMs found matching label "
            f"'{_controller_label_key(label_prefix)}=true': {vm_ids}. "
            f"Expected at most one controller VM. Remove duplicates before proceeding."
        )
    return cast(StandaloneVmHandle, vms[0]) if vms else None


def _build_controller_vm_config(config: config_pb2.IrisClusterConfig) -> config_pb2.VmConfig:
    """Build a VmConfig for the controller VM from cluster config."""
    label_prefix = config.platform.label_prefix or "iris"
    vm_config = config_pb2.VmConfig()
    vm_config.name = f"iris-controller-{label_prefix}"
    vm_config.labels[_controller_label_key(label_prefix)] = "true"

    which = config.controller.WhichOneof("controller")
    if which == "gcp":
        gcp_ctrl = config.controller.gcp

        zone = gcp_ctrl.zone
        if not zone:
            raise RuntimeError("controller.gcp.zone is required for GCP controller")

        vm_config.gcp.zone = zone
        vm_config.gcp.machine_type = gcp_ctrl.machine_type or "n2-standard-4"
        vm_config.gcp.boot_disk_size_gb = gcp_ctrl.boot_disk_size_gb or 50
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
        raise ValueError(f"start_controller() does not support controller type: {which}. Use LocalController for local.")

    return vm_config


def start_controller(
    platform: Platform,
    config: config_pb2.IrisClusterConfig,
) -> tuple[str, StandaloneVmHandle]:
    """Start or discover existing controller. Returns (address, vm_handle).

    1. Try to discover an existing healthy controller
    2. If found and healthy, return it
    3. Otherwise, create a new VM and bootstrap it

    For GCP: creates a GCE instance, SSHs in, bootstraps the controller container.
    For Manual: allocates a host, SSHs in, bootstraps.
    """
    label_prefix = config.platform.label_prefix or "iris"
    zones = _controller_zones(config)
    port = _controller_port(config)

    # Check for existing controller
    existing_vm = _discover_controller_vm(platform, zones, label_prefix)
    if existing_vm:
        logger.info("Found existing controller VM %s, checking health...", existing_vm.vm_id)
        if wait_healthy(existing_vm, port):
            address = f"http://{existing_vm.internal_address}:{port}"
            logger.info("Existing controller at %s is healthy", address)
            return address, existing_vm
        logger.info("Existing controller is unhealthy, terminating and recreating")
        existing_vm.terminate()

    # Create new controller VM
    vm_config = _build_controller_vm_config(config)
    logger.info("Creating controller VM: %s", vm_config.name)
    vm = platform.create_vm(vm_config)

    # Wait for connection
    if not vm.wait_for_connection(timeout=Duration.from_seconds(300)):
        vm.terminate()
        raise RuntimeError(f"Controller VM {vm_config.name} did not become reachable within 300s")

    # Bootstrap
    bootstrap_script = build_controller_bootstrap_script_from_config(config)
    vm.bootstrap(bootstrap_script)

    # Health check
    address = f"http://{vm.internal_address}:{port}"
    if not wait_healthy(vm, port):
        raise RuntimeError(f"Controller at {address} failed health check after bootstrap")

    # Tag for discovery: label for list_vms(), metadata for address
    vm.set_labels({_controller_label_key(label_prefix): "true"})
    vm.set_metadata({_controller_address_metadata_key(label_prefix): address})

    logger.info("Controller started at %s", address)
    return address, vm


def stop_controller(platform: Platform, config: config_pb2.IrisClusterConfig) -> None:
    """Find and terminate the controller VM."""
    label_prefix = config.platform.label_prefix or "iris"
    zones = _controller_zones(config)
    vm = _discover_controller_vm(platform, zones, label_prefix)
    if vm:
        logger.info("Stopping controller VM %s", vm.vm_id)
        vm.terminate()
    else:
        logger.info("No controller VM found to stop")


def reload_controller(
    platform: Platform,
    config: config_pb2.IrisClusterConfig,
) -> str:
    """Re-bootstrap the controller on existing VM. Returns address."""
    label_prefix = config.platform.label_prefix or "iris"
    zones = _controller_zones(config)
    port = _controller_port(config)

    vm = _discover_controller_vm(platform, zones, label_prefix)
    if not vm:
        raise RuntimeError("Controller VM not found. Use start_controller() to create a new one.")

    logger.info("Reloading controller on VM %s", vm.vm_id)
    bootstrap_script = build_controller_bootstrap_script_from_config(config)
    vm.bootstrap(bootstrap_script)

    address = f"http://{vm.internal_address}:{port}"
    if not wait_healthy(vm, port):
        raise RuntimeError(f"Controller at {address} failed health check after reload")

    logger.info("Controller reloaded at %s", address)
    return address
