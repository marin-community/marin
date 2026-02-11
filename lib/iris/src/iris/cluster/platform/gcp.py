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

"""GCP Platform implementation.

Implements the Platform protocol for Google Cloud Platform, providing:
- GcpPlatform: Creates/lists VMs and TPU slices via gcloud CLI
- GcpSliceHandle: Manages a TPU pod (list workers, terminate, status)
- GcpVmHandle: SSH to a TPU worker VM via gcloud
- GcpStandaloneVmHandle: SSH to a GCE instance with terminate/label/metadata support

All gcloud operations shell out to the gcloud CLI. Each run_command() call
creates a new SSH process, making VmHandle implementations thread-safe for
concurrent access.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from iris.cluster.platform._vm_base import SshVmBase
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    PlatformError,
    QuotaExhaustedError,
    SliceStatus,
    VmStatus,
)
from iris.cluster.platform.ssh import (
    GceSshConnection,
    GcloudSshConnection,
)
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

# GCP TPU state mapping
_TPU_STATE_MAP: dict[str, CloudSliceState] = {
    "CREATING": CloudSliceState.CREATING,
    "READY": CloudSliceState.READY,
    "REPAIRING": CloudSliceState.REPAIRING,
    "DELETING": CloudSliceState.DELETING,
}


def _format_labels(labels: dict[str, str]) -> str:
    """Format labels as comma-separated key=value pairs for gcloud --labels flag."""
    return ",".join(f"{k}={v}" for k, v in labels.items())


def _build_label_filter(labels: dict[str, str]) -> str:
    """Build a gcloud --filter expression for label matching."""
    parts = [f"labels.{k}={v}" for k, v in labels.items()]
    return " AND ".join(parts)


def _extract_node_name(resource_name: str) -> str:
    """Extract node name from GCP resource path.

    GCP returns 'projects/proj/locations/zone/nodes/my-tpu'
    but gcloud delete expects just 'my-tpu'.
    """
    if "/" in resource_name:
        return resource_name.split("/")[-1]
    return resource_name


def _parse_tpu_created_at(tpu_data: dict) -> Timestamp:
    """Parse createTime from GCP TPU JSON into a Timestamp."""
    create_time = tpu_data.get("createTime", "")
    if not create_time:
        return Timestamp.now()
    # GCP returns ISO 8601 format like "2024-01-15T10:30:00.000Z"
    # Convert to epoch ms
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        epoch_ms = int(dt.timestamp() * 1000)
        return Timestamp.from_epoch_ms(epoch_ms)
    except (ValueError, AttributeError):
        return Timestamp.now()


def _classify_gcloud_error(stderr: str) -> PlatformError:
    """Classify a gcloud error into a specific PlatformError subclass."""
    lower = stderr.lower()
    if "quota" in lower or "insufficient" in lower or "resource_exhausted" in lower:
        return QuotaExhaustedError(stderr)
    return PlatformError(stderr)


def _validate_slice_config(config: config_pb2.SliceConfig) -> None:
    """Validate required fields on a SliceConfig before creating a TPU.

    Raises ValueError listing all missing fields so operators can fix config
    in one pass rather than discovering issues one-by-one.
    """
    missing: list[str] = []
    if not config.accelerator_variant:
        missing.append("accelerator_variant")
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if not config.gcp.runtime_version:
        missing.append("gcp.runtime_version")
    if missing:
        raise ValueError(f"SliceConfig is missing required fields: {', '.join(missing)}")


def _validate_vm_config(config: config_pb2.VmConfig) -> None:
    """Validate required fields on a VmConfig before creating a GCE instance."""
    missing: list[str] = []
    if not config.name:
        missing.append("name")
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if missing:
        raise ValueError(f"VmConfig is missing required fields: {', '.join(missing)}")


def _wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class GcpVmHandle(SshVmBase):
    """Handle to a TPU worker VM within a slice.

    Uses GcloudSshConnection for SSH via `gcloud compute tpus tpu-vm ssh`.
    Thread-safe: each run_command() spawns a new SSH process.
    """

    def status(self) -> VmStatus:
        # TPU worker VMs don't have independent status queries;
        # their status is derived from the slice status.
        return VmStatus(state=CloudVmState.RUNNING)


@dataclass
class GcpStandaloneVmHandle(SshVmBase):
    """Handle to a standalone GCE instance (e.g., controller VM).

    Uses GceSshConnection for SSH via `gcloud compute ssh`.
    Supports terminate, set_labels, and set_metadata operations.
    """

    _zone: str = ""
    _project_id: str = ""

    def status(self) -> VmStatus:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            self._vm_id,
            f"--project={self._project_id}",
            f"--zone={self._zone}",
            "--format=value(status)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return VmStatus(state=CloudVmState.UNKNOWN)
        status_str = result.stdout.strip().upper()
        state_map = {
            "RUNNING": CloudVmState.RUNNING,
            "STOPPED": CloudVmState.STOPPED,
            "TERMINATED": CloudVmState.TERMINATED,
        }
        return VmStatus(state=state_map.get(status_str, CloudVmState.UNKNOWN))

    def reboot(self) -> None:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "reset",
            self._vm_id,
            f"--project={self._project_id}",
            f"--zone={self._zone}",
            "--quiet",
        ]
        logger.info("Rebooting GCE instance: %s", self._vm_id)
        logger.info("gcloud command: %s", cmd)
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def terminate(self) -> None:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "delete",
            self._vm_id,
            f"--project={self._project_id}",
            f"--zone={self._zone}",
            "--quiet",
        ]
        logger.info("Deleting GCE instance: %s", self._vm_id)
        logger.info("gcloud command: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip()
            if "not found" not in error.lower():
                logger.warning("Failed to delete GCE instance %s: %s", self._vm_id, error)

    def set_labels(self, labels: dict[str, str]) -> None:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "update",
            self._vm_id,
            f"--project={self._project_id}",
            f"--zone={self._zone}",
            f"--update-labels={_format_labels(labels)}",
        ]
        logger.info("Setting labels on GCE instance: %s", self._vm_id)
        logger.info("gcloud command: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to set labels on %s: %s", self._vm_id, result.stderr.strip())

    def set_metadata(self, metadata: dict[str, str]) -> None:
        metadata_str = ",".join(f"{k}={v}" for k, v in metadata.items())
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "add-metadata",
            self._vm_id,
            f"--project={self._project_id}",
            f"--zone={self._zone}",
            f"--metadata={metadata_str}",
        ]
        logger.info("Setting metadata on GCE instance: %s", self._vm_id)
        logger.info("gcloud command: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to set metadata on %s: %s", self._vm_id, result.stderr.strip())


@dataclass
class GcpSliceHandle:
    """Handle to a GCP TPU slice (pod).

    list_vms() performs a live query via `gcloud compute tpus describe` to get
    current network endpoints. The slice is the atomic unit for termination.
    """

    _slice_id: str
    _zone: str
    _project_id: str
    _labels: dict[str, str]
    _created_at: Timestamp
    _label_prefix: str
    _accelerator_variant: str
    _ssh_config: config_pb2.SshConfig | None = None
    _state: str = "READY"

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return self._zone

    @property
    def scale_group(self) -> str:
        return self._labels.get(f"{self._label_prefix}-scale-group", "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def list_vms(self) -> list[GcpVmHandle]:
        """Live query for VMs in this TPU slice.

        Returns vm_count handles based on topology (source of truth for VM count).
        Endpoint data provides IP addresses; VMs still provisioning have empty addresses.
        """
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            self._slice_id,
            f"--zone={self._zone}",
            f"--project={self._project_id}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to describe TPU %s: %s", self._slice_id, result.stderr.strip())
            return []

        tpu_data = json.loads(result.stdout)
        endpoints = tpu_data.get("networkEndpoints", [])

        try:
            vm_count = get_tpu_topology(self._accelerator_variant).vm_count
        except ValueError as e:
            raise PlatformError(
                f"Unknown TPU topology '{self._accelerator_variant}' for slice {self._slice_id}. "
                f"Cannot determine VM count without a known topology."
            ) from e

        # Create handles for all VMs (based on topology count)
        vms: list[GcpVmHandle] = []
        for i in range(vm_count):
            # Get IP addresses from endpoints if available
            ep = endpoints[i] if i < len(endpoints) else {}
            internal_ip = ep.get("ipAddress", "")
            external_ip = ep.get("accessConfig", {}).get("externalIp") if "accessConfig" in ep else None

            if not internal_ip and i < len(endpoints):
                logger.warning(
                    "TPU %s endpoint %d has no IP address; VM may still be provisioning",
                    self._slice_id,
                    i,
                )

            ssh = GcloudSshConnection(
                project_id=self._project_id,
                _zone=self._zone,
                vm_id=self._slice_id,
                worker_index=i,
                _address=internal_ip,
            )
            vms.append(
                GcpVmHandle(
                    _vm_id=f"{self._slice_id}-worker-{i}",
                    _internal_address=internal_ip,
                    _external_address=external_ip,
                    _ssh=ssh,
                )
            )

        return vms

    def terminate(self) -> None:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            self._slice_id,
            f"--zone={self._zone}",
            f"--project={self._project_id}",
            "--quiet",
        ]
        logger.info("Terminating TPU: %s", self._slice_id)
        logger.info("gcloud command: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to delete TPU %s: %s", self._slice_id, result.stderr.strip())

    def status(self) -> SliceStatus:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            self._slice_id,
            f"--zone={self._zone}",
            f"--project={self._project_id}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return SliceStatus(state=CloudSliceState.UNKNOWN, vm_count=0)

        tpu_data = json.loads(result.stdout)
        state_str = tpu_data.get("state", "UNKNOWN")
        state = _TPU_STATE_MAP.get(state_str, CloudSliceState.UNKNOWN)
        vm_count = len(tpu_data.get("networkEndpoints", []))
        return SliceStatus(state=state, vm_count=vm_count)


# ============================================================================
# GcpPlatform
# ============================================================================

DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50


class GcpPlatform:
    """Platform implementation for Google Cloud Platform.

    Manages GCE instances (standalone VMs) and TPU slices via gcloud CLI.
    Zone is not stored on the platform — it comes from VmConfig/SliceConfig.
    """

    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
    ):
        self._project_id = gcp_config.project_id
        self._label_prefix = label_prefix
        self._ssh_config = ssh_config

    def create_vm(self, config: config_pb2.VmConfig) -> GcpStandaloneVmHandle:
        """Create a GCE instance. Returns a handle with SSH and label/metadata support."""
        _validate_vm_config(config)
        gcp = config.gcp
        zone = gcp.zone
        machine_type = gcp.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = gcp.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            config.name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--machine-type={machine_type}",
            f"--boot-disk-size={boot_disk_size}GB",
            "--image-family=debian-12",
            "--image-project=debian-cloud",
            "--scopes=cloud-platform",
            "--format=json",
        ]

        if config.labels:
            cmd.append(f"--labels={_format_labels(dict(config.labels))}")

        if config.metadata:
            metadata_str = ",".join(f"{k}={v}" for k, v in config.metadata.items())
            cmd.append(f"--metadata={metadata_str}")

        logger.info("Creating GCE instance: %s (zone=%s, type=%s)", config.name, zone, machine_type)
        logger.info("gcloud command: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "already exists" in error_msg.lower():
                logger.info("GCE instance %s already exists, getting its IP", config.name)
            else:
                raise _classify_gcloud_error(error_msg)

        # Get internal/external IP
        internal_ip, external_ip = self._get_vm_ips(zone, config.name)

        ssh = GceSshConnection(
            project_id=self._project_id,
            zone=zone,
            vm_name=config.name,
        )

        return GcpStandaloneVmHandle(
            _vm_id=config.name,
            _internal_address=internal_ip,
            _external_address=external_ip,
            _zone=zone,
            _project_id=self._project_id,
            _ssh=ssh,
        )

    def create_slice(self, config: config_pb2.SliceConfig) -> GcpSliceHandle:
        """Create a TPU slice via gcloud."""
        _validate_slice_config(config)
        gcp = config.gcp
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            slice_id,
            f"--zone={gcp.zone}",
            f"--project={self._project_id}",
            f"--accelerator-type={config.accelerator_variant}",
            f"--version={gcp.runtime_version}",
        ]

        if config.labels:
            cmd.extend(["--labels", _format_labels(dict(config.labels))])

        if config.preemptible:
            cmd.append("--preemptible")

        logger.info("Creating TPU slice: %s (type=%s, zone=%s)", slice_id, config.accelerator_variant, gcp.zone)
        logger.info("gcloud command: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

        return GcpSliceHandle(
            _slice_id=slice_id,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _accelerator_variant=config.accelerator_variant,
            _ssh_config=self._ssh_config,
        )

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpSliceHandle]:
        """List TPU slices across zones, optionally filtered by labels."""
        results: list[GcpSliceHandle] = []
        for zone in zones:
            for tpu_data in self._gcloud_list_tpus(zone, labels):
                state = tpu_data.get("state", "UNKNOWN")
                if state not in ("READY", "CREATING"):
                    logger.info("Skipping TPU %s in state %s", tpu_data["name"], state)
                    continue

                tpu_labels = tpu_data.get("labels", {})
                accelerator_type = tpu_data.get("acceleratorType", "")
                # acceleratorType can be a full path like "v5litepod-16" or
                # "projects/proj/locations/zone/acceleratorTypes/v5litepod-16"
                if "/" in accelerator_type:
                    accelerator_type = accelerator_type.split("/")[-1]

                results.append(
                    GcpSliceHandle(
                        _slice_id=tpu_data["name"],
                        _zone=zone,
                        _project_id=self._project_id,
                        _labels=tpu_labels,
                        _created_at=_parse_tpu_created_at(tpu_data),
                        _label_prefix=self._label_prefix,
                        _accelerator_variant=accelerator_type,
                        _ssh_config=self._ssh_config,
                        _state=state,
                    )
                )

        return results

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpStandaloneVmHandle]:
        """List GCE instances across zones, optionally filtered by labels."""
        results: list[GcpStandaloneVmHandle] = []
        for zone in zones:
            for instance in self._gcloud_list_instances(zone, labels):
                name = instance.get("name", "")
                network_interfaces = instance.get("networkInterfaces", [])
                internal_ip = ""
                external_ip = None
                if network_interfaces:
                    internal_ip = network_interfaces[0].get("networkIP", "")
                    access_configs = network_interfaces[0].get("accessConfigs", [])
                    if access_configs:
                        external_ip = access_configs[0].get("natIP")

                ssh = GceSshConnection(
                    project_id=self._project_id,
                    zone=zone,
                    vm_name=name,
                )
                results.append(
                    GcpStandaloneVmHandle(
                        _vm_id=name,
                        _internal_address=internal_ip,
                        _external_address=external_ip,
                        _zone=zone,
                        _project_id=self._project_id,
                        _ssh=ssh,
                    )
                )

        return results

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return _gcp_tunnel(
            project=self._project_id,
            label_prefix=self._label_prefix,
            local_port=local_port or 10000,
        )

    def shutdown(self) -> None:
        pass

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Discover controller by querying GCP for labeled controller VM."""
        gcp = controller_config.gcp
        port = gcp.port or 10000
        label_key = f"{self._label_prefix}-controller"

        vms = self.list_vms(
            zones=[gcp.zone],
            labels={label_key: "true"},
        )
        if not vms:
            raise RuntimeError(f"No controller VM found (label={label_key}=true, project={self._project_id})")
        return f"{vms[0].internal_address}:{port}"

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _get_vm_ips(self, zone: str, vm_name: str) -> tuple[str, str | None]:
        """Get internal and external IPs for a GCE instance."""
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            vm_name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to describe VM {vm_name}: {result.stderr.strip()}")

        data = json.loads(result.stdout)
        network_interfaces = data.get("networkInterfaces", [])
        internal_ip = ""
        external_ip = None
        if network_interfaces:
            internal_ip = network_interfaces[0].get("networkIP", "")
            access_configs = network_interfaces[0].get("accessConfigs", [])
            if access_configs:
                external_ip = access_configs[0].get("natIP")

        if not internal_ip:
            raise RuntimeError(f"VM {vm_name} has no internal IP")

        return internal_ip, external_ip

    def _gcloud_list_tpus(self, zone: str, labels: dict[str, str] | None) -> list[dict]:
        """List TPU VMs in a zone, optionally filtered by labels."""
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--zone={zone}",
            f"--project={self._project_id}",
            "--format=json",
        ]
        if labels:
            cmd.append(f"--filter={_build_label_filter(labels)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to list TPUs in zone %s: %s", zone, result.stderr.strip())
            return []
        if not result.stdout.strip():
            return []

        tpus = json.loads(result.stdout)
        for tpu in tpus:
            tpu["name"] = _extract_node_name(tpu.get("name", ""))
        return tpus

    def _gcloud_list_instances(self, zone: str, labels: dict[str, str] | None) -> list[dict]:
        """List GCE instances in a zone, optionally filtered by labels."""
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={self._project_id}",
            f"--zones={zone}",
            "--format=json",
        ]
        if labels:
            cmd.append(f"--filter={_build_label_filter(labels)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to list instances in zone %s: %s", zone, result.stderr.strip())
            return []
        if not result.stdout.strip():
            return []

        return json.loads(result.stdout)


# ============================================================================
# Tunnel
# ============================================================================


def _discover_controller_vm_name(project: str, zone: str, label_prefix: str) -> str | None:
    """Find controller VM by name pattern."""
    name_filter = f"name~^iris-controller-{re.escape(label_prefix)}$"
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--zones={zone}",
        f"--filter={name_filter}",
        "--format=value(name)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
    return names[0] if names else None


@contextmanager
def _gcp_tunnel(
    project: str,
    label_prefix: str,
    local_port: int = 10000,
    timeout: float = 60.0,
) -> Iterator[str]:
    """SSH tunnel to the controller VM, yielding the local URL."""
    # We need to discover the controller VM and its zone. For now, list across
    # all zones by using a broad filter (gcloud instances list is project-wide
    # when --zones is omitted).
    name_filter = f"name~^iris-controller-{re.escape(label_prefix)}$ AND status=RUNNING"
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter={name_filter}",
        "--format=value(name,zone)",
        "--limit=1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"No controller VM found for prefix '{label_prefix}'")

    parts = result.stdout.strip().split()
    vm_name = parts[0]
    # Zone comes as a full path like us-central2-b
    zone = parts[1] if len(parts) > 1 else ""

    logger.info("Establishing SSH tunnel to %s (zone=%s)...", vm_name, zone)

    proc = subprocess.Popen(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "-L",
            f"{local_port}:localhost:10000",
            "-N",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        if not _wait_for_port(local_port, timeout=timeout):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

        logger.info("Tunnel ready: localhost:%d -> %s:10000", local_port, vm_name)
        yield f"http://localhost:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
