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
- GcpPlatform: Creates/lists workers and TPU slices via gcloud CLI
- GcpSliceHandle: Manages a TPU pod (list workers, terminate, status)
- GcpWorkerHandle: SSH to a TPU worker via gcloud
- GcpStandaloneWorkerHandle: SSH to a GCE instance with terminate/label/metadata support

All gcloud operations shell out to the gcloud CLI. Each run_command() call
creates a new SSH process, making worker handle implementations thread-safe for
concurrent access.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from iris.cluster.platform._worker_base import RemoteExecWorkerBase
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    PlatformError,
    QuotaExhaustedError,
    SliceStatus,
    WorkerStatus,
)
from iris.cluster.platform.bootstrap import build_worker_bootstrap_script
from iris.cluster.platform.debug import wait_for_port
from iris.cluster.platform.remote_exec import (
    GceRemoteExec,
    GcloudRemoteExec,
)
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

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


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Find a free port on the given host by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class GcpWorkerHandle(RemoteExecWorkerBase):
    """Handle to a TPU worker within a slice.

    Uses GcloudRemoteExec for SSH via `gcloud compute tpus tpu-vm ssh`.
    Thread-safe: each run_command() spawns a new SSH process.
    """

    def status(self) -> WorkerStatus:
        # TPU workers don't have independent status queries;
        # their status is derived from the slice status.
        return WorkerStatus(state=CloudWorkerState.RUNNING)


@dataclass
class GcpStandaloneWorkerHandle(RemoteExecWorkerBase):
    """Handle to a standalone GCE instance (e.g., controller VM).

    Uses GceRemoteExec for SSH via `gcloud compute ssh`.
    Supports terminate, set_labels, and set_metadata operations.
    """

    _zone: str = ""
    _project_id: str = ""

    def status(self) -> WorkerStatus:
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
            return WorkerStatus(state=CloudWorkerState.UNKNOWN)
        status_str = result.stdout.strip().upper()
        state_map = {
            "RUNNING": CloudWorkerState.RUNNING,
            "STOPPED": CloudWorkerState.STOPPED,
            "TERMINATED": CloudWorkerState.TERMINATED,
        }
        return WorkerStatus(state=state_map.get(status_str, CloudWorkerState.UNKNOWN))

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


class GcpSliceHandle:
    """Handle to a GCP TPU slice (pod).

    describe() queries TPU state and VM endpoints via `gcloud compute tpus describe`.
    The slice is the atomic unit for termination.
    """

    def __init__(
        self,
        *,
        _slice_id: str,
        _zone: str,
        _project_id: str,
        _labels: dict[str, str],
        _created_at: Timestamp,
        _label_prefix: str,
        _accelerator_variant: str,
        _ssh_config: config_pb2.SshConfig | None = None,
        _state: str = "READY",
    ):
        self._slice_id = _slice_id
        self._zone = _zone
        self._project_id = _project_id
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._accelerator_variant = _accelerator_variant
        self._ssh_config = _ssh_config
        self._state = _state
        # Bootstrap state: None means no bootstrap requested or not yet started.
        # Set by the platform's internal bootstrap thread.
        self._bootstrap_state: CloudSliceState | None = None
        self._bootstrap_lock = threading.Lock()

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

    def describe(self) -> SliceStatus:
        """Query TPU state and VM endpoints, compositing bootstrap state.

        A single `gcloud compute tpus tpu-vm describe` call populates both
        the slice state and the VM handles. When bootstrap is in progress,
        the reported state reflects the bootstrap lifecycle rather than the
        raw cloud state.
        """
        cloud_status = self._describe_cloud()
        cloud_state = cloud_status.state

        # Composite state: bootstrap state overrides cloud READY
        with self._bootstrap_lock:
            bs = self._bootstrap_state

        if cloud_state == CloudSliceState.CREATING:
            effective_state = CloudSliceState.CREATING
        elif cloud_state == CloudSliceState.READY and bs is None:
            # Cloud is ready but bootstrap hasn't completed yet — still bootstrapping.
            # This handles the case where bootstrap_config was provided.
            effective_state = CloudSliceState.BOOTSTRAPPING
        elif bs == CloudSliceState.READY:
            effective_state = CloudSliceState.READY
        elif bs == CloudSliceState.FAILED:
            effective_state = CloudSliceState.FAILED
        else:
            effective_state = cloud_state

        return SliceStatus(
            state=effective_state,
            worker_count=cloud_status.worker_count,
            workers=cloud_status.workers,
        )

    def _describe_cloud(self) -> SliceStatus:
        """Query raw TPU state and VM endpoints from GCP."""
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
            return SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0)

        tpu_data = json.loads(result.stdout)
        state = _TPU_STATE_MAP.get(tpu_data.get("state", "UNKNOWN"), CloudSliceState.UNKNOWN)
        endpoints = tpu_data.get("networkEndpoints", [])

        try:
            worker_count = get_tpu_topology(self._accelerator_variant).vm_count
        except ValueError as e:
            raise PlatformError(
                f"Unknown TPU topology '{self._accelerator_variant}' for slice {self._slice_id}. "
                f"Cannot determine worker count without a known topology."
            ) from e

        workers: list[GcpWorkerHandle] = []
        for i in range(worker_count):
            ep = endpoints[i] if i < len(endpoints) else {}
            internal_ip = ep.get("ipAddress", "")
            external_ip = ep.get("accessConfig", {}).get("externalIp") if "accessConfig" in ep else None

            if not internal_ip and i < len(endpoints):
                logger.warning(
                    "TPU %s endpoint %d has no IP address; worker may still be provisioning",
                    self._slice_id,
                    i,
                )

            remote_exec = GcloudRemoteExec(
                project_id=self._project_id,
                _zone=self._zone,
                vm_id=self._slice_id,
                worker_index=i,
                _address=internal_ip,
            )
            workers.append(
                GcpWorkerHandle(
                    _vm_id=f"{self._slice_id}-worker-{i}",
                    _internal_address=internal_ip,
                    _external_address=external_ip,
                    _remote_exec=remote_exec,
                )
            )

        return SliceStatus(state=state, worker_count=worker_count, workers=workers)

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
            error = result.stderr.strip()
            if "not found" not in error.lower():
                raise RuntimeError(f"Failed to delete TPU {self._slice_id}: {error}")


# ============================================================================
# GcpPlatform
# ============================================================================

DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50


class GcpPlatform:
    """Platform implementation for Google Cloud Platform.

    Manages GCE instances (standalone VMs) and TPU slices via gcloud CLI.
    Zones are stored from GcpPlatformConfig for list_all_slices(); per-slice
    zones come from SliceConfig.
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
        self._zones = list(gcp_config.zones)

    def create_vm(self, config: config_pb2.VmConfig) -> GcpStandaloneWorkerHandle:
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

        remote_exec = GceRemoteExec(
            project_id=self._project_id,
            zone=zone,
            vm_name=config.name,
        )

        return GcpStandaloneWorkerHandle(
            _vm_id=config.name,
            _internal_address=internal_ip,
            _external_address=external_ip,
            _zone=zone,
            _project_id=self._project_id,
            _remote_exec=remote_exec,
        )

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        cluster_config: config_pb2.IrisClusterConfig | None = None,
    ) -> GcpSliceHandle:
        """Create a TPU slice via gcloud.

        When cluster_config is provided, spawns a background thread that waits
        for the slice to reach cloud READY, then runs the bootstrap script on
        each worker. The handle's describe() composites bootstrap state with
        cloud state.
        """
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

        handle = GcpSliceHandle(
            _slice_id=slice_id,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _accelerator_variant=config.accelerator_variant,
            _ssh_config=self._ssh_config,
        )

        if cluster_config:

            def _bootstrap_worker():
                try:
                    self._run_bootstrap(handle, cluster_config)
                except Exception as e:
                    logger.error("Bootstrap failed for slice %s: %s", handle.slice_id, e)
                    with handle._bootstrap_lock:
                        handle._bootstrap_state = CloudSliceState.FAILED

            threading.Thread(
                target=_bootstrap_worker,
                name=f"bootstrap-{handle.slice_id}",
                daemon=True,
            ).start()
        else:
            # No bootstrap requested — mark as immediately ready so describe()
            # doesn't report BOOTSTRAPPING.
            with handle._bootstrap_lock:
                handle._bootstrap_state = CloudSliceState.READY

        return handle

    def _run_bootstrap(
        self,
        handle: GcpSliceHandle,
        cluster_config: config_pb2.IrisClusterConfig,
        poll_interval: float = 10.0,
        cloud_ready_timeout: float = 600.0,
    ) -> None:
        """Wait for slice to reach cloud READY, then bootstrap each worker.

        Polls handle._describe_cloud() until the TPU is READY, then runs the
        bootstrap script on each worker via run_command(). On success, sets
        handle._bootstrap_state to READY. On failure, sets FAILED (the caller
        catches exceptions and sets FAILED as well).
        """
        # Phase 1: Wait for cloud to report READY
        deadline = time.monotonic() + cloud_ready_timeout
        while True:
            cloud_status = handle._describe_cloud()
            if cloud_status.state == CloudSliceState.READY:
                break
            if cloud_status.state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
                raise PlatformError(
                    f"Slice {handle.slice_id} entered {cloud_status.state} while waiting for cloud READY"
                )
            if time.monotonic() > deadline:
                raise PlatformError(f"Slice {handle.slice_id} did not reach cloud READY within {cloud_ready_timeout}s")
            time.sleep(poll_interval)

        # Phase 2: Bootstrap each worker
        cloud_status = handle._describe_cloud()
        for worker in cloud_status.workers:
            if not worker.internal_address:
                raise PlatformError(f"Worker {worker.worker_id} in slice {handle.slice_id} has no internal address")

            script = build_worker_bootstrap_script(cluster_config, worker.internal_address)
            result = worker.run_command(script, timeout=Duration.from_seconds(600))
            if result.returncode != 0:
                raise PlatformError(
                    f"Bootstrap failed for worker {worker.worker_id} in slice {handle.slice_id}: "
                    f"exit code {result.returncode}\n{result.stderr}"
                )

        logger.info("Bootstrap completed for slice %s (%d workers)", handle.slice_id, len(cloud_status.workers))
        with handle._bootstrap_lock:
            handle._bootstrap_state = CloudSliceState.READY

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

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[GcpSliceHandle]:
        if not self._zones:
            raise ValueError(
                "GcpPlatform.list_all_slices() called but no zones configured. "
                "Set platform.gcp.zones in your cluster config."
            )
        return self.list_slices(zones=self._zones, labels=labels)

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpStandaloneWorkerHandle]:
        """List GCE instances across zones, optionally filtered by labels."""
        results: list[GcpStandaloneWorkerHandle] = []
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

                remote_exec = GceRemoteExec(
                    project_id=self._project_id,
                    zone=zone,
                    vm_name=name,
                )
                results.append(
                    GcpStandaloneWorkerHandle(
                        _vm_id=name,
                        _internal_address=internal_ip,
                        _external_address=external_ip,
                        _zone=zone,
                        _project_id=self._project_id,
                        _remote_exec=remote_exec,
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
            local_port=local_port,
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


@contextmanager
def _gcp_tunnel(
    project: str,
    label_prefix: str,
    local_port: int | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """SSH tunnel to the controller VM, yielding the local URL.

    Binds explicitly to 127.0.0.1 to avoid conflicts with other processes
    that may be listening on the same port on a different address family (IPv6).
    Picks a free port automatically if none is specified.
    """
    if local_port is None:
        local_port = _find_free_port()

    label_filter = f"labels.{label_prefix}-controller=true AND status=RUNNING"
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter={label_filter}",
        "--format=value(name,zone)",
        "--limit=1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"No controller VM found (label={label_prefix}-controller=true, project={project})")

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
            f"127.0.0.1:{local_port}:localhost:10000",
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
        if not wait_for_port(local_port, host="127.0.0.1", timeout=timeout):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

        logger.info("Tunnel ready: 127.0.0.1:%d -> %s:10000", local_port, vm_name)
        yield f"http://127.0.0.1:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
