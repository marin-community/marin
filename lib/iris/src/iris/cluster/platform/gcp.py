# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
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

import logging
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import cast

from iris.cluster.controller.vm_lifecycle import restart_controller as vm_restart_controller
from iris.cluster.controller.vm_lifecycle import start_controller as vm_start_controller
from iris.cluster.controller.vm_lifecycle import stop_controller as vm_stop_controller
from iris.cluster.platform._worker_base import RemoteExecWorkerBase
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    Labels,
    PlatformError,
    SliceHandle,
    SliceStatus,
    WorkerStatus,
    default_stop_all,
    find_free_port,
    generate_slice_suffix,
)
from iris.cluster.platform.bootstrap import (
    build_worker_bootstrap_script,
    rewrite_ghcr_to_ar_remote,
    zone_to_multi_region,
)
from iris.cluster.platform.debug import wait_for_port
from iris.cluster.platform.gcp_service import (
    GcpService,
    TpuCreateRequest,
    VmCreateRequest,
)
from iris.cluster.platform.gcp_service_impl import GcpServiceImpl
from iris.cluster.platform.remote_exec import (
    GceRemoteExec,
    GcloudRemoteExec,
)
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import get_tpu_topology
from iris.cluster.worker.env_probe import construct_worker_id
from iris.rpc import config_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

# GCP TPU state mapping
_TPU_STATE_MAP: dict[str, CloudSliceState] = {
    "CREATING": CloudSliceState.CREATING,
    "READY": CloudSliceState.READY,
    "REPAIRING": CloudSliceState.REPAIRING,
    "DELETING": CloudSliceState.DELETING,
}

_VM_STATE_MAP: dict[str, CloudSliceState] = {
    "PROVISIONING": CloudSliceState.CREATING,
    "STAGING": CloudSliceState.CREATING,
    "RUNNING": CloudSliceState.READY,
    "STOPPING": CloudSliceState.DELETING,
    "TERMINATED": CloudSliceState.DELETING,
}

_ACTIVE_VM_SLICE_STATES = frozenset({"PROVISIONING", "STAGING", "RUNNING"})
_GCE_NAME_MAX_LEN = 63
_GCE_NAME_RE = re.compile(r"[^a-z0-9-]+")
_GCE_NAME_EDGE_RE = re.compile(r"^-+|-+$")
_GCE_VM_SLICE_SSH_USER = "iris"


def _build_vm_slice_id(name_prefix: str, suffix: str) -> str:
    """Build a bounded VM slice id valid for both GCE instance names and labels."""
    max_prefix_len = _GCE_NAME_MAX_LEN - len(suffix) - 1
    if max_prefix_len <= 0:
        raise ValueError("Timestamp suffix leaves no room for VM slice id prefix")

    normalized = _GCE_NAME_RE.sub("-", name_prefix.lower())
    normalized = re.sub(r"-+", "-", normalized)
    normalized = _GCE_NAME_EDGE_RE.sub("", normalized)
    if not normalized:
        normalized = "slice"
    if not normalized[0].isalpha():
        normalized = f"slice-{normalized}"

    trimmed = normalized[:max_prefix_len]
    trimmed = _GCE_NAME_EDGE_RE.sub("", trimmed)
    if not trimmed:
        trimmed = "slice"

    return f"{trimmed}-{suffix}"


def _composite_slice_state(
    cloud_state: CloudSliceState,
    bootstrap_state: CloudSliceState | None,
) -> CloudSliceState:
    """Compose cloud lifecycle with bootstrap lifecycle into effective slice state."""
    if cloud_state != CloudSliceState.READY:
        # Never mask non-READY cloud states (DELETING/REPAIRING/UNKNOWN/etc).
        return cloud_state
    if bootstrap_state is None:
        return CloudSliceState.BOOTSTRAPPING
    if bootstrap_state == CloudSliceState.FAILED:
        return CloudSliceState.FAILED
    return CloudSliceState.READY


def _validate_slice_config(config: config_pb2.SliceConfig) -> None:
    """Validate required fields on a SliceConfig before creating a GCP slice.

    Raises ValueError listing all missing fields so operators can fix config
    in one pass rather than discovering issues one-by-one.
    """
    missing: list[str] = []
    violations: list[str] = []
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if config.gcp.mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM:
        if not config.gcp.machine_type:
            missing.append("gcp.machine_type")
        if config.num_vms != 1:
            violations.append("GCP VM slice mode requires num_vms=1")
        if config.preemptible:
            violations.append("GCP VM slice mode does not support preemptible instances")
        if config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
            violations.append("GCP VM slice mode requires accelerator_type=cpu")
        if config.accelerator_variant:
            violations.append("GCP VM slice mode does not support accelerator_variant")
    else:
        if not config.accelerator_variant:
            missing.append("accelerator_variant")
        if not config.gcp.runtime_version:
            missing.append("gcp.runtime_version")
    errors: list[str] = []
    if missing:
        errors.append(f"SliceConfig is missing required fields: {', '.join(missing)}")
    errors.extend(violations)
    if errors:
        raise ValueError("; ".join(errors))


def _validate_vm_config(config: config_pb2.VmConfig) -> None:
    """Validate required fields on a VmConfig before creating a GCE instance."""
    missing: list[str] = []
    if not config.name:
        missing.append("name")
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if missing:
        raise ValueError(f"VmConfig is missing required fields: {', '.join(missing)}")


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
    Supports terminate, set_labels, and set_metadata operations via GcpService.

    _vm_id is the Iris worker ID (may differ from GCE instance name after
    construct_worker_id). gcloud commands use _gce_vm_name which is the real
    GCE instance name from the underlying GceRemoteExec.
    """

    _zone: str = ""
    _project_id: str = ""
    # Always populated at construction; Optional only for dataclass inheritance ordering.
    _gcp_service: GcpService | None = None

    def __post_init__(self) -> None:
        if self._gcp_service is None:
            raise ValueError("_gcp_service is required")

    @property
    def _gce_vm_name(self) -> str:
        """Real GCE instance name for gcloud commands."""
        return cast(GceRemoteExec, self._remote_exec).vm_name

    def status(self) -> WorkerStatus:
        assert self._gcp_service is not None
        info = self._gcp_service.vm_describe(self._gce_vm_name, self._zone)
        if info is None:
            return WorkerStatus(state=CloudWorkerState.UNKNOWN)
        state_map = {
            "RUNNING": CloudWorkerState.RUNNING,
            "STOPPED": CloudWorkerState.STOPPED,
            "TERMINATED": CloudWorkerState.TERMINATED,
        }
        return WorkerStatus(state=state_map.get(info.status, CloudWorkerState.UNKNOWN))

    def reboot(self) -> None:
        assert self._gcp_service is not None
        logger.info("Rebooting GCE instance: %s", self._gce_vm_name)
        self._gcp_service.vm_reset(self._gce_vm_name, self._zone)

    def terminate(self) -> None:
        assert self._gcp_service is not None
        logger.info("Deleting GCE instance: %s", self._gce_vm_name)
        self._gcp_service.vm_delete(self._gce_vm_name, self._zone)

    def set_labels(self, labels: dict[str, str]) -> None:
        assert self._gcp_service is not None
        logger.info("Setting labels on GCE instance: %s", self._gce_vm_name)
        try:
            self._gcp_service.vm_update_labels(self._gce_vm_name, self._zone, labels)
        except PlatformError:
            logger.warning("Failed to set labels on %s", self._gce_vm_name)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        assert self._gcp_service is not None
        logger.info("Setting metadata on GCE instance: %s", self._gce_vm_name)
        try:
            self._gcp_service.vm_set_metadata(self._gce_vm_name, self._zone, metadata)
        except PlatformError:
            logger.warning("Failed to set metadata on %s", self._gce_vm_name)


class GcpSliceHandle:
    """Handle to a GCP TPU slice (pod).

    describe() queries TPU state and VM endpoints via GcpService.
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
        _gcp_service: GcpService,
        _ssh_config: config_pb2.SshConfig | None = None,
        _state: str = "READY",
        _bootstrapping: bool = False,
    ):
        self._slice_id = _slice_id
        self._zone = _zone
        self._project_id = _project_id
        self._gcp_service = _gcp_service
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._iris_labels = Labels(_label_prefix)
        self._accelerator_variant = _accelerator_variant
        self._ssh_config = _ssh_config
        self._state = _state
        self._bootstrap_state: CloudSliceState | None = None if _bootstrapping else CloudSliceState.READY
        self._bootstrap_lock = threading.Lock()

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return self._zone

    @property
    def scale_group(self) -> str:
        return self._labels.get(self._iris_labels.iris_scale_group, "")

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

        effective_state = _composite_slice_state(cloud_state, bs)

        return SliceStatus(
            state=effective_state,
            worker_count=cloud_status.worker_count,
            workers=cloud_status.workers,
        )

    def _describe_cloud(self) -> SliceStatus:
        """Query raw TPU state and VM endpoints via GcpService."""
        tpu_info = self._gcp_service.tpu_describe(self._slice_id, self._zone)
        if tpu_info is None:
            logger.warning("Failed to describe TPU %s", self._slice_id)
            return SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0)

        state = _TPU_STATE_MAP.get(tpu_info.state, CloudSliceState.UNKNOWN)

        try:
            worker_count = get_tpu_topology(self._accelerator_variant).vm_count
        except ValueError as e:
            raise PlatformError(
                f"Unknown TPU topology '{self._accelerator_variant}' for slice {self._slice_id}. "
                f"Cannot determine worker count without a known topology."
            ) from e

        workers: list[GcpWorkerHandle] = []
        for i in range(worker_count):
            internal_ip = tpu_info.network_endpoints[i] if i < len(tpu_info.network_endpoints) else ""

            if not internal_ip and i < len(tpu_info.network_endpoints):
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
                    _external_address=None,
                    _remote_exec=remote_exec,
                )
            )

        return SliceStatus(state=state, worker_count=worker_count, workers=workers)

    def terminate(self) -> None:
        logger.info("Terminating TPU (async): %s", self._slice_id)
        self._gcp_service.tpu_delete(self._slice_id, self._zone)


class GcpVmSliceHandle:
    """Handle to a single-VM GCE-backed slice."""

    def __init__(
        self,
        *,
        _slice_id: str,
        _vm_name: str,
        _zone: str,
        _project_id: str,
        _labels: dict[str, str],
        _created_at: Timestamp,
        _label_prefix: str,
        _gcp_service: GcpService,
        _ssh_config: config_pb2.SshConfig | None = None,
        _bootstrapping: bool = False,
    ):
        self._slice_id = _slice_id
        self._vm_name = _vm_name
        self._zone = _zone
        self._project_id = _project_id
        self._gcp_service = _gcp_service
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._iris_labels = Labels(_label_prefix)
        self._ssh_config = _ssh_config
        self._bootstrap_state: CloudSliceState | None = None if _bootstrapping else CloudSliceState.READY
        self._bootstrap_lock = threading.Lock()

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return self._zone

    @property
    def scale_group(self) -> str:
        return self._labels.get(self._iris_labels.iris_scale_group, "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        cloud_status = self._describe_cloud()
        cloud_state = cloud_status.state

        with self._bootstrap_lock:
            bs = self._bootstrap_state

        effective_state = _composite_slice_state(cloud_state, bs)

        return SliceStatus(
            state=effective_state,
            worker_count=cloud_status.worker_count,
            workers=cloud_status.workers,
        )

    def _describe_cloud(self) -> SliceStatus:
        vm_info = self._gcp_service.vm_describe(self._vm_name, self._zone)
        if vm_info is None:
            logger.warning(
                "Failed to describe VM slice %s (%s)",
                self._slice_id,
                self._vm_name,
            )
            return SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0)

        state = _VM_STATE_MAP.get(vm_info.status, CloudSliceState.UNKNOWN)

        remote_exec = GceRemoteExec(
            project_id=self._project_id,
            zone=self._zone,
            vm_name=self._vm_name,
            ssh_user=_GCE_VM_SLICE_SSH_USER,
        )
        worker = GcpStandaloneWorkerHandle(
            _vm_id=f"{self._slice_id}-worker-0",
            _internal_address=vm_info.internal_ip,
            _external_address=vm_info.external_ip,
            _zone=self._zone,
            _project_id=self._project_id,
            _gcp_service=self._gcp_service,
            _remote_exec=remote_exec,
        )
        return SliceStatus(state=state, worker_count=1, workers=[worker])

    def terminate(self) -> None:
        logger.info("Terminating VM slice: %s (vm=%s)", self._slice_id, self._vm_name)
        self._gcp_service.vm_delete(self._vm_name, self._zone)


# ============================================================================
# GcpPlatform
# ============================================================================

DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50
# pd-ssd provides ~6000 IOPS vs ~38 on pd-standard, critical for controller DB
# and generally better for all GCE VMs. We hardcode this rather than exposing
# it in config since it's GCP-specific and pd-ssd is the right choice.
DEFAULT_BOOT_DISK_TYPE = "pd-ssd"


class GcpPlatform:
    """Platform implementation for Google Cloud Platform.

    Manages GCE instances (standalone VMs) and TPU slices via GcpService.
    Zones are stored from GcpPlatformConfig for list_all_slices(); per-slice
    zones come from SliceConfig.
    """

    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
        gcp_service: GcpService | None = None,
    ):
        self._project_id = gcp_config.project_id
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix)
        self._ssh_config = ssh_config
        self._zones = list(gcp_config.zones)
        self._gcp: GcpService = gcp_service or GcpServiceImpl(mode=ServiceMode.CLOUD, project_id=self._project_id)

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        """Rewrite ``ghcr.io/`` images to the AR remote repo for *zone*'s continent.

        Non-GHCR images pass through unchanged.
        """
        if not image.startswith("ghcr.io/"):
            return image
        if not zone:
            raise ValueError("zone is required for GHCR→AR image rewriting on GCP")
        multi_region = zone_to_multi_region(zone)
        if not multi_region:
            return image
        return rewrite_ghcr_to_ar_remote(image, multi_region, self._project_id)

    def _best_effort_delete_tpu(self, slice_id: str, zone: str) -> None:
        """Try to delete a TPU VM that may have been partially created.

        Silently ignores errors (resource may never have been created).
        """
        logger.info("Best-effort async cleanup of TPU %s in %s", slice_id, zone)
        try:
            self._gcp.tpu_delete(slice_id, zone)
        except PlatformError as e:
            logger.warning("Cleanup of TPU %s failed: %s", slice_id, e)

    def _best_effort_delete_vm(self, vm_name: str, zone: str) -> None:
        """Try to delete a GCE VM that may have been partially created.

        Silently ignores errors (resource may never have been created).
        """
        logger.info("Best-effort cleanup of VM %s in %s", vm_name, zone)
        try:
            self._gcp.vm_delete(vm_name, zone)
        except PlatformError as e:
            logger.warning("Cleanup of VM %s failed: %s", vm_name, e)

    def create_vm(self, config: config_pb2.VmConfig) -> GcpStandaloneWorkerHandle:
        """Create a GCE instance. Returns a handle with SSH and label/metadata support."""
        _validate_vm_config(config)
        gcp = config.gcp
        zone = gcp.zone
        machine_type = gcp.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = gcp.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB

        request = VmCreateRequest(
            name=config.name,
            zone=zone,
            machine_type=machine_type,
            labels=dict(config.labels),
            metadata=dict(config.metadata),
            disk_size_gb=boot_disk_size,
            boot_disk_type=DEFAULT_BOOT_DISK_TYPE,
            image_family="debian-12",
            image_project="debian-cloud",
        )

        logger.info("Creating GCE instance: %s (zone=%s, type=%s)", config.name, zone, machine_type)
        try:
            vm_info = self._gcp.vm_create(request)
        except PlatformError:
            self._best_effort_delete_vm(config.name, zone)
            raise

        remote_exec = GceRemoteExec(
            project_id=self._project_id,
            zone=zone,
            vm_name=config.name,
        )

        return GcpStandaloneWorkerHandle(
            _vm_id=construct_worker_id(config.name, 0),
            _internal_address=vm_info.internal_ip,
            _external_address=vm_info.external_ip,
            _zone=zone,
            _project_id=self._project_id,
            _gcp_service=self._gcp,
            _remote_exec=remote_exec,
        )

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> SliceHandle:
        """Create a GCP-backed slice (TPU pod or single VM).

        In LOCAL mode, delegates to GcpServiceImpl to spawn local worker threads
        and returns a LocalSliceHandle.
        """
        if self._gcp.mode == ServiceMode.LOCAL:
            return self._create_local_slice(config, worker_config)
        _validate_slice_config(config)
        if config.gcp.mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM:
            return self._create_vm_slice(config, worker_config)
        return self._create_tpu_slice(config, worker_config)

    def _create_local_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> SliceHandle:
        """Create a local slice via GcpServiceImpl(LOCAL)."""
        slice_id = f"{config.name_prefix}-{generate_slice_suffix()}"
        return self._gcp.create_local_slice(slice_id, config, worker_config)

    def _create_tpu_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> GcpSliceHandle:
        """Create a TPU slice via GcpService.

        When worker_config is provided the bootstrap script is passed as TPU
        metadata (startup-script) so each worker VM self-bootstraps on first
        boot, matching the pattern used for GCE VM slices. Bootstrap progress
        is monitored via health endpoint polling rather than SSH.
        """
        gcp = config.gcp
        slice_id = f"{config.name_prefix}-{generate_slice_suffix()}"

        # Pre-render bootstrap script for metadata embedding.
        metadata: dict[str, str] = {}
        if worker_config:
            worker_config.docker_image = self.resolve_image(worker_config.docker_image, zone=gcp.zone)
            worker_config.slice_id = slice_id
            startup_script = build_worker_bootstrap_script(worker_config)
            metadata["startup-script"] = startup_script

        request = TpuCreateRequest(
            name=slice_id,
            zone=gcp.zone,
            accelerator_type=config.accelerator_variant,
            runtime_version=gcp.runtime_version,
            labels=dict(config.labels),
            metadata=metadata,
            preemptible=config.preemptible,
        )

        logger.info("Creating TPU slice: %s (type=%s, zone=%s)", slice_id, config.accelerator_variant, gcp.zone)
        try:
            self._gcp.tpu_create(request)
        except PlatformError:
            self._best_effort_delete_tpu(slice_id, gcp.zone)
            raise

        handle = GcpSliceHandle(
            _slice_id=slice_id,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _accelerator_variant=config.accelerator_variant,
            _gcp_service=self._gcp,
            _ssh_config=self._ssh_config,
            _bootstrapping=worker_config is not None,
        )

        if worker_config:

            def _bootstrap_worker():
                try:
                    self._run_tpu_bootstrap(handle, worker_config)
                except Exception as e:
                    logger.error("Bootstrap failed for slice %s: %s", handle.slice_id, e)
                    with handle._bootstrap_lock:
                        handle._bootstrap_state = CloudSliceState.FAILED

            threading.Thread(
                target=_bootstrap_worker,
                name=f"bootstrap-{handle.slice_id}",
                daemon=True,
            ).start()

        return handle

    def _create_vm_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> GcpVmSliceHandle:
        """Create a single GCE VM that behaves as a one-worker slice.

        When worker_config is provided the bootstrap script is passed as GCE
        startup-script metadata so the VM self-bootstraps on first boot (and on
        every subsequent reboot). This eliminates the need to SSH into the VM
        for initial setup.
        """
        gcp = config.gcp
        slice_id = _build_vm_slice_id(config.name_prefix, generate_slice_suffix())
        vm_name = slice_id
        machine_type = gcp.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = config.disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB

        labels = dict(config.labels)
        labels[self._iris_labels.iris_slice_id] = slice_id

        # Pre-render the bootstrap script so we can bake it into VM metadata.
        startup_script: str | None = None
        if worker_config:
            worker_config.docker_image = self.resolve_image(worker_config.docker_image, zone=gcp.zone)
            worker_config.worker_id = construct_worker_id(slice_id, 0)
            startup_script = build_worker_bootstrap_script(worker_config)

        request = VmCreateRequest(
            name=vm_name,
            zone=gcp.zone,
            machine_type=machine_type,
            labels=labels,
            disk_size_gb=boot_disk_size,
            image_family="debian-12",
            image_project="debian-cloud",
            startup_script=startup_script,
        )

        logger.info("Creating VM slice: %s (vm=%s, zone=%s, type=%s)", slice_id, vm_name, gcp.zone, machine_type)
        try:
            self._gcp.vm_create(request)
        except PlatformError:
            self._best_effort_delete_vm(vm_name, gcp.zone)
            raise

        handle = GcpVmSliceHandle(
            _slice_id=slice_id,
            _vm_name=vm_name,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _gcp_service=self._gcp,
            _labels=labels,
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _ssh_config=self._ssh_config,
            _bootstrapping=worker_config is not None,
        )

        if worker_config:

            def _bootstrap_worker():
                try:
                    self._run_vm_slice_bootstrap(handle, worker_config)
                except Exception as e:
                    logger.error("Bootstrap failed for VM slice %s: %s", handle.slice_id, e)
                    with handle._bootstrap_lock:
                        handle._bootstrap_state = CloudSliceState.FAILED

            threading.Thread(
                target=_bootstrap_worker,
                name=f"bootstrap-{handle.slice_id}",
                daemon=True,
            ).start()

        return handle

    def _run_tpu_bootstrap(
        self,
        handle: GcpSliceHandle,
        worker_config: config_pb2.WorkerConfig,
        poll_interval: float = 10.0,
        cloud_ready_timeout: float = 600.0,
        bootstrap_timeout: float = 600.0,
    ) -> None:
        """Monitor TPU startup-script bootstrap via health endpoint polling.

        The bootstrap script was baked into TPU metadata at creation time.
        Phase 1: Wait for cloud READY with all worker IPs.
        Phase 2: Poll worker health endpoints until all respond healthy.
        On timeout: query Cloud Logging for [iris-init] entries for diagnostics.
        """
        # Phase 1: Wait for cloud READY with all worker IPs populated.
        deadline = Deadline.from_now(Duration.from_seconds(cloud_ready_timeout))
        while not deadline.expired():
            cloud_status = handle._describe_cloud()
            if cloud_status.state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
                raise PlatformError(
                    f"Slice {handle.slice_id} entered {cloud_status.state} while waiting for cloud READY"
                )
            if cloud_status.state == CloudSliceState.READY:
                all_have_ips = all(w.internal_address for w in cloud_status.workers)
                if all_have_ips and len(cloud_status.workers) == cloud_status.worker_count:
                    break
                logger.info(
                    "Slice %s is READY but only %d/%d workers have IPs, waiting...",
                    handle.slice_id,
                    sum(1 for w in cloud_status.workers if w.internal_address),
                    cloud_status.worker_count,
                )
            time.sleep(poll_interval)
        else:
            raise PlatformError(f"Slice {handle.slice_id} did not reach cloud READY within {cloud_ready_timeout}s")

        # Phase 2: Poll health endpoints for all workers.
        workers = cloud_status.workers
        worker_addrs = [(w.worker_id, w.internal_address) for w in workers]
        healthy_workers: set[str] = set()
        health_deadline = Deadline.from_now(Duration.from_seconds(bootstrap_timeout))

        logger.info(
            "Polling health endpoints for %d workers in slice %s",
            len(worker_addrs),
            handle.slice_id,
        )

        while not health_deadline.expired():
            for worker_id, addr in worker_addrs:
                if worker_id in healthy_workers:
                    continue
                try:
                    resp = urllib.request.urlopen(
                        f"http://{addr}:{worker_config.port}/health",
                        timeout=5,
                    )
                    if resp.status == 200:
                        healthy_workers.add(worker_id)
                        logger.info("Worker %s is healthy", worker_id)
                except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
                    pass  # not ready yet

            if len(healthy_workers) == len(worker_addrs):
                break
            time.sleep(poll_interval)
        else:
            self._fetch_bootstrap_logs(handle)
            raise PlatformError(
                f"TPU slice {handle.slice_id} bootstrap timed out: "
                f"{len(healthy_workers)}/{len(worker_addrs)} workers healthy"
            )

        logger.info("Bootstrap completed for TPU slice %s (%d workers)", handle.slice_id, len(workers))
        with handle._bootstrap_lock:
            handle._bootstrap_state = CloudSliceState.READY

    def _fetch_bootstrap_logs(self, handle: GcpSliceHandle) -> None:
        """Fetch [iris-init] log entries from Cloud Logging for diagnostics.

        Called only on bootstrap failure/timeout. Queries the last 30 minutes
        of logs for the TPU's VMs.
        """
        log_filter = (
            f'resource.type="gce_instance" '
            f'textPayload:"[iris-init]" '
            f'labels."compute.googleapis.com/resource_name":"{handle._slice_id}"'
        )
        cmd = [
            "gcloud",
            "logging",
            "read",
            log_filter,
            f"--project={self._project_id}",
            "--freshness=30m",
            "--limit=200",
            "--format=value(textPayload)",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Cloud Logging query timed out for %s", handle.slice_id)
            return

        if result.returncode == 0 and result.stdout.strip():
            logger.error("Bootstrap logs for %s:\n%s", handle.slice_id, result.stdout)
        else:
            logger.warning(
                "Could not fetch Cloud Logging for %s (rc=%d): %s",
                handle.slice_id,
                result.returncode,
                result.stderr.strip(),
            )

    def _run_vm_slice_bootstrap(
        self,
        handle: GcpVmSliceHandle,
        worker_config: config_pb2.WorkerConfig,
        poll_interval: float = 5.0,
        cloud_ready_timeout: float = 600.0,
    ) -> None:
        """Monitor GCE startup-script bootstrap via serial port output.

        The bootstrap script was baked into VM metadata at creation time, so the
        VM self-bootstraps on first boot.  This method polls
        ``gcloud compute instances get-serial-port-output`` for ``[iris-init]``
        log lines until the script emits ``Bootstrap complete`` or the timeout
        expires.  No SSH is required.
        """
        deadline = Deadline.from_now(Duration.from_seconds(cloud_ready_timeout))
        poll_duration = Duration.from_seconds(poll_interval)

        # Phase 1: wait for VM to reach RUNNING with an IP.
        while not deadline.expired():
            cloud_status = handle._describe_cloud()
            if cloud_status.state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
                raise PlatformError(
                    f"VM slice {handle.slice_id} entered {cloud_status.state} while waiting for cloud READY"
                )
            if cloud_status.state == CloudSliceState.READY and cloud_status.workers:
                if cloud_status.workers[0].internal_address:
                    break
            time.sleep(poll_duration.to_seconds())
        else:
            raise PlatformError(f"VM slice {handle.slice_id} did not reach cloud READY within {cloud_ready_timeout}s")

        # Phase 2: tail serial port output for [iris-init] progress lines.
        # GCE serial port output is append-only; we track the byte offset so
        # each poll returns only new output.
        serial_offset = 0
        bootstrap_complete = False
        bootstrap_failed = False

        while not deadline.expired():
            output = self._gcp.vm_get_serial_port_output(handle._vm_name, handle._zone, start=serial_offset)
            if output:
                for line in output.splitlines():
                    if "[iris-init]" in line:
                        logger.info("[%s serial] %s", handle.slice_id, line.strip())
                    if "Bootstrap complete" in line:
                        bootstrap_complete = True
                    if "[iris-init] ERROR" in line:
                        bootstrap_failed = True

                # Advance offset past what we already read.
                serial_offset += len(output)

            if bootstrap_complete:
                break
            if bootstrap_failed:
                raise PlatformError(
                    f"Startup-script bootstrap failed for VM slice {handle.slice_id} (see serial output above)"
                )

            time.sleep(poll_duration.to_seconds())
        else:
            raise PlatformError(
                f"VM slice {handle.slice_id} startup-script did not complete within {cloud_ready_timeout}s"
            )

        logger.info("Bootstrap completed for VM slice %s (via startup-script)", handle.slice_id)
        with handle._bootstrap_lock:
            handle._bootstrap_state = CloudSliceState.READY

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpSliceHandle | GcpVmSliceHandle]:
        """List TPU and VM slices across zones, optionally filtered by labels."""
        if self._gcp.mode == ServiceMode.LOCAL:
            return self._gcp.get_local_slices(labels)  # type: ignore[return-value]

        handles: list[GcpSliceHandle | GcpVmSliceHandle] = []

        tpu_infos = self._gcp.tpu_list(zones, labels)
        for tpu in tpu_infos:
            if tpu.state not in ("READY", "CREATING"):
                logger.info("Skipping TPU %s in state %s", tpu.name, tpu.state)
                continue
            handles.append(
                GcpSliceHandle(
                    _slice_id=tpu.name,
                    _zone=tpu.zone,
                    _project_id=self._project_id,
                    _labels=tpu.labels,
                    _created_at=tpu.created_at,
                    _label_prefix=self._label_prefix,
                    _accelerator_variant=tpu.accelerator_type,
                    _gcp_service=self._gcp,
                    _ssh_config=self._ssh_config,
                    _state=tpu.state,
                )
            )

        vm_infos = self._gcp.vm_list(zones, labels)
        for vm in vm_infos:
            if vm.status not in _ACTIVE_VM_SLICE_STATES:
                logger.info("Skipping VM instance %s in state %s", vm.name, vm.status)
                continue
            slice_id = vm.labels.get(self._iris_labels.iris_slice_id, "")
            if not slice_id:
                continue
            handles.append(
                GcpVmSliceHandle(
                    _slice_id=slice_id,
                    _vm_name=vm.name,
                    _zone=vm.zone,
                    _project_id=self._project_id,
                    _gcp_service=self._gcp,
                    _labels=vm.labels,
                    _created_at=vm.created_at,
                    _label_prefix=self._label_prefix,
                    _ssh_config=self._ssh_config,
                )
            )

        return handles

    def list_all_slices(self) -> list[GcpSliceHandle | GcpVmSliceHandle]:
        """List all slices managed by this cluster.

        Uses project-wide queries (empty zones = all zones) via GcpService,
        filtered by iris-{prefix}-managed=true.
        """
        managed_labels = {self._iris_labels.iris_managed: "true"}

        if self._gcp.mode == ServiceMode.LOCAL:
            return self._gcp.get_local_slices(managed_labels)  # type: ignore[return-value]

        tpu_infos = self._gcp.tpu_list(zones=[], labels=managed_labels)
        vm_infos = self._gcp.vm_list(zones=[], labels=managed_labels)

        handles: list[GcpSliceHandle | GcpVmSliceHandle] = []

        for tpu in tpu_infos:
            if tpu.state not in ("READY", "CREATING"):
                continue
            handles.append(
                GcpSliceHandle(
                    _slice_id=tpu.name,
                    _zone=tpu.zone,
                    _project_id=self._project_id,
                    _labels=tpu.labels,
                    _created_at=tpu.created_at,
                    _label_prefix=self._label_prefix,
                    _accelerator_variant=tpu.accelerator_type,
                    _gcp_service=self._gcp,
                    _ssh_config=self._ssh_config,
                    _state=tpu.state,
                )
            )

        for vm in vm_infos:
            if vm.status not in _ACTIVE_VM_SLICE_STATES:
                continue
            slice_id = vm.labels.get(self._iris_labels.iris_slice_id, "")
            if not slice_id:
                continue
            handles.append(
                GcpVmSliceHandle(
                    _slice_id=slice_id,
                    _vm_name=vm.name,
                    _zone=vm.zone,
                    _project_id=self._project_id,
                    _gcp_service=self._gcp,
                    _labels=vm.labels,
                    _created_at=vm.created_at,
                    _label_prefix=self._label_prefix,
                    _ssh_config=self._ssh_config,
                )
            )

        logger.info("list_all_slices: found %d managed slices", len(handles))
        return handles

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpStandaloneWorkerHandle]:
        """List GCE instances across zones, optionally filtered by labels.

        When zones is empty, searches the entire project (empty zones triggers
        project-wide search in GcpService).
        """
        vm_infos = self._gcp.vm_list(zones, labels)

        handles: list[GcpStandaloneWorkerHandle] = []
        for vm in vm_infos:
            remote_exec = GceRemoteExec(
                project_id=self._project_id,
                zone=vm.zone,
                vm_name=vm.name,
            )
            handles.append(
                GcpStandaloneWorkerHandle(
                    _vm_id=construct_worker_id(vm.name, 0),
                    _internal_address=vm.internal_ip,
                    _external_address=vm.external_ip,
                    _zone=vm.zone,
                    _project_id=self._project_id,
                    _gcp_service=self._gcp,
                    _remote_exec=remote_exec,
                )
            )
        return handles

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        if self._gcp.mode == ServiceMode.LOCAL:
            return nullcontext(address)
        return _gcp_tunnel(
            project=self._project_id,
            label_prefix=self._label_prefix,
            local_port=local_port,
        )

    def shutdown(self) -> None:
        self._gcp.shutdown()

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Discover controller by querying GCP for labeled controller VM.

        In LOCAL mode, returns the configured address directly without querying GCP.
        """
        gcp = controller_config.gcp
        port = gcp.port or 10000

        if self._gcp.mode == ServiceMode.LOCAL:
            return f"localhost:{port}"

        vms = self.list_vms(
            zones=[gcp.zone],
            labels={self._iris_labels.iris_controller: "true"},
        )
        if not vms:
            raise RuntimeError(
                f"No controller VM found (label={self._iris_labels.iris_controller}=true, project={self._project_id})"
            )
        return f"{vms[0].internal_address}:{port}"

    def start_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Start or discover existing controller on GCP. Returns address (host:port)."""
        address, _vm = vm_start_controller(self, config)
        return address

    def restart_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Restart controller container in-place on existing GCP VM."""
        address, _vm = vm_restart_controller(self, config)
        return address

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        """Stop the controller on GCP by terminating the controller VM."""
        vm_stop_controller(self, config)

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        return default_stop_all(self, config, dry_run=dry_run, label_prefix=label_prefix)

    # ========================================================================
    # Internal helpers
    # ========================================================================


# ============================================================================
# Tunnel
# ============================================================================


def _check_gcloud_ssh_key() -> None:
    """Verify that the gcloud compute SSH key exists.

    ``gcloud compute ssh`` expects ``~/.ssh/google_compute_engine``.  When the
    key is missing, gcloud tries to generate one interactively — which hangs
    indefinitely in a non-interactive subprocess.  Detect this early and give
    the user a clear remediation path.
    """
    key_path = os.path.expanduser("~/.ssh/google_compute_engine")
    if not os.path.exists(key_path):
        raise RuntimeError(
            f"SSH key not found at {key_path}. "
            "gcloud compute ssh requires this key to connect to VMs.\n"
            "To create it, run:\n"
            "  gcloud compute ssh --dry-run <any-vm> --zone=<zone>\n"
            "or:\n"
            "  ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -C \"$(gcloud config get account)\" -N ''\n"
            "Then re-run your command."
        )


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
    _check_gcloud_ssh_key()

    if local_port is None:
        local_port = find_free_port(start=10000)

    labels = Labels(label_prefix)
    label_filter = f"labels.{labels.iris_controller}=true AND status=RUNNING"
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
        raise RuntimeError(f"No controller VM found (label={labels.iris_controller}=true, project={project})")

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
            "BatchMode=yes",
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
