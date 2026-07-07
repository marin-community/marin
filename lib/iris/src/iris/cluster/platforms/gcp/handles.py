# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP handle implementations for TPU slices, VM slices, and standalone VMs.

GcpWorkerHandle: TPU worker within a slice (SSH via gcloud compute tpus tpu-vm ssh)
GcpStandaloneWorkerHandle: GCE instance with terminate/label/metadata support
GcpSliceHandle: TPU pod slice (describe, terminate)
GcpVmSliceHandle: Single-VM GCE-backed slice
"""

import logging
import re
import threading
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.config import SshConfig
from iris.cluster.platforms._worker_base import RemoteExecWorkerBase
from iris.cluster.platforms.gcp.service import GcpService
from iris.cluster.platforms.gcp.ssh import ssh_impersonate_service_account
from iris.cluster.platforms.remote_exec import GceRemoteExec, GcloudRemoteExec
from iris.cluster.platforms.types import (
    CloudSliceState,
    CloudWorkerState,
    InfraError,
    Labels,
    SliceStatus,
    WorkerStatus,
)
from iris.cluster.tpu_topology import get_tpu_topology

logger = logging.getLogger(__name__)

# GCP TPU state mapping. States not in this map collapse to UNKNOWN; the boot
# reconciler treats anything outside the alive set (CREATING/READY/REPAIRING)
# as a candidate for reclaim.
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

# Queued-resource (reserved TPU) state mapping. Non-live states must surface so
# the boot reconciler can reclaim them; ACTIVE is transient (the matching TPU
# VM is what list_all_slices normally returns) and only appears here briefly.
_QR_STATE_MAP: dict[str, CloudSliceState] = {
    "QUEUED": CloudSliceState.CREATING,
    "WAITING_FOR_RESOURCES": CloudSliceState.CREATING,
    "PROVISIONING": CloudSliceState.CREATING,
    "ACTIVE": CloudSliceState.READY,
    "FAILED": CloudSliceState.FAILED,
    "SUSPENDED": CloudSliceState.FAILED,
    "DELETING": CloudSliceState.DELETING,
}
_GCE_NAME_MAX_LEN = 63
_GCE_NAME_RE = re.compile(r"[^a-z0-9-]+")
_GCE_NAME_EDGE_RE = re.compile(r"^-+|-+$")


def _build_gce_resource_name(name_prefix: str, suffix: str) -> str:
    """Build a GCE-valid resource name from a prefix and suffix.

    Normalizes the prefix to lowercase alphanumeric + hyphens (GCE naming rules),
    truncates to fit within the 63-char GCE limit, and appends the suffix.
    Used for both TPU slice names and VM instance names.
    """
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
    """Compose cloud lifecycle with bootstrap lifecycle into effective slice state.

    Worker health is canonical for liveness: once bootstrap is READY the slice
    is READY even if the GCP-reported cloud state still lags at CREATING — a TPU
    can boot and serve long before its create operation flips to READY. A slice
    rediscovered on controller restart (no bootstrap monitoring) carries a READY
    bootstrap sentinel, so it too becomes READY and is validated by the
    autoscaler's health probe, which reaps it if the workers are in fact dead.

    A bootstrap that definitively FAILED is authoritative — even when the cloud
    is no longer describable (UNKNOWN). A stockout/quota create failure leaves no
    TPU resource to describe, so describe() reports UNKNOWN; without surfacing the
    FAILED bootstrap verdict the autoscaler would wait out the full
    unresolvable-timeout grace period (and keep re-describing the dead TPU) before
    reaping a slice it already knows never came up.

    Cloud states that mean "gone or doomed" — FAILED and DELETING — are likewise
    authoritative. A bare UNKNOWN (cloud no longer describable, bootstrap still in
    progress or a stale READY sentinel) is reported as UNKNOWN so a vanished node
    never lingers as READY and the unresolvable-timeout path can reap it.
    """
    if cloud_state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
        return cloud_state
    # A definitive bootstrap failure wins over UNKNOWN: the slice never came up,
    # so reap it now instead of waiting out the unresolvable timeout.
    if bootstrap_state == CloudSliceState.FAILED:
        return CloudSliceState.FAILED
    if cloud_state == CloudSliceState.UNKNOWN:
        return CloudSliceState.UNKNOWN
    if bootstrap_state == CloudSliceState.READY:
        return CloudSliceState.READY
    # Bootstrap still in progress: surface BOOTSTRAPPING once cloud is READY,
    # otherwise reflect the raw cloud state (CREATING/REPAIRING).
    if cloud_state == CloudSliceState.READY:
        return CloudSliceState.BOOTSTRAPPING
    return cloud_state


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
    construct_worker_id). _gce_vm_name is the real GCE instance name used for
    gcloud commands.
    """

    _gce_vm_name: str = ""
    _zone: str = ""
    _project_id: str = ""
    _service_account: str | None = None
    # Always populated at construction; Optional only for dataclass inheritance ordering.
    _gcp_service: GcpService | None = None

    def __post_init__(self) -> None:
        if self._gcp_service is None:
            raise ValueError("_gcp_service is required")

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

    def terminate(self, *, wait: bool = False) -> None:
        assert self._gcp_service is not None
        logger.info("Deleting GCE instance: %s", self._gce_vm_name)
        self._gcp_service.vm_delete(self._gce_vm_name, self._zone, wait=wait)

    def set_labels(self, labels: dict[str, str]) -> None:
        assert self._gcp_service is not None
        logger.info("Setting labels on GCE instance: %s", self._gce_vm_name)
        try:
            self._gcp_service.vm_update_labels(self._gce_vm_name, self._zone, labels)
        except InfraError as e:
            logger.warning("Failed to set labels on %s: %s", self._gce_vm_name, e)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        assert self._gcp_service is not None
        logger.info("Setting metadata on GCE instance: %s", self._gce_vm_name)
        try:
            self._gcp_service.vm_set_metadata(self._gce_vm_name, self._zone, metadata)
        except InfraError as e:
            logger.warning("Failed to set metadata on %s: %s", self._gce_vm_name, e)

    def bootstrap(self, script: str) -> None:
        # Persist the script as the instance's startup-script before executing it
        # over SSH, so a future GCE-initiated reboot (host maintenance, manual
        # reset) re-runs *this* script — not the original baked in at VM creation,
        # which would pin the controller to the day-one image.
        self.set_metadata({"startup-script": script})
        super().bootstrap(script)


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
        _worker_port: int,
        _accelerator_variant: str,
        _gcp_service: GcpService,
        _ssh_config: SshConfig | None = None,
        _service_account: str | None = None,
        _bootstrapping: bool = False,
        _is_queued_resource: bool = False,
        _create_operation: str = "",
    ):
        self._slice_id = _slice_id
        self._zone = _zone
        self._project_id = _project_id
        self._gcp_service = _gcp_service
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._worker_port = _worker_port
        self._iris_labels = Labels(_label_prefix)
        self._accelerator_variant = _accelerator_variant
        self._ssh_config = _ssh_config
        self._service_account = _service_account
        self.is_queued_resource: bool = _is_queued_resource
        self._create_operation = _create_operation
        self._bootstrap_state: CloudSliceState | None = None if _bootstrapping else CloudSliceState.READY
        # Failure detail captured when bootstrap fails (e.g. the create-LRO error
        # "no more capacity ..."); surfaced via describe() so the autoscaler can
        # classify the outcome (stockout vs error) instead of losing it.
        self._bootstrap_error: str = ""
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

        with self._bootstrap_lock:
            bs = self._bootstrap_state
            bootstrap_error = self._bootstrap_error

        effective_state = _composite_slice_state(cloud_state, bs)

        return SliceStatus(
            state=effective_state,
            worker_count=cloud_status.worker_count,
            workers=cloud_status.workers,
            error_message=bootstrap_error if effective_state == CloudSliceState.FAILED else "",
        )

    def _describe_cloud(self) -> SliceStatus:
        """Query raw TPU state and VM endpoints via GcpService."""
        tpu_info = self._gcp_service.tpu_describe(self._slice_id, self._zone)
        if tpu_info is None:
            if self.is_queued_resource:
                return self._describe_queued_resource()
            logger.warning("Failed to describe TPU %s", self._slice_id)
            return SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0)

        state = _TPU_STATE_MAP.get(tpu_info.state, CloudSliceState.UNKNOWN)

        # Derive the topology from the live TPU rather than self._accelerator_variant:
        # queued-resource handles adopted during boot recovery carry an empty variant
        # (the QR API doesn't report one), and that handle is never refreshed once the
        # backing TPU VM provisions. tpu_info always reports the real accelerator_type.
        accelerator_variant = tpu_info.accelerator_type
        try:
            worker_count = get_tpu_topology(accelerator_variant).vm_count
        except ValueError as e:
            raise InfraError(
                f"Unknown TPU topology '{accelerator_variant}' for slice {self._slice_id}. "
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
                impersonate_service_account=ssh_impersonate_service_account(self._ssh_config),
                _address=internal_ip,
            )
            workers.append(
                GcpWorkerHandle(
                    _vm_id=f"{self._slice_id}-worker-{i}",
                    _internal_address=internal_ip,
                    _port=self._worker_port,
                    _remote_exec=remote_exec,
                )
            )

        return SliceStatus(state=state, worker_count=worker_count, workers=workers)

    def _describe_queued_resource(self) -> SliceStatus:
        """Query queued resource state when the TPU VM doesn't exist yet."""
        qr = self._gcp_service.queued_resource_describe(self._slice_id, self._zone)
        if qr is None:
            return SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0)
        if qr.state in ("FAILED", "SUSPENDED"):
            return SliceStatus(state=CloudSliceState.FAILED, worker_count=0)
        # QUEUED, PROVISIONING, WAITING_FOR_RESOURCES → still creating
        return SliceStatus(state=CloudSliceState.CREATING, worker_count=0)

    def terminate(self, *, wait: bool = False) -> None:
        if self.is_queued_resource:
            logger.info("Terminating queued resource (force): %s", self._slice_id)
            self._gcp_service.queued_resource_delete(self._slice_id, self._zone)
        else:
            logger.info("Terminating TPU (async): %s", self._slice_id)
            self._gcp_service.tpu_delete(self._slice_id, self._zone)

    def cleanup_bootstrap_failure(self) -> None:
        """Clean up provider state after bootstrap fails."""
        if self.is_queued_resource:
            self.terminate()


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
        _worker_port: int,
        _gcp_service: GcpService,
        _ssh_config: SshConfig | None = None,
        _service_account: str | None = None,
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
        self._worker_port = _worker_port
        self._iris_labels = Labels(_label_prefix)
        self._ssh_config = _ssh_config
        self._service_account = _service_account
        self._bootstrap_state: CloudSliceState | None = None if _bootstrapping else CloudSliceState.READY
        # See GcpSliceHandle._bootstrap_error.
        self._bootstrap_error: str = ""
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
            bootstrap_error = self._bootstrap_error

        effective_state = _composite_slice_state(cloud_state, bs)

        return SliceStatus(
            state=effective_state,
            worker_count=cloud_status.worker_count,
            workers=cloud_status.workers,
            error_message=bootstrap_error if effective_state == CloudSliceState.FAILED else "",
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
            impersonate_service_account=ssh_impersonate_service_account(self._ssh_config),
        )
        worker = GcpStandaloneWorkerHandle(
            _vm_id=f"{self._slice_id}-worker-0",
            _internal_address=vm_info.internal_ip,
            _port=self._worker_port,
            _gce_vm_name=self._vm_name,
            _zone=self._zone,
            _project_id=self._project_id,
            _gcp_service=self._gcp_service,
            _remote_exec=remote_exec,
            _service_account=self._service_account,
        )
        return SliceStatus(state=state, worker_count=1, workers=[worker])

    def terminate(self, *, wait: bool = False) -> None:
        logger.info("Terminating VM slice: %s (vm=%s)", self._slice_id, self._vm_name)
        self._gcp_service.vm_delete(self._vm_name, self._zone, wait=wait)

    def cleanup_bootstrap_failure(self) -> None:
        """Clean up provider state after bootstrap fails."""
