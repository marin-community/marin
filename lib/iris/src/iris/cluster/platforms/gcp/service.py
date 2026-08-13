# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP service interface and shared request types."""

import re
from dataclasses import dataclass, field
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.config import SliceConfig, WorkerConfig
from iris.cluster.platforms.types import InfraError, QuotaExhaustedError, ResourceNotFoundError, SliceHandle
from iris.cluster.service_mode import ServiceMode
from iris.cluster.tpu_topology import TPU_TOPOLOGIES
from iris.cluster.types import CapacityType

KNOWN_GCP_ZONES: frozenset[str] = frozenset(
    {
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-central1-f",
        "us-central2-b",
        "us-east1-b",
        "us-east1-d",
        "us-east5-a",
        "us-east5-b",
        "us-east5-c",
        "us-west1-a",
        "us-west1-c",
        "us-west4-a",
        "us-south1-a",
        "europe-west4-a",
        "europe-west4-b",
        "asia-northeast1-b",
    }
)
KNOWN_TPU_TYPES: frozenset[str] = frozenset(t.name for t in TPU_TOPOLOGIES)

_LABEL_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_LABEL_VALUE_RE = re.compile(r"^[a-z0-9_-]{0,63}$")
_RESOURCE_NAME_RE = re.compile(r"^[a-z]([a-z0-9-]*[a-z0-9])?$")
MAX_RESOURCE_NAME_LENGTH = 63

CAPACITY_TYPE_LABEL = "capacity-type"
CAPACITY_TYPE_RESERVED_VALUE = "reserved"

_RPC_CODE_RESOURCE_EXHAUSTED = 8


@dataclass
class TpuInfo:
    """Parsed TPU state from GCP API."""

    name: str
    state: str
    accelerator_type: str
    zone: str
    labels: dict[str, str]
    metadata: dict[str, str]
    service_account: str | None
    network_endpoints: list[str]
    external_network_endpoints: list[str | None]
    created_at: Timestamp
    # Present only on results returned by tpu_create while creation is in flight.
    operation_name: str = ""


@dataclass(frozen=True)
class OperationStatus:
    """Status of an async GCP long-running operation."""

    done: bool
    error_code: int | None = None
    error_message: str | None = None


def operation_error(status: OperationStatus) -> InfraError | None:
    """Map resource exhaustion to autoscaler backoff and other failures to ``InfraError``."""
    if status.error_code is None:
        return None
    if status.error_code == _RPC_CODE_RESOURCE_EXHAUSTED:
        return QuotaExhaustedError(status.error_message or "resource exhausted")
    return InfraError(f"TPU operation failed: {status.error_message}")


@dataclass
class VmInfo:
    """Parsed GCE VM state from GCP API."""

    name: str
    status: str
    zone: str
    internal_ip: str
    external_ip: str | None
    labels: dict[str, str]
    metadata: dict[str, str]
    service_account: str | None
    created_at: Timestamp


@dataclass
class TpuCreateRequest:
    """Parameters for creating a TPU slice."""

    name: str
    zone: str
    accelerator_type: str
    runtime_version: str
    capacity_type: CapacityType | None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    service_account: str | None = None
    network: str | None = None
    subnetwork: str | None = None
    enable_external_ip: bool = True


@dataclass
class QueuedResourceInfo:
    """Status of a GCP queued resource."""

    name: str
    state: str
    zone: str = ""
    labels: dict[str, str] | None = None


@dataclass
class VmCreateRequest:
    """Parameters for creating a GCE VM."""

    name: str
    zone: str
    machine_type: str
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    startup_script: str | None = None
    service_account: str | None = None
    disk_size_gb: int = 200
    boot_disk_type: str = "pd-standard"
    image_family: str = "cos-stable"
    image_project: str = "cos-cloud"


def validate_resource_name(name: str, resource_kind: str) -> None:
    if len(name) > MAX_RESOURCE_NAME_LENGTH:
        raise ValueError(f"{resource_kind} name exceeds {MAX_RESOURCE_NAME_LENGTH} chars: {name!r}")
    if not _RESOURCE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {resource_kind} name (must be lowercase alphanumeric/hyphens, start with letter): {name!r}"
        )


def validate_labels(labels: dict[str, str]) -> None:
    for key, value in labels.items():
        if not _LABEL_KEY_RE.match(key):
            raise ValueError(f"Invalid label key: {key!r}")
        if not _LABEL_VALUE_RE.match(value):
            raise ValueError(f"Invalid label value for {key!r}: {value!r}")


def validate_zone(zone: str, valid_zones: set[str]) -> None:
    if zone not in valid_zones:
        raise InfraError(f"Zone {zone!r} not available")


def validate_tpu_create(request: TpuCreateRequest, valid_zones: set[str], valid_types: set[str]) -> None:
    validate_resource_name(request.name, "TPU")
    validate_zone(request.zone, valid_zones)
    if request.accelerator_type not in valid_types:
        raise ResourceNotFoundError(f"Unknown accelerator type: {request.accelerator_type!r}")
    if not request.runtime_version:
        raise ValueError("runtime_version must be non-empty")
    validate_labels(request.labels)


def validate_vm_create(request: VmCreateRequest, valid_zones: set[str]) -> None:
    validate_resource_name(request.name, "VM")
    validate_zone(request.zone, valid_zones)
    if request.disk_size_gb <= 0:
        raise ValueError(f"disk_size_gb must be positive, got {request.disk_size_gb}")
    validate_labels(request.labels)


class GcpService(Protocol):
    """Provision and inspect Iris compute resources through GCP semantics."""

    @property
    def mode(self) -> ServiceMode: ...

    @property
    def project_id(self) -> str: ...

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo: ...
    def tpu_operation_status(self, operation_name: str) -> OperationStatus: ...
    def tpu_delete(self, name: str, zone: str) -> None: ...
    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None: ...
    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]: ...

    def queued_resource_create(self, request: TpuCreateRequest) -> None: ...
    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None: ...
    def queued_resource_delete(self, name: str, zone: str) -> None: ...
    def queued_resource_list(
        self, zones: list[str], labels: dict[str, str] | None = None
    ) -> list[QueuedResourceInfo]: ...

    def vm_create(self, request: VmCreateRequest, *, wait: bool = False) -> VmInfo: ...
    def vm_delete(self, name: str, zone: str, *, wait: bool = False) -> None: ...
    def vm_describe(self, name: str, zone: str) -> VmInfo | None: ...
    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]: ...
    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None: ...
    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None: ...
    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str: ...

    def logging_read(self, filter_str: str, limit: int = 200) -> list[str]: ...

    def create_local_slice(
        self,
        slice_id: str,
        config: SliceConfig,
        worker_config: WorkerConfig | None = None,
    ) -> SliceHandle: ...

    def get_local_slices(self, labels: dict[str, str] | None = None) -> list[SliceHandle]: ...
    def shutdown(self) -> None: ...
