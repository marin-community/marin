# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from iris.cluster.platform.service_mode import ServiceMode
from iris.time_utils import Timestamp


@dataclass
class TpuInfo:
    """Parsed TPU state from GCP API."""

    name: str
    state: str  # "CREATING", "READY", "DELETING", etc.
    accelerator_type: str
    zone: str
    labels: dict[str, str]
    metadata: dict[str, str]
    network_endpoints: list[str]  # IP addresses
    created_at: Timestamp


@dataclass
class VmInfo:
    """Parsed GCE VM state from GCP API."""

    name: str
    status: str  # "RUNNING", "TERMINATED", etc.
    zone: str
    internal_ip: str
    external_ip: str | None
    labels: dict[str, str]
    metadata: dict[str, str]


@dataclass
class TpuCreateRequest:
    """Parameters for creating a TPU slice."""

    name: str
    zone: str
    accelerator_type: str
    runtime_version: str
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    preemptible: bool = False
    service_account: str | None = None
    network: str | None = None
    subnetwork: str | None = None


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


class GcpService(Protocol):
    """Service boundary for GCP operations.

    All methods raise PlatformError (or subclass) on failure.
    Implementations: GcpServiceImpl with ServiceMode determining behavior.
    """

    @property
    def mode(self) -> ServiceMode: ...

    @property
    def project_id(self) -> str: ...

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo: ...
    def tpu_delete(self, name: str, zone: str) -> None: ...
    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None: ...
    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]: ...

    def vm_create(self, request: VmCreateRequest) -> VmInfo: ...
    def vm_delete(self, name: str, zone: str) -> None: ...
    def vm_describe(self, name: str, zone: str) -> VmInfo | None: ...
    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]: ...
    def vm_reset(self, name: str, zone: str) -> None: ...
    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None: ...
    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None: ...
    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str: ...

    def shutdown(self) -> None:
        """Stop all managed resources. No-op in CLOUD/DRY_RUN modes."""
        ...
