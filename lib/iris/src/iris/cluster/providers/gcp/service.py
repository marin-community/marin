# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from iris.cluster.providers.gcp.api import GCPApi
from iris.cluster.providers.types import (
    InfraError,
    ResourceNotFoundError,
)
from iris.cluster.providers.gcp.local import LocalSliceHandle
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import TPU_TOPOLOGIES
from iris.rpc import config_pb2
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)

# GCP zones where TPUs are available
KNOWN_GCP_ZONES: frozenset[str] = frozenset(
    {
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-central1-f",
        "us-central2-b",
        "us-east1-d",
        "us-east5-a",
        "us-east5-b",
        "us-east5-c",
        "us-west1-c",
        "us-west4-a",
        "us-south1-a",
        "europe-west4-a",
        "europe-west4-b",
        "asia-northeast1-b",
    }
)

# Accelerator type names derived from the TPU_TOPOLOGIES registry
KNOWN_TPU_TYPES: frozenset[str] = frozenset(t.name for t in TPU_TOPOLOGIES)

# GCP label key/value constraints
_LABEL_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_LABEL_VALUE_RE = re.compile(r"^[a-z0-9_-]{0,63}$")

# GCP resource name constraints
_RESOURCE_NAME_RE = re.compile(r"^[a-z]([a-z0-9-]*[a-z0-9])?$")
MAX_RESOURCE_NAME_LENGTH = 63

# GCP label key/value used to tag reserved (queued-resource) TPUs for rediscovery.
CAPACITY_TYPE_LABEL = "capacity-type"
CAPACITY_TYPE_RESERVED_VALUE = "reserved"


# ============================================================================
# Data types
# ============================================================================


@dataclass
class TpuInfo:
    """Parsed TPU state from GCP API."""

    name: str
    state: str  # "CREATING", "READY", "DELETING", etc.
    accelerator_type: str
    zone: str
    labels: dict[str, str]
    metadata: dict[str, str]
    service_account: str | None
    network_endpoints: list[str]  # Internal IP addresses
    external_network_endpoints: list[str | None]
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
    service_account: str | None
    created_at: Timestamp


@dataclass
class TpuCreateRequest:
    """Parameters for creating a TPU slice."""

    name: str
    zone: str
    accelerator_type: str
    runtime_version: str
    capacity_type: int  # config_pb2.CapacityType enum value
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    service_account: str | None = None
    network: str | None = None
    subnetwork: str | None = None


@dataclass
class QueuedResourceInfo:
    """Status of a GCP queued resource."""

    name: str
    state: str  # QUEUED, PROVISIONING, ACTIVE, FAILED, SUSPENDED
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


# ============================================================================
# Shared validation functions
# ============================================================================


def validate_resource_name(name: str, resource_kind: str) -> None:
    if len(name) > MAX_RESOURCE_NAME_LENGTH:
        raise ValueError(f"{resource_kind} name exceeds {MAX_RESOURCE_NAME_LENGTH} chars: {name!r}")
    if not _RESOURCE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {resource_kind} name (must be lowercase alphanumeric/hyphens, " f"start with letter): {name!r}"
        )


def validate_labels(labels: dict[str, str]) -> None:
    for key, val in labels.items():
        if not _LABEL_KEY_RE.match(key):
            raise ValueError(f"Invalid label key: {key!r}")
        if not _LABEL_VALUE_RE.match(val):
            raise ValueError(f"Invalid label value for {key!r}: {val!r}")


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


# ============================================================================
# Protocol
# ============================================================================


class GcpService(Protocol):
    """Service boundary for GCP operations.

    All methods raise InfraError (or subclass) on failure.
    Implementations: CloudGcpService (CLOUD), InMemoryGcpService (DRY_RUN/LOCAL).
    """

    @property
    def mode(self) -> ServiceMode: ...

    @property
    def project_id(self) -> str: ...

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo: ...
    def tpu_delete(self, name: str, zone: str) -> None: ...
    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None: ...
    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]: ...

    def queued_resource_create(self, request: TpuCreateRequest) -> None: ...
    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None: ...
    def queued_resource_delete(self, name: str, zone: str) -> None: ...
    def queued_resource_list(
        self, zones: list[str], labels: dict[str, str] | None = None
    ) -> list[QueuedResourceInfo]: ...

    def vm_create(self, request: VmCreateRequest) -> VmInfo: ...
    def vm_delete(self, name: str, zone: str) -> None: ...
    def vm_describe(self, name: str, zone: str) -> VmInfo | None: ...
    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]: ...
    def vm_reset(self, name: str, zone: str) -> None: ...
    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None: ...
    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None: ...
    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str: ...

    def logging_read(self, filter_str: str, limit: int = 200) -> list[str]:
        """Return matching Cloud Logging textPayload entries (newest first)."""
        ...

    def create_local_slice(
        self,
        slice_id: str,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        """Create an in-process slice. Only valid in LOCAL mode."""
        ...

    def get_local_slices(self, labels: dict[str, str] | None = None) -> list[LocalSliceHandle]:
        """Return tracked local slices, optionally filtered by labels. Only valid in LOCAL mode."""
        ...

    def shutdown(self) -> None:
        """Stop all managed resources. No-op in CLOUD/DRY_RUN modes."""
        ...


# ============================================================================
# CloudGcpService — REST API implementation via GCPApi
# ============================================================================


def _build_label_filter(labels: dict[str, str]) -> str:
    parts = [f"labels.{k}={v}" for k, v in labels.items()]
    return " AND ".join(parts)


def _labels_match(resource_labels: dict[str, str], required: dict[str, str]) -> bool:
    return all(resource_labels.get(k) == v for k, v in required.items())


def _extract_node_name(resource_name: str) -> str:
    if "/" in resource_name:
        return resource_name.split("/")[-1]
    return resource_name


def _parse_tpu_created_at(tpu_data: dict) -> Timestamp:
    create_time = tpu_data.get("createTime", "")
    if not create_time:
        return Timestamp.now()
    try:
        dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        epoch_ms = int(dt.timestamp() * 1000)
        return Timestamp.from_ms(epoch_ms)
    except (ValueError, AttributeError):
        return Timestamp.now()


def _parse_vm_created_at(vm_data: dict) -> Timestamp:
    create_time = vm_data.get("creationTimestamp", "")
    if not create_time:
        return Timestamp.now()
    try:
        dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        epoch_ms = int(dt.timestamp() * 1000)
        return Timestamp.from_ms(epoch_ms)
    except (ValueError, AttributeError):
        return Timestamp.now()


def _parse_tpu_info(tpu_data: dict, zone: str) -> TpuInfo:
    """Parse raw GCP TPU JSON into a TpuInfo dataclass."""
    name = _extract_node_name(tpu_data.get("name", ""))

    accelerator_type = tpu_data.get("acceleratorType", "")
    if "/" in accelerator_type:
        accelerator_type = accelerator_type.split("/")[-1]

    endpoints = tpu_data.get("networkEndpoints", [])
    ips = [ep.get("ipAddress", "") for ep in endpoints if ep.get("ipAddress")]
    external_ips = [(ep.get("accessConfig") or {}).get("externalIp") for ep in endpoints]

    return TpuInfo(
        name=name,
        state=tpu_data.get("state", "UNKNOWN"),
        accelerator_type=accelerator_type,
        zone=zone,
        labels=tpu_data.get("labels", {}),
        metadata=tpu_data.get("metadata", {}),
        service_account=(tpu_data.get("serviceAccount", {}) or {}).get("email"),
        network_endpoints=ips,
        external_network_endpoints=external_ips,
        created_at=_parse_tpu_created_at(tpu_data),
    )


def _parse_vm_info(vm_data: dict, fallback_zone: str = "") -> VmInfo:
    """Parse raw GCP VM JSON into a VmInfo dataclass."""
    zone_url = vm_data.get("zone", "")
    zone = zone_url.split("/")[-1] if zone_url else fallback_zone

    network_interfaces = vm_data.get("networkInterfaces", [])
    internal_ip = ""
    external_ip = None
    if network_interfaces:
        internal_ip = network_interfaces[0].get("networkIP", "")
        access_configs = network_interfaces[0].get("accessConfigs", [])
        if access_configs:
            external_ip = access_configs[0].get("natIP")

    # Metadata in GCP JSON is {"items": [{"key": ..., "value": ...}]}
    raw_metadata = vm_data.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(raw_metadata, dict):
        for item in raw_metadata.get("items", []):
            metadata[item["key"]] = item.get("value", "")

    service_accounts = vm_data.get("serviceAccounts") or []
    first_service_account = service_accounts[0] if service_accounts else None
    service_account_email = first_service_account.get("email") if isinstance(first_service_account, dict) else None

    return VmInfo(
        name=vm_data.get("name", ""),
        status=vm_data.get("status", "UNKNOWN"),
        zone=zone,
        internal_ip=internal_ip,
        external_ip=external_ip,
        labels=vm_data.get("labels", {}),
        metadata=metadata,
        service_account=service_account_email,
        created_at=_parse_vm_created_at(vm_data),
    )


class CloudGcpService:
    """GcpService backed by GCP REST APIs via GCPApi. Used in CLOUD mode."""

    def __init__(self, project_id: str, api: GCPApi | None = None) -> None:
        self._project_id = project_id
        self._api = api if api is not None else GCPApi(project_id)
        self._valid_zones: set[str] = set(KNOWN_GCP_ZONES)
        self._valid_accelerator_types: set[str] = set(KNOWN_TPU_TYPES)

    @property
    def mode(self) -> ServiceMode:
        return ServiceMode.CLOUD

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def api(self) -> GCPApi:
        return self._api

    # ========================================================================
    # TPU operations
    # ========================================================================

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        validate_tpu_create(request, self._valid_zones, self._valid_accelerator_types)

        body: dict = {
            "acceleratorType": request.accelerator_type,
            "runtimeVersion": request.runtime_version,
        }
        if request.labels:
            body["labels"] = request.labels
        if request.metadata:
            body["metadata"] = request.metadata
        if request.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE:
            body["schedulingConfig"] = {"preemptible": True}
        if request.service_account:
            body["serviceAccount"] = {"email": request.service_account}
        if request.network or request.subnetwork:
            network_config: dict = {}
            if request.network:
                network_config["network"] = request.network
            if request.subnetwork:
                network_config["subnetwork"] = request.subnetwork
            body["networkConfig"] = network_config

        logger.info("Creating TPU: %s (type=%s, zone=%s)", request.name, request.accelerator_type, request.zone)
        self._api.tpu_create(request.name, request.zone, body)

        # REST create returns an operation, not the node — fetch it explicitly.
        info = self.tpu_describe(request.name, request.zone)
        if info is None:
            raise InfraError(f"TPU {request.name} created but could not be described")
        return info

    def tpu_delete(self, name: str, zone: str) -> None:
        logger.info("Deleting TPU (async): %s", name)
        self._api.tpu_delete(name, zone)

    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None:
        try:
            tpu_data = self._api.tpu_get(name, zone)
        except ResourceNotFoundError:
            return None
        except InfraError:
            logger.warning("Failed to describe TPU %s", name, exc_info=True)
            return None
        return _parse_tpu_info(tpu_data, zone)

    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]:
        results: list[TpuInfo] = []
        # TPU v2 API requires a real zone; empty zones = scan all known zones.
        zone_list = zones if zones else list(self._valid_zones)

        for zone in zone_list:
            try:
                items = self._api.tpu_list(zone)
            except InfraError:
                logger.warning("Failed to list TPUs in zone %s", zone, exc_info=True)
                continue
            for tpu_data in items:
                if labels and not _labels_match(tpu_data.get("labels", {}), labels):
                    continue
                # Extract zone from resource name if present
                tpu_zone = zone
                raw_name = tpu_data.get("name", "")
                if "/" in raw_name:
                    parts = raw_name.split("/")
                    if len(parts) >= 4:
                        tpu_zone = parts[3]
                results.append(_parse_tpu_info(tpu_data, tpu_zone))

        return results

    # ========================================================================
    # Queued resource operations (for reserved TPUs)
    # ========================================================================

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        validate_tpu_create(request, self._valid_zones, self._valid_accelerator_types)

        node_spec: dict = {
            "node": {
                "acceleratorType": request.accelerator_type,
                "runtimeVersion": request.runtime_version,
                "labels": request.labels or {},
                "metadata": request.metadata or {},
            },
            "nodeId": request.name,
        }
        if request.service_account:
            node_spec["node"]["serviceAccount"] = {"email": request.service_account}
        if request.network or request.subnetwork:
            network_config: dict = {}
            if request.network:
                network_config["network"] = request.network
            if request.subnetwork:
                network_config["subnetwork"] = request.subnetwork
            node_spec["node"]["networkConfig"] = network_config

        body = {
            "tpu": {"nodeSpec": [node_spec]},
            "guaranteed": {"reserved": True},
        }

        logger.info(
            "Creating queued resource: %s (type=%s, zone=%s)",
            request.name,
            request.accelerator_type,
            request.zone,
        )
        self._api.queued_resource_create(request.name, request.zone, body)

    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None:
        try:
            data = self._api.queued_resource_get(name, zone)
        except ResourceNotFoundError:
            return None
        state = data.get("state", {}).get("state", "UNKNOWN")
        return QueuedResourceInfo(name=name, state=state, zone=zone)

    def queued_resource_delete(self, name: str, zone: str) -> None:
        logger.info("Deleting queued resource (force): %s", name)
        self._api.queued_resource_delete(name, zone)

    def queued_resource_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[QueuedResourceInfo]:
        zone_list = zones if zones else list(self._valid_zones)
        results: list[QueuedResourceInfo] = []
        for zone in zone_list:
            try:
                items = self._api.queued_resource_list(zone)
            except InfraError:
                logger.warning("Failed to list queued resources in %s", zone, exc_info=True)
                continue
            for item in items:
                qr_name = item.get("name", "").rsplit("/", 1)[-1]
                state = item.get("state", {}).get("state", "UNKNOWN")
                item_labels = item.get("tpu", {}).get("nodeSpec", [{}])[0].get("node", {}).get("labels", {})
                if labels and not all(item_labels.get(k) == v for k, v in labels.items()):
                    continue
                results.append(QueuedResourceInfo(name=qr_name, state=state, zone=zone, labels=item_labels))
        return results

    # ========================================================================
    # VM operations
    # ========================================================================

    def vm_create(self, request: VmCreateRequest) -> VmInfo:
        validate_vm_create(request, self._valid_zones)

        all_metadata = dict(request.metadata)
        if request.startup_script:
            all_metadata["startup-script"] = request.startup_script

        body: dict = {
            "name": request.name,
            "machineType": f"zones/{request.zone}/machineTypes/{request.machine_type}",
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "initializeParams": {
                        "diskSizeGb": str(request.disk_size_gb),
                        "diskType": f"zones/{request.zone}/diskTypes/{request.boot_disk_type}",
                        "sourceImage": f"projects/{request.image_project}/global/images/family/{request.image_family}",
                    },
                }
            ],
            "networkInterfaces": [{"accessConfigs": [{"type": "ONE_TO_ONE_NAT"}]}],
            "serviceAccounts": [
                {
                    "email": request.service_account or "default",
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                }
            ],
        }
        if request.labels:
            body["labels"] = request.labels
        if all_metadata:
            body["metadata"] = {"items": [{"key": k, "value": v} for k, v in all_metadata.items()]}

        logger.info("Creating VM: %s (zone=%s, type=%s)", request.name, request.zone, request.machine_type)
        try:
            self._api.instance_insert(request.zone, body)
        except InfraError as e:
            if "already exists" not in str(e).lower():
                raise

        info = self.vm_describe(request.name, request.zone)
        if info is None:
            raise InfraError(f"VM {request.name} created but could not be described")
        return info

    def vm_delete(self, name: str, zone: str) -> None:
        logger.info("Deleting VM: %s", name)
        self._api.instance_delete(name, zone)

    def vm_reset(self, name: str, zone: str) -> None:
        logger.info("Resetting VM: %s", name)
        self._api.instance_reset(name, zone)

    def vm_describe(self, name: str, zone: str) -> VmInfo | None:
        try:
            data = self._api.instance_get(name, zone)
        except ResourceNotFoundError:
            return None
        except InfraError:
            logger.warning("Failed to describe VM %s", name, exc_info=True)
            return None
        return _parse_vm_info(data, fallback_zone=zone)

    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]:
        results: list[VmInfo] = []
        filter_str = _build_label_filter(labels) if labels else ""

        if not zones:
            try:
                items = self._api.instance_list(zone=None, filter_str=filter_str)
            except InfraError:
                logger.warning("Failed to list instances", exc_info=True)
                return []
            for vm_data in items:
                results.append(_parse_vm_info(vm_data))
            return results

        for zone in zones:
            try:
                items = self._api.instance_list(zone=zone, filter_str=filter_str)
            except InfraError:
                logger.warning("Failed to list instances in zone %s", zone, exc_info=True)
                continue
            for vm_data in items:
                results.append(_parse_vm_info(vm_data, fallback_zone=zone))

        return results

    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None:
        validate_labels(labels)
        logger.info("Updating labels on VM %s", name)
        # Read-modify-write: GET the current labelFingerprint, merge, then POST.
        data = self._api.instance_get(name, zone)
        current_labels = data.get("labels", {})
        current_labels.update(labels)
        fingerprint = data.get("labelFingerprint", "")
        self._api.instance_set_labels(name, zone, current_labels, fingerprint)

    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None:
        logger.info("Setting metadata on VM %s", name)
        # Read-modify-write: GET current metadata with fingerprint, merge items, POST.
        data = self._api.instance_get(name, zone)
        raw_metadata = data.get("metadata", {})
        fingerprint = raw_metadata.get("fingerprint", "")
        existing_items: dict[str, str] = {}
        for item in raw_metadata.get("items", []):
            existing_items[item["key"]] = item.get("value", "")
        existing_items.update(metadata)
        body = {
            "fingerprint": fingerprint,
            "items": [{"key": k, "value": v} for k, v in existing_items.items()],
        }
        self._api.instance_set_metadata(name, zone, body)

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        try:
            data = self._api.instance_get_serial_port_output(name, zone, start=start)
        except InfraError:
            logger.warning("Failed to get serial port output for %s", name, exc_info=True)
            return ""
        return data.get("contents", "")

    def logging_read(self, filter_str: str, limit: int = 200) -> list[str]:
        try:
            entries = self._api.logging_list_entries(filter_str, limit=limit)
        except InfraError:
            logger.warning("Cloud Logging query failed", exc_info=True)
            return []
        return [e.get("textPayload", "") for e in entries if e.get("textPayload")]

    def create_local_slice(
        self,
        slice_id: str,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        raise RuntimeError("create_local_slice is not supported in CLOUD mode")

    def get_local_slices(self, labels: dict[str, str] | None = None) -> list[LocalSliceHandle]:
        raise RuntimeError("get_local_slices is not supported in CLOUD mode")

    def shutdown(self) -> None:
        self._api.close()
