# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import google.api_core.exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.requests
import httpx
from google.cloud import tpu_v2alpha1

from iris.cluster.providers.types import (
    InfraError,
    QuotaExhaustedError,
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

# REST API base URLs
_TPU_BASE = "https://tpu.googleapis.com/v2"
_COMPUTE_BASE = "https://compute.googleapis.com/compute/v1"
_LOGGING_BASE = "https://logging.googleapis.com/v2"

# HTTP/auth constants
_REFRESH_MARGIN = 300  # seconds before expiry to refresh token
_DEFAULT_TIMEOUT = 120  # seconds
_OPERATION_POLL_INTERVAL = 2  # seconds between operation status polls
_OPERATION_TIMEOUT = 600  # seconds to wait for an operation to complete

# google.rpc.Code value for RESOURCE_EXHAUSTED. Used to classify LRO failures
# where the initial HTTP response was 200 but the async operation ended with a
# quota/stockout error (e.g. "no more capacity in the zone").
_RPC_CODE_RESOURCE_EXHAUSTED = 8


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
    def vm_delete(self, name: str, zone: str, *, wait: bool = False) -> None: ...
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
# CloudGcpService — REST API implementation
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
    """GcpService backed by GCP REST APIs. Used in CLOUD mode."""

    def __init__(self, project_id: str, http_client: httpx.Client | None = None) -> None:
        self._project_id = project_id
        self._client = http_client if http_client is not None else httpx.Client(timeout=_DEFAULT_TIMEOUT)
        self._creds: google.auth.credentials.Credentials | None = None
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._valid_zones: set[str] = set(KNOWN_GCP_ZONES)
        self._valid_accelerator_types: set[str] = set(KNOWN_TPU_TYPES)
        self._tpu_alpha_client_cached: tpu_v2alpha1.TpuClient | None = None

    @property
    def _tpu_alpha_client(self) -> tpu_v2alpha1.TpuClient:
        if self._tpu_alpha_client_cached is None:
            self._tpu_alpha_client_cached = tpu_v2alpha1.TpuClient()
        return self._tpu_alpha_client_cached

    @property
    def mode(self) -> ServiceMode:
        return ServiceMode.CLOUD

    @property
    def project_id(self) -> str:
        return self._project_id

    # ========================================================================
    # HTTP helpers (auth, errors, pagination, operation polling)
    # ========================================================================

    def _headers(self) -> dict[str, str]:
        if self._token is None or time.monotonic() >= self._expires_at:
            self._refresh_token()
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _refresh_token(self) -> None:
        if self._creds is None:
            self._creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._creds.refresh(google.auth.transport.requests.Request())
        self._token = self._creds.token
        now = time.monotonic()
        if self._creds.expiry is not None:
            self._expires_at = now + (self._creds.expiry.timestamp() - time.time()) - _REFRESH_MARGIN
        else:
            self._expires_at = now + _REFRESH_MARGIN

    def _classify_response(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
            error = body.get("error", {})
            message = error.get("message", resp.text)
            status = error.get("status", "")
            code = error.get("code", resp.status_code)
        except (json.JSONDecodeError, AttributeError):
            message = resp.text
            status = ""
            code = resp.status_code

        if code == 404 or status == "NOT_FOUND":
            raise ResourceNotFoundError(message)
        if code == 429 or status in ("RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"):
            raise QuotaExhaustedError(message)
        raise InfraError(f"GCP API error {code}: {message}")

    def _paginate(self, url: str, items_key: str, params: dict[str, str] | None = None) -> list[dict]:
        results: list[dict] = []
        p = dict(params or {})
        while True:
            resp = self._client.get(url, headers=self._headers(), params=p)
            self._classify_response(resp)
            data = resp.json()
            results.extend(data.get(items_key, []))
            token = data.get("nextPageToken")
            if not token:
                break
            p["pageToken"] = token
        return results

    def _paginate_raw(self, url: str, params: dict[str, str] | None = None) -> list[dict]:
        pages: list[dict] = []
        p = dict(params or {})
        while True:
            resp = self._client.get(url, headers=self._headers(), params=p)
            self._classify_response(resp)
            data = resp.json()
            pages.append(data)
            token = data.get("nextPageToken")
            if not token:
                break
            p["pageToken"] = token
        return pages

    def _wait_zone_operation(self, zone: str, operation_name: str, timeout: float = _OPERATION_TIMEOUT) -> dict:
        url = f"{_COMPUTE_BASE}/projects/{self._project_id}/zones/{zone}/operations/{operation_name}"
        deadline = time.monotonic() + timeout
        while True:
            resp = self._client.get(url, headers=self._headers())
            self._classify_response(resp)
            data = resp.json()
            if data.get("status") == "DONE":
                if "error" in data:
                    errors = data["error"].get("errors", [])
                    msg = "; ".join(e.get("message", str(e)) for e in errors)
                    raise InfraError(f"Operation {operation_name} failed: {msg}")
                return data
            if time.monotonic() >= deadline:
                raise InfraError(f"Operation {operation_name} timed out after {timeout}s")
            time.sleep(_OPERATION_POLL_INTERVAL)

    def _wait_tpu_operation(self, operation_name: str, timeout: float = _OPERATION_TIMEOUT) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            data = self._get_tpu_operation(operation_name)
            if data.get("done"):
                if "error" in data:
                    self._raise_tpu_operation_error(data["error"])
                return data
            if time.monotonic() >= deadline:
                raise InfraError(f"TPU operation {operation_name} timed out after {timeout}s")
            time.sleep(_OPERATION_POLL_INTERVAL)

    def _get_tpu_operation(self, operation_name: str) -> dict:
        url = f"{_TPU_BASE}/{operation_name}"
        resp = self._client.get(url, headers=self._headers())
        self._classify_response(resp)
        return resp.json()

    def _raise_tpu_operation_error(self, error: dict) -> None:
        msg = error.get("message", str(error))
        # Zone stockouts ("no more capacity in the zone ...") come back as
        # RESOURCE_EXHAUSTED on the LRO rather than on the initial HTTP response.
        # Surface them as QuotaExhaustedError so the autoscaler treats them like
        # any other quota hit (terse warning + backoff, no stack trace).
        if error.get("code") == _RPC_CODE_RESOURCE_EXHAUSTED:
            raise QuotaExhaustedError(msg)
        raise InfraError(f"TPU operation failed: {msg}")

    # ========================================================================
    # Low-level REST helpers
    # ========================================================================

    def _tpu_parent(self, zone: str) -> str:
        return f"projects/{self._project_id}/locations/{zone}"

    def _instance_url(self, zone: str, name: str = "") -> str:
        path = f"{_COMPUTE_BASE}/projects/{self._project_id}/zones/{zone}/instances"
        if name:
            path += f"/{name}"
        return path

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
        network_config: dict = {"enableExternalIps": True}
        if request.network:
            network_config["network"] = request.network
        if request.subnetwork:
            network_config["subnetwork"] = request.subnetwork
        body["networkConfig"] = network_config

        logger.info("Creating TPU: %s (type=%s, zone=%s)", request.name, request.accelerator_type, request.zone)

        # POST to create, probe the LRO once, then describe the node if it is
        # already visible. Do not block slice bookkeeping on the full TPU LRO.
        url = f"{_TPU_BASE}/{self._tpu_parent(request.zone)}/nodes"
        resp = self._client.post(url, params={"nodeId": request.name}, headers=self._headers(), json=body)
        self._classify_response(resp)
        data = resp.json()
        op_name = data.get("name", "")
        if op_name and "/operations/" in op_name:
            op = self._get_tpu_operation(op_name)
            if op.get("done") and "error" in op:
                existing = self.tpu_describe(request.name, request.zone)
                if existing is not None:
                    logger.warning(
                        "TPU operation %s reported failure after TPU %s became visible: %s",
                        op_name,
                        request.name,
                        op["error"].get("message", op["error"]),
                    )
                    return existing
                self._raise_tpu_operation_error(op["error"])

        existing = self.tpu_describe(request.name, request.zone)
        if existing is not None:
            return existing

        return TpuInfo(
            name=request.name,
            state="CREATING",
            accelerator_type=request.accelerator_type,
            zone=request.zone,
            labels=dict(request.labels),
            metadata=dict(request.metadata),
            service_account=request.service_account,
            network_endpoints=[],
            external_network_endpoints=[],
            created_at=Timestamp.now(),
        )

    def tpu_delete(self, name: str, zone: str) -> None:
        logger.info("Deleting TPU (async): %s", name)
        url = f"{_TPU_BASE}/{self._tpu_parent(zone)}/nodes/{name}"
        resp = self._client.delete(url, headers=self._headers())
        if resp.status_code != 404:
            self._classify_response(resp)

    def _tpu_get(self, name: str, zone: str) -> dict:
        url = f"{_TPU_BASE}/{self._tpu_parent(zone)}/nodes/{name}"
        resp = self._client.get(url, headers=self._headers())
        self._classify_response(resp)
        return resp.json()

    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None:
        try:
            tpu_data = self._tpu_get(name, zone)
        except ResourceNotFoundError:
            return None
        except InfraError:
            logger.warning("Failed to describe TPU %s", name, exc_info=True)
            return None
        return _parse_tpu_info(tpu_data, zone)

    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]:
        results: list[TpuInfo] = []
        # Use locations/- for project-wide listing (matches gcloud --zone=-).
        zone_list = zones if zones else ["-"]

        for zone in zone_list:
            try:
                items = self._paginate(f"{_TPU_BASE}/{self._tpu_parent(zone)}/nodes", "nodes")
            except InfraError:
                logger.warning("Failed to list TPUs in zone %s", zone, exc_info=True)
                continue
            for tpu_data in items:
                if labels and not _labels_match(tpu_data.get("labels", {}), labels):
                    continue
                tpu_zone = zone
                raw_name = tpu_data.get("name", "")
                if "/" in raw_name:
                    parts = raw_name.split("/")
                    if len(parts) >= 4:
                        tpu_zone = parts[3]
                results.append(_parse_tpu_info(tpu_data, tpu_zone))

        return results

    # ========================================================================
    # Queued resource operations (for reserved TPUs via typed v2alpha1 client)
    # ========================================================================

    def _qr_api_error(self, exc: google.api_core.exceptions.GoogleAPICallError) -> InfraError:
        """Map google-cloud-tpu exceptions to the GcpService error hierarchy."""
        if isinstance(exc, google.api_core.exceptions.NotFound):
            return ResourceNotFoundError(str(exc))
        if isinstance(exc, google.api_core.exceptions.ResourceExhausted):
            return QuotaExhaustedError(str(exc))
        return InfraError(str(exc))

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        validate_tpu_create(request, self._valid_zones, self._valid_accelerator_types)

        node = tpu_v2alpha1.Node(
            accelerator_type=request.accelerator_type,
            runtime_version=request.runtime_version,
            labels=request.labels or {},
            metadata=request.metadata or {},
            network_config=tpu_v2alpha1.NetworkConfig(
                enable_external_ips=True,
                network=request.network or "",
                subnetwork=request.subnetwork or "",
            ),
        )
        if request.service_account:
            node.service_account = tpu_v2alpha1.ServiceAccount(email=request.service_account)

        queued_resource = tpu_v2alpha1.QueuedResource(
            tpu=tpu_v2alpha1.QueuedResource.Tpu(
                node_spec=[
                    tpu_v2alpha1.QueuedResource.Tpu.NodeSpec(
                        parent=f"projects/{self._project_id}/locations/{request.zone}",
                        node_id=request.name,
                        node=node,
                    )
                ]
            ),
            guaranteed=tpu_v2alpha1.QueuedResource.Guaranteed(reserved=True),
        )

        parent = f"projects/{self._project_id}/locations/{request.zone}"
        logger.info(
            "Creating queued resource: %s (type=%s, zone=%s)",
            request.name,
            request.accelerator_type,
            request.zone,
        )
        try:
            self._tpu_alpha_client.create_queued_resource(
                parent=parent,
                queued_resource=queued_resource,
                queued_resource_id=request.name,
            )
        except google.api_core.exceptions.GoogleAPICallError as exc:
            raise self._qr_api_error(exc) from exc

    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None:
        qr_name = f"projects/{self._project_id}/locations/{zone}/queuedResources/{name}"
        try:
            qr = self._tpu_alpha_client.get_queued_resource(name=qr_name)
        except google.api_core.exceptions.NotFound:
            return None
        except google.api_core.exceptions.GoogleAPICallError as exc:
            raise self._qr_api_error(exc) from exc
        state = qr.state.state.name if qr.state else "UNKNOWN"
        return QueuedResourceInfo(name=name, state=state, zone=zone)

    def queued_resource_delete(self, name: str, zone: str) -> None:
        logger.info("Deleting queued resource (force): %s", name)
        qr_name = f"projects/{self._project_id}/locations/{zone}/queuedResources/{name}"
        try:
            self._tpu_alpha_client.delete_queued_resource(
                request=tpu_v2alpha1.DeleteQueuedResourceRequest(name=qr_name, force=True)
            )
        except google.api_core.exceptions.NotFound:
            pass
        except google.api_core.exceptions.GoogleAPICallError as exc:
            raise self._qr_api_error(exc) from exc

    def queued_resource_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[QueuedResourceInfo]:
        zone_list = zones if zones else ["-"]
        results: list[QueuedResourceInfo] = []
        for zone in zone_list:
            parent = f"projects/{self._project_id}/locations/{zone}"
            try:
                for qr in self._tpu_alpha_client.list_queued_resources(parent=parent):
                    qr_short_name = qr.name.rsplit("/", 1)[-1]
                    state = qr.state.state.name if qr.state else "UNKNOWN"
                    node_specs = list(qr.tpu.node_spec) if qr.tpu else []
                    item_labels = dict(node_specs[0].node.labels) if node_specs and node_specs[0].node else {}
                    if labels and not all(item_labels.get(k) == v for k, v in labels.items()):
                        continue
                    # When listing with a wildcard zone ("-"), extract the actual zone
                    # from the full resource name so handles don't get zone="-".
                    actual_zone = zone
                    if zone == "-":
                        parts = qr.name.split("/")
                        loc_idx = next((i for i, p in enumerate(parts) if p == "locations"), -1)
                        if loc_idx >= 0 and loc_idx + 1 < len(parts):
                            actual_zone = parts[loc_idx + 1]
                    results.append(
                        QueuedResourceInfo(name=qr_short_name, state=state, zone=actual_zone, labels=item_labels)
                    )
            except google.api_core.exceptions.GoogleAPICallError:
                logger.warning("Failed to list queued resources in %s", zone, exc_info=True)
                continue
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
            # POST to insert, wait for zone operation
            url = self._instance_url(request.zone)
            resp = self._client.post(url, headers=self._headers(), json=body)
            self._classify_response(resp)
            data = resp.json()
            op_name = data.get("name", "")
            if op_name:
                self._wait_zone_operation(request.zone, op_name)
        except InfraError as e:
            if "already exists" not in str(e).lower():
                raise

        info = self.vm_describe(request.name, request.zone)
        if info is None:
            raise InfraError(f"VM {request.name} created but could not be described")
        return info

    def vm_delete(self, name: str, zone: str, *, wait: bool = False) -> None:
        logger.info("Deleting VM: %s", name)
        url = self._instance_url(zone, name)
        resp = self._client.delete(url, headers=self._headers())
        if resp.status_code == 404:
            return
        self._classify_response(resp)
        if wait:
            op_name = resp.json().get("name", "")
            if op_name:
                self._wait_zone_operation(zone, op_name)

    def vm_reset(self, name: str, zone: str) -> None:
        logger.info("Resetting VM: %s", name)
        url = self._instance_url(zone, name) + "/reset"
        resp = self._client.post(url, headers=self._headers())
        self._classify_response(resp)

    def _instance_get(self, name: str, zone: str) -> dict:
        url = self._instance_url(zone, name)
        resp = self._client.get(url, headers=self._headers())
        self._classify_response(resp)
        return resp.json()

    def vm_describe(self, name: str, zone: str) -> VmInfo | None:
        try:
            data = self._instance_get(name, zone)
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
            # Project-wide: aggregatedList, flatten across zones
            url = f"{_COMPUTE_BASE}/projects/{self._project_id}/aggregated/instances"
            params: dict[str, str] = {}
            if filter_str:
                params["filter"] = filter_str
            try:
                for page in self._paginate_raw(url, params):
                    for scope in page.get("items", {}).values():
                        for vm_data in scope.get("instances", []):
                            results.append(_parse_vm_info(vm_data))
            except InfraError:
                logger.warning("Failed to list instances", exc_info=True)
                return []
            return results

        for zone in zones:
            params = {}
            if filter_str:
                params["filter"] = filter_str
            try:
                items = self._paginate(self._instance_url(zone), "items", params)
            except InfraError:
                logger.warning("Failed to list instances in zone %s", zone, exc_info=True)
                continue
            for vm_data in items:
                results.append(_parse_vm_info(vm_data, fallback_zone=zone))

        return results

    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None:
        validate_labels(labels)
        logger.info("Updating labels on VM %s", name)
        data = self._instance_get(name, zone)
        current_labels = data.get("labels", {})
        current_labels.update(labels)
        fingerprint = data.get("labelFingerprint", "")
        url = self._instance_url(zone, name) + "/setLabels"
        resp = self._client.post(
            url,
            headers=self._headers(),
            json={"labels": current_labels, "labelFingerprint": fingerprint},
        )
        self._classify_response(resp)
        op_name = resp.json().get("name", "")
        if op_name:
            self._wait_zone_operation(zone, op_name)

    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None:
        logger.info("Setting metadata on VM %s", name)
        data = self._instance_get(name, zone)
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
        url = self._instance_url(zone, name) + "/setMetadata"
        resp = self._client.post(url, headers=self._headers(), json=body)
        self._classify_response(resp)
        op_name = resp.json().get("name", "")
        if op_name:
            self._wait_zone_operation(zone, op_name)

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        try:
            url = self._instance_url(zone, name) + "/serialPort"
            resp = self._client.get(url, headers=self._headers(), params={"start": str(start)})
            self._classify_response(resp)
            data = resp.json()
        except InfraError:
            logger.warning("Failed to get serial port output for %s", name, exc_info=True)
            return ""
        return data.get("contents", "")

    def logging_read(self, filter_str: str, limit: int = 200) -> list[str]:
        try:
            url = f"{_LOGGING_BASE}/entries:list"
            body = {
                "resourceNames": [f"projects/{self._project_id}"],
                "filter": filter_str,
                "pageSize": min(limit, 1000),
                "orderBy": "timestamp desc",
            }
            resp = self._client.post(url, headers=self._headers(), json=body, timeout=30)
            self._classify_response(resp)
            entries = resp.json().get("entries", [])
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
        self._client.close()
