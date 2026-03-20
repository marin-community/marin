# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GcpService implementations: DryRunGcpService, LocalGcpService, CloudGcpService.

DryRunGcpService: In-memory fake with validation and failure injection for testing.
LocalGcpService: Extends DryRunGcpService with real local worker thread spawning.
CloudGcpService: Shells out to gcloud CLI for real GCP operations.

Factory: create_gcp_service(mode, ...) returns the appropriate implementation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from iris.cluster.net import find_free_port
from iris.cluster.platform.base import (
    Labels,
    PlatformError,
    QuotaExhaustedError,
    ResourceNotFoundError,
)
from iris.cluster.platform.gcp_service import (
    GcpService,
    TpuCreateRequest,
    TpuInfo,
    VmCreateRequest,
    VmInfo,
)
from iris.cluster.platform.local import LocalSliceHandle
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import TPU_TOPOLOGIES
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

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


def _format_labels(labels: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in labels.items())


def _build_label_filter(labels: dict[str, str]) -> str:
    parts = [f"labels.{k}={v}" for k, v in labels.items()]
    return " AND ".join(parts)


def _classify_gcloud_error(stderr: str) -> PlatformError:
    lower = stderr.lower()
    if "quota" in lower or "insufficient" in lower or "resource_exhausted" in lower:
        return QuotaExhaustedError(stderr)
    return PlatformError(stderr)


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

    return TpuInfo(
        name=name,
        state=tpu_data.get("state", "UNKNOWN"),
        accelerator_type=accelerator_type,
        zone=zone,
        labels=tpu_data.get("labels", {}),
        metadata=tpu_data.get("metadata", {}),
        network_endpoints=ips,
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

    return VmInfo(
        name=vm_data.get("name", ""),
        status=vm_data.get("status", "UNKNOWN"),
        zone=zone,
        internal_ip=internal_ip,
        external_ip=external_ip,
        labels=vm_data.get("labels", {}),
        metadata=metadata,
        created_at=_parse_vm_created_at(vm_data),
    )


def _validate_resource_name(name: str, resource_kind: str) -> None:
    if len(name) > MAX_RESOURCE_NAME_LENGTH:
        raise ValueError(f"{resource_kind} name exceeds {MAX_RESOURCE_NAME_LENGTH} chars: {name!r}")
    if not _RESOURCE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {resource_kind} name (must be lowercase alphanumeric/hyphens, " f"start with letter): {name!r}"
        )


def _validate_labels(labels: dict[str, str]) -> None:
    for key, val in labels.items():
        if not _LABEL_KEY_RE.match(key):
            raise ValueError(f"Invalid label key: {key!r}")
        if not _LABEL_VALUE_RE.match(val):
            raise ValueError(f"Invalid label value for {key!r}: {val!r}")


def _validate_zone(zone: str, valid_zones: set[str]) -> None:
    if zone not in valid_zones:
        raise PlatformError(f"Zone {zone!r} not available")


def _validate_tpu_create(request: TpuCreateRequest, valid_zones: set[str], valid_accelerator_types: set[str]) -> None:
    _validate_resource_name(request.name, "TPU")
    _validate_zone(request.zone, valid_zones)
    if request.accelerator_type not in valid_accelerator_types:
        raise ResourceNotFoundError(f"Unknown accelerator type: {request.accelerator_type!r}")
    if not request.runtime_version:
        raise ValueError("runtime_version must be non-empty")
    _validate_labels(request.labels)


def _validate_vm_create(request: VmCreateRequest, valid_zones: set[str]) -> None:
    _validate_resource_name(request.name, "VM")
    _validate_zone(request.zone, valid_zones)
    if request.disk_size_gb <= 0:
        raise ValueError(f"disk_size_gb must be positive, got {request.disk_size_gb}")
    _validate_labels(request.labels)


# ============================================================================
# DryRunGcpService
# ============================================================================


class DryRunGcpService:
    """In-memory GCP fake with validation and failure injection for testing."""

    DEFAULT_QUOTA = 100

    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id
        self._valid_zones: set[str] = set(KNOWN_GCP_ZONES)
        self._valid_accelerator_types: set[str] = set(KNOWN_TPU_TYPES)

        self._tpus: dict[tuple[str, str], TpuInfo] = {}
        self._vms: dict[tuple[str, str], VmInfo] = {}

        self._injected_failures: dict[str, PlatformError] = {}
        self._zone_quotas: dict[str, int] = {}
        self._vm_zone_quotas: dict[str, int] = {}
        self._available_types_by_zone: dict[str, set[str]] | None = None

    @property
    def mode(self) -> ServiceMode:
        return ServiceMode.DRY_RUN

    @property
    def project_id(self) -> str:
        return self._project_id

    # -- Failure injection --

    def inject_failure(self, operation: str, error: PlatformError) -> None:
        """Make the next call to `operation` raise `error`, then auto-clear."""
        self._injected_failures[operation] = error

    def set_zone_quota(self, zone: str, max_tpus: int) -> None:
        self._zone_quotas[zone] = max_tpus

    def set_tpu_type_unavailable(self, accelerator_type: str) -> None:
        self._valid_accelerator_types.discard(accelerator_type)

    def add_tpu_type(self, accelerator_type: str) -> None:
        self._valid_accelerator_types.add(accelerator_type)

    def set_vm_zone_quota(self, zone: str, max_vms: int) -> None:
        self._vm_zone_quotas[zone] = max_vms

    def set_available_types_by_zone(self, mapping: dict[str, set[str]]) -> None:
        self._available_types_by_zone = mapping

    def advance_tpu_state(self, name: str, zone: str, state: str = "READY") -> None:
        key = (name, zone)
        if key not in self._tpus:
            raise ValueError(f"TPU {name!r} not found in {zone}")
        self._tpus[key] = dataclasses.replace(self._tpus[key], state=state)

    def _check_injected_failure(self, operation: str) -> None:
        err = self._injected_failures.pop(operation, None)
        if err is not None:
            raise err

    # -- TPU operations --

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        self._check_injected_failure("tpu_create")
        _validate_tpu_create(request, self._valid_zones, self._valid_accelerator_types)
        return self._tpu_create_in_memory(request)

    def _tpu_create_in_memory(self, request: TpuCreateRequest) -> TpuInfo:
        if (request.name, request.zone) in self._tpus:
            raise PlatformError(f"TPU {request.name!r} already exists in {request.zone}")

        zone_count = sum(1 for (_, z) in self._tpus if z == request.zone)
        max_quota = self._zone_quotas.get(request.zone, self.DEFAULT_QUOTA)
        if zone_count >= max_quota:
            raise QuotaExhaustedError(f"Quota exhausted in {request.zone}")

        if self._available_types_by_zone is not None:
            zone_types = self._available_types_by_zone.get(request.zone, set())
            if request.accelerator_type not in zone_types:
                raise QuotaExhaustedError(
                    f"Accelerator type {request.accelerator_type!r} not available in {request.zone}"
                )

        from iris.cluster.types import get_tpu_topology

        try:
            topo = get_tpu_topology(request.accelerator_type)
            vm_count = topo.vm_count
        except ValueError:
            logger.debug("Unknown accelerator type %r; TPU topology not available", request.accelerator_type)
            vm_count = 1

        seq = len(self._tpus)
        endpoints = [f"10.0.{seq}.{i}" for i in range(vm_count)]

        info = TpuInfo(
            name=request.name,
            state="CREATING",
            accelerator_type=request.accelerator_type,
            zone=request.zone,
            labels=dict(request.labels),
            metadata=dict(request.metadata),
            network_endpoints=endpoints,
            created_at=Timestamp.now(),
        )
        self._tpus[(request.name, request.zone)] = info
        return info

    def tpu_delete(self, name: str, zone: str) -> None:
        self._check_injected_failure("tpu_delete")
        self._tpus.pop((name, zone), None)

    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None:
        self._check_injected_failure("tpu_describe")
        return self._tpus.get((name, zone))

    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]:
        self._check_injected_failure("tpu_list")
        results: list[TpuInfo] = []
        for (_, z), info in self._tpus.items():
            if zones and z not in zones:
                continue
            if labels and not all(info.labels.get(k) == v for k, v in labels.items()):
                continue
            results.append(info)
        return results

    # -- VM operations --

    def vm_create(self, request: VmCreateRequest) -> VmInfo:
        self._check_injected_failure("vm_create")
        _validate_vm_create(request, self._valid_zones)
        return self._vm_create_in_memory(request)

    def _vm_create_in_memory(self, request: VmCreateRequest) -> VmInfo:
        if (request.name, request.zone) in self._vms:
            raise PlatformError(f"VM {request.name!r} already exists in {request.zone}")

        vm_zone_count = sum(1 for (_, z) in self._vms if z == request.zone)
        max_vm_quota = self._vm_zone_quotas.get(request.zone, self.DEFAULT_QUOTA)
        if vm_zone_count >= max_vm_quota:
            raise QuotaExhaustedError(f"VM quota exhausted in {request.zone}")

        seq = len(self._vms)
        info = VmInfo(
            name=request.name,
            status="RUNNING",
            zone=request.zone,
            internal_ip=f"10.1.{seq}.1",
            external_ip=None,
            labels=dict(request.labels),
            metadata=dict(request.metadata),
            created_at=Timestamp.now(),
        )
        self._vms[(request.name, request.zone)] = info
        return info

    def vm_delete(self, name: str, zone: str) -> None:
        self._check_injected_failure("vm_delete")
        self._vms.pop((name, zone), None)

    def vm_reset(self, name: str, zone: str) -> None:
        self._check_injected_failure("vm_reset")

    def vm_describe(self, name: str, zone: str) -> VmInfo | None:
        self._check_injected_failure("vm_describe")
        return self._vms.get((name, zone))

    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]:
        self._check_injected_failure("vm_list")
        results: list[VmInfo] = []
        for (_, z), info in self._vms.items():
            if zones and z not in zones:
                continue
            if labels and not all(info.labels.get(k) == v for k, v in labels.items()):
                continue
            results.append(info)
        return results

    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None:
        self._check_injected_failure("vm_update_labels")
        _validate_labels(labels)
        vm = self._vms.get((name, zone))
        if vm is None:
            raise PlatformError(f"VM {name!r} not found in zone {zone!r}")
        vm.labels.update(labels)

    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None:
        self._check_injected_failure("vm_set_metadata")
        vm = self._vms.get((name, zone))
        if vm is None:
            raise PlatformError(f"VM {name!r} not found in zone {zone!r}")
        vm.metadata.update(metadata)

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        self._check_injected_failure("vm_get_serial_port_output")
        return ""

    # -- Local slice operations (not supported in DRY_RUN) --

    def create_local_slice(
        self,
        slice_id: str,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        raise ValueError("DRY_RUN mode does not support local slices")

    def get_local_slices(self, labels: dict[str, str] | None = None) -> list[LocalSliceHandle]:
        return []

    def shutdown(self) -> None:
        pass


# ============================================================================
# LocalGcpService
# ============================================================================


class LocalGcpService(DryRunGcpService):
    """Extends DryRunGcpService with local worker thread spawning."""

    def __init__(
        self,
        project_id: str = "",
        controller_address: str | None = None,
        cache_path: Path | None = None,
        fake_bundle: Path | None = None,
        port_allocator: PortAllocator | None = None,
        threads: ThreadContainer | None = None,
        worker_attributes_by_group: dict[str, dict[str, str | int | float]] | None = None,
        gpu_count_by_group: dict[str, int] | None = None,
        storage_prefix: str = "",
        label_prefix: str = "iris",
    ) -> None:
        super().__init__(project_id=project_id)
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._port_allocator = port_allocator
        self._threads = threads or ThreadContainer(name="gcp-service-local")
        self._worker_attributes_by_group = worker_attributes_by_group or {}
        self._gpu_count_by_group = gpu_count_by_group or {}
        self._storage_prefix = storage_prefix
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix)
        self._local_slices: dict[str, LocalSliceHandle] = {}

    @property
    def mode(self) -> ServiceMode:
        return ServiceMode.LOCAL

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        self._check_injected_failure("tpu_create")
        # LOCAL mode: skip strict GCP zone/accelerator validation
        _validate_resource_name(request.name, "TPU")
        _validate_labels(request.labels)
        return self._tpu_create_in_memory(request)

    def tpu_delete(self, name: str, zone: str) -> None:
        self._check_injected_failure("tpu_delete")
        self._tpus.pop((name, zone), None)
        local_slice = self._local_slices.pop(name, None)
        if local_slice is not None:
            local_slice.terminate()

    def vm_create(self, request: VmCreateRequest) -> VmInfo:
        self._check_injected_failure("vm_create")
        # LOCAL mode: skip strict GCP zone/disk validation
        _validate_resource_name(request.name, "VM")
        _validate_labels(request.labels)
        return self._vm_create_in_memory(request)

    def create_local_slice(
        self,
        slice_id: str,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        num_vms = config.num_vms or 1

        if self._controller_address is not None:
            handle = self._create_slice_with_workers(slice_id, num_vms, config, worker_config)
        else:
            vm_ids = [f"{slice_id}-worker-{i}" for i in range(num_vms)]
            addresses = [f"localhost:{9000 + i}" for i in range(num_vms)]
            handle = LocalSliceHandle(
                _slice_id=slice_id,
                _vm_ids=vm_ids,
                _addresses=addresses,
                _labels=dict(config.labels),
                _created_at=Timestamp.now(),
                _label_prefix=self._label_prefix,
            )

        self._local_slices[slice_id] = handle
        return handle

    def _create_slice_with_workers(
        self,
        slice_id: str,
        num_vms: int,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        from iris.cluster.bundle import BundleStore
        from iris.cluster.runtime.process import ProcessRuntime
        from iris.cluster.types import get_tpu_topology
        from iris.cluster.worker.env_probe import FixedEnvironmentProvider, HardwareProbe, build_worker_metadata
        from iris.cluster.worker.worker import Worker, WorkerConfig

        if self._cache_path is None:
            raise ValueError("cache_path required for LOCAL mode")
        if self._threads is None:
            raise ValueError("threads required for LOCAL mode")

        workers: list[Worker] = []
        vm_ids: list[str] = []
        addresses: list[str] = []

        worker_count = num_vms
        is_tpu = config.accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        is_gpu = config.accelerator_type == config_pb2.ACCELERATOR_TYPE_GPU
        if is_tpu and config.accelerator_variant:
            try:
                topo = get_tpu_topology(config.accelerator_variant)
                worker_count = topo.vm_count
            except ValueError:
                logger.debug("Unknown accelerator variant %r; TPU topology not available", config.accelerator_variant)

        for tpu_worker_id in range(worker_count):
            worker_id = f"worker-{slice_id}-{tpu_worker_id}-{uuid.uuid4().hex[:8]}"
            bundle_store = BundleStore(
                storage_dir=str(self._cache_path / f"bundles-{worker_id}"),
                controller_address=self._controller_address,
            )
            container_runtime = ProcessRuntime(cache_dir=self._cache_path / worker_id)
            worker_port = find_free_port()

            extra_attrs: dict[str, str] = {}
            sg_name = config.labels.get(self._iris_labels.iris_scale_group, "")
            if sg_name and sg_name in self._worker_attributes_by_group:
                for k, v in self._worker_attributes_by_group[sg_name].items():
                    extra_attrs.setdefault(k, str(v))

            if worker_config is not None:
                for k, v in worker_config.worker_attributes.items():
                    extra_attrs.setdefault(k, v)

            extra_attrs.setdefault("region", "local")
            preemptible = extra_attrs.pop("preemptible", "false").lower() == "true"

            gpu_count = 0
            if is_gpu:
                gpu_count = self._gpu_count_by_group.get(sg_name, 1)

            hardware = HardwareProbe(
                hostname="local",
                ip_address="127.0.0.1",
                cpu_count=1000,
                memory_bytes=1000 * 1024**3,
                disk_bytes=100 * 1024**3,
                gpu_count=0,
                gpu_name="",
                gpu_memory_mb=0,
                tpu_name=slice_id if is_tpu else "",
                tpu_type=config.accelerator_variant if is_tpu else "",
                tpu_worker_hostnames="",
                tpu_worker_id=str(tpu_worker_id) if is_tpu else "",
                tpu_chips_per_host_bounds="",
            )

            metadata = build_worker_metadata(
                hardware=hardware,
                accelerator_type=config.accelerator_type,
                accelerator_variant=config.accelerator_variant,
                gpu_count_override=gpu_count,
                preemptible=preemptible,
                worker_attributes=extra_attrs,
            )

            env_provider = FixedEnvironmentProvider(metadata)

            wc = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=self._cache_path / worker_id,
                controller_address=self._controller_address,
                worker_id=worker_id,
                default_task_image="process-runtime-unused",
                poll_interval=Duration.from_seconds(0.1),
                storage_prefix=self._storage_prefix,
                auth_token=worker_config.auth_token if worker_config is not None else "",
            )
            worker_threads = self._threads.create_child(f"worker-{worker_id}")
            worker = Worker(
                wc,
                bundle_store=bundle_store,
                container_runtime=container_runtime,
                environment_provider=env_provider,
                port_allocator=self._port_allocator,
                threads=worker_threads,
            )
            worker.start()
            workers.append(worker)
            vm_ids.append(worker_id)
            addresses.append(f"127.0.0.1:{worker_port}")

        logger.info(
            "LocalGcpService created slice %s with %d workers",
            slice_id,
            len(workers),
        )

        return LocalSliceHandle(
            _slice_id=slice_id,
            _vm_ids=vm_ids,
            _addresses=addresses,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _workers=workers,
        )

    def get_local_slices(self, labels: dict[str, str] | None = None) -> list[LocalSliceHandle]:
        results = list(self._local_slices.values())
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def shutdown(self) -> None:
        for s in list(self._local_slices.values()):
            s.terminate()
        self._local_slices.clear()
        if self._threads is not None:
            self._threads.stop(timeout=Duration.from_seconds(5.0))


# ============================================================================
# CloudGcpService
# ============================================================================


class CloudGcpService:
    """Shells out to gcloud CLI for real GCP operations."""

    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id
        self._valid_zones: set[str] = set(KNOWN_GCP_ZONES)
        self._valid_accelerator_types: set[str] = set(KNOWN_TPU_TYPES)

    @property
    def mode(self) -> ServiceMode:
        return ServiceMode.CLOUD

    @property
    def project_id(self) -> str:
        return self._project_id

    # -- TPU operations --

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        _validate_tpu_create(request, self._valid_zones, self._valid_accelerator_types)

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            request.name,
            f"--zone={request.zone}",
            f"--project={self._project_id}",
            f"--accelerator-type={request.accelerator_type}",
            f"--version={request.runtime_version}",
            "--format=json",
        ]

        if request.labels:
            cmd.extend(["--labels", _format_labels(request.labels)])
        if request.preemptible:
            cmd.append("--preemptible")
        if request.service_account:
            cmd.append(f"--service-account={request.service_account}")
        if request.network:
            cmd.append(f"--network={request.network}")
        if request.subnetwork:
            cmd.append(f"--subnetwork={request.subnetwork}")

        metadata_files: dict[str, str] = {}
        inline_metadata: dict[str, str] = {}
        for k, v in request.metadata.items():
            if len(v) > 256:
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
                f.write(v)
                f.close()
                metadata_files[k] = f.name
            else:
                inline_metadata[k] = v

        if inline_metadata:
            metadata_str = ",".join(f"{k}={v}" for k, v in inline_metadata.items())
            cmd.append(f"--metadata={metadata_str}")
        if metadata_files:
            file_str = ",".join(f"{k}={path}" for k, path in metadata_files.items())
            cmd.append(f"--metadata-from-file={file_str}")

        logger.info("Creating TPU: %s (type=%s, zone=%s)", request.name, request.accelerator_type, request.zone)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            for path in metadata_files.values():
                os.unlink(path)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

        if result.stdout.strip():
            tpu_data = json.loads(result.stdout)
            return _parse_tpu_info(tpu_data, request.zone)

        info = self.tpu_describe(request.name, request.zone)
        if info is None:
            raise PlatformError(f"TPU {request.name} created but could not be described")
        return info

    def tpu_delete(self, name: str, zone: str) -> None:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            f"--zone={zone}",
            f"--project={self._project_id}",
            "--quiet",
            "--async",
        ]
        logger.info("Deleting TPU (async): %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip()
            if "not found" not in error.lower():
                raise _classify_gcloud_error(error)

    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            f"--zone={zone}",
            f"--project={self._project_id}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip().lower()
            if "not found" in error or "could not be found" in error:
                return None
            logger.warning("Failed to describe TPU %s: %s", name, result.stderr.strip())
            return None

        tpu_data = json.loads(result.stdout)
        return _parse_tpu_info(tpu_data, zone)

    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]:
        results: list[TpuInfo] = []
        zone_list = zones if zones else ["-"]

        for zone in zone_list:
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
                continue
            if not result.stdout.strip():
                continue

            for tpu_data in json.loads(result.stdout):
                tpu_zone = zone
                raw_name = tpu_data.get("name", "")
                if zone == "-" and "/" in raw_name:
                    parts = raw_name.split("/")
                    if len(parts) >= 4:
                        tpu_zone = parts[3]
                results.append(_parse_tpu_info(tpu_data, tpu_zone))

        return results

    # -- VM operations --

    def vm_create(self, request: VmCreateRequest) -> VmInfo:
        _validate_vm_create(request, self._valid_zones)

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            request.name,
            f"--project={self._project_id}",
            f"--zone={request.zone}",
            f"--machine-type={request.machine_type}",
            f"--boot-disk-size={request.disk_size_gb}GB",
            f"--boot-disk-type={request.boot_disk_type}",
            f"--image-family={request.image_family}",
            f"--image-project={request.image_project}",
            "--scopes=cloud-platform",
            "--format=json",
        ]

        if request.labels:
            cmd.append(f"--labels={_format_labels(request.labels)}")

        metadata_files: dict[str, str] = {}
        all_metadata = dict(request.metadata)
        if request.startup_script:
            all_metadata["startup-script"] = request.startup_script

        inline_metadata: dict[str, str] = {}
        for k, v in all_metadata.items():
            if len(v) > 256:
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
                f.write(v)
                f.close()
                metadata_files[k] = f.name
            else:
                inline_metadata[k] = v

        if inline_metadata:
            metadata_str = ",".join(f"{k}={v}" for k, v in inline_metadata.items())
            cmd.append(f"--metadata={metadata_str}")
        if metadata_files:
            file_str = ",".join(f"{k}={path}" for k, path in metadata_files.items())
            cmd.append(f"--metadata-from-file={file_str}")

        if request.service_account:
            cmd.append(f"--service-account={request.service_account}")

        logger.info("Creating VM: %s (zone=%s, type=%s)", request.name, request.zone, request.machine_type)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            for path in metadata_files.values():
                os.unlink(path)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "already exists" not in error_msg.lower():
                raise _classify_gcloud_error(error_msg)

        info = self.vm_describe(request.name, request.zone)
        if info is None:
            raise PlatformError(f"VM {request.name} created but could not be described")
        return info

    def vm_delete(self, name: str, zone: str) -> None:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "delete",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            "--quiet",
        ]
        logger.info("Deleting VM: %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip()
            if "not found" not in error.lower():
                raise _classify_gcloud_error(error)

    def vm_reset(self, name: str, zone: str) -> None:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "reset",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            "--quiet",
        ]
        logger.info("Resetting VM: %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

    def vm_describe(self, name: str, zone: str) -> VmInfo | None:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip().lower()
            if "not found" in error or "could not be found" in error:
                return None
            logger.warning("Failed to describe VM %s: %s", name, result.stderr.strip())
            return None

        data = json.loads(result.stdout)
        return _parse_vm_info(data, fallback_zone=zone)

    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]:
        results: list[VmInfo] = []

        if not zones:
            cmd = [
                "gcloud",
                "compute",
                "instances",
                "list",
                f"--project={self._project_id}",
                "--format=json",
            ]
            if labels:
                cmd.append(f"--filter={_build_label_filter(labels)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Failed to list instances: %s", result.stderr.strip())
                return []
            if not result.stdout.strip():
                return []
            for vm_data in json.loads(result.stdout):
                results.append(_parse_vm_info(vm_data))
            return results

        for zone in zones:
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
                continue
            if not result.stdout.strip():
                continue

            for vm_data in json.loads(result.stdout):
                results.append(_parse_vm_info(vm_data, fallback_zone=zone))

        return results

    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None:
        _validate_labels(labels)
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "update",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--update-labels={_format_labels(labels)}",
        ]
        logger.info("Updating labels on VM %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None:
        metadata_str = ",".join(f"{k}={v}" for k, v in metadata.items())
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "add-metadata",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--metadata={metadata_str}",
        ]
        logger.info("Setting metadata on VM %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "get-serial-port-output",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--start={start}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to get serial port output for %s: %s", name, result.stderr.strip())
            return ""
        return result.stdout

    # -- Local slice operations (not supported in CLOUD) --

    def create_local_slice(
        self,
        slice_id: str,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        raise ValueError("CLOUD mode does not support local slices")

    def get_local_slices(self, labels: dict[str, str] | None = None) -> list[LocalSliceHandle]:
        return []

    def shutdown(self) -> None:
        pass


# ============================================================================
# Factory + backward compat
# ============================================================================


def create_gcp_service(mode: ServiceMode, project_id: str = "", **kwargs) -> GcpService:
    if mode == ServiceMode.CLOUD:
        return CloudGcpService(project_id=project_id)
    if mode == ServiceMode.LOCAL:
        return LocalGcpService(project_id=project_id, **kwargs)
    return DryRunGcpService(project_id=project_id)
