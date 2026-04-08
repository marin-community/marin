# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory GcpService for DRY_RUN and LOCAL modes.

DRY_RUN mode validates requests and returns synthetic responses with in-memory state.
LOCAL mode validates, tracks in-memory state, and spawns real local worker threads.

Validation runs in ALL modes — zone, accelerator type, label format, name length.
Failure injection (inject_failure, set_zone_quota, set_tpu_type_unavailable) is
supported in DRY_RUN and LOCAL modes for testing.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from pathlib import Path

from iris.cluster.providers.types import (
    InfraError,
    Labels,
    QuotaExhaustedError,
    find_free_port,
)
from iris.cluster.providers.gcp.service import (
    KNOWN_GCP_ZONES,
    KNOWN_TPU_TYPES,
    QueuedResourceInfo,
    TpuCreateRequest,
    TpuInfo,
    VmCreateRequest,
    VmInfo,
    validate_labels,
    validate_resource_name,
    validate_tpu_create,
    validate_vm_create,
)
from iris.cluster.providers.gcp.local import LocalSliceHandle
from iris.cluster.service_mode import ServiceMode
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)


class InMemoryGcpService:
    """In-memory GcpService for DRY_RUN and LOCAL testing.

    Validation runs in all modes. The mode determines what happens after validation:
    - DRY_RUN: return synthetic response, maintain in-memory state
    - LOCAL: return synthetic response, maintain in-memory state, spawn local workers
    """

    DEFAULT_QUOTA = 100

    def __init__(
        self,
        mode: ServiceMode,
        project_id: str = "",
        # LOCAL mode params
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
        assert mode in (
            ServiceMode.DRY_RUN,
            ServiceMode.LOCAL,
        ), f"InMemoryGcpService only supports DRY_RUN and LOCAL modes, got {mode}"
        self._mode = mode
        self._project_id = project_id

        # Mutable copies so tests can add/remove types
        self._valid_zones: set[str] = set(KNOWN_GCP_ZONES)
        self._valid_accelerator_types: set[str] = set(KNOWN_TPU_TYPES)

        # In-memory state for DRY_RUN/LOCAL modes
        self._tpus: dict[tuple[str, str], TpuInfo] = {}
        self._vms: dict[tuple[str, str], VmInfo] = {}
        self._queued_resources: set[tuple[str, str]] = set()  # TPUs created via queued_resource_create

        # Failure injection (DRY_RUN/LOCAL only)
        self._injected_failures: dict[str, InfraError] = {}
        self._zone_quotas: dict[str, int] = {}
        self._vm_zone_quotas: dict[str, int] = {}
        self._available_types_by_zone: dict[str, set[str]] | None = None

        # LOCAL mode: worker spawning params
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._port_allocator = port_allocator
        self._threads = threads or (ThreadContainer(name="gcp-service-local") if mode == ServiceMode.LOCAL else None)
        self._worker_attributes_by_group = worker_attributes_by_group or {}
        self._gpu_count_by_group = gpu_count_by_group or {}
        self._storage_prefix = storage_prefix
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix) if mode == ServiceMode.LOCAL else None

        # Serial port output injection for testing bootstrap monitoring
        self._serial_port_output: dict[tuple[str, str], str] = {}

        # LOCAL mode: track spawned workers per slice for cleanup
        self._local_slices: dict[str, LocalSliceHandle] = {}

    @property
    def mode(self) -> ServiceMode:
        return self._mode

    @property
    def project_id(self) -> str:
        return self._project_id

    # ========================================================================
    # Failure injection (DRY_RUN/LOCAL)
    # ========================================================================

    def inject_failure(self, operation: str, error: InfraError) -> None:
        """Make the next call to `operation` raise `error`, then auto-clear."""
        self._injected_failures[operation] = error

    def set_zone_quota(self, zone: str, max_tpus: int) -> None:
        """Set TPU quota for a zone. Enforced in DRY_RUN/LOCAL modes."""
        self._zone_quotas[zone] = max_tpus

    def set_tpu_type_unavailable(self, accelerator_type: str) -> None:
        """Remove an accelerator type from the valid set."""
        self._valid_accelerator_types.discard(accelerator_type)

    def add_tpu_type(self, accelerator_type: str) -> None:
        """Add an accelerator type to the valid set."""
        self._valid_accelerator_types.add(accelerator_type)

    def set_vm_zone_quota(self, zone: str, max_vms: int) -> None:
        """Set VM quota for a zone. Enforced in DRY_RUN/LOCAL modes."""
        self._vm_zone_quotas[zone] = max_vms

    def set_available_types_by_zone(self, mapping: dict[str, set[str]]) -> None:
        """Restrict which accelerator types are available per zone."""
        self._available_types_by_zone = mapping

    def advance_tpu_state(self, name: str, zone: str, state: str = "READY") -> None:
        """Transition a TPU to a new state (DRY_RUN/LOCAL only)."""
        key = (name, zone)
        if key not in self._tpus:
            raise ValueError(f"TPU {name!r} not found in {zone}")
        self._tpus[key] = dataclasses.replace(self._tpus[key], state=state)

    def _check_injected_failure(self, operation: str) -> None:
        err = self._injected_failures.pop(operation, None)
        if err is not None:
            raise err

    # ========================================================================
    # TPU operations
    # ========================================================================

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        self._check_injected_failure("tpu_create")

        if self._mode == ServiceMode.LOCAL:
            # LOCAL mode: skip strict GCP validation (zones, accelerator types)
            validate_resource_name(request.name, "TPU")
            validate_labels(request.labels)
        else:
            validate_tpu_create(request, self._valid_zones, self._valid_accelerator_types)

        # DRY_RUN / LOCAL: duplicate detection
        if (request.name, request.zone) in self._tpus:
            raise InfraError(f"TPU {request.name!r} already exists in {request.zone}")

        # Check quota
        zone_count = sum(1 for (_, z) in self._tpus if z == request.zone)
        max_quota = self._zone_quotas.get(request.zone, self.DEFAULT_QUOTA)
        if zone_count >= max_quota:
            raise QuotaExhaustedError(f"Quota exhausted in {request.zone}")

        # Per-type-per-zone availability
        if self._available_types_by_zone is not None:
            zone_types = self._available_types_by_zone.get(request.zone, set())
            if request.accelerator_type not in zone_types:
                raise QuotaExhaustedError(
                    f"Accelerator type {request.accelerator_type!r} not available in {request.zone}"
                )

        # Synthetic network endpoints based on TPU topology
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
            service_account=request.service_account,
            network_endpoints=endpoints,
            external_network_endpoints=[None] * len(endpoints),
            created_at=Timestamp.now(),
        )
        self._tpus[(request.name, request.zone)] = info
        return info

    def tpu_delete(self, name: str, zone: str) -> None:
        self._check_injected_failure("tpu_delete")

        # DRY_RUN / LOCAL: remove from in-memory state
        self._tpus.pop((name, zone), None)

        # LOCAL: stop worker threads for this slice
        local_slice = self._local_slices.pop(name, None)
        if local_slice is not None:
            local_slice.terminate()

    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None:
        self._check_injected_failure("tpu_describe")
        return self._tpus.get((name, zone))

    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]:
        self._check_injected_failure("tpu_list")

        # DRY_RUN / LOCAL: filter in-memory state
        results: list[TpuInfo] = []
        for (_, z), info in self._tpus.items():
            if zones and z not in zones:
                continue
            if labels and not all(info.labels.get(k) == v for k, v in labels.items()):
                continue
            results.append(info)
        return results

    # ========================================================================
    # Queued resource operations (for reserved TPUs)
    # ========================================================================

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        self._check_injected_failure("queued_resource_create")
        # In DRY_RUN/LOCAL mode, simulate immediate provisioning by creating
        # the TPU directly (as if the queued resource instantly became ACTIVE).
        self.tpu_create(request)
        self._queued_resources.add((request.name, request.zone))

    def queued_resource_describe(self, name: str, zone: str) -> QueuedResourceInfo | None:
        self._check_injected_failure("queued_resource_describe")
        if (name, zone) not in self._queued_resources:
            return None
        state = "ACTIVE" if (name, zone) in self._tpus else "PROVISIONING"
        return QueuedResourceInfo(name=name, state=state, zone=zone)

    def queued_resource_delete(self, name: str, zone: str) -> None:
        self._check_injected_failure("queued_resource_delete")
        self._queued_resources.discard((name, zone))
        self.tpu_delete(name, zone)

    def queued_resource_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[QueuedResourceInfo]:
        self._check_injected_failure("queued_resource_list")
        results: list[QueuedResourceInfo] = []
        for name, zone in self._queued_resources:
            if zones and zone not in zones:
                continue
            tpu_info = self._tpus.get((name, zone))
            tpu_labels = tpu_info.labels if tpu_info else {}
            if labels and not all(tpu_labels.get(k) == v for k, v in labels.items()):
                continue
            state = "ACTIVE" if tpu_info else "PROVISIONING"
            results.append(QueuedResourceInfo(name=name, state=state, zone=zone, labels=tpu_labels))
        return results

    # ========================================================================
    # VM operations
    # ========================================================================

    def vm_create(self, request: VmCreateRequest) -> VmInfo:
        self._check_injected_failure("vm_create")

        if self._mode == ServiceMode.LOCAL:
            validate_resource_name(request.name, "VM")
            validate_labels(request.labels)
        else:
            validate_vm_create(request, self._valid_zones)

        # DRY_RUN / LOCAL: duplicate detection
        if (request.name, request.zone) in self._vms:
            raise InfraError(f"VM {request.name!r} already exists in {request.zone}")

        # Check VM quota
        vm_zone_count = sum(1 for (_, z) in self._vms if z == request.zone)
        max_vm_quota = self._vm_zone_quotas.get(request.zone, self.DEFAULT_QUOTA)
        if vm_zone_count >= max_vm_quota:
            raise QuotaExhaustedError(f"VM quota exhausted in {request.zone}")

        # DRY_RUN / LOCAL: create in-memory
        seq = len(self._vms)
        info = VmInfo(
            name=request.name,
            status="RUNNING",
            zone=request.zone,
            internal_ip=f"10.1.{seq}.1",
            external_ip=None,
            labels=dict(request.labels),
            metadata=dict(request.metadata),
            service_account=request.service_account,
            created_at=Timestamp.now(),
        )
        self._vms[(request.name, request.zone)] = info
        return info

    def vm_delete(self, name: str, zone: str, *, wait: bool = False) -> None:
        self._check_injected_failure("vm_delete")
        self._vms.pop((name, zone), None)

    def vm_reset(self, name: str, zone: str) -> None:
        self._check_injected_failure("vm_reset")
        # DRY_RUN / LOCAL: no-op (VM stays in same state)

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
        validate_labels(labels)

        # DRY_RUN / LOCAL: update in-memory state
        vm = self._vms.get((name, zone))
        if vm is None:
            raise InfraError(f"VM {name!r} not found in zone {zone!r}")
        vm.labels.update(labels)

    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None:
        self._check_injected_failure("vm_set_metadata")

        vm = self._vms.get((name, zone))
        if vm is None:
            raise InfraError(f"VM {name!r} not found in zone {zone!r}")
        vm.metadata.update(metadata)

    def set_serial_port_output(self, name: str, zone: str, output: str) -> None:
        """Inject serial port output for a VM. Used by tests to simulate GCE serial console."""
        self._serial_port_output[(name, zone)] = output

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        self._check_injected_failure("vm_get_serial_port_output")
        full_output = self._serial_port_output.get((name, zone), "")
        return full_output[start:]

    def logging_read(self, filter_str: str, limit: int = 200) -> list[str]:
        return []

    # ========================================================================
    # LOCAL mode: worker spawning
    # ========================================================================

    def create_local_slice(
        self,
        slice_id: str,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        """Create a local slice, spawning real Worker threads if controller_address is set.

        Validates the slice_id against GCE naming rules so local tests catch
        naming issues that would fail on real GCP infrastructure.
        """
        validate_resource_name(slice_id, "local slice")
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
        """Spawn real Worker threads for a slice."""
        from iris.cluster.bundle import BundleStore
        from iris.cluster.runtime.process import ProcessRuntime
        from iris.cluster.types import get_tpu_topology
        from iris.cluster.worker.env_probe import FixedEnvironmentProvider, HardwareProbe, build_worker_metadata
        from iris.cluster.worker.worker import Worker, WorkerConfig

        assert self._cache_path is not None
        assert self._threads is not None
        assert self._iris_labels is not None

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
            # Use the canonical capacity_type from the slice config proto.
            capacity_type_val = config.capacity_type or config_pb2.CAPACITY_TYPE_ON_DEMAND

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
                capacity_type=capacity_type_val,
                worker_attributes=extra_attrs,
            )

            env_provider = FixedEnvironmentProvider(metadata)

            wc = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=self._cache_path / worker_id,
                controller_address=self._controller_address,
                worker_id=worker_id,
                slice_id=slice_id,
                worker_attributes=dict(extra_attrs),
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
            "InMemoryGcpService(LOCAL) created slice %s with %d workers",
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
        """Return tracked local slices, optionally filtered by labels."""
        results = list(self._local_slices.values())
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def shutdown(self) -> None:
        """Stop all local worker threads. No-op in DRY_RUN mode."""
        if self._mode != ServiceMode.LOCAL:
            return
        for s in list(self._local_slices.values()):
            s.terminate()
        self._local_slices.clear()
        if self._threads is not None:
            self._threads.stop(timeout=Duration.from_seconds(5.0))
