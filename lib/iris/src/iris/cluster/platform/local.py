# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LocalPlatform implementation for in-process testing.

Implements the full Platform interface. "VMs" are in-memory stubs,
"slices" are groups of tracked entries, and "standalone VMs" (controller)
are in-memory handles. No SSH, no cloud resources.

When constructed with worker-spawning parameters (controller_address, cache_path,
fake_bundle, port_allocator), create_slice() spawns real Worker threads that
register with the controller — enabling full E2E testing. Without these params,
create_slice() only creates in-memory stubs (for unit tests).

LocalPlatform.shutdown() is critical — it stops worker threads managed by
the ThreadContainer.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from iris.cluster.bundle import BundleStore
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    Labels,
    SliceStatus,
    WorkerStatus,
    find_free_port,
    generate_slice_suffix,
)
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class _LocalWorkerHandle:
    """Handle to a local in-process worker.

    run_command() executes commands locally via subprocess.
    wait_for_connection() returns True immediately (local process).
    """

    _vm_id: str
    _internal_address: str
    _bootstrap_log_lines: list[str] = field(default_factory=list)

    @property
    def worker_id(self) -> str:
        return self._vm_id

    @property
    def vm_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        return True

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        timeout_secs = timeout.to_seconds() if timeout else 30.0
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        if on_line:
            for line in result.stdout.splitlines():
                on_line(line)
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def bootstrap(self, script: str) -> None:
        self._bootstrap_log_lines.clear()
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
        )
        self._bootstrap_log_lines.extend(result.stdout.splitlines())
        if result.returncode != 0:
            self._bootstrap_log_lines.extend(result.stderr.splitlines())
            raise RuntimeError(f"Bootstrap failed on {self._vm_id}: exit code {result.returncode}\n{result.stderr}")

    @property
    def bootstrap_log(self) -> str:
        return "\n".join(self._bootstrap_log_lines)

    def reboot(self) -> None:
        logger.info("Reboot requested for local VM %s (no-op)", self._vm_id)


@dataclass
class _LocalStandaloneWorkerHandle:
    """Handle to a standalone local worker (e.g., controller).

    Extends _LocalWorkerHandle with terminate, set_labels, set_metadata -- all
    operating on in-memory state.
    """

    _vm_id: str
    _internal_address: str
    _labels: dict[str, str] = field(default_factory=dict)
    _metadata: dict[str, str] = field(default_factory=dict)
    _terminated: bool = False
    _bootstrap_log_lines: list[str] = field(default_factory=list)

    @property
    def worker_id(self) -> str:
        return self._vm_id

    @property
    def vm_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def status(self) -> WorkerStatus:
        if self._terminated:
            return WorkerStatus(state=CloudWorkerState.TERMINATED)
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        return True

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        timeout_secs = timeout.to_seconds() if timeout else 30.0
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        if on_line:
            for line in result.stdout.splitlines():
                on_line(line)
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def bootstrap(self, script: str) -> None:
        self._bootstrap_log_lines.clear()
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
        )
        self._bootstrap_log_lines.extend(result.stdout.splitlines())
        if result.returncode != 0:
            self._bootstrap_log_lines.extend(result.stderr.splitlines())
            raise RuntimeError(f"Bootstrap failed on {self._vm_id}: exit code {result.returncode}\n{result.stderr}")

    @property
    def bootstrap_log(self) -> str:
        return "\n".join(self._bootstrap_log_lines)

    def reboot(self) -> None:
        logger.info("Reboot requested for local VM %s (no-op)", self._vm_id)

    def terminate(self) -> None:
        self._terminated = True

    def set_labels(self, labels: dict[str, str]) -> None:
        self._labels.update(labels)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        self._metadata.update(metadata)


@dataclass
class LocalSliceHandle:
    """Handle to a local in-process slice.

    list_vms() returns _LocalWorkerHandle instances for each "worker" in the slice.
    terminate() marks the slice as terminated and stops any real Worker instances.
    """

    _slice_id: str
    _vm_ids: list[str]
    _addresses: list[str]
    _labels: dict[str, str]
    _created_at: Timestamp
    _label_prefix: str
    _workers: list = field(default_factory=list)
    _terminated: bool = False

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return "local"

    @property
    def scale_group(self) -> str:
        return self._labels.get(Labels(self._label_prefix).iris_scale_group, "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        if self._terminated:
            return SliceStatus(state=CloudSliceState.DELETING, worker_count=0)
        workers = [
            _LocalWorkerHandle(_vm_id=vm_id, _internal_address=addr)
            for vm_id, addr in zip(self._vm_ids, self._addresses, strict=True)
        ]
        return SliceStatus(state=CloudSliceState.READY, worker_count=len(self._vm_ids), workers=workers)

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        for worker in self._workers:
            worker.stop()


# ============================================================================
# LocalPlatform
# ============================================================================


class LocalPlatform:
    """Platform for local testing — workers run as in-process threads.

    Implements the full Platform interface. "VMs" are threads, "slices"
    are groups of worker threads, and "standalone VMs" (controller) are
    in-memory handles.

    When constructed with worker-spawning params (controller_address, cache_path,
    fake_bundle, port_allocator), create_slice() spawns real Worker threads.
    Without these params, create_slice() creates in-memory stubs only.

    shutdown() stops all worker threads via the ThreadContainer. This is
    critical for clean test teardown.
    """

    def __init__(
        self,
        label_prefix: str,
        threads: ThreadContainer | None = None,
        controller_address: str | None = None,
        cache_path: Path | None = None,
        fake_bundle: Path | None = None,
        port_allocator: PortAllocator | None = None,
        worker_attributes_by_group: dict[str, dict[str, str | int | float]] | None = None,
        gpu_count_by_group: dict[str, int] | None = None,
        storage_prefix: str = "",
    ):
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix)
        self._threads = threads or ThreadContainer(name="local-platform")
        self._slices: dict[str, LocalSliceHandle] = {}
        self._vms: dict[str, _LocalStandaloneWorkerHandle] = {}
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._port_allocator = port_allocator
        self._worker_attributes_by_group = worker_attributes_by_group or {}
        self._gpu_count_by_group = gpu_count_by_group or {}
        self._storage_prefix = storage_prefix

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def create_vm(self, config: config_pb2.VmConfig) -> _LocalStandaloneWorkerHandle:
        """Create an in-process "VM" (e.g., for the controller in local mode)."""
        handle = _LocalStandaloneWorkerHandle(
            _vm_id=config.name,
            _internal_address="localhost",
            _labels=dict(config.labels),
            _metadata=dict(config.metadata),
        )
        self._vms[config.name] = handle
        return handle

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        """Create a local slice, optionally spawning real Worker instances.

        When controller_address was provided at construction, spawns real Workers
        that register with the controller (E2E mode). Otherwise creates in-memory
        stubs (unit test mode).
        """
        slice_id = f"{config.name_prefix}-{generate_slice_suffix()}"
        num_vms = config.num_vms or 1

        if self._controller_address is not None:
            return self._create_slice_with_workers(slice_id, num_vms, config, worker_config)

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
        self._slices[slice_id] = handle
        return handle

    def _create_slice_with_workers(
        self,
        slice_id: str,
        num_vms: int,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> LocalSliceHandle:
        """Spawn real Worker threads for a slice."""
        from iris.cluster.runtime.process import ProcessRuntime
        from iris.cluster.types import get_tpu_topology
        from iris.cluster.worker.env_probe import FixedEnvironmentProvider, HardwareProbe, build_worker_metadata
        from iris.cluster.worker.worker import Worker, WorkerConfig

        workers: list[Worker] = []
        vm_ids: list[str] = []
        addresses: list[str] = []
        assert self._cache_path is not None
        # Determine worker count from TPU topology if applicable
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

            # Collect extra worker attributes from scale group config
            extra_attrs: dict[str, str] = {}
            sg_name = config.labels.get(self._iris_labels.iris_scale_group, "")
            if sg_name and sg_name in self._worker_attributes_by_group:
                for k, v in self._worker_attributes_by_group[sg_name].items():
                    extra_attrs.setdefault(k, str(v))

            if worker_config is not None:
                for k, v in worker_config.worker_attributes.items():
                    extra_attrs.setdefault(k, v)

            # Local workers always export region="local" so that region
            # constraints work in tests. Real workers get region from GCP zone.
            extra_attrs.setdefault("region", "local")

            # Determine preemptible from worker attributes
            preemptible = extra_attrs.pop("preemptible", "false").lower() == "true"

            # Determine GPU count from scale group config
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
            "LocalPlatform created slice %s with %d workers",
            slice_id,
            len(workers),
        )

        handle = LocalSliceHandle(
            _slice_id=slice_id,
            _vm_ids=vm_ids,
            _addresses=addresses,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _workers=workers,
        )
        self._slices[slice_id] = handle
        return handle

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[LocalSliceHandle]:
        """List all local slices, optionally filtered by labels.

        The zones parameter is accepted for interface compatibility but ignored —
        all local slices report zone="local".
        """
        results = list(self._slices.values())
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def list_all_slices(self) -> list[LocalSliceHandle]:
        return self.list_slices(zones=[], labels={self._iris_labels.iris_managed: "true"})

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[_LocalStandaloneWorkerHandle]:
        """List all local standalone VMs, optionally filtered by labels.

        The zones parameter is accepted for interface compatibility but ignored —
        local VMs have no zone concept.
        """
        results = list(self._vms.values())
        if labels:
            results = [v for v in results if all(v.labels.get(k) == val for k, val in labels.items())]
        return results

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(address)

    def shutdown(self) -> None:
        """Stop all worker threads. Critical for clean test teardown."""
        for s in self._slices.values():
            s.terminate()
        self._threads.stop(timeout=Duration.from_seconds(5.0))
        self._slices.clear()
        self._vms.clear()

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Return controller address. Uses configured address or localhost."""
        if self._controller_address:
            return self._controller_address
        port = controller_config.local.port or 10000
        return f"localhost:{port}"

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        """Terminate all managed slices. No external controller to stop in local mode."""
        all_slices = self.list_all_slices()
        names = [f"slice:{s.slice_id}" for s in all_slices]
        if not dry_run:
            for s in all_slices:
                s.terminate()
        return names

    @property
    def threads(self) -> ThreadContainer:
        """Expose the ThreadContainer for callers that need to spawn worker threads."""
        return self._threads
