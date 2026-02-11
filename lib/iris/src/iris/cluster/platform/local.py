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
import socket
import subprocess
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    CommandResult,
    SliceStatus,
    VmStatus,
)
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2, config_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


# ============================================================================
# Local Providers (in-process implementations for testing)
# ============================================================================


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _LocalBundleProvider:
    def __init__(self, bundle_path: Path):
        self._bundle_path = bundle_path

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        del gcs_path, expected_hash
        return self._bundle_path


class LocalEnvironmentProvider:
    def __init__(
        self,
        cpu: int = 1000,
        memory_gb: int = 1000,
        attributes: dict[str, str | int | float] | None = None,
        device: cluster_pb2.DeviceConfig | None = None,
    ):
        self._cpu = cpu
        self._memory_gb = memory_gb
        self._attributes = attributes or {}
        self._device = device

    def probe(self) -> cluster_pb2.WorkerMetadata:
        if self._device is not None:
            device = self._device
        else:
            device = cluster_pb2.DeviceConfig()
            device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        proto_attrs = {}
        for key, value in self._attributes.items():
            if isinstance(value, str):
                proto_attrs[key] = cluster_pb2.AttributeValue(string_value=value)
            elif isinstance(value, int):
                proto_attrs[key] = cluster_pb2.AttributeValue(int_value=value)
            elif isinstance(value, float):
                proto_attrs[key] = cluster_pb2.AttributeValue(float_value=value)

        return cluster_pb2.WorkerMetadata(
            hostname="local",
            ip_address="127.0.0.1",
            cpu_count=self._cpu,
            memory_bytes=self._memory_gb * 1024**3,
            disk_bytes=100 * 1024**3,  # Default 100GB for local
            device=device,
            attributes=proto_attrs,
        )


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class _LocalVmHandle:
    """Handle to a local in-process "VM".

    run_command() executes commands locally via subprocess.
    wait_for_connection() returns True immediately (local process).
    """

    _vm_id: str
    _internal_address: str
    _bootstrap_log_lines: list[str] = field(default_factory=list)

    @property
    def vm_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    def status(self) -> VmStatus:
        return VmStatus(state=CloudVmState.RUNNING)

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
class _LocalStandaloneVmHandle:
    """Handle to a standalone local "VM" (e.g., controller).

    Extends _LocalVmHandle with terminate, set_labels, set_metadata — all
    operating on in-memory state.
    """

    _vm_id: str
    _internal_address: str
    _labels: dict[str, str] = field(default_factory=dict)
    _metadata: dict[str, str] = field(default_factory=dict)
    _terminated: bool = False
    _bootstrap_log_lines: list[str] = field(default_factory=list)

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

    def status(self) -> VmStatus:
        if self._terminated:
            return VmStatus(state=CloudVmState.TERMINATED)
        return VmStatus(state=CloudVmState.RUNNING)

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

    list_vms() returns _LocalVmHandle instances for each "worker" in the slice.
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
        return self._labels.get(f"{self._label_prefix}-scale-group", "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def list_vms(self) -> list[_LocalVmHandle]:
        return [
            _LocalVmHandle(_vm_id=vm_id, _internal_address=addr)
            for vm_id, addr in zip(self._vm_ids, self._addresses, strict=True)
        ]

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        for worker in self._workers:
            worker.stop()

    def status(self) -> SliceStatus:
        if self._terminated:
            return SliceStatus(state=CloudSliceState.DELETING, vm_count=0)
        return SliceStatus(state=CloudSliceState.READY, vm_count=len(self._vm_ids))


# ============================================================================
# LocalPlatform
# ============================================================================


class LocalPlatform:
    """Platform for local testing — workers run as in-process threads.

    Implements the full Platform interface. "VMs" are threads, "slices"
    are groups of worker threads, and "standalone VMs" (controller) are
    in-process server instances.

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
    ):
        self._label_prefix = label_prefix
        self._threads = threads or ThreadContainer(name="local-platform")
        self._slices: dict[str, LocalSliceHandle] = {}
        self._vms: dict[str, _LocalStandaloneVmHandle] = {}
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._port_allocator = port_allocator

    def create_vm(self, config: config_pb2.VmConfig) -> _LocalStandaloneVmHandle:
        """Create an in-process "VM". Used by start_controller() for local mode."""
        handle = _LocalStandaloneVmHandle(
            _vm_id=config.name,
            _internal_address="localhost",
            _labels=dict(config.labels),
            _metadata=dict(config.metadata),
        )
        self._vms[config.name] = handle
        return handle

    def create_slice(self, config: config_pb2.SliceConfig) -> LocalSliceHandle:
        """Create a local slice, optionally spawning real Worker instances.

        When controller_address was provided at construction, spawns real Workers
        that register with the controller (E2E mode). Otherwise creates in-memory
        stubs (unit test mode).
        """
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"
        slice_size = config.slice_size or 1

        if self._controller_address is not None:
            return self._create_slice_with_workers(slice_id, slice_size, config)

        vm_ids = [f"{slice_id}-worker-{i}" for i in range(slice_size)]
        addresses = [f"localhost:{9000 + i}" for i in range(slice_size)]

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
        slice_size: int,
        config: config_pb2.SliceConfig,
    ) -> LocalSliceHandle:
        """Spawn real Worker threads for a slice."""
        from iris.cluster.runtime.process import ProcessRuntime
        from iris.cluster.types import get_tpu_topology, tpu_device
        from iris.cluster.worker.worker import Worker, WorkerConfig

        workers: list[Worker] = []
        vm_ids: list[str] = []
        addresses: list[str] = []

        # Determine worker count from TPU topology if applicable
        worker_count = slice_size
        if config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU and config.accelerator_variant:
            try:
                topo = get_tpu_topology(config.accelerator_variant)
                worker_count = topo.vm_count
            except ValueError:
                logger.debug("Unknown accelerator variant %r; TPU topology not available", config.accelerator_variant)

        for tpu_worker_id in range(worker_count):
            bundle_provider = _LocalBundleProvider(self._fake_bundle)
            container_runtime = ProcessRuntime()
            worker_id = f"worker-{slice_id}-{tpu_worker_id}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()

            attributes: dict[str, str | int | float] = {}
            device = None
            if config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU and config.accelerator_variant:
                attributes["tpu-name"] = slice_id
                attributes["tpu-worker-id"] = tpu_worker_id
                attributes["tpu-topology"] = config.accelerator_variant
                topo = get_tpu_topology(config.accelerator_variant)
                device = tpu_device(config.accelerator_variant, count=topo.chips_per_vm)

            environment_provider = LocalEnvironmentProvider(
                cpu=1000,
                memory_gb=1000,
                attributes=attributes,
                device=device,
            )

            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=self._cache_path,
                controller_address=self._controller_address,
                worker_id=worker_id,
                poll_interval=Duration.from_seconds(0.1),
            )
            worker_threads = self._threads.create_child(f"worker-{worker_id}")
            worker = Worker(
                worker_config,
                bundle_provider=bundle_provider,
                container_runtime=container_runtime,
                environment_provider=environment_provider,
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

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[_LocalStandaloneVmHandle]:
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
        self._threads.stop(timeout=Duration.from_seconds(5.0))
        self._slices.clear()
        self._vms.clear()

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Return controller address. Uses configured address or localhost."""
        if self._controller_address:
            return self._controller_address
        port = controller_config.local.port or 10000
        return f"localhost:{port}"

    @property
    def threads(self) -> ThreadContainer:
        """Expose the ThreadContainer for callers that need to spawn worker threads."""
        return self._threads
