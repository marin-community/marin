# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake implementations for testing.

Provides:
- FakePlatform / FakeSliceHandle / FakeVmHandle: In-memory Platform that
  simulates VM lifecycle with configurable delays and failure injection.
- FakeTpuApi / FakeComputeApi: Protocol-based fakes implementing TpuApi and
  ComputeApi protocols for testing GcpPlatform.

Usage (FakePlatform):
    config = config_pb2.ScaleGroupConfig(name="test-group", ...)
    platform = FakePlatform(FakePlatformConfig(config=config))
    handle = platform.create_slice(slice_config)
    platform.tick(ts=now_ms())

Usage (FakeTpuApi/FakeComputeApi):
    fake_apis = FakeGcpApis(tpu=FakeTpuApi(), compute=FakeComputeApi())
    fake_apis.set_tpu_failure("create_node", "RESOURCE_EXHAUSTED: no capacity")
    platform = GcpPlatform(gcp_config, label_prefix="iris",
                          tpu_api=fake_apis.tpu, compute_api=fake_apis.compute)
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto

import pytest

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    CommandResult,
    QuotaExhaustedError,
    SliceStatus,
    VmStatus,
)
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Failure injection modes for FakePlatform."""

    NONE = auto()
    CREATE_FAILS = auto()
    QUOTA_EXCEEDED = auto()


class FakeVmHandle:
    """In-memory VmHandle that simulates state transitions.

    State transitions happen during explicit tick() calls on FakeSliceHandle,
    making tests deterministic.
    """

    def __init__(
        self,
        vm_id: str,
        address: str,
        created_at_ms: int,
        boot_delay_ms: int = 0,
        init_delay_ms: int = 0,
    ):
        self._vm_id = vm_id
        self._address = address
        self._state = CloudVmState.UNKNOWN  # Starts as "not yet running"
        self._state_changed_at_ms = created_at_ms
        self._boot_delay_ms = boot_delay_ms
        self._init_delay_ms = init_delay_ms
        # Internal iris-level state for FakeVm compatibility
        self._iris_state_booting = True
        self._iris_state_initializing = False
        self._iris_state_ready = False
        self._bootstrap_count = 0
        self._bootstrap_log_lines: list[str] = []
        # Test control flag for wait_for_connection behavior
        self._wait_for_connection_succeeds = True

    @property
    def vm_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._address

    @property
    def external_address(self) -> str | None:
        return None

    def status(self) -> VmStatus:
        return VmStatus(state=self._state)

    def wait_for_connection(self, timeout: Duration, poll_interval: Duration = Duration.from_seconds(5)) -> bool:
        return self._wait_for_connection_succeeds

    def run_command(self, command: str, timeout: Duration | None = None, on_line=None) -> CommandResult:
        return CommandResult(returncode=0, stdout="", stderr="")

    def bootstrap(self, script: str) -> None:
        self._bootstrap_count += 1
        self._bootstrap_log_lines.append(f"[fake] bootstrap called (count={self._bootstrap_count})")

    @property
    def bootstrap_log(self) -> str:
        return "\n".join(self._bootstrap_log_lines)

    def reboot(self) -> None:
        pass

    def tick(self, ts: int) -> None:
        """Process state transitions based on elapsed time."""
        if self._iris_state_booting:
            elapsed = ts - self._state_changed_at_ms
            if elapsed >= self._boot_delay_ms:
                self._iris_state_booting = False
                self._iris_state_initializing = True
                self._state_changed_at_ms = ts

        if self._iris_state_initializing:
            elapsed = ts - self._state_changed_at_ms
            if elapsed >= self._init_delay_ms:
                self._iris_state_initializing = False
                self._iris_state_ready = True
                self._state = CloudVmState.RUNNING
                self._state_changed_at_ms = ts

    def set_terminated(self) -> None:
        self._state = CloudVmState.TERMINATED
        self._iris_state_booting = False
        self._iris_state_initializing = False
        self._iris_state_ready = False


class FakeSliceHandle:
    """In-memory SliceHandle for testing.

    Holds FakeVmHandle instances and computes status from their states.
    State transitions happen during tick() calls.
    """

    def __init__(
        self,
        slice_id: str,
        scale_group: str,
        zone: str,
        vms: list[FakeVmHandle],
        labels: dict[str, str] | None = None,
        created_at_ms: int | None = None,
    ):
        self._slice_id = slice_id
        self._scale_group = scale_group
        self._zone = zone
        self._vms = vms
        self._labels = labels or {}
        self._created_at = Timestamp.from_ms(created_at_ms) if created_at_ms is not None else Timestamp.now()
        self._terminated = False

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return self._zone

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def labels(self) -> dict[str, str]:
        return self._labels

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        if self._terminated:
            return SliceStatus(state=CloudSliceState.DELETING, vm_count=len(self._vms), vms=list(self._vms))
        all_running = all(vm._state == CloudVmState.RUNNING for vm in self._vms)
        state = CloudSliceState.READY if all_running else CloudSliceState.CREATING
        return SliceStatus(state=state, vm_count=len(self._vms), vms=list(self._vms))

    def terminate(self) -> None:
        for vm in self._vms:
            vm.set_terminated()
        self._terminated = True

    def tick(self, ts: int) -> None:
        """Advance VM state transitions."""
        for vm in self._vms:
            vm.tick(ts)


@dataclass
class FakePlatformConfig:
    """Configuration for FakePlatform behavior.

    Args:
        config: Scale group configuration
        boot_delay_ms: Time in ms for VM to transition from BOOTING to INITIALIZING
        init_delay_ms: Time in ms for VM to transition from INITIALIZING to READY
        failure_mode: Failure injection mode for create_slice
        controller_address: Controller address returned by discover_controller
    """

    config: config_pb2.ScaleGroupConfig
    boot_delay_ms: int = 0
    init_delay_ms: int = 0
    failure_mode: FailureMode = field(default=FailureMode.NONE)
    controller_address: str = "10.0.0.1:10000"


class FakePlatform:
    """In-memory Platform for testing.

    Implements the Platform protocol. Creates FakeSliceHandle instances with
    FakeVmHandle that transition states during tick() calls.

    Thread-safe for use in concurrent tests.
    """

    def __init__(self, config: FakePlatformConfig):
        self._config = config
        self._lock = threading.Lock()
        self._slices: dict[str, FakeSliceHandle] = {}
        self._slice_counter = 0

    def create_vm(self, config: config_pb2.VmConfig):
        raise NotImplementedError("FakePlatform does not support standalone VMs")

    def create_slice(self, config: config_pb2.SliceConfig) -> FakeSliceHandle:
        """Create a new fake slice."""
        if self._config.failure_mode == FailureMode.QUOTA_EXCEEDED:
            raise QuotaExhaustedError(f"Quota exceeded for {self._config.config.name}")
        if self._config.failure_mode == FailureMode.CREATE_FAILS:
            raise RuntimeError("FakePlatform configured to fail on create")

        with self._lock:
            self._slice_counter += 1
            slice_id = f"fake-slice-{self._config.config.name}-{self._slice_counter}"
            ts = Timestamp.now().epoch_ms()

            topology = get_tpu_topology(self._config.config.accelerator_variant)
            vm_count = topology.vm_count
            zone = self._config.config.slice_template.gcp.zone or "us-central1-a"

            vms = []
            for i in range(vm_count):
                vm = FakeVmHandle(
                    vm_id=f"{slice_id}-vm-{i}",
                    address=f"10.128.0.{self._slice_counter * 10 + i}",
                    created_at_ms=ts,
                    boot_delay_ms=self._config.boot_delay_ms,
                    init_delay_ms=self._config.init_delay_ms,
                )
                vms.append(vm)

            labels = dict(config.labels) if config.labels else {}
            fake_slice = FakeSliceHandle(
                slice_id=slice_id,
                scale_group=self._config.config.name,
                zone=zone,
                vms=vms,
                labels=labels,
                created_at_ms=ts,
            )
            self._slices[slice_id] = fake_slice

            logger.debug("Created fake slice %s with %d VMs", slice_id, vm_count)
            return fake_slice

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[FakeSliceHandle]:
        """List slices, optionally filtered by labels."""
        with self._lock:
            slices = list(self._slices.values())
        if labels:
            slices = [s for s in slices if all(s.labels.get(k) == v for k, v in labels.items())]
        return slices

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[FakeSliceHandle]:
        return self.list_slices(zones=[], labels=labels)

    def list_vms(self, zones: list[str], labels: dict[str, str] | None = None) -> list:
        return []

    def tunnel(self, address: str, local_port: int | None = None):
        return nullcontext(address)

    def shutdown(self) -> None:
        """No-op: FakePlatform has no background threads to stop."""

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        return self._config.controller_address

    def tick(self, ts: int | None = None) -> None:
        """Advance all slice state transitions.

        Call this to simulate time passing and VMs completing boot/init.
        """
        ts = ts or Timestamp.now().epoch_ms()
        with self._lock:
            for fake_slice in self._slices.values():
                fake_slice.tick(ts)

    def set_failure_mode(self, mode: FailureMode) -> None:
        """Set the failure mode for subsequent operations."""
        self._config.failure_mode = mode

    def get_slice(self, slice_id: str) -> FakeSliceHandle | None:
        """Get a specific slice by ID."""
        with self._lock:
            return self._slices.get(slice_id)


# =============================================================================
# FakeTpuApi and FakeComputeApi — protocol-based fakes for testing GcpPlatform
# =============================================================================


@dataclass
class FakeTpuApi:
    """In-memory fake implementation of TpuApi protocol.

    Maintains TPU state dictionary keyed by (name, zone). Supports failure
    injection for testing error paths.
    """

    _tpus: dict[tuple[str, str], dict] = field(default_factory=dict)
    _failures: dict[str, str] = field(default_factory=dict)

    def set_failure(self, operation: str, error: str) -> None:
        """Make a specific operation type fail on the next call.

        Args:
            operation: One of "get_node", "create_node", "delete_node", "list_nodes".
            error: The error message to raise.
        """
        self._failures[operation] = error

    def clear_failure(self) -> None:
        """Remove all injected failures."""
        self._failures.clear()

    def _check_failure(self, operation: str) -> None:
        if operation in self._failures:
            error = self._failures.pop(operation)
            from iris.cluster.platform.base import PlatformError, QuotaExhaustedError

            if "RESOURCE_EXHAUSTED" in error or "Quota exceeded" in error:
                raise QuotaExhaustedError(error)
            raise PlatformError(error)

    def get_node(self, project: str, zone: str, name: str) -> dict:
        self._check_failure("get_node")
        key = (name, zone)
        if key not in self._tpus:
            from iris.cluster.platform.base import PlatformError

            raise PlatformError(f"TPU node {name} not found in zone {zone}")
        return dict(self._tpus[key])

    def create_node(
        self,
        project: str,
        zone: str,
        node_id: str,
        accelerator_type: str,
        runtime_version: str,
        labels: dict[str, str] | None = None,
        preemptible: bool = False,
    ) -> None:
        self._check_failure("create_node")

        idx = len(self._tpus)
        tpu_data = {
            "name": node_id,
            "state": "READY",
            "acceleratorType": accelerator_type,
            "labels": dict(labels) if labels else {},
            "networkEndpoints": [{"ipAddress": f"10.0.0.{idx + 1}"}],
            "createTime": "2024-01-15T10:30:00.000Z",
        }
        self._tpus[(node_id, zone)] = tpu_data

    def delete_node(self, project: str, zone: str, name: str) -> None:
        self._check_failure("delete_node")
        self._tpus.pop((name, zone), None)

    def list_nodes(self, project: str, zone: str) -> list[dict]:
        self._check_failure("list_nodes")
        matching = []
        for (_, tpu_zone), tpu in self._tpus.items():
            if tpu_zone == zone:
                matching.append(dict(tpu))
        return matching


@dataclass
class FakeComputeApi:
    """In-memory fake implementation of ComputeApi protocol.

    Maintains VM state dictionary keyed by (name, zone). Supports failure
    injection for testing error paths.
    """

    _vms: dict[tuple[str, str], dict] = field(default_factory=dict)
    _failures: dict[str, str] = field(default_factory=dict)

    def set_failure(self, operation: str, error: str) -> None:
        """Make a specific operation type fail on the next call.

        Args:
            operation: One of "get_instance", "create_instance", "delete_instance",
                       "list_instances", "reset_instance", "set_labels", "set_metadata".
            error: The error message to raise.
        """
        self._failures[operation] = error

    def clear_failure(self) -> None:
        """Remove all injected failures."""
        self._failures.clear()

    def _check_failure(self, operation: str) -> None:
        if operation in self._failures:
            error = self._failures.pop(operation)
            from iris.cluster.platform.base import PlatformError, QuotaExhaustedError

            if "RESOURCE_EXHAUSTED" in error or "Quota exceeded" in error:
                raise QuotaExhaustedError(error)
            raise PlatformError(error)

    def get_instance(self, project: str, zone: str, instance: str) -> dict:
        self._check_failure("get_instance")
        key = (instance, zone)
        if key not in self._vms:
            from iris.cluster.platform.base import PlatformError

            raise PlatformError(f"Instance {instance} not found in zone {zone}")
        return dict(self._vms[key])

    def create_instance(
        self,
        project: str,
        zone: str,
        instance_name: str,
        machine_type: str,
        boot_disk_size_gb: int,
        labels: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        self._check_failure("create_instance")

        idx = len(self._vms) + 1
        vm_data = {
            "name": instance_name,
            "status": "RUNNING",
            "networkInterfaces": [
                {
                    "networkIP": f"10.128.0.{idx}",
                    "accessConfigs": [{"natIP": f"34.1.2.{idx}"}],
                }
            ],
            "labels": dict(labels) if labels else {},
            "metadata": dict(metadata) if metadata else {},
        }
        self._vms[(instance_name, zone)] = vm_data
        return dict(vm_data)

    def delete_instance(self, project: str, zone: str, instance: str) -> None:
        self._check_failure("delete_instance")
        self._vms.pop((instance, zone), None)

    def list_instances(self, project: str, zone: str) -> list[dict]:
        self._check_failure("list_instances")
        matching = []
        for (_, vm_zone), vm in self._vms.items():
            if vm_zone == zone:
                matching.append(dict(vm))
        return matching

    def reset_instance(self, project: str, zone: str, instance: str) -> None:
        self._check_failure("reset_instance")
        key = (instance, zone)
        if key not in self._vms:
            from iris.cluster.platform.base import PlatformError

            raise PlatformError(f"Instance {instance} not found in zone {zone}")

    def set_labels(self, project: str, zone: str, instance: str, labels: dict[str, str]) -> None:
        self._check_failure("set_labels")
        key = (instance, zone)
        if key not in self._vms:
            from iris.cluster.platform.base import PlatformError

            raise PlatformError(f"Instance {instance} not found in zone {zone}")
        self._vms[key].setdefault("labels", {}).update(labels)

    def set_metadata(self, project: str, zone: str, instance: str, metadata: dict[str, str]) -> None:
        self._check_failure("set_metadata")
        key = (instance, zone)
        if key not in self._vms:
            from iris.cluster.platform.base import PlatformError

            raise PlatformError(f"Instance {instance} not found in zone {zone}")
        self._vms[key].setdefault("metadata", {}).update(metadata)


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def fake_tpu_api() -> FakeTpuApi:
    """Pytest fixture providing a FakeTpuApi instance."""
    return FakeTpuApi()


@pytest.fixture
def fake_compute_api() -> FakeComputeApi:
    """Pytest fixture providing a FakeComputeApi instance."""
    return FakeComputeApi()


@dataclass
class FakeGcpApis:
    """Container for fake GCP APIs."""

    tpu: FakeTpuApi
    compute: FakeComputeApi

    def set_tpu_failure(self, operation: str, error: str) -> None:
        """Convenience method to set TPU API failures."""
        self.tpu.set_failure(operation, error)

    def set_compute_failure(self, operation: str, error: str) -> None:
        """Convenience method to set Compute API failures."""
        self.compute.set_failure(operation, error)

    def clear_failures(self) -> None:
        """Clear all failures from both APIs."""
        self.tpu.clear_failure()
        self.compute.clear_failure()


@pytest.fixture
def fake_gcp_apis() -> FakeGcpApis:
    """Pytest fixture providing both fake GCP APIs in a container."""
    return FakeGcpApis(tpu=FakeTpuApi(), compute=FakeComputeApi())
