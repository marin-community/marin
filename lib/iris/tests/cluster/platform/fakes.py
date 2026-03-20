# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake implementations for testing.

Provides FakePlatform / FakeSliceHandle / FakeWorkerHandle: In-memory Platform
that simulates worker lifecycle with configurable delays and failure injection.

Usage:
    config = config_pb2.ScaleGroupConfig(name="test-group", ...)
    platform = FakePlatform(FakePlatformConfig(config=config))
    handle = platform.create_slice(slice_config)
    platform.tick(ts=now_ms())
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    QuotaExhaustedError,
    SliceStatus,
    WorkerStatus,
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


class FakeWorkerHandle:
    """In-memory worker handle that simulates state transitions.

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
        self._state = CloudWorkerState.UNKNOWN  # Starts as "not yet running"
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
    def worker_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._address

    @property
    def external_address(self) -> str | None:
        return None

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=self._state)

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
                self._state = CloudWorkerState.RUNNING
                self._state_changed_at_ms = ts

    def set_terminated(self) -> None:
        self._state = CloudWorkerState.TERMINATED
        self._iris_state_booting = False
        self._iris_state_initializing = False
        self._iris_state_ready = False


class FakeSliceHandle:
    """In-memory SliceHandle for testing.

    Holds FakeWorkerHandle instances and computes status from their states.
    State transitions happen during tick() calls.
    """

    def __init__(
        self,
        slice_id: str,
        scale_group: str,
        zone: str,
        vms: list[FakeWorkerHandle],
        labels: dict[str, str] | None = None,
        created_at_ms: int | None = None,
        worker_config: config_pb2.WorkerConfig | None = None,
    ):
        self._slice_id = slice_id
        self._scale_group = scale_group
        self._zone = zone
        self._vms = vms
        self._labels = labels or {}
        self._created_at = Timestamp.from_ms(created_at_ms) if created_at_ms is not None else Timestamp.now()
        self._terminated = False
        self._worker_config = worker_config
        self._bootstrapped = False

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
            return SliceStatus(state=CloudSliceState.DELETING, worker_count=len(self._vms), workers=list(self._vms))
        all_running = all(vm._state == CloudWorkerState.RUNNING for vm in self._vms)
        if all_running and self._worker_config and not self._bootstrapped:
            state = CloudSliceState.BOOTSTRAPPING
        elif all_running and (not self._worker_config or self._bootstrapped):
            state = CloudSliceState.READY
        else:
            state = CloudSliceState.CREATING
        return SliceStatus(state=state, worker_count=len(self._vms), workers=list(self._vms))

    def terminate(self) -> None:
        for vm in self._vms:
            vm.set_terminated()
        self._terminated = True

    def tick(self, ts: int) -> None:
        """Advance VM state transitions and simulate bootstrap when configured."""
        for vm in self._vms:
            vm.tick(ts)
        if self._worker_config and not self._bootstrapped:
            all_running = all(vm._state == CloudWorkerState.RUNNING for vm in self._vms)
            if all_running:
                for vm in self._vms:
                    vm._bootstrap_count += 1
                self._bootstrapped = True


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
    FakeWorkerHandle that transition states during tick() calls.

    Thread-safe for use in concurrent tests.
    """

    def __init__(self, config: FakePlatformConfig):
        self._config = config
        self._lock = threading.Lock()
        self._slices: dict[str, FakeSliceHandle] = {}
        self._slice_counter = 0

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def create_vm(self, config: config_pb2.VmConfig):
        raise NotImplementedError("FakePlatform does not support standalone VMs")

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> FakeSliceHandle:
        """Create a new fake slice.

        When worker_config is provided, the slice starts in CREATING state and
        transitions through BOOTSTRAPPING to READY during tick(). Without it,
        slices go straight from CREATING to READY (no bootstrap simulation).
        """
        if self._config.failure_mode == FailureMode.QUOTA_EXCEEDED:
            raise QuotaExhaustedError(f"Quota exceeded for {self._config.config.name}")
        if self._config.failure_mode == FailureMode.CREATE_FAILS:
            raise RuntimeError("FakePlatform configured to fail on create")

        with self._lock:
            self._slice_counter += 1
            slice_id = f"fake-slice-{self._config.config.name}-{self._slice_counter}"
            ts = Timestamp.now().epoch_ms()

            device_variant = (
                self._config.config.resources.device_variant if self._config.config.HasField("resources") else ""
            )
            topology = get_tpu_topology(device_variant)
            vm_count = topology.vm_count
            zone = self._config.config.slice_template.gcp.zone or "us-central1-a"

            workers = []
            for i in range(vm_count):
                worker = FakeWorkerHandle(
                    vm_id=f"{slice_id}-vm-{i}",
                    address=f"10.128.0.{self._slice_counter * 10 + i}",
                    created_at_ms=ts,
                    boot_delay_ms=self._config.boot_delay_ms,
                    init_delay_ms=self._config.init_delay_ms,
                )
                workers.append(worker)

            labels = dict(config.labels) if config.labels else {}
            fake_slice = FakeSliceHandle(
                slice_id=slice_id,
                scale_group=self._config.config.name,
                zone=zone,
                vms=workers,
                labels=labels,
                created_at_ms=ts,
                worker_config=worker_config,
            )
            self._slices[slice_id] = fake_slice

            logger.debug("Created fake slice %s with %d workers", slice_id, vm_count)
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

    def list_all_slices(self) -> list[FakeSliceHandle]:
        return self.list_slices(zones=[], labels=None)

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

    def inject_slice(self, handle: FakeSliceHandle) -> None:
        """Add a slice without going through create_slice(). Simulates pre-existing cloud slices."""
        with self._lock:
            self._slices[handle.slice_id] = handle

    def remove_slice(self, slice_id: str) -> None:
        """Remove a slice without calling terminate(). Simulates termination during restart."""
        with self._lock:
            self._slices.pop(slice_id, None)

    def set_failure_mode(self, mode: FailureMode) -> None:
        """Set the failure mode for subsequent operations."""
        self._config.failure_mode = mode

    def get_slice(self, slice_id: str) -> FakeSliceHandle | None:
        """Get a specific slice by ID."""
        with self._lock:
            return self._slices.get(slice_id)
