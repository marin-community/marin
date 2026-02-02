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

"""Fake implementations for testing the autoscaler.

Provides in-memory implementations of VmGroupProtocol and VmManagerProtocol
that simulate VM lifecycle with configurable delays and failure injection.

Usage:
    # Create a FakeVmManager with a scale group config
    config = config_pb2.ScaleGroupConfig(name="test-group", ...)
    manager = FakeVmManager(FakeVmManagerConfig(config=config))

    # Create VM groups (they start in BOOTING state)
    vm_group = manager.create_vm_group()
    assert vm_group.status().vms[0].state == vm_pb2.VM_STATE_BOOTING

    # Advance time to trigger state transitions (BOOTING -> INITIALIZING -> READY)
    manager.tick(ts=now_ms())
    assert vm_group.status().vms[0].state == vm_pb2.VM_STATE_READY

    # Inject failures at the manager level
    manager.set_failure_mode(FailureMode.QUOTA_EXCEEDED)
    manager.create_vm_group()  # raises QuotaExceededError

Key concepts:
    - tick(ts): Advances VM state transitions based on elapsed time.
      With default delays of 0, VMs transition instantly on tick().
    - FailureMode: Inject quota errors or generic creation failures at the manager level.
    - FakeVm state machine: BOOTING -> INITIALIZING -> READY
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto

from iris.cluster.types import get_tpu_topology
from iris.cluster.vm.managed_vm import QuotaExceededError
from iris.cluster.vm.vm_platform import VmGroupStatus, VmSnapshot
from iris.rpc import time_pb2, config_pb2, vm_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Failure injection modes for FakeVmManager."""

    NONE = auto()
    CREATE_FAILS = auto()
    QUOTA_EXCEEDED = auto()


class FakeVm:
    """Fake VM that simulates state transitions with configurable delays.

    Unlike ManagedVm, this doesn't run a thread - state transitions happen
    during explicit tick() calls, making tests deterministic.
    """

    def __init__(
        self,
        vm_id: str,
        slice_id: str,
        scale_group: str,
        zone: str,
        created_at_ms: int,
        boot_delay_ms: int = 0,
        init_delay_ms: int = 0,
    ):
        self.info = vm_pb2.VmInfo(
            vm_id=vm_id,
            slice_id=slice_id,
            scale_group=scale_group,
            state=vm_pb2.VM_STATE_BOOTING,
            zone=zone,
            created_at=time_pb2.Timestamp(epoch_ms=created_at_ms),
            state_changed_at=time_pb2.Timestamp(epoch_ms=created_at_ms),
        )
        self._boot_delay_ms = boot_delay_ms
        self._init_delay_ms = init_delay_ms

    def tick(self, ts: int) -> None:
        """Process state transitions based on elapsed time."""
        if self.info.state == vm_pb2.VM_STATE_BOOTING:
            elapsed = ts - self.info.state_changed_at.epoch_ms
            if elapsed >= self._boot_delay_ms:
                self.info.state = vm_pb2.VM_STATE_INITIALIZING
                self.info.state_changed_at.CopyFrom(time_pb2.Timestamp(epoch_ms=ts))

        if self.info.state == vm_pb2.VM_STATE_INITIALIZING:
            elapsed = ts - self.info.state_changed_at.epoch_ms
            if elapsed >= self._init_delay_ms:
                self.info.state = vm_pb2.VM_STATE_READY
                self.info.worker_id = f"worker-{self.info.vm_id}"
                self.info.worker_healthy = True
                self.info.state_changed_at.CopyFrom(time_pb2.Timestamp(epoch_ms=ts))


class FakeVmGroup:
    """Fake VM group for testing.

    Holds FakeVm instances and computes status from their states.
    State transitions happen during tick() calls.
    """

    def __init__(
        self,
        slice_id: str,
        scale_group: str,
        zone: str,
        vms: list[FakeVm],
        created_at_ms: int | None = None,
    ):
        self._slice_id = slice_id
        self._scale_group = scale_group
        self._zone = zone
        self._vms = vms
        self._created_at = Timestamp.from_ms(created_at_ms) if created_at_ms is not None else Timestamp.now()
        self._terminated = False

    @property
    def group_id(self) -> str:
        return self._slice_id

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def created_at_ms(self) -> int:
        """Timestamp when this VM group was created (milliseconds since epoch)."""
        return self._created_at.epoch_ms()

    def status(self) -> VmGroupStatus:
        """Compute status from current VM states."""
        snapshots = [
            VmSnapshot(
                vm_id=vm.info.vm_id,
                state=vm.info.state,
                address=vm.info.address,
                init_phase=vm.info.init_phase,
                init_error=vm.info.init_error,
            )
            for vm in self._vms
        ]
        return VmGroupStatus(vms=snapshots)

    def vms(self) -> list[FakeVm]:
        """Get VMs in this VM group.

        Returns FakeVm instances (not ManagedVm). The autoscaler uses this
        to find workers by worker_id, so FakeVm must have a compatible interface.
        """
        return list(self._vms)

    def terminate(self) -> None:
        """Mark VM group as terminated."""
        ts = Timestamp.now().epoch_ms()
        for vm in self._vms:
            vm.info.state = vm_pb2.VM_STATE_TERMINATED
            vm.info.state_changed_at.CopyFrom(time_pb2.Timestamp(epoch_ms=ts))
        self._terminated = True

    def to_proto(self) -> vm_pb2.SliceInfo:
        """Convert to proto for RPC APIs."""
        return vm_pb2.SliceInfo(
            slice_id=self._slice_id,
            scale_group=self._scale_group,
            created_at=self._created_at.to_proto(),
            vms=[vm.info for vm in self._vms],
        )

    def tick(self, ts: int) -> None:
        """Advance VM state transitions."""
        for vm in self._vms:
            vm.tick(ts)


@dataclass
class FakeVmManagerConfig:
    """Configuration for FakeVmManager behavior.

    Args:
        config: Scale group configuration
        boot_delay_ms: Time in ms for VM to transition from BOOTING to INITIALIZING
        init_delay_ms: Time in ms for VM to transition from INITIALIZING to READY
        failure_mode: Failure injection mode for create_vm_group
    """

    config: config_pb2.ScaleGroupConfig
    boot_delay_ms: int = 0
    init_delay_ms: int = 0
    failure_mode: FailureMode = field(default=FailureMode.NONE)


class FakeVmManager:
    """In-memory VmManager for testing.

    Implements VmManagerProtocol for the new architecture. Creates FakeVmGroup
    instances with FakeVm that transition states during tick() calls.

    Thread-safe for use in concurrent tests.
    """

    def __init__(self, config: FakeVmManagerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._slices: dict[str, FakeVmGroup] = {}
        self._slice_counter = 0

    def create_vm_group(self, tags: dict[str, str] | None = None) -> FakeVmGroup:
        """Create a new fake VM group."""
        del tags  # Unused
        if self._config.failure_mode == FailureMode.QUOTA_EXCEEDED:
            raise QuotaExceededError(f"Quota exceeded for {self._config.config.name}")
        if self._config.failure_mode == FailureMode.CREATE_FAILS:
            raise RuntimeError("FakeVmManager configured to fail on create")

        with self._lock:
            self._slice_counter += 1
            slice_id = f"fake-slice-{self._config.config.name}-{self._slice_counter}"
            ts = Timestamp.now().epoch_ms()

            topology = get_tpu_topology(self._config.config.accelerator_variant)
            vm_count = topology.vm_count
            zone = self._config.config.zones[0] if self._config.config.zones else "us-central1-a"

            vms = []
            for i in range(vm_count):
                vm_id = f"{slice_id}-vm-{i}"
                vm = FakeVm(
                    vm_id=vm_id,
                    slice_id=slice_id,
                    scale_group=self._config.config.name,
                    zone=zone,
                    created_at_ms=ts,
                    boot_delay_ms=self._config.boot_delay_ms,
                    init_delay_ms=self._config.init_delay_ms,
                )
                vms.append(vm)

            fake_vm_group = FakeVmGroup(
                slice_id=slice_id,
                scale_group=self._config.config.name,
                zone=zone,
                vms=vms,
                created_at_ms=ts,
            )
            self._slices[slice_id] = fake_vm_group

            logger.debug("Created fake VM group %s with %d VMs", slice_id, vm_count)
            return fake_vm_group

    def discover_vm_groups(self) -> list[FakeVmGroup]:
        """No-op for fake - returns empty list."""
        return []

    def tick(self, ts: int | None = None) -> None:
        """Advance all VM group state transitions.

        Call this to simulate time passing and VMs completing boot/init.
        """
        ts = ts or Timestamp.now().epoch_ms()
        with self._lock:
            for fake_vm_group in self._slices.values():
                fake_vm_group.tick(ts)

    def set_failure_mode(self, mode: FailureMode) -> None:
        """Set the failure mode for subsequent operations."""
        self._config.failure_mode = mode

    def get_slice(self, slice_id: str) -> FakeVmGroup | None:
        """Get a specific VM group by ID."""
        with self._lock:
            return self._slices.get(slice_id)

    def list_slices(self) -> list[FakeVmGroup]:
        """List all VM groups."""
        with self._lock:
            return list(self._slices.values())
