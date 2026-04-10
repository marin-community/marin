# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for provider unit tests.

FakeWorkerHandle and FakeSliceHandle are in-memory implementations of the
RemoteWorkerHandle and SliceHandle protocols. They replace MagicMock-based
fakes with type-checkable dataclasses that behave like LocalSliceHandle
but support failure injection for testing error paths.
"""

from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

from iris.cluster.providers.types import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    Labels,
    SliceStatus,
    WorkerStatus as CloudWorkerStatus,
)
from iris.rpc import config_pb2, vm_pb2
from rigging.timing import Duration, Timestamp


def _cloud_worker_state_from_iris(state: vm_pb2.VmState) -> CloudWorkerState:
    """Reverse map from Iris VM state to CloudWorkerState for test setup."""
    if state == vm_pb2.VM_STATE_READY:
        return CloudWorkerState.RUNNING
    if state == vm_pb2.VM_STATE_FAILED:
        return CloudWorkerState.STOPPED
    if state == vm_pb2.VM_STATE_TERMINATED:
        return CloudWorkerState.TERMINATED
    return CloudWorkerState.UNKNOWN


@dataclass
class FakeWorkerHandle:
    """In-memory implementation of RemoteWorkerHandle for tests."""

    _vm_id: str
    _internal_address: str
    _state: CloudWorkerState = CloudWorkerState.RUNNING
    _bootstrap_log: str = ""

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
    def bootstrap_log(self) -> str:
        return self._bootstrap_log

    def status(self) -> CloudWorkerStatus:
        return CloudWorkerStatus(state=self._state)

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        return CommandResult(returncode=0, stdout="", stderr="")

    def reboot(self) -> None:
        pass


@dataclass
class FakeSliceHandle:
    """In-memory implementation of SliceHandle for tests.

    Supports injecting a terminate error via terminate_error for testing
    error-handling paths in ScalingGroup.
    """

    _slice_id: str
    _scale_group: str
    _zone: str
    _labels: dict[str, str]
    _created_at: Timestamp
    _status: SliceStatus
    terminate_error: Exception | None = None
    terminated: bool = False

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
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        return self._status

    def terminate(self, *, wait: bool = False) -> None:
        self.terminated = True
        if self.terminate_error is not None:
            raise self.terminate_error


def make_fake_worker_handle(
    vm_id: str,
    address: str,
    state: vm_pb2.VmState,
    bootstrap_log: str = "",
) -> FakeWorkerHandle:
    return FakeWorkerHandle(
        _vm_id=vm_id,
        _internal_address=address,
        _state=_cloud_worker_state_from_iris(state),
        _bootstrap_log=bootstrap_log,
    )


def make_fake_slice_handle(
    slice_id: str,
    scale_group: str = "test-group",
    all_ready: bool = False,
    any_failed: bool = False,
    vm_states: list[vm_pb2.VmState] | None = None,
    bootstrap_logs: list[str] | None = None,
    created_at_ms: int = 1000000,
) -> FakeSliceHandle:
    if vm_states is None:
        if any_failed:
            vm_states = [vm_pb2.VM_STATE_FAILED]
        elif all_ready:
            vm_states = [vm_pb2.VM_STATE_READY]
        else:
            vm_states = [vm_pb2.VM_STATE_BOOTING]

    # Derive slice state from VM states
    if any(s == vm_pb2.VM_STATE_FAILED for s in vm_states):
        slice_state = CloudSliceState.READY
    elif all(s == vm_pb2.VM_STATE_READY for s in vm_states):
        slice_state = CloudSliceState.READY
    elif all(s == vm_pb2.VM_STATE_TERMINATED for s in vm_states):
        slice_state = CloudSliceState.DELETING
    else:
        slice_state = CloudSliceState.CREATING

    slice_hash = abs(hash(slice_id)) % 256
    worker_handles: list[FakeWorkerHandle] = []
    for i, state in enumerate(vm_states):
        bootstrap_log = bootstrap_logs[i] if bootstrap_logs and i < len(bootstrap_logs) else ""
        worker_handle = make_fake_worker_handle(
            vm_id=f"{slice_id}-vm-{i}",
            address=f"10.0.{slice_hash}.{i}",
            state=state,
            bootstrap_log=bootstrap_log,
        )
        worker_handles.append(worker_handle)

    iris_labels = Labels("iris")
    return FakeSliceHandle(
        _slice_id=slice_id,
        _scale_group=scale_group,
        _zone="us-central1-a",
        _labels={iris_labels.iris_scale_group: scale_group, iris_labels.iris_managed: "true"},
        _created_at=Timestamp.from_ms(created_at_ms),
        _status=SliceStatus(state=slice_state, worker_count=len(vm_states), workers=worker_handles),
    )


def make_mock_platform(
    slices_to_discover: list[FakeSliceHandle] | None = None,
) -> MagicMock:
    """Create a mock WorkerInfraProvider for testing.

    The platform itself stays as a MagicMock because tests need mock assertions
    on create_slice (assert_called_once, side_effect, call_args). The slice and
    worker handles it returns are real FakeSliceHandle/FakeWorkerHandle instances.
    """
    platform = MagicMock()
    platform.list_slices.return_value = slices_to_discover or []

    create_count = [0]

    def create_slice_side_effect(config: config_pb2.SliceConfig, worker_config=None) -> FakeSliceHandle:
        create_count[0] += 1
        slice_id = f"new-slice-{create_count[0]}"
        return make_fake_slice_handle(slice_id)

    platform.create_slice.side_effect = create_slice_side_effect
    return platform


# Keep old names as aliases for backward compatibility with imports in other test files.
make_mock_slice_handle = make_fake_slice_handle
make_mock_worker_handle = make_fake_worker_handle
