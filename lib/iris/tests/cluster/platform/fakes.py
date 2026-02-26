# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake implementations for testing.

Provides:
- FakePlatform / FakeSliceHandle / FakeWorkerHandle: In-memory Platform that
  simulates worker lifecycle with configurable delays and failure injection.
- FakeGcloud: In-memory gcloud CLI fake that intercepts subprocess.run calls,
  simulating TPU and GCE instance lifecycle for testing GcpPlatform.

Usage (FakePlatform):
    config = config_pb2.ScaleGroupConfig(name="test-group", ...)
    platform = FakePlatform(FakePlatformConfig(config=config))
    handle = platform.create_slice(slice_config)
    platform.tick(ts=now_ms())

Usage (FakeGcloud):
    fake = FakeGcloud()
    fake.set_failure("tpu_create", "RESOURCE_EXHAUSTED: no capacity")
    # Use via the fake_gcloud pytest fixture, which patches subprocess.run.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto
from unittest.mock import patch

import pytest

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

            topology = get_tpu_topology(self._config.config.accelerator_variant)
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
# FakeGcloud — in-memory gcloud CLI fake for testing GcpPlatform
# =============================================================================


@dataclass
class FakeResult:
    """Drop-in for subprocess.CompletedProcess returned by FakeGcloud."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _parse_flag(cmd: list[str], flag: str) -> str | None:
    """Extract value from --flag=value style args in a command list.

    Returns None if the flag is not present.
    """
    prefix = f"--{flag}="
    for arg in cmd:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _parse_labels_string(s: str) -> dict[str, str]:
    """Parse 'k1=v1,k2=v2' into {'k1': 'v1', 'k2': 'v2'}."""
    if not s:
        return {}
    result: dict[str, str] = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _parse_filter_labels(filter_str: str) -> dict[str, str]:
    """Parse gcloud filter expressions like 'labels.k=v AND labels.k2=v2' into a dict."""
    if not filter_str:
        return {}
    labels: dict[str, str] = {}
    for match in re.finditer(r"labels\.([^=\s]+)=([^\s]+)", filter_str):
        labels[match.group(1)] = match.group(2)
    return labels


@dataclass
class FakeGcloud:
    """In-memory gcloud CLI fake that intercepts subprocess.run calls.

    Maintains TPU and VM state dictionaries keyed by (name, zone). Each gcloud
    subcommand is dispatched to a handler method. Unrecognized commands raise
    ValueError so test bugs surface immediately.
    """

    _tpus: dict[tuple[str, str], dict] = field(default_factory=dict)
    _vms: dict[tuple[str, str], dict] = field(default_factory=dict)
    _failures: dict[str, tuple[str, int]] = field(default_factory=dict)
    # Serial port output per VM, appended to by tests to simulate startup-script progress.
    _serial_output: dict[tuple[str, str], str] = field(default_factory=dict)

    def set_failure(self, operation: str, error: str, code: int = 1) -> None:
        """Make a specific operation type fail on the next call.

        Args:
            operation: One of "tpu_create", "tpu_list", "tpu_describe", "tpu_delete",
                       "vm_create", "vm_list", "vm_describe", "vm_delete".
            error: The stderr message to return.
            code: The returncode to return (default 1).
        """
        self._failures[operation] = (error, code)

    def clear_failure(self) -> None:
        """Remove all injected failures."""
        self._failures.clear()

    def _check_failure(self, operation: str) -> FakeResult | None:
        if operation in self._failures:
            error, code = self._failures.pop(operation)
            return FakeResult(returncode=code, stderr=error)
        return None

    def __call__(self, cmd: list[str], **kwargs) -> FakeResult:
        """Drop-in replacement for subprocess.run. Dispatches by gcloud subcommand."""
        if not cmd or cmd[0] != "gcloud":
            raise ValueError(f"FakeGcloud: unrecognized command: {cmd}")

        # Extract subcommand tokens, skipping flags and their arguments.
        # Handles both --flag=value and --flag value forms.
        tokens: list[str] = []
        skip_next = False
        for arg in cmd[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg.startswith("--"):
                # If it's --flag (no =), the next arg is the value
                if "=" not in arg:
                    skip_next = True
                continue
            tokens.append(arg)

        if _matches_gcloud(tokens, ["compute", "tpus", "tpu-vm", "create", None]):
            return self._tpu_create(cmd, tokens[4])
        if _matches_gcloud(tokens, ["compute", "tpus", "tpu-vm", "list"]):
            return self._tpu_list(cmd)
        if _matches_gcloud(tokens, ["compute", "tpus", "tpu-vm", "describe", None]):
            return self._tpu_describe(cmd, tokens[4])
        if _matches_gcloud(tokens, ["compute", "tpus", "tpu-vm", "delete", None]):
            return self._tpu_delete(cmd, tokens[4])
        if _matches_gcloud(tokens, ["compute", "instances", "create", None]):
            return self._vm_create(cmd, tokens[3])
        if _matches_gcloud(tokens, ["compute", "instances", "describe", None]):
            return self._vm_describe(cmd, tokens[3])
        if _matches_gcloud(tokens, ["compute", "instances", "list"]):
            return self._vm_list(cmd)
        if _matches_gcloud(tokens, ["compute", "instances", "delete", None]):
            return self._vm_delete(cmd, tokens[3])
        if _matches_gcloud(tokens, ["compute", "instances", "update", None]):
            return self._vm_update(cmd, tokens[3])
        if _matches_gcloud(tokens, ["compute", "instances", "add-metadata", None]):
            return self._vm_add_metadata(cmd, tokens[3])
        if _matches_gcloud(tokens, ["compute", "instances", "get-serial-port-output", None]):
            return self._vm_get_serial_port_output(cmd, tokens[3])

        raise ValueError(f"FakeGcloud: unrecognized command: {cmd}")

    # -------------------------------------------------------------------------
    # TPU handlers
    # -------------------------------------------------------------------------

    def _tpu_create(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("tpu_create"):
            return failure

        accel_type = _parse_flag(cmd, "accelerator-type")
        if not accel_type:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.tpus.tpu-vm.create) argument --accelerator-type: expected one argument",
            )

        zone = _parse_flag(cmd, "zone")
        if not zone:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.tpus.tpu-vm.create) argument --zone: expected one argument",
            )

        labels: dict[str, str] = {}
        # Labels can appear as --labels=K=V,... or --labels K=V,...
        labels_str = _parse_flag(cmd, "labels")
        if labels_str is None:
            # Check for the two-token form: --labels K=V,...
            for i, arg in enumerate(cmd):
                if arg == "--labels" and i + 1 < len(cmd):
                    labels_str = cmd[i + 1]
                    break
        if labels_str:
            labels = _parse_labels_string(labels_str)

        idx = len(self._tpus)
        tpu_data = {
            "name": name,
            "state": "READY",
            "acceleratorType": accel_type,
            "labels": labels,
            "networkEndpoints": [{"ipAddress": f"10.0.0.{idx + 1}"}],
            "createTime": "2024-01-15T10:30:00.000Z",
        }
        self._tpus[(name, zone)] = tpu_data
        return FakeResult(returncode=0)

    def _tpu_list(self, cmd: list[str]) -> FakeResult:
        if failure := self._check_failure("tpu_list"):
            return failure

        zone = _parse_flag(cmd, "zone")
        filter_str = _parse_flag(cmd, "filter")
        required_labels = _parse_filter_labels(filter_str) if filter_str else {}

        matching = []
        for (_, tpu_zone), tpu in self._tpus.items():
            if zone and tpu_zone != zone:
                continue
            tpu_labels = tpu.get("labels", {})
            if all(tpu_labels.get(k) == v for k, v in required_labels.items()):
                matching.append(tpu)

        return FakeResult(returncode=0, stdout=json.dumps(matching))

    def _tpu_describe(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("tpu_describe"):
            return failure

        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._tpus:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        return FakeResult(returncode=0, stdout=json.dumps(self._tpus[key]))

    def _tpu_delete(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("tpu_delete"):
            return failure

        zone = _parse_flag(cmd, "zone")
        self._tpus.pop((name, zone), None)
        return FakeResult(returncode=0)

    # -------------------------------------------------------------------------
    # VM handlers
    # -------------------------------------------------------------------------

    def _vm_create(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("vm_create"):
            return failure

        machine_type = _parse_flag(cmd, "machine-type")
        if not machine_type:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.instances.create) argument --machine-type: expected one argument",
            )

        zone = _parse_flag(cmd, "zone")
        if not zone:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.instances.create) argument --zone: expected one argument",
            )

        labels_str = _parse_flag(cmd, "labels")
        labels = _parse_labels_string(labels_str) if labels_str else {}

        metadata_str = _parse_flag(cmd, "metadata")
        metadata = _parse_labels_string(metadata_str) if metadata_str else {}

        # Parse --metadata-from-file=key=path and read file contents into metadata.
        metadata_from_file_str = _parse_flag(cmd, "metadata-from-file")
        if metadata_from_file_str and "=" in metadata_from_file_str:
            key, path = metadata_from_file_str.split("=", 1)
            try:
                with open(path) as f:
                    metadata[key] = f.read()
            except OSError:
                pass

        idx = len(self._vms) + 1
        vm_data = {
            "name": name,
            "status": "RUNNING",
            "networkInterfaces": [
                {
                    "networkIP": f"10.128.0.{idx}",
                    "accessConfigs": [{"natIP": f"34.1.2.{idx}"}],
                }
            ],
            "labels": labels,
            "metadata": metadata,
        }
        self._vms[(name, zone)] = vm_data
        return FakeResult(returncode=0, stdout=json.dumps([vm_data]))

    def _vm_describe(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("vm_describe"):
            return failure

        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        fmt = _parse_flag(cmd, "format")
        if fmt == "value(status)":
            return FakeResult(returncode=0, stdout=self._vms[key]["status"] + "\n")

        return FakeResult(returncode=0, stdout=json.dumps(self._vms[key]))

    def _vm_list(self, cmd: list[str]) -> FakeResult:
        if failure := self._check_failure("vm_list"):
            return failure

        # gcloud instances list uses --zones (plural)
        zones_str = _parse_flag(cmd, "zones")
        zones = set(zones_str.split(",")) if zones_str else set()
        filter_str = _parse_flag(cmd, "filter")
        required_labels = _parse_filter_labels(filter_str) if filter_str else {}

        matching = []
        for (_, vm_zone), vm in self._vms.items():
            if zones and vm_zone not in zones:
                continue
            vm_labels = vm.get("labels", {})
            if all(vm_labels.get(k) == v for k, v in required_labels.items()):
                matching.append(vm)

        return FakeResult(returncode=0, stdout=json.dumps(matching))

    def _vm_delete(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("vm_delete"):
            return failure

        zone = _parse_flag(cmd, "zone")
        self._vms.pop((name, zone), None)
        return FakeResult(returncode=0)

    def _vm_update(self, cmd: list[str], name: str) -> FakeResult:
        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        update_labels_str = _parse_flag(cmd, "update-labels")
        if update_labels_str:
            new_labels = _parse_labels_string(update_labels_str)
            self._vms[key].setdefault("labels", {}).update(new_labels)

        return FakeResult(returncode=0)

    def _vm_add_metadata(self, cmd: list[str], name: str) -> FakeResult:
        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        metadata_str = _parse_flag(cmd, "metadata")
        if metadata_str:
            new_metadata = _parse_labels_string(metadata_str)
            self._vms[key].setdefault("metadata", {}).update(new_metadata)

        return FakeResult(returncode=0)

    def _vm_get_serial_port_output(self, cmd: list[str], name: str) -> FakeResult:
        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        full_output = self._serial_output.get(key, "")
        start_str = _parse_flag(cmd, "start")
        start = int(start_str) if start_str else 0
        return FakeResult(returncode=0, stdout=full_output[start:])

    def append_serial_output(self, name: str, zone: str, text: str) -> None:
        """Append text to a VM's serial port output buffer for testing."""
        key = (name, zone)
        self._serial_output[key] = self._serial_output.get(key, "") + text


def _matches_gcloud(tokens: list[str], pattern: list[str | None]) -> bool:
    """Check if tokens match a pattern where None means 'any single token'."""
    if len(tokens) != len(pattern):
        return False
    return all(p is None or t == p for t, p in zip(tokens, pattern, strict=True))


@pytest.fixture
def fake_gcloud() -> FakeGcloud:
    """Pytest fixture that patches iris.cluster.platform.gcp.subprocess.run with a FakeGcloud."""
    fake = FakeGcloud()
    with patch("iris.cluster.platform.gcp.subprocess.run", fake):
        yield fake
