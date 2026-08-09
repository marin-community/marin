# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protobuf enum and value display helpers."""

import signal

import humanfriendly
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from iris.cluster.resources.execution import GpuDevice, ResourceSpec, TpuDevice
from iris.cluster.resources.state import JobState, PriorityBand, TaskState
from iris.rpc import vm_pb2


def signal_name(signum: int) -> str:
    """Return the canonical signal name for ``signum`` (e.g. 9 -> 'SIGKILL').

    Falls back to ``'signal N'`` for numbers that are not valid signals. Used to
    interpret process exit codes above 128, where the killing signal is
    ``exit_code - 128``.
    """
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _enum_name(enum_wrapper: EnumTypeWrapper, value: int) -> str:
    """Return the proto enum name for ``value``, or ``UNKNOWN(value)`` if unmapped."""
    try:
        return enum_wrapper.Name(value)
    except ValueError:
        return f"UNKNOWN({value})"


def vm_state_name(state: int) -> str:
    """Return enum name like 'VM_STATE_READY'."""
    return _enum_name(vm_pb2.VmState, state)


def job_state_name(state: int) -> str:
    """Return enum name like 'JOB_STATE_RUNNING'."""
    try:
        return f"JOB_STATE_{JobState(state).name}"
    except ValueError:
        return f"UNKNOWN({state})"


def job_state_friendly(state: int) -> str:
    """Return human-friendly lowercase name like 'running'."""
    return job_state_name(state).removeprefix("JOB_STATE_").lower()


def task_state_name(state: int) -> str:
    """Return enum name like 'TASK_STATE_RUNNING'."""
    try:
        return f"TASK_STATE_{TaskState(state).name}"
    except ValueError:
        return f"UNKNOWN({state})"


def task_state_friendly(state: int) -> str:
    """Return human-friendly lowercase name like 'running'."""
    return task_state_name(state).removeprefix("TASK_STATE_").lower()


def format_resources(resources: ResourceSpec | None) -> str:
    """Format a native resource specification as a compact summary.

    Examples:
        format_resources(...) -> "0.5 cpu, 8 GiB, 5 GiB disk, v5litepod-16"
        format_resources(...) -> "8 cpu, 32 GiB, 8xH100"
        format_resources(None) -> "-"
    """
    if not resources:
        return "-"
    parts: list[str] = []
    if resources.cpu:
        parts.append(f"{resources.cpu:g} cpu")
    if resources.memory:
        parts.append(humanfriendly.format_size(resources.memory, binary=True))
    if resources.disk:
        parts.append(f"{humanfriendly.format_size(resources.disk, binary=True)} disk")
    if isinstance(resources.device, TpuDevice):
        parts.append(resources.device.variant)
    elif isinstance(resources.device, GpuDevice):
        gpu = resources.device
        parts.append(f"{gpu.count}x{gpu.variant}" if gpu.variant else f"{gpu.count}gpu")
    return ", ".join(parts) if parts else "-"


def format_accelerator_display(device_type: str, variant: str = "") -> str:
    """Format an accelerator device type and variant for display.

    Examples:
        format_accelerator_display("tpu", "v5litepod-16") -> "tpu (v5litepod-16)"
        format_accelerator_display("gpu", "A100") -> "gpu (A100)"
        format_accelerator_display("cpu", "") -> "cpu"
    """
    friendly = device_type or "unspecified"
    if variant:
        return f"{friendly} ({variant})"
    return friendly


# ---------------------------------------------------------------------------
# PriorityBand helpers
# ---------------------------------------------------------------------------


def priority_band_name(band: int) -> str:
    """Human-friendly lowercase name for a priority band value."""
    return PriorityBand(band).name.lower()


def priority_band_value(name: str) -> int:
    """Integer value from a human-friendly band name like 'interactive'."""
    return int(PriorityBand[name.upper()])


PRIORITY_BAND_VALUES: list[int] = [
    int(PriorityBand.PRODUCTION),
    int(PriorityBand.INTERACTIVE),
    int(PriorityBand.BATCH),
]

PRIORITY_BAND_NAMES: list[str] = [priority_band_name(b) for b in PRIORITY_BAND_VALUES]
