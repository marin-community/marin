# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protobuf enum utilities."""

from iris.rpc import vm_pb2, job_pb2, config_pb2


def vm_state_name(state: int) -> str:
    """Return enum name like 'VM_STATE_READY'."""
    try:
        return vm_pb2.VmState.Name(state)
    except ValueError:
        return f"UNKNOWN({state})"


def job_state_name(state: int) -> str:
    """Return enum name like 'JOB_STATE_RUNNING'."""
    try:
        return job_pb2.JobState.Name(state)
    except ValueError:
        return f"UNKNOWN({state})"


def job_state_friendly(state: int) -> str:
    """Return human-friendly lowercase name like 'running'."""
    return job_state_name(state).removeprefix("JOB_STATE_").lower()


def task_state_name(state: int) -> str:
    """Return enum name like 'TASK_STATE_RUNNING'."""
    try:
        return job_pb2.TaskState.Name(state)
    except ValueError:
        return f"UNKNOWN({state})"


def task_state_friendly(state: int) -> str:
    """Return human-friendly lowercase name like 'running'."""
    return task_state_name(state).removeprefix("TASK_STATE_").lower()


def accelerator_type_name(accel_type: int) -> str:
    """Return enum name like 'ACCELERATOR_TYPE_TPU'."""
    try:
        return config_pb2.AcceleratorType.Name(accel_type)
    except ValueError:
        return f"UNKNOWN({accel_type})"


def job_state_friendly(state: int) -> str:
    """Human-friendly lowercase name for a JobState proto value."""
    return job_state_name(state).removeprefix("JOB_STATE_").lower()


def task_state_friendly(state: int) -> str:
    """Human-friendly lowercase name for a TaskState proto value."""
    return task_state_name(state).removeprefix("TASK_STATE_").lower()


def accelerator_type_friendly(accel_type: int) -> str:
    """Return human-friendly accelerator type name.

    Examples:
        ACCELERATOR_TYPE_UNSPECIFIED (0) -> "unspecified"
        ACCELERATOR_TYPE_CPU (1) -> "cpu"
        ACCELERATOR_TYPE_GPU (2) -> "gpu"
        ACCELERATOR_TYPE_TPU (3) -> "tpu"
    """
    name = accelerator_type_name(accel_type)
    if name.startswith("ACCELERATOR_TYPE_"):
        return name.replace("ACCELERATOR_TYPE_", "").lower()
    return name.lower()


def format_accelerator_display(accel_type: int, variant: str = "") -> str:
    """Format accelerator type and variant for display.

    Examples:
        format_accelerator_display(3, "v5litepod-16") -> "tpu (v5litepod-16)"
        format_accelerator_display(2, "A100") -> "gpu (A100)"
        format_accelerator_display(1, "") -> "cpu"
    """
    friendly = accelerator_type_friendly(accel_type)
    if variant:
        return f"{friendly} ({variant})"
    return friendly


# ---------------------------------------------------------------------------
# PriorityBand helpers
# ---------------------------------------------------------------------------


def priority_band_name(band: int) -> str:
    """Human-friendly lowercase name for a PriorityBand proto value."""
    return job_pb2.PriorityBand.Name(band).removeprefix("PRIORITY_BAND_").lower()


def priority_band_value(name: str) -> int:
    """Proto int value from a human-friendly band name like 'interactive'."""
    return job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{name.upper()}")


PRIORITY_BAND_VALUES: list[int] = [
    job_pb2.PRIORITY_BAND_PRODUCTION,
    job_pb2.PRIORITY_BAND_INTERACTIVE,
    job_pb2.PRIORITY_BAND_BATCH,
]

PRIORITY_BAND_NAMES: list[str] = [priority_band_name(b) for b in PRIORITY_BAND_VALUES]
