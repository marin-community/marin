# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protobuf enum utilities."""

import humanfriendly
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from iris.rpc import config_pb2, job_pb2, vm_pb2


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
    return _enum_name(job_pb2.JobState, state)


def job_state_friendly(state: int) -> str:
    """Return human-friendly lowercase name like 'running'."""
    return job_state_name(state).removeprefix("JOB_STATE_").lower()


def task_state_name(state: int) -> str:
    """Return enum name like 'TASK_STATE_RUNNING'."""
    return _enum_name(job_pb2.TaskState, state)


def task_state_friendly(state: int) -> str:
    """Return human-friendly lowercase name like 'running'."""
    return task_state_name(state).removeprefix("TASK_STATE_").lower()


def accelerator_type_name(accel_type: int) -> str:
    """Return enum name like 'ACCELERATOR_TYPE_TPU'."""
    return _enum_name(config_pb2.AcceleratorType, accel_type)


def accelerator_type_friendly(accel_type: int) -> str:
    """Return human-friendly accelerator type name.

    Examples:
        ACCELERATOR_TYPE_UNSPECIFIED (0) -> "unspecified"
        ACCELERATOR_TYPE_CPU (1) -> "cpu"
        ACCELERATOR_TYPE_GPU (2) -> "gpu"
        ACCELERATOR_TYPE_TPU (3) -> "tpu"
    """
    return accelerator_type_name(accel_type).removeprefix("ACCELERATOR_TYPE_").lower()


def format_resources(resources: job_pb2.ResourceSpecProto | None) -> str:
    """Format a ResourceSpec proto as a compact comma-separated summary.

    Examples:
        format_resources(...) -> "0.5 cpu, 8 GiB, 5 GiB disk, v5litepod-16"
        format_resources(...) -> "8 cpu, 32 GiB, 8xH100"
        format_resources(None) -> "-"
    """
    if not resources:
        return "-"
    parts: list[str] = []
    if resources.cpu_millicores:
        parts.append(f"{resources.cpu_millicores / 1000:g} cpu")
    if resources.memory_bytes:
        parts.append(humanfriendly.format_size(resources.memory_bytes, binary=True))
    if resources.disk_bytes:
        parts.append(f"{humanfriendly.format_size(resources.disk_bytes, binary=True)} disk")
    if resources.HasField("device"):
        device = resources.device
        if device.HasField("tpu"):
            parts.append(device.tpu.variant)
        elif device.HasField("gpu"):
            gpu = device.gpu
            parts.append(f"{gpu.count}x{gpu.variant}" if gpu.variant else f"{gpu.count}gpu")
    return ", ".join(parts) if parts else "-"


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


# ---------------------------------------------------------------------------
# ContainerProfile helpers
# ---------------------------------------------------------------------------


# User-selectable profiles, ordered ascending by privilege.
CONTAINER_PROFILE_VALUES: list[int] = [
    job_pb2.CONTAINER_PROFILE_RESTRICTED,
    job_pb2.CONTAINER_PROFILE_DEFAULT,
    job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS,
    job_pb2.CONTAINER_PROFILE_PRIVILEGED,
]

CONTAINER_PROFILE_NAMES: list[str] = [job_pb2.ContainerProfile.Name(p) for p in CONTAINER_PROFILE_VALUES]


def resolve_container_profile(profile: int) -> int:
    """Resolve UNSPECIFIED to the DEFAULT profile; pass others through."""
    if profile == job_pb2.CONTAINER_PROFILE_UNSPECIFIED:
        return job_pb2.CONTAINER_PROFILE_DEFAULT
    return profile
