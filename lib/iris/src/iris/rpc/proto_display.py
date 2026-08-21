# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protobuf enum and value display helpers."""

import signal

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from iris.rpc import job_pb2, vm_pb2


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
    """Human-friendly lowercase name for a PriorityBand proto value."""
    return job_pb2.PriorityBand.Name(band).removeprefix("PRIORITY_BAND_").lower()


def priority_band_value(name: str) -> int:
    """Proto int value from a human-friendly band name like 'interactive'."""
    return job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{name.upper()}")


PRIORITY_BAND_VALUES: list[int] = [
    job_pb2.PRIORITY_BAND_PRODUCTION,
    job_pb2.PRIORITY_BAND_PRIORITY,
    job_pb2.PRIORITY_BAND_INTERACTIVE,
    job_pb2.PRIORITY_BAND_BATCH,
]

PRIORITY_BAND_NAMES: list[str] = [priority_band_name(b) for b in PRIORITY_BAND_VALUES]
_PRIORITY_BAND_RANKS = {band: rank for rank, band in enumerate(PRIORITY_BAND_VALUES)}


def priority_band_rank(band: int) -> int:
    """Scheduling rank for a real PriorityBand; lower ranks run first."""
    try:
        return _PRIORITY_BAND_RANKS[band]
    except KeyError as err:
        raise ValueError(f"Unknown priority band {band}") from err


# ---------------------------------------------------------------------------
# ContainerProfile helpers
# ---------------------------------------------------------------------------


# User-selectable profiles, ordered ascending by privilege.
CONTAINER_PROFILE_VALUES: list[int] = [
    job_pb2.CONTAINER_PROFILE_RESTRICTED,
    job_pb2.CONTAINER_PROFILE_DEFAULT,
    job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS,
    job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    job_pb2.CONTAINER_PROFILE_GVISOR,
]

CONTAINER_PROFILE_NAMES: list[str] = [job_pb2.ContainerProfile.Name(p) for p in CONTAINER_PROFILE_VALUES]


def resolve_container_profile(profile: int) -> int:
    """Resolve UNSPECIFIED to the DEFAULT profile; pass others through."""
    if profile == job_pb2.CONTAINER_PROFILE_UNSPECIFIED:
        return job_pb2.CONTAINER_PROFILE_DEFAULT
    return profile
