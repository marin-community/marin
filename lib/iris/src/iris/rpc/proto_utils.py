# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protobuf enum utilities."""

from google.protobuf.descriptor import EnumDescriptor

from iris.rpc import vm_pb2, cluster_pb2, config_pb2

_VM_STATE = vm_pb2.DESCRIPTOR.enum_types_by_name["VmState"]
_JOB_STATE = cluster_pb2.DESCRIPTOR.enum_types_by_name["JobState"]
_TASK_STATE = cluster_pb2.DESCRIPTOR.enum_types_by_name["TaskState"]
_ACCELERATOR_TYPE = config_pb2.DESCRIPTOR.enum_types_by_name["AcceleratorType"]


def _enum_name(descriptor: EnumDescriptor, value: int) -> str:
    """Look up the name of a protobuf enum value, returning UNKNOWN(value) for missing entries."""
    try:
        return descriptor.values_by_number[value].name
    except KeyError:
        return f"UNKNOWN({value})"


def vm_state_name(state: int) -> str:
    """Return enum name like 'VM_STATE_READY'."""
    return _enum_name(_VM_STATE, state)


def job_state_name(state: int) -> str:
    """Return enum name like 'JOB_STATE_RUNNING'."""
    return _enum_name(_JOB_STATE, state)


def task_state_name(state: int) -> str:
    """Return enum name like 'TASK_STATE_RUNNING'."""
    return _enum_name(_TASK_STATE, state)


def accelerator_type_name(accel_type: int) -> str:
    """Return enum name like 'ACCELERATOR_TYPE_TPU'."""
    return _enum_name(_ACCELERATOR_TYPE, accel_type)


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
