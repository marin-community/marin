# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protobuf enum utilities."""

from iris.rpc import vm_pb2, cluster_pb2, config_pb2


def vm_state_name(state: int) -> str:
    """Return enum name like 'VM_STATE_READY'."""
    try:
        return vm_pb2.VmState.Name(state)
    except ValueError:
        return f"UNKNOWN({state})"


def job_state_name(state: int) -> str:
    """Return enum name like 'JOB_STATE_RUNNING'."""
    try:
        return cluster_pb2.JobState.Name(state)
    except ValueError:
        return f"UNKNOWN({state})"


def task_state_name(state: int) -> str:
    """Return enum name like 'TASK_STATE_RUNNING'."""
    try:
        return cluster_pb2.TaskState.Name(state)
    except ValueError:
        return f"UNKNOWN({state})"


def accelerator_type_name(accel_type: int) -> str:
    """Return enum name like 'ACCELERATOR_TYPE_TPU'."""
    try:
        return config_pb2.AcceleratorType.Name(accel_type)
    except ValueError:
        return f"UNKNOWN({accel_type})"


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
