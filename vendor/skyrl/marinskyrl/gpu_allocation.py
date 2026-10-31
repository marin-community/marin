"""Bind CUDA discovery to the device-plugin allocation, including privileged tasks."""

import os
from pathlib import Path

NVIDIA_DEVICE_ALLOCATION_DIRECTORY = Path("/var/run/nvidia-container-devices")


def pin_task_cuda_devices() -> str | None:
    """Apply and return the GPU allocation mask, or None if no bounded allocation exists."""
    allocation = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    # Some task images keep NVIDIA_VISIBLE_DEVICES=void even when the plugin
    # has mounted its authoritative allocation. Privilege still exposes all
    # host device nodes, so the mounted UUID files must take precedence.
    if allocation in {None, "all", "void"} and NVIDIA_DEVICE_ALLOCATION_DIRECTORY.is_dir():
        allocation = str(NVIDIA_DEVICE_ALLOCATION_DIRECTORY)
    if not allocation or allocation in {"all", "none", "void"}:
        return None
    if allocation.startswith("/"):
        # CoreWeave uses the NVIDIA device plugin's volume-mount strategy.
        devices = sorted(path.name for path in Path(allocation).iterdir())
        if not devices or any(not device.startswith(("GPU-", "MIG-")) for device in devices):
            raise ValueError("NVIDIA device allocation directory contains no valid GPU allocation")
        allocation = ",".join(devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = allocation
    return allocation
