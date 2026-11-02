"""Keep CUDA discovery within the GPU allocation exposed to an Iris task."""

import os
from pathlib import Path

NVIDIA_DEVICE_ALLOCATION_DIRECTORY = Path("/var/run/nvidia-container-devices")
UNBOUNDED_NVIDIA_VISIBILITY = frozenset({None, "all", "none", "void"})


def constrain_cuda_device_discovery() -> str | None:
    """Apply the task allocation as CUDA's discovery mask when one is available.

    This limits ordinary CUDA enumeration in a privileged task container. It is
    not a security boundary and does not prevent direct access to host devices.
    """
    allocation = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    # Some task images retain NVIDIA_VISIBLE_DEVICES=void even when the device
    # plugin mounts its authoritative allocation as UUID-named files.
    if allocation in {None, "all", "void"} and NVIDIA_DEVICE_ALLOCATION_DIRECTORY.is_dir():
        allocation = str(NVIDIA_DEVICE_ALLOCATION_DIRECTORY)
    if allocation in UNBOUNDED_NVIDIA_VISIBILITY:
        return None
    assert allocation is not None
    if Path(allocation).is_absolute():
        devices = sorted(path.name for path in Path(allocation).iterdir())
        if not devices or any(not device.startswith(("GPU-", "MIG-")) for device in devices):
            raise ValueError("NVIDIA device allocation directory contains no valid GPU allocation")
        allocation = ",".join(devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = allocation
    return allocation
