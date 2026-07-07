# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from levanter.config import JsonAtom

logger = logging.getLogger(__name__)


def jsonable_topology_value(value: Any) -> JsonAtom | list[Any] | dict[str, Any]:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [jsonable_topology_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): jsonable_topology_value(item) for key, item in value.items()}
    return str(value)


def device_topology_entry(device: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    for attr in (
        "id",
        "process_index",
        "host_id",
        "task_id",
        "platform",
        "device_kind",
        "local_hardware_id",
        "slice_index",
        "coords",
        "core_on_chip",
    ):
        if hasattr(device, attr):
            entry[attr] = jsonable_topology_value(getattr(device, attr))
    return entry


def tpu_topology_shape(devices: Sequence[Any]) -> str | None:
    coords = []
    for device in devices:
        if getattr(device, "platform", None) != "tpu" or not hasattr(device, "coords"):
            continue

        coord = getattr(device, "coords")
        if not isinstance(coord, Sequence) or isinstance(coord, str | bytes | bytearray):
            continue

        coords.append(tuple(int(axis) for axis in coord))

    if not coords:
        return None

    rank = len(coords[0])
    if rank == 0 or any(len(coord) != rank for coord in coords):
        return None

    axis_sizes = []
    for axis in range(rank):
        axis_values = [coord[axis] for coord in coords]
        axis_sizes.append(max(axis_values) - min(axis_values) + 1)

    return "x".join(str(axis_size) for axis_size in axis_sizes)


def nvidia_topology_matrix_summary(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    header: list[str] | None = None
    rows: dict[str, list[str]] = {}

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        if parts[0].startswith("GPU") and header is None:
            header_end = parts.index("CPU") if "CPU" in parts else len(parts)
            header = parts[:header_end]
            continue

        if header is None:
            continue

        row_name = parts[0]
        if row_name.startswith("GPU"):
            rows[row_name] = parts[1 : 1 + len(header)]

    if header is None or not rows:
        return {}

    gpu_labels = [label for label in header if label.startswith("GPU")]
    interface_labels = [label for label in header if not label.startswith("GPU")]
    gpu_gpu_link_counts: dict[str, int] = {}
    gpu_nic_link_counts: dict[str, int] = {}

    for row_index, gpu in enumerate(gpu_labels):
        row = rows.get(gpu)
        if row is None:
            continue

        for col_index, peer_gpu in enumerate(gpu_labels[row_index + 1 :], start=row_index + 1):
            if col_index >= len(row):
                continue
            link = row[col_index]
            if peer_gpu == gpu or link == "X":
                continue
            gpu_gpu_link_counts[link] = gpu_gpu_link_counts.get(link, 0) + 1

        for interface in interface_labels:
            col_index = header.index(interface)
            if col_index >= len(row):
                continue
            link = row[col_index]
            gpu_nic_link_counts[link] = gpu_nic_link_counts.get(link, 0) + 1

    summary: dict[str, Any] = {}
    if gpu_gpu_link_counts:
        summary["gpu_gpu_link_counts"] = gpu_gpu_link_counts
    if gpu_nic_link_counts:
        summary["gpu_nic_link_counts"] = gpu_nic_link_counts
    return summary


def nvidia_smi_topology() -> dict[str, Any] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None

    command = [nvidia_smi, "topo", "-m"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    topology: dict[str, Any] = {
        "command": " ".join(command),
        "returncode": result.returncode,
    }
    if result.stdout:
        topology["stdout"] = result.stdout
        topology_summary = nvidia_topology_matrix_summary(result.stdout)
        if topology_summary:
            topology["summary"] = topology_summary
    if result.stderr:
        topology["stderr"] = result.stderr
    return topology


def hardware_topology_summary() -> dict[str, Any]:
    devices = list(jax.devices())
    local_devices = list(jax.local_devices())
    backend = jax.default_backend()
    topology: dict[str, Any] = {
        "devices": [device_topology_entry(device) for device in devices],
        "local_devices": [device_topology_entry(device) for device in local_devices],
    }

    if devices:
        platform_version = getattr(getattr(devices[0], "client", None), "platform_version", None)
        if platform_version is not None:
            topology["platform_version"] = str(platform_version)

    topology_shape = tpu_topology_shape(devices)
    if topology_shape is not None:
        topology["tpu_topology_shape"] = topology_shape

    if backend == "gpu":
        try:
            gpu_topology = nvidia_smi_topology()
        except (OSError, subprocess.SubprocessError):
            logger.info("Failed to query NVIDIA topology with nvidia-smi.", exc_info=True)
        else:
            if gpu_topology is not None:
                topology["nvidia_smi_topology"] = gpu_topology

    return topology
