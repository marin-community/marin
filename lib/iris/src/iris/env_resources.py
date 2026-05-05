# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query the resource limits visible to the current process.

Provides a :class:`TaskResources` dataclass with a :meth:`from_environment`
factory that discovers limits using an escalation strategy:

1. ``IRIS_TASK_RESOURCES`` env var (proto-JSON for ``ResourceSpecProto``,
   set by the Iris worker on task launch).
2. cgroup v2 / v1 files (container memory/cpu limits).
3. ``/proc/meminfo`` and ``os.cpu_count()`` — bare-metal fallback with warning.

GPU/TPU counts are only available via the ``IRIS_TASK_RESOURCES`` env var.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os

from google.protobuf import json_format

from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

_IRIS_TASK_RESOURCES_ENV = "IRIS_TASK_RESOURCES"

# cgroup paths in preference order (v2 first, then v1).
_CGROUP_MEMORY_PATHS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)
_CGROUP_CPU_QUOTA_PATH = "/sys/fs/cgroup/cpu.max"  # cgroup v2: "quota period"
_CGROUP_CPU_QUOTA_V1_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_CPU_PERIOD_V1_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


@dataclasses.dataclass(frozen=True)
class TaskResources:
    """Snapshot of the resource limits visible to this process.

    Use :meth:`from_environment` to build an instance from the current
    process environment.
    """

    memory_bytes: int
    cpu_cores: float
    gpu_count: int
    tpu_count: int

    @staticmethod
    def from_environment() -> TaskResources:
        """Discover resource limits from env var, cgroups, and /proc.

        CPU/memory fall back to host values when no Iris env var or cgroup
        limit is found. GPU/TPU counts default to 0 when running outside an
        Iris task.
        """
        proto = _read_iris_resource_proto()

        memory = _resolve_memory(proto)
        cpu = _resolve_cpu(proto)
        gpu = (
            proto.device.gpu.count
            if proto is not None and proto.HasField("device") and proto.device.HasField("gpu")
            else 0
        )
        tpu = (
            proto.device.tpu.count
            if proto is not None and proto.HasField("device") and proto.device.HasField("tpu")
            else 0
        )

        result = TaskResources(memory_bytes=memory, cpu_cores=cpu, gpu_count=gpu, tpu_count=tpu)
        logger.info("TaskResources: %s", result)
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@functools.cache
def _read_iris_resource_proto() -> job_pb2.ResourceSpecProto | None:
    """Parse ``IRIS_TASK_RESOURCES`` into a ``ResourceSpecProto``, or None."""
    raw = os.environ.get(_IRIS_TASK_RESOURCES_ENV)
    if not raw:
        return None
    try:
        proto = job_pb2.ResourceSpecProto()
        json_format.Parse(raw, proto)
        return proto
    except json_format.ParseError:
        logger.warning("Failed to parse %s: %s", _IRIS_TASK_RESOURCES_ENV, raw)
        return None


def _read_cgroup_file(path: str) -> str | None:
    try:
        with open(path) as f:
            value = f.read().strip()
        if value and value != "max":
            return value
    except (FileNotFoundError, PermissionError):
        pass
    return None


def _read_proc_meminfo_total() -> int | None:
    """Read total memory from ``/proc/meminfo`` (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:       16384000 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB → bytes
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass
    return None


def _resolve_memory(proto: job_pb2.ResourceSpecProto | None) -> int:
    """Return memory limit in bytes using the escalation chain."""
    if proto is not None and proto.memory_bytes:
        return proto.memory_bytes

    for path in _CGROUP_MEMORY_PATHS:
        value = _read_cgroup_file(path)
        if value is not None:
            return int(value)

    total = _read_proc_meminfo_total()
    if total is not None:
        logger.warning(
            "No IRIS_TASK_RESOURCES or cgroup memory limit found, falling back to /proc/meminfo: %d bytes",
            total,
        )
        return total

    # Last resort: 0 signals unknown
    logger.warning("Cannot determine memory limit from env, cgroups, or /proc/meminfo")
    return 0


def _resolve_cpu(proto: job_pb2.ResourceSpecProto | None) -> float:
    """Return CPU limit in cores using the escalation chain."""
    if proto is not None and proto.cpu_millicores:
        return proto.cpu_millicores / 1000

    # cgroup v2: "quota period" (e.g., "200000 100000" means 2 cores)
    value = _read_cgroup_file(_CGROUP_CPU_QUOTA_PATH)
    if value is not None:
        parts = value.split()
        if len(parts) == 2 and parts[0] != "max":
            return int(parts[0]) / int(parts[1])

    # cgroup v1
    quota = _read_cgroup_file(_CGROUP_CPU_QUOTA_V1_PATH)
    period = _read_cgroup_file(_CGROUP_CPU_PERIOD_V1_PATH)
    if quota is not None and period is not None and int(quota) > 0:
        return int(quota) / int(period)

    cpus = float(os.cpu_count() or 1)
    logger.warning(
        "No IRIS_TASK_RESOURCES or cgroup CPU limit found, falling back to os.cpu_count(): %.1f cores",
        cpus,
    )
    return cpus
