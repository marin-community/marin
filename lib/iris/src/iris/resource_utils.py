# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for querying the resource limits visible to the current process.

Uses an escalation strategy:
1. ``IRIS_TASK_RESOURCES`` env var (JSON with ``memory_bytes``, ``cpu_millicores``,
   ``disk_bytes``) — set by the Iris worker when launching task containers.
2. cgroup v2 / v1 files — when running inside a container with cgroup limits.
3. psutil — bare-metal fallback using host-level information.
"""

from __future__ import annotations

import functools
import json
import logging
import os

import psutil

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


@functools.cache
def _read_iris_env() -> dict | None:
    raw = os.environ.get(_IRIS_TASK_RESOURCES_ENV)
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
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


def get_memory_limit() -> int:
    """Return the memory limit in bytes visible to this process.

    Escalation: IRIS_TASK_RESOURCES → cgroup → psutil.
    """
    env = _read_iris_env()
    if env and "memory_bytes" in env:
        return int(env["memory_bytes"])

    for path in _CGROUP_MEMORY_PATHS:
        value = _read_cgroup_file(path)
        if value is not None:
            return int(value)

    total = psutil.virtual_memory().total
    logger.warning(
        "No IRIS_TASK_RESOURCES or cgroup memory limit found, falling back to psutil: %d bytes",
        total,
    )
    return total


def get_cpu_limit() -> float:
    """Return the CPU limit in cores visible to this process.

    Escalation: IRIS_TASK_RESOURCES → cgroup → psutil.
    """
    env = _read_iris_env()
    if env and "cpu_millicores" in env:
        return int(env["cpu_millicores"]) / 1000

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
        "No IRIS_TASK_RESOURCES or cgroup CPU limit found, falling back to psutil: %.1f cores",
        cpus,
    )
    return cpus


def get_gpu_count() -> int:
    """Return the number of GPUs allocated to this task.

    Raises ValueError if IRIS_TASK_RESOURCES is not set.
    """
    env = _read_iris_env()
    if env is None:
        raise ValueError(f"{_IRIS_TASK_RESOURCES_ENV} not set — cannot determine GPU count outside an Iris task")
    return int(env.get("gpu_count", 0))


def get_tpu_count() -> int:
    """Return the number of TPU chips allocated to this task.

    Raises ValueError if IRIS_TASK_RESOURCES is not set.
    """
    env = _read_iris_env()
    if env is None:
        raise ValueError(f"{_IRIS_TASK_RESOURCES_ENV} not set — cannot determine TPU count outside an Iris task")
    return int(env.get("tpu_count", 0))
