# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fray cluster abstraction for job scheduling.
Example:
    >>> from fray.v1.cluster import LocalCluster, JobRequest, EnvironmentConfig
    >>> cluster = LocalCluster()
    >>> request = JobRequest(
    ...     name="hello-world",
    ...     entrypoint="json.tool",
    ...     environment=EnvironmentConfig.create(),
    ... )
    >>> job_id = cluster.launch(request)
    >>> job_info = cluster.monitor(job_id)  # Logs stream to logger
    >>> print(job_info.status)
"""

import logging
import os
from contextvars import ContextVar

from fray.v1.cluster.base import (
    Cluster,
    CpuConfig,
    DeviceConfig,
    DeviceKind,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    GpuType,
    JobId,
    JobInfo,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    TpuTopologyInfo,
    TpuType,
    get_tpu_topology,
)
from fray.v1.cluster.device_flops import FlopDtype
from fray.v1.cluster.local_cluster import LocalCluster

logger = logging.getLogger(__name__)

# Context variable for current cluster
_cluster_context: ContextVar[Cluster | None] = ContextVar("fray_cluster", default=None)


def set_current_cluster(cluster: Cluster) -> None:
    _cluster_context.set(cluster)


def current_cluster() -> Cluster:
    """Return a cluster context.

    For local execution (including with GPUs), uses LocalCluster by default.
    This runs everything in the current process using the user's venv, avoiding
    Ray's virtualenv creation and worker process overhead.

    Priority order:
    1. Already-set cluster context
    2. FRAY_CLUSTER_SPEC environment variable (for explicit cluster configuration)
    3. LocalCluster (default for local execution)

    Note: To use Ray for distributed execution, set FRAY_CLUSTER_SPEC=ray?namespace=...
    """
    cluster = _cluster_context.get()
    if cluster is not None:
        return cluster

    cluster_spec = os.environ.get("FRAY_CLUSTER_SPEC")
    if cluster_spec is not None:
        logger.info(f"Using cluster from FRAY_CLUSTER_SPEC={cluster_spec}")
        cluster = create_cluster(cluster_spec)
        set_current_cluster(cluster)
        return cluster

    cluster = LocalCluster()
    set_current_cluster(cluster)
    logger.info("No active cluster found and Ray is not initialized, using LocalCluster")
    return cluster


def create_cluster(cluster_spec: str) -> Cluster:
    """Create a cluster from a specification string.

    Args:
        cluster_spec: Cluster specification:
            - "local" -> LocalCluster
            - "ray?namespace=x" -> RayCluster

    Returns:
        Configured cluster instance
    """
    from pathlib import Path
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(cluster_spec)
    query_params = parse_qs(parsed.query)

    if cluster_spec.startswith("local"):
        return LocalCluster.from_spec(query_params)

    if cluster_spec.startswith("ray"):
        from fray.v1.cluster.ray.cluster import RayCluster

        return RayCluster.from_spec(query_params)

    raise ValueError(f"Unknown cluster spec: {cluster_spec}")


__all__ = [
    "Cluster",
    "CpuConfig",
    "DeviceConfig",
    "DeviceKind",
    "Entrypoint",
    "EnvironmentConfig",
    "FlopDtype",
    "GpuConfig",
    "GpuType",
    "JobId",
    "JobInfo",
    "JobRequest",
    "LocalCluster",
    "ResourceConfig",
    "TPUConfig",
    "TpuConfig",
    "TpuType",
    "create_cluster",
    "current_cluster",
    "get_tpu_topology",
    "set_current_cluster",
]
