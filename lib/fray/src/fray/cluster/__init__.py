# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fray cluster abstraction for job scheduling.
Example:
    >>> from fray.cluster import LocalCluster, JobRequest, create_environment
    >>> cluster = LocalCluster()
    >>> request = JobRequest(
    ...     name="hello-world",
    ...     entrypoint="json.tool",
    ...     environment=create_environment(),
    ... )
    >>> job_id = cluster.launch(request)
    >>> for line in cluster.monitor(job_id):
    ...     print(line)
"""

import logging
import os
from contextvars import ContextVar

from fray.cluster.base import (
    Cluster,
    CpuConfig,
    DeviceConfig,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    GpuType,
    JobId,
    JobInfo,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    TpuType,
    create_environment,
)
from fray.cluster.local_cluster import LocalCluster

logger = logging.getLogger(__name__)

# Context variable for current cluster
_cluster_context: ContextVar[Cluster | None] = ContextVar("fray_cluster", default=None)


def set_current_cluster(cluster: Cluster) -> None:
    _cluster_context.set(cluster)


def current_cluster() -> Cluster:
    """Get the current cluster from context.

    Auto-detection priority:
    1. Context variable (set via set_current_cluster())
    2. Ray cluster (if ray.is_initialized())
    3. FRAY_CLUSTER_SPEC environment variable
    4. LocalCluster (default fallback)

    Returns:
        The cluster instance

    Raises:
        RuntimeError: If cluster creation fails
    """
    cluster = _cluster_context.get()
    if cluster is not None:
        return cluster

    try:
        import ray

        if ray.is_initialized():
            from fray.cluster.ray.cluster import RayCluster

            cluster = RayCluster()
            set_current_cluster(cluster)
            logger.info("Auto-detected Ray cluster from ray.is_initialized()")
            return cluster
    except ImportError:
        # Ray is not installed; fall back to other cluster types
        pass

    # Check for FRAY_CLUSTER_SPEC
    cluster_spec = os.environ.get("FRAY_CLUSTER_SPEC")
    if cluster_spec is not None:
        cluster = create_cluster(cluster_spec)
        set_current_cluster(cluster)
        logger.info(f"Auto-created cluster from FRAY_CLUSTER_SPEC={cluster_spec}")
        return cluster

    # Default to LocalCluster
    cluster = LocalCluster()
    set_current_cluster(cluster)
    logger.info("Using default LocalCluster")
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
        return LocalCluster()

    if cluster_spec.startswith("ray"):
        from fray.cluster.ray.cluster import RayCluster

        return RayCluster.from_spec(query_params)

    raise ValueError(f"Unknown cluster spec: {cluster_spec}")


__all__ = [
    "Cluster",
    "CpuConfig",
    "DeviceConfig",
    "Entrypoint",
    "EnvironmentConfig",
    "GpuConfig",
    "GpuType",
    "JobId",
    "JobInfo",
    "JobRequest",
    "JobStatus",
    "LocalCluster",
    "ResourceConfig",
    "TpuConfig",
    "TpuType",
    "create_cluster",
    "create_environment",
    "current_cluster",
    "set_current_cluster",
]
