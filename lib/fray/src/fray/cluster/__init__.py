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
    >>> from fray.cluster import LocalCluster, JobRequest, EnvironmentConfig
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

from fray.cluster.base import (
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
from fray.cluster.device_flops import FlopDtype
from fray.cluster.local_cluster import LocalCluster

logger = logging.getLogger(__name__)

# Context variable for current cluster
_cluster_context: ContextVar[Cluster | None] = ContextVar("fray_cluster", default=None)


def set_current_cluster(cluster: Cluster) -> None:
    _cluster_context.set(cluster)


def _is_running_in_ray_context() -> bool:
    try:
        import ray

        ray.get_runtime_context().get_job_id()
        return True
    except (ImportError, RuntimeError):
        return False


def _has_local_accelerators() -> bool:
    """Check if local accelerators (GPUs/TPUs) are available.

    Uses nvidia-smi for GPU detection and environment variables for TPU detection.
    Does not call jax.devices() to avoid taking ownership of devices.
    """
    import shutil
    import subprocess

    # Detect GPUs using nvidia-smi (doesn't take ownership like jax.devices())
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_count = len([line for line in result.stdout.strip().split("\n") if line.strip()])
                if gpu_count > 0:
                    logger.info(f"Detected {gpu_count} GPU(s)")
                    return True
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")

    # Detect TPUs via environment variable (doesn't call jax.devices())
    tpu_name = os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK_ID")
    if tpu_name:
        logger.info(f"Detected TPU environment: {tpu_name}")
        return True

    return False


def current_cluster() -> Cluster:
    """Return a cluster context.

    If a cluster is already set in context, return it.
    If a FRAY_CLUSTER_SPEC is set, create a cluster from it.
    If running inside of Ray or Ray is already initialized, use RayCluster.
    Otherwise, use LocalCluster (works fine for Zephyr and local execution).
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

    try:
        import ray
        from fray.cluster.ray.cluster import RayCluster

        # Use RayCluster if Ray is already initialized or we're in a Ray context
        if ray.is_initialized() or _is_running_in_ray_context():
            logger.info("Auto-detected Ray cluster")
            cluster = RayCluster()
            set_current_cluster(cluster)
            return cluster
    except ImportError:
        pass

    # For local execution, use LocalCluster
    # This avoids Ray's virtualenv creation and pip installation overhead
    # Note: Zephyr data processing works fine with LocalCluster
    if _has_local_accelerators():
        logger.info("Using LocalCluster for local execution with accelerators")
    else:
        logger.info("Using LocalCluster for local execution")

    cluster = LocalCluster()
    set_current_cluster(cluster)
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
        from fray.cluster.ray.cluster import RayCluster

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
