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


def _detect_local_accelerators() -> tuple[bool, dict[str, float]]:
    """Detect local accelerators (GPUs/TPUs).

    Returns a tuple of (has_accelerators, custom_resources).
    - has_accelerators: True if GPUs or TPUs are detected
    - custom_resources: Additional Ray custom resources (e.g., accelerator_type)

    Note: We don't include "GPU" in custom_resources because Ray auto-detects GPUs.
    The num_gpus parameter in ray.remote() uses Ray's built-in GPU tracking.
    """
    import shutil
    import subprocess

    has_accelerators = False
    custom_resources: dict[str, float] = {}

    # detect GPUs using nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_names = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                gpu_count = len(gpu_names)
                if gpu_count > 0:
                    has_accelerators = True
                    # Try to map GPU name to Ray accelerator type (as custom resource)
                    gpu_name = gpu_names[0].upper()
                    accel_type = None
                    if "A100" in gpu_name:
                        accel_type = "A100-80G" if "80G" in gpu_name else "A100-40G"
                    elif "H100" in gpu_name:
                        accel_type = "H100"
                    elif "V100" in gpu_name:
                        accel_type = "V100"
                    elif "A10" in gpu_name:
                        accel_type = "A10G"
                    elif "L4" in gpu_name:
                        accel_type = "L4"
                    elif "T4" in gpu_name:
                        accel_type = "T4"

                    if accel_type:
                        custom_resources[f"accelerator_type:{accel_type}"] = float(gpu_count)
                        logger.info(f"Detected {gpu_count} GPU(s) of type {accel_type}")
                    else:
                        logger.info(f"Detected {gpu_count} GPU(s): {gpu_names[0]}")
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")

    # detect TPUs via environment variable
    if not has_accelerators:
        tpu_name = os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK_ID")
        if tpu_name:
            # On TPU VMs, detect chip count via JAX (safe since no GPU conflict)
            try:
                import jax

                devices = jax.devices()
                tpu_count = sum(1 for d in devices if d.platform == "tpu")
                if tpu_count > 0:
                    has_accelerators = True
                    custom_resources["TPU"] = float(tpu_count)
                    logger.info(f"Detected {tpu_count} TPU(s)")
            except Exception as e:
                logger.debug(f"Could not detect TPUs via JAX: {e}")

    return has_accelerators, custom_resources


def current_cluster() -> Cluster:
    """Return a cluster context.

    If a cluster is already set in context, return it.
    If a FRAY_CLUSTER_SPEC is set, create a cluster from it.
    If running inside of Ray, use a Ray cluster.
    If accelerators are detected locally, initialize Ray with them.
    Otherwise, use a LocalCluster for CPU-only work.

    Note: For local single-node clusters (auto-detected GPUs), RayCluster will
    run callable jobs directly in the current process to avoid Python version
    mismatch issues with Ray's virtualenv.
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

        if ray.is_initialized() or _is_running_in_ray_context():
            logger.info("Auto-detected Ray cluster")
            from fray.cluster.ray.cluster import RayCluster

            cluster = RayCluster()
            set_current_cluster(cluster)
            return cluster

        # Check for local accelerators and initialize Ray with them
        has_accelerators, custom_resources = _detect_local_accelerators()
        if has_accelerators:
            logger.info(f"Detected local accelerators, initializing Ray (custom resources: {custom_resources})")
            # Let Ray auto-detect GPUs; only pass additional custom resources
            ray.init(
                namespace="marin",
                ignore_reinit_error=True,
                resources=custom_resources if custom_resources else None,
            )
            from fray.cluster.ray.cluster import RayCluster

            cluster = RayCluster()
            set_current_cluster(cluster)
            return cluster
    except ImportError:
        pass

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
