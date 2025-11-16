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

"""Ray-based cluster implementation."""

import asyncio
import logging
import time
from collections.abc import Iterator
from typing import cast

import ray
from ray.job_submission import JobStatus as RayJobStatus
from ray.job_submission import JobSubmissionClient

from fray.cluster.base import Cluster
from fray.cluster.ray.deps import build_runtime_env_for_packages
from fray.cluster.types import (
    GpuConfig,
    JobId,
    JobInfo,
    JobRequest,
    JobStatus,
    TpuConfig,
    create_environment,
)

logger = logging.getLogger(__name__)


class RayCluster(Cluster):
    """Ray-based cluster implementation.

    Submits jobs to a Ray cluster via the JobSubmissionClient API.
    Supports CPU, GPU, and TPU resources with dependency isolation
    via Ray runtime environments.

    Resource mapping:
    - TpuConfig(type="v5e-16", count=8) -> {"TPU": 8, "v5e-16-head": 1}
    - GpuConfig(type="A100", count=4) -> {"GPU": 4}
    - CpuConfig() -> {} (no special resources)

    The RayCluster handles both CLI-style job submissions and can be used
    to configure ray.remote decorators with appropriate runtime environments.
    """

    def __init__(
        self,
        address: str = "auto",
        dashboard_address: str | None = None,
    ):
        """Initialize Ray cluster connection.

        Args:
            address: Ray cluster address (default: "auto" for local)
            dashboard_address: Dashboard address for job submission
                             (if None, derived from address)
        """
        self._address = address
        self._dashboard_address = dashboard_address or self._get_dashboard_address()
        self._client = JobSubmissionClient(self._dashboard_address)

    def launch(self, request: JobRequest) -> JobId:
        """Launch job on Ray cluster, returning job identifier."""
        runtime_env = self._get_runtime_env(request)
        entrypoint = self._build_entrypoint(request)

        submission_id = self._client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            metadata={"name": request.name},
        )
        return JobId(submission_id)

    def _get_runtime_env(self, request: JobRequest) -> dict:
        """Build Ray runtime environment for the given job request."""
        environment = request.environment if request.environment else create_environment()

        return build_runtime_env_for_packages(
            extra=list(environment.extra_dependency_groups),
            pip_packages=list(environment.pip_packages),
            env_vars=dict(environment.env_vars),
        )

    def monitor(self, job_id: JobId) -> Iterator[str]:
        """Stream logs from Ray job, returning an iterator over log lines."""

        async def _tail_logs():
            async for line in self._client.tail_job_logs(job_id):
                yield line

        # Consume async generator and yield results
        async def _consume():
            results = []
            async for line in _tail_logs():
                results.append(line)
            return results

        yield from asyncio.run(_consume())

    def poll(self, job_id: JobId) -> JobInfo:
        """Poll Ray job status, returning the current job information or raising KeyError."""
        try:
            info = self._client.get_job_info(job_id)
        except Exception as e:
            raise KeyError(f"Job {job_id} not found") from e

        status = self._convert_ray_status(info.status)

        return JobInfo(
            job_id=job_id,
            status=status,
            name=info.metadata.get("name", "") if info.metadata else "",
            start_time=info.start_time / 1000 if info.start_time else None,
            end_time=info.end_time / 1000 if info.end_time else None,
            error_message=info.message if status == "failed" else None,
        )

    def terminate(self, job_id: JobId) -> None:
        """Stop a Ray job with the given job identifier.

        Waits for the job to actually stop before returning.
        """
        try:
            self._client.stop_job(job_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", job_id, e)
            return

        # Wait for job to actually stop
        for _ in range(100):  # 10 seconds max
            try:
                info = self._client.get_job_info(job_id)
                status = self._convert_ray_status(info.status)
                if status in ["stopped", "failed", "succeeded"]:
                    break
            except Exception:
                # Job no longer exists
                break
            time.sleep(1.0)

    def list_jobs(self) -> list[JobInfo]:
        """List all Ray jobs.

        Returns:
            List of all job information
        """
        jobs = self._client.list_jobs()
        result = []
        for job_info in jobs:
            result.append(
                JobInfo(
                    job_id=JobId(job_info.submission_id),
                    status=self._convert_ray_status(job_info.status),
                    name=job_info.metadata.get("name", "") if job_info.metadata else "",
                    start_time=job_info.start_time / 1000 if job_info.start_time else None,
                    end_time=job_info.end_time / 1000 if job_info.end_time else None,
                    error_message=job_info.message if self._convert_ray_status(job_info.status) == "failed" else None,
                )
            )
        return result

    def get_ray_resources(self, request: JobRequest) -> dict[str, float]:
        """Convert ResourceConfig to Ray resource specification.

        Maps structured device configs to Ray's resource format:
        - TpuConfig(type="v5e-16", count=8) -> {"TPU": 8, "v5e-16-head": 1}
        - GpuConfig(type="A100", count=4) -> {"GPU": 4}
        - CpuConfig() -> {}

        Args:
            request: Job specification

        Returns:
            Ray resources dict for use with ray.remote()

        Example:
            >>> cluster = RayCluster()
            >>> request = JobRequest(
            ...     name="tpu-job",
            ...     entrypoint="my_module",
            ...     resources=ResourceConfig(device=TpuConfig(type="v5e-16", count=8))
            ... )
            >>> resources = cluster.get_ray_resources(request)
            >>> # resources = {"TPU": 8, "v5e-16-head": 1}
            >>> @ray.remote(resources=resources)
            ... def my_function():
            ...     pass
        """
        resources: dict[str, float] = {}

        device = request.resources.device

        if isinstance(device, TpuConfig):
            # TPU resources include:
            # 1. Generic "TPU" resource for chip count
            # 2. Specific type-head resource for exclusive access to a TPU pod
            resources["TPU"] = float(device.count)
            resources[f"{device.type}-head"] = 1.0
        elif isinstance(device, GpuConfig):
            # GPU resources just specify the count
            resources["GPU"] = float(device.count)
        # CpuConfig requires no special resources

        return resources

    def _build_entrypoint(self, request: JobRequest) -> str:
        args = " ".join(request.entrypoint_args)
        return f"python -m {request.entrypoint} {args}".strip()

    def _convert_ray_status(self, ray_status: RayJobStatus) -> JobStatus:
        mapping = {
            RayJobStatus.PENDING: "pending",
            RayJobStatus.RUNNING: "running",
            RayJobStatus.SUCCEEDED: "succeeded",
            RayJobStatus.FAILED: "failed",
            RayJobStatus.STOPPED: "stopped",
        }
        return cast(JobStatus, mapping.get(ray_status, "failed"))

    def _get_dashboard_address(self) -> str:
        if self._address == "auto":
            try:
                ray.init(address="auto", ignore_reinit_error=True)
                dashboard_url = ray.get_runtime_context().dashboard_url
                if dashboard_url:
                    return dashboard_url
            except Exception:
                pass
            return "http://127.0.0.1:8265"

        # For remote addresses, assume dashboard is on port 8265
        return self._address
