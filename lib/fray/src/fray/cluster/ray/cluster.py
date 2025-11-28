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
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import ray
from ray.job_submission import JobStatus as RayJobStatus
from ray.job_submission import JobSubmissionClient

from fray.cluster.base import (
    Cluster,
    GpuConfig,
    JobId,
    JobInfo,
    JobRequest,
    JobStatus,
    TpuConfig,
    create_environment,
)
from fray.cluster.ray.deps import build_runtime_env_for_packages
from fray.cluster.ray.tpu import run_on_pod_ray
from fray.fn_thunk import create_thunk_entrypoint

logger = logging.getLogger(__name__)


class RayCluster(Cluster):
    def __init__(
        self,
        address: str = "auto",
        dashboard_address: str | None = None,
        config_path: str | None = None,
    ):
        """Initialize Ray cluster connection.

        Args:
            address: Ray cluster address (default: "auto" for local)
            dashboard_address: Dashboard address for job submission
                             (if None, derived from address)
            config_path: Path to cluster config YAML for SSH tunnel setup
                       (if None, no SSH tunnel will be created)
        """
        self._address = os.environ.get("RAY_ADDRESS", "auto") if address == "auto" else address
        self._config_path = config_path
        self._dashboard_address = dashboard_address or self._get_dashboard_address()
        self._tpu_jobs: dict[str, dict] = {}  # Track TPU jobs: job_id -> {ref, name, start_time}

    def _job_client(self) -> JobSubmissionClient:
        """Create JobSubmissionClient on demand (after connect() establishes tunnel)."""
        return JobSubmissionClient(self._dashboard_address)

    def launch(self, request: JobRequest) -> JobId:
        """Launch job on Ray cluster, returning job identifier."""
        logger.info("Launching job: %s", request.name)

        # We currently only launch TPU jobs from an existing Ray cluster. The TPU slice actor
        # bouncing prevents us from using a traditional job submission for TPU workloads.
        if request.entrypoint.callable is not None:
            if isinstance(request.resources.device, TpuConfig):
                # TPU jobs execute callable directly via run_on_pod
                return self._launch_tpu_job(request)
            else:
                # Non-TPU callable: use thunk helper
                return self._launch_callable_job(request)

        # Command-line entrypoint
        runtime_env = self._get_runtime_env(request)
        entrypoint = f"{request.entrypoint.binary} {' '.join(request.entrypoint.args)}"
        logger.info("Submitting job with entrypoint: %s", entrypoint)
        logger.debug("Runtime env: %s", runtime_env)

        # Map ResourceConfig to Ray entrypoint parameters
        entrypoint_params = self._get_entrypoint_params(request)
        logger.debug("Entrypoint params: %s", entrypoint_params)

        submission_id = self._job_client().submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            metadata={"name": request.name},
            **entrypoint_params,
        )
        logger.info("Job submitted with ID: %s", submission_id)
        return JobId(submission_id)

    def _get_runtime_env(self, request: JobRequest) -> dict | None:
        """Build Ray runtime environment for the given job request.

        For local clusters (address='auto'), returns None to skip runtime_env setup
        since dependencies are already available in the current environment.
        For remote clusters, builds full runtime_env with package dependencies.
        """
        # Skip runtime_env for local clusters - dependencies already available
        if self._address == "auto":
            return None

        environment = request.environment if request.environment else create_environment()

        runtime_env = build_runtime_env_for_packages(
            extra=list(environment.extra_dependency_groups),
            pip_packages=list(environment.pip_packages),
            env_vars=dict(environment.env_vars),
        )

        # Set working directory and excludes
        if environment.workspace:
            runtime_env["working_dir"] = environment.workspace
        runtime_env["excludes"] = [".git", "tests/", "docs/", "**/*.pack"]
        runtime_env["config"] = {"setup_timeout_seconds": 1800}

        return runtime_env

    def _get_entrypoint_params(self, request: JobRequest) -> dict:
        """Map ResourceConfig to Ray entrypoint parameters."""
        import humanfriendly

        params = {}

        # Map CPU count
        if request.resources.cpu > 0:
            params["entrypoint_num_cpus"] = float(request.resources.cpu)

        # Map memory
        if request.resources.ram:
            params["entrypoint_memory"] = humanfriendly.parse_size(request.resources.ram, binary=True)

        # Map device-specific resources
        device = request.resources.device
        if isinstance(device, GpuConfig):
            params["entrypoint_num_gpus"] = float(device.count)
        elif isinstance(device, TpuConfig):
            # Build TPU resources dict
            params["entrypoint_resources"] = {
                f"TPU-{device.type}-head": 1.0,
                "TPU": float(device.count),
            }

        return params

    def monitor(self, job_id: JobId) -> Iterator[str]:
        """Stream logs from job, returning an iterator over log lines."""
        logger.info("Starting log monitoring for job %s", job_id)

        if job_id.startswith("tpu-"):
            logger.warning("Log streaming not supported for TPU jobs")
            yield "Log streaming not supported for TPU jobs. Use Ray dashboard to view logs.\n"
            return

        async def _tail_logs():
            line_count = 0
            async for line in self._job_client().tail_job_logs(job_id):
                line_count += 1
                yield line
            logger.info("Finished tailing logs for job %s, got %d lines", job_id, line_count)

        # Consume async generator and yield results
        async def _consume():
            results = []
            async for line in _tail_logs():
                results.append(line)
            logger.debug("Collected %d log lines for job %s", len(results), job_id)
            return results

        logs = asyncio.run(_consume())
        logger.info("Yielding %d log lines for job %s", len(logs), job_id)
        yield from logs

    def poll(self, job_id: JobId) -> JobInfo:
        """Poll job status, returning the current job information or raising KeyError."""
        if job_id.startswith("tpu-"):
            return self._poll_tpu_job(job_id)

        try:
            info = self._job_client().get_job_info(job_id)
        except Exception as e:
            logger.error("Failed to get job info for %s: %s", job_id, e)
            raise KeyError(f"Job {job_id} not found") from e

        status = self._convert_ray_status(info.status)
        logger.debug("Job %s status: %s (raw: %s)", job_id, status, info.status)

        if status == "failed":
            logger.warning("Job %s failed with message: %s", job_id, info.message)

        return JobInfo(
            job_id=job_id,
            status=status,
            name=info.metadata.get("name", "") if info.metadata else "",
            start_time=info.start_time / 1000 if info.start_time else None,
            end_time=info.end_time / 1000 if info.end_time else None,
            error_message=info.message if status == "failed" else None,
        )

    def terminate(self, job_id: JobId) -> None:
        """Stop a job with the given job identifier.

        Waits for the job to actually stop before returning.
        """
        if job_id.startswith("tpu-"):
            self._terminate_tpu_job(job_id)
            return

        try:
            self._job_client().stop_job(job_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", job_id, e)
            return

        # Wait for job to actually stop
        for _ in range(100):
            try:
                info = self._job_client().get_job_info(job_id)
                status = self._convert_ray_status(info.status)
                if status in ["stopped", "failed", "succeeded"]:
                    break
            except Exception:
                break
            time.sleep(1.0)

    def list_jobs(self) -> list[JobInfo]:
        jobs = self._job_client().list_jobs()
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

    def _convert_ray_status(self, ray_status: RayJobStatus) -> JobStatus:
        mapping = {
            RayJobStatus.PENDING: "pending",
            RayJobStatus.RUNNING: "running",
            RayJobStatus.SUCCEEDED: "succeeded",
            RayJobStatus.FAILED: "failed",
            RayJobStatus.STOPPED: "stopped",
        }
        return cast(JobStatus, mapping.get(ray_status, "failed"))

    def _launch_callable_job(self, request: JobRequest) -> JobId:
        """Launch non-TPU callable job using thunk helper."""
        entrypoint = request.entrypoint
        thunk_entrypoint = create_thunk_entrypoint(entrypoint.callable, prefix=f"/tmp/{request.name}")
        runtime_env = self._get_runtime_env(request)
        entrypoint_cmd = f"{thunk_entrypoint.binary} {' '.join(thunk_entrypoint.args)}"
        entrypoint_params = self._get_entrypoint_params(request)

        submission_id = self._job_client().submit_job(
            entrypoint=entrypoint_cmd,
            runtime_env=runtime_env,
            metadata={"name": request.name},
            **entrypoint_params,
        )
        return JobId(submission_id)

    def _launch_tpu_job(self, request: JobRequest) -> JobId:
        """Launch TPU job using run_on_pod."""
        entrypoint = request.entrypoint
        assert entrypoint.callable is not None, "TPU jobs require callable entrypoint"

        device = request.resources.device
        assert isinstance(device, TpuConfig), "TPU job requires TpuConfig"

        runtime_env = self._get_runtime_env(request)

        # Wrap callable with ray.remote
        remote_fn = ray.remote(max_calls=1, runtime_env=runtime_env)(entrypoint.callable)

        # Submit to TPU execution
        object_ref = run_on_pod_ray.remote(
            remote_fn,
            tpu_type=device.type,
            num_slices=device.num_slices,
            max_retries_preemption=10000,
            max_retries_failure=10,
        )

        # Track via ObjectRef
        job_id = f"tpu-{object_ref.hex()}"
        self._tpu_jobs[job_id] = {
            "ref": object_ref,
            "name": request.name,
            "start_time": time.time(),
        }
        return JobId(job_id)

    def _poll_tpu_job(self, job_id: JobId) -> JobInfo:
        """Poll TPU job status via ObjectRef.

        Helper function to check TPU job status by inspecting the Ray ObjectRef.
        """
        info = self._tpu_jobs.get(job_id)
        if not info:
            raise KeyError(f"TPU job {job_id} not found")

        ready, _ = ray.wait([info["ref"]], timeout=0)

        if ready:
            try:
                ray.get(info["ref"])
                status = "succeeded"
                error_msg = None
                end_time = time.time()
            except Exception as e:
                status = "failed"
                error_msg = str(e)
                end_time = time.time()
        else:
            status = "running"
            error_msg = None
            end_time = None

        return JobInfo(
            job_id=job_id,
            status=status,
            name=info["name"],
            start_time=info["start_time"],
            end_time=end_time,
            error_message=error_msg,
        )

    def _terminate_tpu_job(self, job_id: JobId) -> None:
        """Cancel TPU job by canceling the ObjectRef."""
        info = self._tpu_jobs.get(job_id)
        if info:
            ray.cancel(info["ref"])
            del self._tpu_jobs[job_id]

    def _get_dashboard_address(self) -> str:
        if self._address == "auto":
            try:
                ray.init(address="auto", ignore_reinit_error=True)
                dashboard_url = ray.get_runtime_context().dashboard_url
                if dashboard_url:
                    logger.info("Using Ray dashboard at: %s", dashboard_url)
                    return dashboard_url
            except Exception as e:
                logger.warning("Failed to get dashboard URL from Ray context: %s", e)

            default_url = "http://127.0.0.1:8265"
            logger.info("Using default dashboard URL: %s", default_url)
            return default_url

        # For remote addresses, assume dashboard is on port 8265
        logger.info("Using remote dashboard address: %s", self._address)
        return self._address

    @contextmanager
    def connect(self) -> Iterator[None]:
        """Establish SSH tunnel and dashboard connection to Ray cluster.

        For remote Ray clusters with a config_path, this creates an SSH tunnel
        via the ray_dashboard context manager. Otherwise this is a no-op.
        """
        if self._config_path:
            from fray.cluster.ray.dashboard import DashboardConfig, ray_dashboard

            dashboard_cfg = DashboardConfig.from_cluster(self._config_path)
            with ray_dashboard(dashboard_cfg):
                yield
        else:
            # Local cluster or no config path specified
            yield
