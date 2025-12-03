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
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import humanfriendly
import ray
from ray.job_submission import JobStatus as RayJobStatus
from ray.job_submission import JobSubmissionClient

from fray.cluster.base import (
    Cluster,
    GpuConfig,
    JobId,
    JobInfo,
    JobRequest,
    TpuConfig,
    create_environment,
)
from fray.cluster.ray.config import find_config_by_region
from fray.cluster.ray.deps import build_runtime_env_for_packages
from fray.cluster.ray.tpu import run_on_pod_ray
from fray.fn_thunk import create_thunk_entrypoint

logger = logging.getLogger(__name__)


# We can't launch TPU jobs directly via Ray, as it doesn't support gang-scheduling and jobs are always
# started with a single task in Ray. Instead we use the ray_tpu helper/actor to stage TPU execution.
# We store TPU "job" information separately here to report to the user.
@dataclass
class _TpuJobInfo:
    ref: ray.ObjectRef
    name: str
    start_time: float


class RayCluster(Cluster):
    def __init__(
        self,
        address: str = "auto",
        dashboard_address: str | None = None,
        config_path: str | None = None,
        namespace: str | None = None,
    ):
        """Initialize Ray cluster connection.

        Args:
            address: Ray cluster address (default: "auto" for local)
            dashboard_address: Dashboard address for job submission
                             (if None, derived from address)
            config_path: Path to cluster config YAML for SSH tunnel setup
                       (if None, no SSH tunnel will be created)
            namespace: Ray namespace for actor isolation
                      (if None, creates a unique namespace)
        """
        self._address = os.environ.get("RAY_ADDRESS", "auto") if address == "auto" else address
        self._config_path = config_path

        # Use provided namespace or create a unique one
        if namespace is None:
            self._namespace = f"fray_{uuid.uuid4().hex[:8]}"
            logger.info(f"Created new namespace: {self._namespace}")
        else:
            self._namespace = namespace
            logger.info(f"Using provided namespace: {self._namespace}")

        self._dashboard_address = dashboard_address or self._get_dashboard_address()
        self._tpu_jobs: dict[str, _TpuJobInfo] = {}  # Track TPU jobs: job_id -> {ref, name, start_time}

    @classmethod
    def from_spec(cls, query_params: dict[str, list[str]]) -> "RayCluster":
        namespace = None
        config_path = None
        address = "auto"

        if "namespace" in query_params:
            namespace = query_params["namespace"][0]

        if "cluster" in query_params:
            config_path = find_config_by_region(query_params["cluster"][0])

        return cls(address=address, config_path=config_path, namespace=namespace)

    def _job_client(self) -> JobSubmissionClient:
        return JobSubmissionClient(self._dashboard_address)

    def _get_cluster_spec(self) -> str:
        if self._address == "auto":
            base = "ray"
        elif self._config_path:
            base = f"ray:{self._config_path}"
        else:
            base = "ray"

        # Append namespace as query param if present
        if self._namespace:
            base += f"?namespace={self._namespace}"

        return base

    def launch(self, request: JobRequest) -> JobId:
        """Launch job on Ray cluster, returning job identifier."""
        logger.info("Launching job: %s", request.name)

        # We currently only launch TPU jobs from an existing Ray cluster. The TPU slice actor
        # bouncing prevents us from using a traditional job submission for TPU workloads.
        if isinstance(request.resources.device, TpuConfig):
            return self._launch_tpu_job(request)

        if request.entrypoint.callable is not None:
            entrypoint = create_thunk_entrypoint(
                request.entrypoint.callable,
                prefix=f"/tmp/{request.name}",
                function_args=request.entrypoint.function_args,
            )
        else:
            entrypoint = request.entrypoint

        runtime_env = self._get_runtime_env(request)
        entrypoint_cmd = f"{entrypoint.binary} {' '.join(entrypoint.args)}"
        logger.info("Submitting job with entrypoint: %s", entrypoint_cmd)
        logger.debug("Runtime env: %s", runtime_env)

        entrypoint_params = self._get_entrypoint_params(request)
        logger.debug("Entrypoint params: %s", entrypoint_params)

        submission_id = self._job_client().submit_job(
            entrypoint=entrypoint_cmd,
            runtime_env=runtime_env,
            metadata={"name": request.name},
            **entrypoint_params,
        )
        logger.info("Job submitted with ID: %s", submission_id)
        return JobId(submission_id)

    def _get_runtime_env(self, request: JobRequest) -> dict | None:
        """Build Ray runtime environment for the given job request."""
        environment = request.environment if request.environment else create_environment()

        env_vars = dict(environment.env_vars)

        if "FRAY_CLUSTER_SPEC" not in env_vars:
            env_vars["FRAY_CLUSTER_SPEC"] = self._get_cluster_spec()

        # skip building the package environment for local clusters
        if self._address == "auto" or self._address == "local":
            logger.info("Skipping package environment for local cluster")
            runtime_env = {
                "env_vars": env_vars,
            }
        else:
            runtime_env = build_runtime_env_for_packages(
                extra=list(environment.extras),
                pip_packages=list(environment.pip_packages),
                env_vars=env_vars,
            )

        runtime_env["working_dir"] = environment.workspace
        runtime_env["excludes"] = [".git", "tests/", "docs/", "**/*.pack"]
        runtime_env["config"] = {"setup_timeout_seconds": 1800}
        return runtime_env

    def _get_entrypoint_params(self, request: JobRequest) -> dict:
        params = {}

        if request.resources.cpu > 0:
            params["entrypoint_num_cpus"] = float(request.resources.cpu)

        if request.resources.ram:
            params["entrypoint_memory"] = humanfriendly.parse_size(request.resources.ram, binary=True)

        device = request.resources.device
        if isinstance(device, GpuConfig):
            params["entrypoint_num_gpus"] = float(device.count)
        elif isinstance(device, TpuConfig):
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

        def synchronize():
            async def get_next_item() -> str:
                try:
                    return await self._job_client().tail_job_logs(job_id).__anext__()
                except StopAsyncIteration:
                    raise StopIteration from None

            yield from asyncio.run(get_next_item())

        yield from synchronize()

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
            tasks=[],
            name=info.metadata.get("name", "") if info.metadata else "",
            error_message=info.message if status == "failed" else None,
        )

    def terminate(self, job_id: JobId) -> None:
        """Stop a job with the given job identifier."""
        if job_id.startswith("tpu-"):
            self._terminate_tpu_job(job_id)
            return

        try:
            self._job_client().stop_job(job_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", job_id, e)
            return

        # Wait for job to stop
        for _ in range(100):
            try:
                info = self._job_client().get_job_info(job_id)
                status = self._convert_ray_status(info.status)
                if status in ["stopped", "failed", "succeeded"]:
                    break
            except Exception:
                break
            logger.info("Waiting for job %s to stop", job_id)
            time.sleep(1.0)

    def list_jobs(self) -> list[JobInfo]:
        jobs = self._job_client().list_jobs()
        result = []
        for job_info in jobs:
            result.append(
                JobInfo(
                    job_id=JobId(job_info.submission_id),
                    status=self._convert_ray_status(job_info.status),
                    tasks=[],
                    name=job_info.metadata.get("name", "") if job_info.metadata else "",
                    error_message=job_info.message if self._convert_ray_status(job_info.status) == "failed" else None,
                )
            )

        for tpu_job_info in self._tpu_jobs.values():
            result.append(self._poll_tpu_job(tpu_job_info.job_id))
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

    def _convert_ray_status(
        self, ray_status: RayJobStatus
    ) -> Literal["pending", "running", "succeeded", "failed", "stopped"]:
        mapping = {
            RayJobStatus.PENDING: "pending",
            RayJobStatus.RUNNING: "running",
            RayJobStatus.SUCCEEDED: "succeeded",
            RayJobStatus.FAILED: "failed",
            RayJobStatus.STOPPED: "stopped",
        }
        return mapping.get(ray_status, "failed")

    def _launch_tpu_job(self, request: JobRequest) -> JobId:
        entrypoint = request.entrypoint
        assert entrypoint.callable is not None, "TPU jobs require callable entrypoint"

        device = request.resources.device
        runtime_env = self._get_runtime_env(request)

        # For nested ray.remote() calls, filter out job-level keys that can only be set via ray.init().
        # These include working_dir, excludes, and config which are already set at the job level.
        nested_runtime_env = {k: v for k, v in runtime_env.items() if k not in ["working_dir", "excludes", "config"]}

        if entrypoint.function_args:
            remote_fn = ray.remote(max_calls=1, runtime_env=nested_runtime_env)(
                lambda: entrypoint.callable(**entrypoint.function_args)
            )
        else:
            remote_fn = ray.remote(max_calls=1, runtime_env=nested_runtime_env)(entrypoint.callable)

        object_ref = run_on_pod_ray.remote(
            remote_fn,
            tpu_type=device.type,
            num_slices=request.resources.replicas,
            max_retries_preemption=10000,
            max_retries_failure=1,
        )

        # Track via ObjectRef
        job_id = f"tpu-{object_ref.hex()}"
        self._tpu_jobs[job_id] = _TpuJobInfo(
            ref=object_ref,
            name=request.name,
            start_time=time.time(),
        )
        return JobId(job_id)

    def _poll_tpu_job(self, job_id: JobId) -> JobInfo:
        """Poll TPU job status via ObjectRef.

        Helper function to check TPU job status by inspecting the Ray ObjectRef.
        """
        info = self._tpu_jobs.get(job_id)
        if not info:
            raise KeyError(f"TPU job {job_id} not found")

        ready, _ = ray.wait([info.ref], timeout=0)

        if ready:
            try:
                ray.get(info.ref)
                status = "succeeded"
                error_msg = None
            except Exception as e:
                status = "failed"
                error_msg = str(e)
        else:
            status = "running"
            error_msg = None

        return JobInfo(
            job_id=job_id,
            status=status,
            name=info.name,
            error_message=error_msg,
        )

    def _terminate_tpu_job(self, job_id: JobId) -> None:
        """Cancel TPU job by canceling the ObjectRef."""
        info = self._tpu_jobs.get(job_id)
        if info:
            ray.cancel(info.ref)
            del self._tpu_jobs[job_id]

    def _get_dashboard_address(self) -> str:
        """Get Ray dashboard address for job submission.

        When running inside a Ray worker, we can't access the dashboard URL
        directly, so we fall back to using the Ray GCS address or the
        configured address.
        """
        if ray.is_initialized():
            try:
                ctx = ray.get_runtime_context()
                if hasattr(ctx, "gcs_address"):
                    return ctx.gcs_address
            except Exception:
                # Fall back to self._address
                pass
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
            yield
