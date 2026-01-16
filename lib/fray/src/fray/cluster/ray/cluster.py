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
from dataclasses import dataclass

import humanfriendly
import ray
from ray.job_submission import JobStatus as RayJobStatus
from ray.job_submission import JobSubmissionClient

from fray.cluster.base import (
    Cluster,
    CpuConfig,
    EnvironmentConfig,
    GpuConfig,
    JobId,
    JobInfo,
    JobRequest,
    JobStatus,
    TpuConfig,
)
from fray.cluster.ray.config import find_config_by_region
from fray.cluster.ray.deps import build_python_path, build_runtime_env_for_packages
from fray.cluster.ray.tpu import run_on_pod_ray
from fray.job.context import RayContext, fray_default_job_ctx

logger = logging.getLogger("ray")


# We can't launch TPU or callable entrypoint jobs directly via Ray, as it
# doesn't support gang-scheduling and jobs are always started with a single task
# in Ray. Instead we use the ray_tpu helper/actor to stage TPU execution.  We
# store "job" information separately here to report to the user.
@dataclass
class RayJobInfo:
    ref: ray.ObjectRef | None
    submission_id: str | None
    name: str

    @staticmethod
    def from_ref(ref: ray.ObjectRef, name: str) -> "RayJobInfo":
        return RayJobInfo(ref=ref, submission_id=None, name=name)

    @staticmethod
    def from_submission_id(ray_job_id: str, name: str) -> "RayJobInfo":
        return RayJobInfo(ref=None, submission_id=ray_job_id, name=name)


def _convert_ray_status(ray_status: RayJobStatus) -> JobStatus:
    mapping = {
        RayJobStatus.PENDING: JobStatus.PENDING,
        RayJobStatus.RUNNING: JobStatus.RUNNING,
        RayJobStatus.SUCCEEDED: JobStatus.SUCCEEDED,
        RayJobStatus.FAILED: JobStatus.FAILED,
        RayJobStatus.STOPPED: JobStatus.STOPPED,
    }
    return mapping.get(ray_status, JobStatus.FAILED)


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
        self._jobs: dict[JobId, RayJobInfo] = {}

        # Set default job context permanently to this cluster.
        ray_ctx = RayContext(ray_options={})
        fray_default_job_ctx(ray_ctx).__enter__()
        logger.info("Set default job context to RayContext")

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

        if isinstance(request.resources.device, TpuConfig):
            return self._launch_tpu_job(request)

        if request.entrypoint.binary_entrypoint is not None:
            return self._launch_binary_job(request)

        return self._launch_callable_job(request)

    def _launch_binary_job(self, request: JobRequest) -> JobId:
        entrypoint = request.entrypoint.binary_entrypoint
        runtime_env = self._get_runtime_env(request)
        entrypoint_cmd = f"{entrypoint.command} {' '.join(entrypoint.args)}"
        logger.info("Submitting job with entrypoint: %s", entrypoint_cmd)
        logger.debug("Runtime env: %s", runtime_env)

        entrypoint_params = self._get_entrypoint_params(request)
        logger.debug("Entrypoint params: %s", entrypoint_params)

        client = self._job_client()
        submission_timeout_s = float(os.environ.get("FRAY_RAY_JOB_SUBMIT_TIMEOUT_S", "30"))
        deadline = time.time() + submission_timeout_s
        sleep_s = 0.5
        while True:
            try:
                submission_id = client.submit_job(
                    entrypoint=entrypoint_cmd,
                    runtime_env=runtime_env,
                    submission_id=f"{request.name}-{uuid.uuid4()}",
                    metadata={"name": request.name},
                    **entrypoint_params,
                )
                break
            except RuntimeError as e:
                # Ray can briefly return "No available agent to submit job" right after startup.
                if "No available agent to submit job" not in str(e) or time.time() >= deadline:
                    raise
                logger.info("Ray job agent not ready yet, retrying submit in %.1fs...", sleep_s)
                time.sleep(sleep_s)
                sleep_s = min(5.0, sleep_s * 1.5)
        logger.info("Job submitted with ID: %s", submission_id)
        job_id = JobId(submission_id)
        self._jobs[job_id] = RayJobInfo.from_submission_id(submission_id, request.name)
        return job_id

    def _launch_callable_job(self, request: JobRequest) -> JobId:
        """Launch a callable job on Ray cluster.

        For backwards compatibility with existing Marin usage, we do _not_ start a new job
        in this case, but instead spawn the callable as a ray.remote task.
        """
        entrypoint = request.entrypoint.callable_entrypoint
        runtime_env = self._get_runtime_env(request)
        # strip out keys that can only be set at the Job level
        runtime_env = {k: v for k, v in runtime_env.items() if k not in ["working_dir", "excludes", "config"]}

        if isinstance(request.resources.device, GpuConfig):
            logger.info(
                f"Launching callable job {request.name} on GPU device {request.resources.device.variant} "
                f"with {request.resources.device.count} GPUs."
            )
            num_gpus = request.resources.device.count
        else:
            num_gpus = 0

        remote_fn = ray.remote(num_gpus=num_gpus)(entrypoint.callable)
        ref = remote_fn.options(runtime_env=runtime_env).remote(*entrypoint.args, **entrypoint.kwargs)
        job_id = JobId(str(id(ref)))
        self._jobs[job_id] = RayJobInfo.from_ref(ref, request.name)
        return job_id

    def _launch_tpu_job(self, request: JobRequest) -> JobId:
        # We only launch TPU jobs from an existing Ray cluster. The TPU slice actor
        # bouncing prevents us from using a traditional job submission for TPU workloads.
        callable_ep = request.entrypoint.callable_entrypoint
        assert callable_ep is not None, "TPU jobs require callable entrypoint"

        device = request.resources.device
        runtime_env = self._get_runtime_env(request)

        # strip out keys that can only be set at the Job level
        runtime_env = {k: v for k, v in runtime_env.items() if k not in ["excludes", "config", "working_dir"]}

        logger.info(f"Launching TPU job {request.name} with runtime env: {runtime_env}")

        if callable_ep.args or callable_ep.kwargs:
            remote_fn = ray.remote(max_calls=1, runtime_env=runtime_env)(
                lambda: callable_ep.callable(*callable_ep.args, **callable_ep.kwargs)
            )
        else:
            remote_fn = ray.remote(max_calls=1, runtime_env=runtime_env)(callable_ep.callable)

        object_ref = run_on_pod_ray.remote(
            remote_fn,
            tpu_type=device.variant,
            num_slices=request.resources.replicas,
            max_retries_preemption=10000,
            max_retries_failure=1,
        )

        job_id = JobId(str(id(object_ref)))
        self._jobs[job_id] = RayJobInfo.from_ref(object_ref, name=request.name)
        return job_id


    def _get_runtime_env(self, request: JobRequest) -> dict | None:
        """Build Ray runtime environment for the given job request."""
        environment = request.environment if request.environment else EnvironmentConfig.create()

        env_vars = dict(environment.env_vars)
        # disable access to the TPU if we're not a TPU job, otherwise
        # any import of JAX will claim the TPU and block other users.
        if isinstance(request.resources.device, CpuConfig):
            if "JAX_PLATFORMS" in env_vars and env_vars["JAX_PLATFORMS"] != "cpu":
                logger.warning(
                    "Found existing JAX_PLATFORMS=%s, overriding for CPU only job.",
                    env_vars["JAX_PLATFORMS"],
                )
            env_vars["JAX_PLATFORMS"] = "cpu"
        elif isinstance(request.resources.device, TpuConfig):
            # Don't add tpu extra if vllm is present (they conflict)
            if "tpu" not in environment.extras and "vllm" not in environment.extras:
                environment.extras.append("tpu")
            env_vars["JAX_PLATFORMS"] = ""
        elif isinstance(request.resources.device, GpuConfig):
            if "gpu" not in environment.extras:
                environment.extras.append("gpu")
            env_vars["JAX_PLATFORMS"] = ""

        logger.info(
            "Building environment for device: %s/%s", request.resources.device.kind, request.resources.device.variant
        )

        env_vars["FRAY_CLUSTER_SPEC"] = self._get_cluster_spec()
        logger.info(
            f"Building environment with {environment.pip_packages}, extras {environment.extras} for job: {request.name}"
        )
        if environment.pip_packages or environment.extras:
            runtime_env = build_runtime_env_for_packages(
                extra=list(environment.extras),
                pip_packages=list(environment.pip_packages),
                env_vars=env_vars,
            )
            runtime_env["working_dir"] = environment.workspace
            runtime_env["excludes"] = [".git", "tests/", "docs/", "**/*.pack"]
            runtime_env["config"] = {"setup_timeout_seconds": 1800}
        else:
            # No runtime package installation: rely on the existing environment.
            # This is primarily used for local clusters (including CI), where workers share
            # the submitting machine's filesystem and already have dependencies installed.
            python_path = build_python_path(submodules_dir=os.path.join(environment.workspace, "submodules"))
            python_path = [
                p if os.path.isabs(p) else os.path.join(environment.workspace, p) for p in python_path if p.strip()
            ]
            if "PYTHONPATH" in env_vars:
                python_path.extend([p for p in env_vars["PYTHONPATH"].split(":") if p.strip()])
            env_vars["PYTHONPATH"] = ":".join(python_path)
            runtime_env = {"env_vars": env_vars}

        logger.info("Ray runtime env: %s", runtime_env)
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
                f"TPU-{device.variant}-head": 1.0,
                "TPU": float(device.count),
            }

        return params

    def monitor(self, job_id: JobId) -> JobInfo:
        logger.info("Starting log monitoring for job %s", job_id)

        job = self._jobs[job_id]
        if job.submission_id is None:
            logger.info("Job is a remote ref, monitoring is automatic, waiting.")
            return self.wait(job_id)

        async def stream_logs():
            async for line in self._job_client().tail_job_logs(job_id):
                logger.info(line.rstrip())

        asyncio.run(stream_logs())
        return self.poll(job_id)

    def _poll_ref_job(self, job_id: JobId, job: RayJobInfo) -> JobInfo:
        """Poll a ref-based job (callable/TPU) for status."""
        ready, _ = ray.wait([job.ref], timeout=0)
        if not ready:
            return JobInfo(
                job_id=job_id,
                status=JobStatus.PENDING,
                tasks=[],
                name=job.name,
                error_message=None,
            )
        try:
            ray.get(job.ref)
        except Exception as e:
            logger.warning("Job %s failed with message: %s", job_id, e)
            return JobInfo(
                job_id=job_id,
                status=JobStatus.FAILED,
                tasks=[],
                name=job.name,
                error_message=str(e),
            )
        return JobInfo(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            tasks=[],
            name=job.name,
            error_message=None,
        )

    def poll(self, job_id: JobId) -> JobInfo:
        """Poll job status, returning the current job information."""
        job = self._jobs[job_id]
        if job.submission_id is None:
            return self._poll_ref_job(job_id, job)

        info = self._job_client().get_job_info(job_id)
        status = _convert_ray_status(info.status)
        logger.info("Job %s status: %s (raw: %s)", job_id, status, info.status)

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
        job = self._jobs[job_id]
        if job.submission_id is None:
            ray.cancel(job.ref)
            return

        try:
            self._job_client().stop_job(job.submission_id)
        except Exception as e:
            logger.warning("Failed to stop job %s: %s", job_id, e)
            return

        # Wait for job to stop
        for _ in range(100):
            try:
                info = self._job_client().get_job_info(job_id)
                status = _convert_ray_status(info.status)
                if status in ["stopped", "failed", "succeeded"]:
                    break
            except Exception:
                break
            logger.info("Waiting for job %s to stop", job_id)
            time.sleep(1.0)

    def list_jobs(self) -> list[JobInfo]:
        """List all jobs tracked by this cluster."""
        result = []

        # Get submitted jobs from Ray's job client
        try:
            for job_info in self._job_client().list_jobs():
                result.append(
                    JobInfo(
                        job_id=JobId(job_info.submission_id),
                        status=_convert_ray_status(job_info.status),
                        tasks=[],
                        name=job_info.metadata.get("name", "") if job_info.metadata else "",
                        error_message=(
                            job_info.message if _convert_ray_status(job_info.status) == JobStatus.FAILED else None
                        ),
                    )
                )
        except Exception as e:
            logger.warning("Failed to list jobs from job client: %s", e)

        # Add ref-based jobs (callable/TPU jobs not tracked by submission client)
        for job_id, job in self._jobs.items():
            if job.submission_id is None:
                result.append(self.poll(job_id))

        return result

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
