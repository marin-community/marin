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

"""Ray backend implementation for ClusterContext."""

import os
import time
from collections.abc import Callable
from typing import Any

import ray
from ray.job_submission import JobSubmissionClient

from fray.backend.ray.ray_job import RayJobContext
from fray.backend.ray.ray_tpu import run_on_pod
from fray.cluster import ClusterContext
from fray.types import EntryPoint, JobInfo, RuntimeEnv, TpuRunConfig


class RayClusterContext(ClusterContext):
    """Ray-based cluster context implementation."""

    def __init__(self, address: str | None = None, dashboard_address: str | None = None):
        """
        Initialize Ray cluster context.

        Args:
            address: Ray cluster address for ray.init() (None for local mode)
            dashboard_address: Dashboard URL for job submission (defaults to address)
                             Format: "http://hostname:8265" or None to use RAY_ADDRESS env var
        """
        self._address = address
        self._dashboard_address = dashboard_address or address

        # Initialize Ray runtime (for run_on_tpu and in-cluster operations)
        if not ray.is_initialized():
            ray.init(address=address)

        # Create job submission client (for create_job, list_jobs, delete_job)
        self._job_client = JobSubmissionClient(self._dashboard_address)

    def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
        """
        Submit a job using Ray JobSubmissionClient.

        Args:
            entrypoint: Shell command to execute (e.g., "python train.py --config foo.yaml")
            env: Runtime environment specification

        Returns:
            job_id: Unique identifier for the submitted job

        Example:
            cluster = RayClusterContext(dashboard_address="http://ray:8265")
            job_id = cluster.create_job(
                entrypoint="python train.py --batch-size 32",
                env=RuntimeEnv(
                    package_requirements=["torch>=2.0"],
                    env={"WANDB_PROJECT": "my-project"}
                )
            )
        """
        # Build Ray runtime environment from Fray RuntimeEnv
        runtime_env = self._build_ray_runtime_env(env)

        # Add current working directory for code access
        runtime_env["working_dir"] = os.getcwd()

        # Add setup timeout configuration
        runtime_env["config"] = {"setup_timeout_seconds": 1800}

        # Generate unique job ID
        job_id = f"fray-{time.time_ns()}"

        # Submit job to Ray cluster
        job_id = self._job_client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            submission_id=job_id,
        )

        return job_id

    def list_jobs(self) -> list[JobInfo]:
        """
        List all jobs in the cluster.

        Returns:
            List of JobInfo objects with job metadata
        """
        from ray.job_submission import JobType

        jobs = []
        # list_jobs() returns JobDetails objects, not job ID strings
        for job_details in self._job_client.list_jobs():
            # Only include SUBMISSION type jobs, not DRIVER jobs (those are pytest/internal processes)
            if job_details.type != JobType.SUBMISSION:
                continue

            # Use submission_id as the job identifier
            job_id = job_details.submission_id or job_details.job_id or "unknown"

            jobs.append(
                JobInfo(
                    id=job_id,
                    status=str(job_details.status),
                    submission_time=job_details.start_time / 1000 if job_details.start_time else 0,
                    start_time=job_details.start_time / 1000 if job_details.start_time else None,
                    end_time=job_details.end_time / 1000 if job_details.end_time else None,
                )
            )
        return jobs

    def delete_job(self, job_id: str) -> None:
        """
        Stop and delete a job from the cluster.

        This stops a running job or marks a pending job as stopped.
        Equivalent to ray_run.py's --auto-stop functionality.

        Args:
            job_id: Identifier of job to stop/delete
        """
        self._job_client.stop_job(job_id)

    def run_on_tpu(self, fn: Callable, config: TpuRunConfig, runtime_env: RuntimeEnv | None = None) -> list[Any]:
        """
        Execute function on TPU slices using ray_tpu.run_on_pod.

        Args:
            fn: Function to execute. Receives a JobContext as its argument.
            config: TPU execution configuration
            runtime_env: Runtime environment (packages, env vars, etc.)

        Returns:
            List of results, one per TPU host across all slices
        """

        # Wrap user function to inject JobContext
        def wrapped_fn():
            job_ctx = RayJobContext()
            return fn(job_ctx)

        # Convert to Ray remote function with max_calls=1 (required for TPUs)
        remote_fn = ray.remote(max_calls=1)(wrapped_fn)

        # Build Ray runtime_env from Fray RuntimeEnv
        ray_runtime_env = self._build_ray_runtime_env(runtime_env) if runtime_env else {}

        # Apply runtime_env to remote function
        if ray_runtime_env:
            remote_fn = remote_fn.options(runtime_env=ray_runtime_env)

        # Delegate to ray_tpu.run_on_pod
        return run_on_pod(
            remote_fn,
            tpu_type=config.tpu_type,
            num_slices=config.num_slices,
            max_retries_preemption=config.max_retries_preemption,
            max_retries_failure=config.max_retries_failure,
        )

    def _build_ray_runtime_env(self, env: RuntimeEnv) -> dict:
        """
        Convert Fray RuntimeEnv to Ray runtime_env dict.

        Args:
            env: Fray RuntimeEnv

        Returns:
            Ray runtime_env dict
        """
        ray_env = {}

        if env.package_requirements:
            ray_env["pip"] = env.package_requirements

        if env.env:
            ray_env["env_vars"] = env.env

        # Note: Fray's RuntimeEnv has minimum_resources and maximum_resources,
        # but Ray's runtime_env doesn't support these directly. They should be
        # handled at the task/actor level when scheduling.

        return ray_env
