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

"""Cluster context interface for job management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from fray.job import JobContext
from fray.types import EntryPoint, JobInfo, RuntimeEnv, TpuRunConfig


class ClusterContext(ABC):
    """
    Context for managing jobs on a cluster.

    Provides the ability to create, list, and delete jobs. A job is a set of
    resources and an isolated execution context. Tasks within a job are
    launched via JobContext.
    """

    @abstractmethod
    def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
        """
        Create and start a new job.

        Args:
            entrypoint: Shell command to execute as the job's main entry point.
                       For example: "python train.py --config foo.yaml"
            env: Runtime environment specification for the job

        Returns:
            job_id: Unique identifier for the created job

        Example:
            cluster = RayClusterContext(dashboard_address="http://ray:8265")
            job_id = cluster.create_job(
                entrypoint="python train.py --batch-size 32",
                env=RuntimeEnv(
                    package_requirements=["torch", "wandb"],
                    env={"WANDB_API_KEY": "xxx"}
                )
            )
        """
        pass

    @abstractmethod
    def list_jobs(self) -> list[JobInfo]:
        """
        List all jobs in the cluster.

        Returns:
            List of JobInfo objects containing job metadata

        Example:
            jobs = cluster.list_jobs()
            for job in jobs:
                print(f"{job.id}: {job.status}")
        """
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> None:
        """
        Delete a job from the cluster.

        Removes the job from tracking. Behavior depends on implementation:
        - May terminate running job
        - May just remove metadata for completed job

        Args:
            job_id: Identifier of job to delete

        Example:
            cluster.delete_job("job_123")
        """
        pass

    @abstractmethod
    def run_on_tpu(
        self, fn: Callable[[JobContext], Any], config: TpuRunConfig, runtime_env: RuntimeEnv | None = None
    ) -> list[Any]:
        """
        Execute a function across TPU slices.

        The function is executed once per TPU host across all allocated slices.
        For multislice jobs, MEGASCALE_* environment variables are automatically
        injected to enable cross-slice coordination.

        Args:
            fn: Function to execute. Receives a JobContext as its argument.
                Called once per TPU host. Should be stateless and idempotent.
            config: TPU execution configuration (TpuRunConfig)
            runtime_env: Runtime environment (packages, env vars, etc.)

        Returns:
            List of results, one per TPU host across all slices.
            For a v4-32 with 4 VMs per slice and 2 slices, returns 8 results.

        Raises:
            RuntimeError: If max retries exceeded for preemption or failure

        Behavior:
            - Automatically retries on preemption (up to max_retries_preemption)
            - Retries on application failures (up to max_retries_failure)
            - Cancels all pending work if any task fails
            - Cleans up resources (actors, lockfiles) on exit

        Example:
            from fray import TpuRunConfig

            def compute(ctx: JobContext):
                import jax
                import jax.numpy as jnp

                # JAX discovers TPUs automatically via MEGASCALE_* env vars
                devices = jax.devices()
                result = jnp.sum(jnp.arange(1000))
                return float(result)

            config = TpuRunConfig(tpu_type="v4-32", num_slices=2)
            results = cluster.run_on_tpu(compute, config)
            # Returns 8 results (4 hosts/slice * 2 slices)
        """
        pass
