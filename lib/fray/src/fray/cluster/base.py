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

"""Cluster interface and type definitions for job scheduling."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, NewType, Self

logger = logging.getLogger(__name__)

TpuType = Literal[
    "v4-8",
    "v4-16",
    "v4-32",
    "v4-64",
    "v4-128",
    "v4-256",
    "v4-512",
    "v4-1024",
    "v4-2048",
    "v4-4096",
    "v5litepod-1",
    "v5litepod-4",
    "v5litepod-8",
    "v5litepod-16",
    "v5litepod-32",
    "v5litepod-64",
    "v5litepod-128",
    "v5litepod-256",
    "v5p-8",
    "v5p-16",
    "v5p-32",
    "v5p-64",
    "v5p-128",
    "v5p-256",
    "v5p-384",
    "v5p-512",
    "v5p-640",
    "v5p-768",
    "v5p-896",
    "v5p-1024",
    "v5p-1152",
    "v5p-1280",
    "v5p-1408",
    "v5p-1536",
    "v5p-1664",
    "v5p-1792",
    "v5p-1920",
    "v5p-2048",
    "v5p-2176",
    "v5p-2304",
    "v5p-2432",
    "v5p-2560",
    "v5p-2688",
    "v5p-2816",
    "v5p-2944",
    "v5p-3072",
    "v5p-3200",
    "v5p-3328",
    "v5p-3456",
    "v5p-3584",
    "v5p-3712",
    "v5p-3840",
    "v5p-3968",
    "v5p-4096",
    "v5p-4224",
    "v5p-4352",
    "v5p-4480",
    "v5p-4608",
    "v5p-4736",
    "v5p-4864",
    "v5p-4992",
    "v5p-5120",
    "v5p-5248",
    "v5p-5376",
    "v5p-5504",
    "v5p-5632",
    "v5p-5760",
    "v5p-5888",
    "v5p-6016",
    "v5p-6144",
    "v5p-6272",
    "v5p-6400",
    "v5p-6528",
    "v5p-6656",
    "v5p-6784",
    "v5p-6912",
    "v5p-7040",
    "v5p-7168",
    "v5p-7296",
    "v5p-7424",
    "v5p-7552",
    "v5p-7680",
    "v5p-7808",
    "v5p-7936",
    "v5p-8064",
    "v5p-8192",
    "v5p-8320",
    "v5p-8448",
    "v5p-8576",
    "v5p-8704",
    "v5p-8832",
    "v5p-8960",
    "v5p-9088",
    "v5p-9216",
    "v5p-9344",
    "v5p-9472",
    "v5p-9600",
    "v5p-9728",
    "v5p-9856",
    "v5p-9984",
    "v5p-10240",
    "v5p-10368",
    "v5p-10496",
    "v5p-10624",
    "v5p-10752",
    "v5p-10880",
    "v5p-11008",
    "v5p-11136",
    "v5p-11264",
    "v5p-11392",
    "v5p-11520",
    "v5p-11648",
    "v5p-11776",
    "v5p-11904",
    "v5p-12032",
    "v5p-12160",
    "v5p-12288",
    "v6e-1",
    "v6e-4",
    "v6e-8",
    "v6e-16",
    "v6e-32",
    "v6e-64",
    "v6e-128",
    "v6e-256",
]

GpuType = Literal[
    "A10",
    "A100-40G",
    "A100-80G",
    "A10G",
    "B100",
    "H100",
    "H200",
    "L4",
    "T4",
    "V100",
    "auto",
]


# TPU configurations are complicated. The number of chips per host is
# always the same for a particular generation, but the number of VMs per host
# can vary based on the pod size.
#
# Even more confusingly, Google sometimes refers to TPU cores
# as chips and vice-versa: v4 and v5p topologies refer to "core", but
# v5e and v6e topologies refer to "chips". It's doubly confusing as some
# topologies split 2 VMs per host, while others do not. We just write them
# all down here.
@dataclass(frozen=True)
class TpuTopologyInfo:
    name: str
    chip_count: int
    host_count: int
    vm_count: int
    chips_per_vm: int


TPU_TOPOLOGIES: list[TpuTopologyInfo] = [
    # https://cloud.google.com/tpu/docs/v4
    TpuTopologyInfo("v4-8", 4, 1, 1, 4),
    TpuTopologyInfo("v4-16", 8, 2, 2, 4),
    TpuTopologyInfo("v4-32", 16, 4, 4, 4),
    TpuTopologyInfo("v4-64", 32, 8, 8, 4),
    TpuTopologyInfo("v4-128", 64, 16, 16, 4),
    TpuTopologyInfo("v4-256", 128, 32, 32, 4),
    TpuTopologyInfo("v4-512", 256, 64, 64, 4),
    TpuTopologyInfo("v4-1024", 512, 128, 128, 4),
    TpuTopologyInfo("v4-2048", 1024, 256, 256, 4),
    TpuTopologyInfo("v4-4096", 2048, 512, 512, 4),
    # https://cloud.google.com/tpu/docs/v5e
    TpuTopologyInfo("v5litepod-1", 1, 1, 1, 1),
    TpuTopologyInfo("v5litepod-2", 2, 1, 1, 2),
    TpuTopologyInfo("v5litepod-4", 4, 1, 1, 4),
    TpuTopologyInfo("v5litepod-8", 8, 1, 1, 8),
    TpuTopologyInfo("v5litepod-16", 16, 2, 4, 4),
    TpuTopologyInfo("v5litepod-32", 32, 4, 8, 4),
    TpuTopologyInfo("v5litepod-64", 64, 8, 16, 4),
    TpuTopologyInfo("v5litepod-128", 128, 16, 32, 4),
    TpuTopologyInfo("v5litepod-256", 256, 32, 64, 4),
    # https://cloud.google.com/tpu/docs/v5p
    TpuTopologyInfo("v5p-8", 4, 1, 1, 4),
    TpuTopologyInfo("v5p-16", 8, 2, 2, 4),
    TpuTopologyInfo("v5p-32", 16, 4, 4, 4),
    TpuTopologyInfo("v5p-64", 32, 8, 8, 4),
    TpuTopologyInfo("v5p-128", 64, 16, 16, 4),
    TpuTopologyInfo("v5p-256", 128, 32, 32, 4),
    TpuTopologyInfo("v5p-512", 256, 64, 64, 4),
    TpuTopologyInfo("v5p-1024", 512, 128, 128, 4),
    TpuTopologyInfo("v5p-2048", 1024, 256, 256, 4),
    TpuTopologyInfo("v5p-4096", 2048, 512, 512, 4),
    TpuTopologyInfo("v5p-8192", 4096, 1024, 1024, 4),
    TpuTopologyInfo("v5p-12288", 6144, 1536, 1536, 4),
    # https://cloud.google.com/tpu/docs/v6e
    TpuTopologyInfo("v6e-1", 1, 1, 1, 1),
    TpuTopologyInfo("v6e-4", 4, 1, 1, 4),
    TpuTopologyInfo("v6e-8", 8, 1, 1, 8),
    TpuTopologyInfo("v6e-16", 16, 4, 4, 4),
    TpuTopologyInfo("v6e-32", 32, 8, 8, 4),
    TpuTopologyInfo("v6e-64", 64, 16, 16, 4),
    TpuTopologyInfo("v6e-128", 128, 32, 32, 4),
    TpuTopologyInfo("v6e-256", 256, 64, 64, 4),
]


def get_tpu_topology(tpu_type: str) -> TpuTopologyInfo:
    """Get TPU topology by type name."""
    for config in TPU_TOPOLOGIES:
        if config.name == tpu_type:
            return config
    raise ValueError(f"Unknown TPU type: {tpu_type}")


DeviceKind = Literal["cpu", "gpu", "tpu"]


@dataclass(frozen=True)
class CpuConfig:
    """CPU-only device configuration."""

    kind: DeviceKind = "cpu"
    variant: str = "cpu"

    def chip_count(self) -> int:
        """CPU has no accelerator chips."""
        return 0

    def device_flops(self, dtype: str = "bf16") -> float:
        """CPU FLOPS not tracked."""
        raise NotImplementedError("CPU FLOPS not available")


@dataclass(frozen=True)
class GpuConfig:
    """GPU device configuration."""

    variant: GpuType
    kind: DeviceKind = "gpu"
    count: int = 1

    def chip_count(self) -> int:
        """Total number of GPU chips."""
        return self.count

    def device_flops(self, dtype: str = "bf16") -> float:
        """Peak FLOP/s for a single GPU."""
        from fray.cluster.device_flops import device_flops

        flops = device_flops(self.variant, dtype)
        if flops is None:
            raise ValueError(f"Unknown device/dtype: {self.variant}/{dtype}")
        return flops

    def total_flops(self, dtype: str = "bf16") -> float:
        """Total peak FLOP/s across all GPUs."""
        return self.device_flops(dtype) * self.count


@dataclass(frozen=True)
class TpuConfig:
    """TPU device configuration.

    Args:
        variant: TPU accelerator type (e.g., "v5litepod-16", "v4-8")
        topology: Optional topology specification (e.g., "2x2x1")
    """

    variant: TpuType
    kind: DeviceKind = "tpu"
    topology: str | None = None

    def chip_count(self) -> int:
        """Total number of TPU chips."""
        return get_tpu_topology(self.variant).chip_count

    def vm_count(self) -> int:
        """Number of VMs in the TPU pod."""
        return get_tpu_topology(self.variant).vm_count

    def device_flops(self, dtype: str = "bf16") -> float:
        """Peak FLOP/s for a single TPU chip."""
        from fray.cluster.device_flops import device_flops

        flops = device_flops(self.variant, dtype)
        if flops is None:
            raise ValueError(f"Unknown device/dtype: {self.variant}/{dtype}")
        return flops

    def total_flops(self, dtype: str = "bf16") -> float:
        """Total peak FLOP/s across all TPU chips."""
        return self.device_flops(dtype) * self.chip_count()


DeviceConfig = CpuConfig | GpuConfig | TpuConfig


@dataclass
class ResourceConfig:
    """Resource requirements for a job.

    Args:
        cpu: Number of CPU cores
        ram: RAM requirement (e.g., "8g", "16g")
        disk: Disk space requirement (e.g., "10g", "100g")
        device: Device configuration (CPU, GPU, or TPU)
        replicas: Number of replicas/slices (for multislice TPU training)
        preemptible: Whether the job can be preempted
        regions: Preferred cloud regions for job placement
    """

    cpu: int = 1
    ram: str = "128m"
    disk: str = "1g"
    device: DeviceConfig = field(default_factory=CpuConfig)
    replicas: int = 1
    preemptible: bool = True
    regions: Sequence[str] | None = None

    def chip_count(self) -> int:
        """Total accelerator chips across all replicas/slices."""
        return self.device.chip_count() * self.replicas

    def device_flops(self, dtype: str = "bf16") -> float:
        """Peak FLOP/s for a single device."""
        return self.device.device_flops(dtype)

    def total_flops(self, dtype: str = "bf16") -> float:
        """Total peak FLOP/s across all devices."""
        if isinstance(self.device, CpuConfig):
            # just use some reasonable number
            return 100e9
        return self.device_flops(dtype) * self.chip_count()

    @staticmethod
    def with_tpu(tpu_type: str, slice_count: int = 1, **kwargs) -> ResourceConfig:
        device = TpuConfig(variant=tpu_type)
        return ResourceConfig(device=device, replicas=slice_count, **kwargs)

    @staticmethod
    def with_gpu(gpu_type: str = "auto", count: int = 1, **kwargs) -> ResourceConfig:
        device = GpuConfig(variant=gpu_type, count=count)
        return ResourceConfig(device=device, **kwargs)

    @staticmethod
    def with_cpu(**kwargs) -> ResourceConfig:
        return ResourceConfig(device=CpuConfig(), **kwargs)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Job environment configuration.

    Can specify either a workspace (for uv-based dependency resolution)
    or a docker image (for containerized execution).

    Args:
        workspace: Path to workspace root for uv-based execution
        docker_image: Docker image for containerized execution
        pip_packages: Additional pip packages to install
        env_vars: Environment variables to set
        extras: Extra dependency groups for uv (e.g., ["tpu", "eval"])
    """

    workspace: str | None = None
    docker_image: str | None = None
    pip_packages: Sequence[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    extras: Sequence[str] = field(default_factory=list)

    def __post_init__(self):
        if self.workspace and self.docker_image:
            raise ValueError("Cannot specify both workspace and docker_image")
        if not self.workspace and not self.docker_image:
            raise ValueError("Must specify either workspace or docker_image")

    @staticmethod
    def create(*args, **kw):
        return create_environment(*args, **kw)


@dataclass(frozen=True)
class BinaryEntrypoint:
    command: str
    args: Sequence[str]


@dataclass(frozen=True)
class CallableEntrypoint:
    callable: Callable[..., Any]
    args: Sequence[Any] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entrypoint:
    callable_entrypoint: CallableEntrypoint | None = None
    binary_entrypoint: BinaryEntrypoint | None = None

    @staticmethod
    def from_callable(
        c: Callable[..., Any],
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Self:
        return Entrypoint(callable_entrypoint=CallableEntrypoint(callable=c, args=args, kwargs=kwargs or {}))

    @staticmethod
    def from_binary(command: str, args: Sequence[str]) -> Self:
        return Entrypoint(binary_entrypoint=BinaryEntrypoint(command=command, args=args))


@dataclass
class JobRequest:
    """Complete job specification for cluster submission.

    Args:
        name: Human-readable job name
        entrypoint: Job entrypoint (command-line or callable)
        resources: Resource requirements for the job
        environment: Environment configuration (dependencies, env vars)
    """

    name: str
    entrypoint: Entrypoint
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    environment: EnvironmentConfig | None = None

    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    def __post_init__(self):
        if " " in self.name:
            raise ValueError("Job name must not contain spaces")


JobId = NewType("JobId", str)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"

    @staticmethod
    def finished(status: Self) -> bool:
        return status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED)


@dataclass
class TaskStatus:
    status: JobStatus
    error_message: str | None = None


@dataclass
class JobInfo:
    job_id: JobId
    status: JobStatus
    tasks: list[TaskStatus]
    name: str
    error_message: str | None = None


def create_environment(
    workspace: str | None = None,
    docker_image: str | None = None,
    pip_packages: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
    extras: Sequence[str] | None = None,
) -> EnvironmentConfig:
    """Create an EnvironmentConfig

    By default, sets the following environment variables:
    - HF_DATASETS_TRUST_REMOTE_CODE: "1" (allows custom dataset code)
    - TOKENIZERS_PARALLELISM: "false" (avoids tokenizer deadlocks)
    - HF_TOKEN
    - WANDB_API_KEY

    Args:
        workspace: Path to workspace root (default: current directory)
        docker_image: Docker image (mutually exclusive with workspace)
        pip_packages: Additional pip packages to install
        env_vars: Custom environment variables (merged with defaults)
        extras: Extra dependency groups for uv

    Returns:
        EnvironmentConfig with defaults applied
    """

    if workspace is None and docker_image is None:
        workspace = os.getcwd()

    default_env_vars = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "MARIN_CI_DISABLE_RUNTIME_ENVS": os.getenv("MARIN_CI_DISABLE_RUNTIME_ENVS"),
    }

    # Filter out None values - Ray requires all env var values to be strings
    merged_env_vars = {k: v for k, v in {**default_env_vars, **(env_vars or {})}.items() if v is not None}

    return EnvironmentConfig(
        workspace=workspace,
        docker_image=docker_image,
        pip_packages=list(pip_packages or []),
        env_vars=merged_env_vars,
        extras=list(extras or []),
    )


class Cluster(ABC):
    """Abstract interface for cluster job scheduling."""

    @abstractmethod
    def launch(self, request: JobRequest) -> JobId:
        """Launch a job on the cluster.

        Args:
            request: Job specification including resources, environment, and entrypoint

        Returns:
            Unique identifier for the launched job

        Raises:
            ValueError: If the request is invalid
            RuntimeError: If job submission fails
        """
        ...

    @abstractmethod
    def monitor(self, job_id: JobId) -> JobInfo:
        """Stream logs from a running job, blocking until completion.

        Logs are emitted directly via the logger. Blocks until the job
        completes or is terminated.

        Args:
            job_id: Job identifier returned by launch()

        Returns:
            JobInfo with final job status

        Raises:
            KeyError: If job_id is not found
        """
        ...

    @abstractmethod
    def poll(self, job_id: JobId) -> JobInfo:
        """Get current status of a job without blocking.

        Args:
            job_id: Job identifier

        Returns:
            Current job information including status

        Raises:
            KeyError: If job_id is not found
        """
        ...

    @abstractmethod
    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Attempts graceful termination first, then forceful kill if needed.

        Args:
            job_id: Job identifier

        Raises:
            KeyError: If job_id is not found
        """
        ...

    @abstractmethod
    def list_jobs(self) -> list[JobInfo]:
        """List all jobs managed by this cluster.

        Returns:
            List of job information for all jobs (running, completed, and failed)
        """
        ...

    def wait(self, job_id: JobId | Sequence[JobId], raise_on_failure: bool = False) -> JobInfo | list[JobInfo]:
        """Block until the specified job(s) complete, returning final status.

        Args:
            job_id: Single job ID or sequence of job IDs to wait for
            raise_on_failure: If True, raises RuntimeError when any job fails

        Returns:
            JobInfo for single job, or list of JobInfo for multiple jobs
        """
        if isinstance(job_id, str):
            return self._wait_single(job_id, raise_on_failure)

        # Multiple jobs: wait for all, then check for failures
        results = [self._wait_single(jid, raise_on_failure=False) for jid in job_id]
        if raise_on_failure:
            failed = [r for r in results if r.status in (JobStatus.FAILED, JobStatus.STOPPED)]
            if failed:
                msg = "; ".join(f"{r.job_id}: {r.error_message}" for r in failed)
                raise RuntimeError(f"{len(failed)} job(s) failed: {msg}")
        return results

    def _wait_single(self, job_id: JobId, raise_on_failure: bool = False) -> JobInfo:
        """Wait for a single job to complete."""
        logger.info(f"Starting wait for job {job_id}")
        sleep_secs = 0.1
        max_sleep_secs = 10.0

        while True:
            info = self.poll(job_id)
            if JobStatus.finished(info.status):
                logger.info(f"Job {job_id} completed with status {info.status}")
                if raise_on_failure and info.status in (JobStatus.FAILED, JobStatus.STOPPED):
                    raise RuntimeError(f"Job {job_id} failed with status {info.status} and error: {info.error_message}")
                return info
            time.sleep(sleep_secs)
            sleep_secs = min(sleep_secs * 2, max_sleep_secs)
