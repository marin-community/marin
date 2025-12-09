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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, NewType

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
    "A100",
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


@dataclass(frozen=True)
class CpuConfig:
    """CPU-only device configuration."""

    type: str = "cpu"

    def chip_count(self) -> int:
        """CPU has no accelerator chips."""
        return 0

    def device_flops(self, dtype: str = "bf16") -> float:
        """CPU FLOPS not tracked."""
        raise NotImplementedError("CPU FLOPS not available")


@dataclass(frozen=True)
class GpuConfig:
    """GPU device configuration."""

    type: GpuType
    count: int = 1

    def chip_count(self) -> int:
        """Total number of GPU chips."""
        return self.count

    def device_flops(self, dtype: str = "bf16") -> float:
        """Peak FLOP/s for a single GPU."""
        from fray.cluster.device_flops import device_flops

        flops = device_flops(self.type, dtype)
        if flops is None:
            raise ValueError(f"Unknown device/dtype: {self.type}/{dtype}")
        return flops

    def total_flops(self, dtype: str = "bf16") -> float:
        """Total peak FLOP/s across all GPUs."""
        return self.device_flops(dtype) * self.count


@dataclass(frozen=True)
class TpuConfig:
    """TPU device configuration.

    Args:
        type: TPU accelerator type (e.g., "v5litepod-16", "v4-8")
        topology: Optional topology specification (e.g., "2x2x1")
    """

    type: TpuType
    topology: str | None = None

    def chip_count(self) -> int:
        """Total number of TPU chips."""
        return get_tpu_topology(self.type).chip_count

    def vm_count(self) -> int:
        """Number of VMs in the TPU pod."""
        return get_tpu_topology(self.type).vm_count

    def device_flops(self, dtype: str = "bf16") -> float:
        """Peak FLOP/s for a single TPU chip."""
        from fray.cluster.device_flops import device_flops

        flops = device_flops(self.type, dtype)
        if flops is None:
            raise ValueError(f"Unknown device/dtype: {self.type}/{dtype}")
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
    def with_tpu(tpu_type: str, slice_count: int = 1) -> ResourceConfig:
        """Create a TPU resource config with sensible defaults."""
        device = TpuConfig(type=tpu_type)  # type: ignore[arg-type]
        return ResourceConfig(device=device, replicas=slice_count)

    @staticmethod
    def with_gpu(gpu_type: str = "auto", count: int = 1) -> ResourceConfig:
        """Create a GPU resource config with sensible defaults."""
        device = GpuConfig(type=gpu_type, count=count)  # type: ignore[arg-type]
        return ResourceConfig(device=device)

    @staticmethod
    def with_cpu() -> ResourceConfig:
        """Create a CPU-only resource config."""
        return ResourceConfig(device=CpuConfig())


@dataclass
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


@dataclass(frozen=True)
class Entrypoint:
    """Job entrypoint specification.

    Args:
        binary: Binary to execute (e.g., "python", "uv", "bash")
        args: Command-line arguments for the binary
        callable: Callable for direct execution
        function_args: Keyword arguments to pass to callable

    Examples:
        Entrypoint(binary="python", args=["train.py", "--config", "config.yaml"])
        Entrypoint(callable=train_fn, function_args={"config": config, "epochs": 100})
        Entrypoint(callable=lambda: train_fn(config))
    """

    binary: str | None = None
    args: Sequence[str] = field(default_factory=list)
    callable: Callable[..., Any] | None = None
    function_args: dict[str, Any] | None = None

    def __post_init__(self):
        if self.binary is None and self.callable is None:
            raise ValueError("Must specify either binary or callable")
        if self.binary is not None and self.callable is not None:
            raise ValueError("Cannot specify both binary and callable")
        if self.args and self.callable is not None:
            raise ValueError("args only valid with binary, not callable")
        if self.function_args is not None and self.callable is None:
            raise ValueError("function_args only valid with callable, not binary")
        if self.function_args is not None and not isinstance(self.function_args, dict):
            raise ValueError("function_args must be a dictionary")


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

    def __post_init__(self):
        if " " in self.name:
            raise ValueError("Job name must not contain spaces")


JobId = NewType("JobId", str)


@dataclass
class TaskStatus:
    status: Literal["pending", "running", "succeeded", "failed", "stopped"]
    error_message: str | None = None


@dataclass
class JobInfo:
    job_id: JobId
    status: Literal["pending", "running", "succeeded", "failed", "stopped"]
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

    @abstractmethod
    @contextmanager
    def connect(self):
        """Establish connection to cluster."""
        ...

    def wait(self, job_id: JobId) -> JobInfo:
        """Block until the specified job completes, returning its final status."""
        logger.info(f"Starting wait for job {job_id}")

        while True:
            info = self.poll(job_id)
            if info.status in ["succeeded", "failed", "stopped"]:
                logger.info(f"Job {job_id} completed with status {info.status}")
                return info
            time.sleep(10.0)
