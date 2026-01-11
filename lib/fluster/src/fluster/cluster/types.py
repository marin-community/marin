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

"""Core types for the fluster cluster layer.

This module contains all cluster-level types with NO knowledge of actors.
Types are ported from fray but simplified according to the fray-zero design.
"""

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NewType
from collections.abc import Callable, Sequence

# Type aliases for clarity
JobId = NewType("JobId", str)
Namespace = NewType("Namespace", str)
WorkerId = NewType("WorkerId", str)
VMId = NewType("VMId", str)
EndpointId = NewType("EndpointId", str)


class JobStatus(StrEnum):
    """Status of a job in the cluster."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"

    @staticmethod
    def finished(status: "JobStatus") -> bool:
        """Check if job has reached terminal state."""
        return status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED)


# TPU Topology Information
# Port from lib/fray/src/fray/cluster/base.py


@dataclass(frozen=True)
class TpuTopologyInfo:
    """TPU topology configuration.

    Args:
        name: TPU type name (e.g., "v5litepod-16", "v4-8")
        chip_count: Total number of TPU chips
        host_count: Number of physical hosts
        vm_count: Number of VMs in the pod
        chips_per_vm: Number of chips per VM
    """

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
    """Get TPU topology by type name.

    Args:
        tpu_type: TPU type name (e.g., "v5litepod-16", "v4-8")

    Returns:
        TpuTopologyInfo for the given type

    Raises:
        ValueError: If TPU type is unknown
    """
    for config in TPU_TOPOLOGIES:
        if config.name == tpu_type:
            return config
    raise ValueError(f"Unknown TPU type: {tpu_type}")


# Device Configurations


@dataclass(frozen=True)
class CpuConfig:
    """CPU-only device configuration."""

    kind: str = "cpu"
    variant: str = "cpu"

    def chip_count(self) -> int:
        """CPU has no accelerator chips."""
        return 0


@dataclass(frozen=True)
class GpuConfig:
    """GPU device configuration.

    Args:
        variant: GPU type (e.g., "A100", "H100", "auto")
        count: Number of GPUs
    """

    variant: str
    kind: str = "gpu"
    count: int = 1

    def chip_count(self) -> int:
        """Total number of GPU chips."""
        return self.count


@dataclass(frozen=True)
class TpuConfig:
    """TPU device configuration.

    Args:
        variant: TPU type (e.g., "v5litepod-16", "v4-8")
        topology: Optional topology specification (e.g., "2x2x1")
    """

    variant: str
    kind: str = "tpu"
    topology: str | None = None

    def chip_count(self) -> int:
        """Total number of TPU chips."""
        return get_tpu_topology(self.variant).chip_count

    def vm_count(self) -> int:
        """Number of VMs in the TPU pod."""
        return get_tpu_topology(self.variant).vm_count


DeviceConfig = CpuConfig | GpuConfig | TpuConfig


@dataclass
class ResourceConfig:
    """Resource requirements for a job.

    Args:
        cpu: Number of CPU cores
        ram: RAM requirement (e.g., "8g", "16g")
        disk: Disk space requirement (e.g., "1g", "100g")
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

    @staticmethod
    def with_tpu(tpu_type: str, slice_count: int = 1, **kwargs) -> "ResourceConfig":
        """Factory method for TPU configuration.

        Args:
            tpu_type: TPU variant (e.g., "v5litepod-16", "v4-8")
            slice_count: Number of TPU slices
            **kwargs: Additional ResourceConfig fields

        Returns:
            ResourceConfig configured for TPU
        """
        device = TpuConfig(variant=tpu_type)
        return ResourceConfig(device=device, replicas=slice_count, **kwargs)

    @staticmethod
    def with_gpu(gpu_type: str = "auto", count: int = 1, **kwargs) -> "ResourceConfig":
        """Factory method for GPU configuration.

        Args:
            gpu_type: GPU variant (e.g., "A100", "H100")
            count: Number of GPUs
            **kwargs: Additional ResourceConfig fields

        Returns:
            ResourceConfig configured for GPU
        """
        device = GpuConfig(variant=gpu_type, count=count)
        return ResourceConfig(device=device, **kwargs)

    @staticmethod
    def with_cpu(**kwargs) -> "ResourceConfig":
        """Factory method for CPU-only configuration.

        Args:
            **kwargs: Additional ResourceConfig fields

        Returns:
            ResourceConfig configured for CPU only
        """
        return ResourceConfig(device=CpuConfig(), **kwargs)


# Environment Configuration


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
        # Validation: exactly one of workspace or docker_image must be set
        if self.workspace and self.docker_image:
            raise ValueError("Cannot specify both workspace and docker_image")
        if not self.workspace and not self.docker_image:
            raise ValueError("Must specify either workspace or docker_image")

    @staticmethod
    def create(*args, **kw):
        """Convenience method that calls create_environment()."""
        return create_environment(*args, **kw)


def create_environment(
    workspace: str | None = None,
    docker_image: str | None = None,
    pip_packages: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
    extras: Sequence[str] | None = None,
) -> EnvironmentConfig:
    """Create an EnvironmentConfig with sensible defaults.

    Default environment variables:
    - HF_DATASETS_TRUST_REMOTE_CODE: "1" (allows custom dataset code)
    - TOKENIZERS_PARALLELISM: "false" (avoids tokenizer deadlocks)
    - HF_TOKEN: from os.environ (if set)
    - WANDB_API_KEY: from os.environ (if set)

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

    # Filter out None values
    merged_env_vars = {k: v for k, v in {**default_env_vars, **(env_vars or {})}.items() if v is not None}

    return EnvironmentConfig(
        workspace=workspace,
        docker_image=docker_image,
        pip_packages=list(pip_packages or []),
        env_vars=merged_env_vars,
        extras=list(extras or []),
    )


# Job Specification


@dataclass
class Entrypoint:
    """Job entrypoint specification.

    Simplified from fray - just a callable with args/kwargs.
    The callable must be picklable.

    Args:
        callable: Python callable to execute
        args: Positional arguments to pass
        kwargs: Keyword arguments to pass
    """

    callable: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobRequest:
    """Request to launch a job on the cluster.

    Args:
        name: Human-readable job name (must not contain spaces)
        entrypoint: Job entrypoint (callable with args/kwargs)
        resources: Resource requirements for the job
        environment: Environment configuration (dependencies, env vars)
        max_retries_failure: Maximum retries on failure
        max_retries_preemption: Maximum retries on preemption
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


@dataclass
class TaskStatus:
    """Status of an individual task in a job.

    Args:
        status: Current status of the task
        error_message: Error message if task failed
    """

    status: JobStatus
    error_message: str | None = None


@dataclass
class JobInfo:
    """Complete job state information.

    Args:
        job_id: Unique job identifier
        status: Current job status
        tasks: List of task statuses
        name: Job name
        error_message: Error message if job failed
    """

    job_id: JobId
    status: JobStatus
    tasks: list[TaskStatus]
    name: str
    error_message: str | None = None


# Generic Endpoint Registry (NOT actor-specific)


@dataclass
class Endpoint:
    """A registered endpoint in the cluster's registry.

    This is a generic service discovery primitive. The actor layer
    uses this to register actors, but the cluster doesn't know about actors.

    Args:
        endpoint_id: Unique endpoint identifier
        name: Endpoint name for discovery
        address: Network address (host:port)
        job_id: Job that registered this endpoint
        namespace: Namespace for scoping
        metadata: Optional key-value metadata
    """

    endpoint_id: EndpointId
    name: str
    address: str
    job_id: JobId
    namespace: Namespace
    metadata: dict[str, str] = field(default_factory=dict)


# VM Information


@dataclass
class VMInfo:
    """Information about a VM managed by the cluster.

    Args:
        vm_id: Unique VM identifier
        address: Network address (IP or hostname)
        status: VM status ("starting", "ready", "stopping", "stopped")
        resources: Resource configuration for this VM
    """

    vm_id: VMId
    address: str
    status: str
    resources: ResourceConfig
