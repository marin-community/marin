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

"""Type definitions for the Fray cluster abstraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NewType

JobId = NewType("JobId", str)
JobStatus = Literal["pending", "running", "succeeded", "failed", "stopped"]

# Device type literals
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
    "v5e-1",
    "v5e-4",
    "v5e-8",
    "v5e-16",
    "v5e-32",
    "v5e-64",
    "v5e-128",
    "v5e-256",
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
]

GpuType = Literal[
    "A100",
    "H100",
    "V100",
    "T4",
    "L4",
    "A10G",
    "A10",
]


@dataclass(frozen=True)
class CpuConfig:
    """CPU-only device configuration."""

    pass


@dataclass(frozen=True)
class GpuConfig:
    """GPU device configuration."""

    type: GpuType
    count: int = 1


@dataclass(frozen=True)
class TpuConfig:
    """TPU device configuration.

    Args:
        type: TPU accelerator type (e.g., "v5e-16", "v4-8")
        count: Number of TPU chips to request
        topology: Optional topology specification (e.g., "2x2x1")
    """

    type: TpuType
    count: int
    topology: str | None = None


DeviceConfig = CpuConfig | GpuConfig | TpuConfig


@dataclass
class ResourceConfig:
    """Resource requirements for a job.

    Args:
        cpu: Number of CPU cores
        ram: RAM requirement (e.g., "8g", "16g")
        disk: Disk space requirement (e.g., "10g", "100g")
        device: Device configuration (CPU, GPU, or TPU)
        count: Number of workers with these resources
        regions: Preferred cloud regions for job placement
    """

    cpu: int = 1
    ram: str = "4g"
    disk: str = "10g"
    device: DeviceConfig = field(default_factory=CpuConfig)
    count: int = 1
    regions: Sequence[str] | None = None


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
        extra_dependency_groups: Extra dependency groups for uv (e.g., ["tpu", "eval"])
    """

    workspace: str | None = None
    docker_image: str | None = None
    pip_packages: Sequence[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    extra_dependency_groups: Sequence[str] = field(default_factory=list)

    def __post_init__(self):
        if self.workspace and self.docker_image:
            raise ValueError("Cannot specify both workspace and docker_image")
        if not self.workspace and not self.docker_image:
            raise ValueError("Must specify either workspace or docker_image")


@dataclass
class JobRequest:
    """Complete job specification for cluster submission.

    Args:
        name: Human-readable job name
        entrypoint: Python module or script path to execute
        entrypoint_args: Arguments to pass to the entrypoint
        resources: Resource requirements for the job
        environment: Environment configuration (dependencies, env vars)
    """

    name: str
    entrypoint: str
    entrypoint_args: Sequence[str] = field(default_factory=list)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    environment: EnvironmentConfig | None = None


@dataclass
class JobInfo:
    """Information about a job.

    Args:
        job_id: Unique job identifier
        status: Current job status
        name: Job name
        start_time: Unix timestamp when job started (None if not started)
        end_time: Unix timestamp when job ended (None if not ended)
        error_message: Error message if job failed (None otherwise)
    """

    job_id: JobId
    status: JobStatus
    name: str
    start_time: float | None = None
    end_time: float | None = None
    error_message: str | None = None


def create_environment(
    workspace: str | None = None,
    docker_image: str | None = None,
    pip_packages: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
    extra_dependency_groups: Sequence[str] | None = None,
) -> EnvironmentConfig:
    """Create an EnvironmentConfig with sensible defaults.

    Sets default environment variables commonly needed for ML workloads:
    - HF_DATASETS_TRUST_REMOTE_CODE: "1" (allows custom dataset code)
    - TOKENIZERS_PARALLELISM: "false" (avoids tokenizer deadlocks)

    Args:
        workspace: Path to workspace root (default: current directory)
        docker_image: Docker image (mutually exclusive with workspace)
        pip_packages: Additional pip packages to install
        env_vars: Custom environment variables (merged with defaults)
        extra_dependency_groups: Extra dependency groups for uv

    Returns:
        EnvironmentConfig with defaults applied
    """
    import os

    # Use current directory as workspace if neither workspace nor docker_image specified
    if workspace is None and docker_image is None:
        workspace = os.getcwd()

    # Default environment variables for ML workloads
    default_env_vars = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    # Merge user env vars with defaults (user values take precedence)
    merged_env_vars = {**default_env_vars, **(env_vars or {})}

    return EnvironmentConfig(
        workspace=workspace,
        docker_image=docker_image,
        pip_packages=list(pip_packages or []),
        env_vars=merged_env_vars,
        extra_dependency_groups=list(extra_dependency_groups or []),
    )
