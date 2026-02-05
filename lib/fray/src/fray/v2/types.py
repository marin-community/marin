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

"""Type definitions for fray v2.

Standalone dataclasses with no v1 backend dependencies. Copied from
fray.cluster.base with one structural change: `replicas` moves from
ResourceConfig to JobRequest (it's a job-level gang-scheduling concern,
not a per-task resource requirement).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Self

# ---------------------------------------------------------------------------
# TPU topology
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class TpuTopologyInfo:
    name: str
    chip_count: int
    host_count: int
    vm_count: int
    chips_per_vm: int


TPU_TOPOLOGIES: list[TpuTopologyInfo] = [
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
    TpuTopologyInfo("v5litepod-1", 1, 1, 1, 1),
    TpuTopologyInfo("v5litepod-2", 2, 1, 1, 2),
    TpuTopologyInfo("v5litepod-4", 4, 1, 1, 4),
    TpuTopologyInfo("v5litepod-8", 8, 1, 1, 8),
    TpuTopologyInfo("v5litepod-16", 16, 2, 4, 4),
    TpuTopologyInfo("v5litepod-32", 32, 4, 8, 4),
    TpuTopologyInfo("v5litepod-64", 64, 8, 16, 4),
    TpuTopologyInfo("v5litepod-128", 128, 16, 32, 4),
    TpuTopologyInfo("v5litepod-256", 256, 32, 64, 4),
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
        return 0

    def device_flops(self, dtype: str = "bf16") -> float:
        raise NotImplementedError("CPU FLOPS not available")


@dataclass(frozen=True)
class GpuConfig:
    """GPU device configuration."""

    variant: GpuType
    kind: DeviceKind = "gpu"
    count: int = 1

    def chip_count(self) -> int:
        return self.count

    def device_flops(self, dtype: str = "bf16") -> float:
        from fray.v2.device_flops import device_flops

        flops = device_flops(self.variant, dtype)
        if flops is None:
            raise ValueError(f"Unknown device/dtype: {self.variant}/{dtype}")
        return flops

    def total_flops(self, dtype: str = "bf16") -> float:
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
        return get_tpu_topology(self.variant).chip_count

    def vm_count(self) -> int:
        return get_tpu_topology(self.variant).vm_count

    def device_flops(self, dtype: str = "bf16") -> float:
        from fray.v2.device_flops import device_flops

        flops = device_flops(self.variant, dtype)
        if flops is None:
            raise ValueError(f"Unknown device/dtype: {self.variant}/{dtype}")
        return flops

    def total_flops(self, dtype: str = "bf16") -> float:
        return self.device_flops(dtype) * self.chip_count()


DeviceConfig = CpuConfig | GpuConfig | TpuConfig


@dataclass
class ResourceConfig:
    """Resource requirements for a single task/replica.

    `replicas` specifies gang-scheduled replica count (e.g. TPU slices for
    multislice training). It is also on JobRequest; when both are set,
    JobRequest.replicas takes precedence. This field exists here so that
    convenience builders like `with_tpu(..., slice_count=4)` can carry the
    replica count alongside the resource spec.

    `max_concurrency` controls how many method calls can run in parallel on
    the actor (Ray's max_concurrency). Use >1 for actors that need to handle
    concurrent calls, e.g. coordinators that block while workers call back.
    """

    cpu: int = 1
    ram: str = "128m"
    disk: str = "1g"
    device: DeviceConfig = field(default_factory=CpuConfig)
    preemptible: bool = True
    regions: Sequence[str] | None = None
    replicas: int = 1
    max_concurrency: int = 1

    def chip_count(self) -> int:
        return self.device.chip_count()

    def device_flops(self, dtype: str = "bf16") -> float:
        return self.device.device_flops(dtype)

    def total_flops(self, dtype: str = "bf16") -> float:
        if isinstance(self.device, CpuConfig):
            return 100e9
        return self.device_flops(dtype) * self.chip_count()

    @staticmethod
    def with_tpu(tpu_type: str, *, slice_count: int | None = None, **kwargs: Any) -> ResourceConfig:
        topo = get_tpu_topology(tpu_type)
        if slice_count is None:
            slice_count = topo.vm_count
        elif slice_count != topo.vm_count:
            raise ValueError(
                f"slice_count must match TPU slice size ({topo.vm_count}) for {tpu_type}, got {slice_count}"
            )

        device = TpuConfig(variant=tpu_type)
        kwargs = dict(kwargs)
        kwargs.setdefault("cpu", 32)
        kwargs.setdefault("ram", "128g")
        kwargs.setdefault("disk", "50g")
        return ResourceConfig(device=device, replicas=slice_count, **kwargs)

    @staticmethod
    def with_gpu(gpu_type: str = "auto", count: int = 1, **kwargs: Any) -> ResourceConfig:
        device = GpuConfig(variant=gpu_type, count=count)
        return ResourceConfig(device=device, **kwargs)

    @staticmethod
    def with_cpu(**kwargs: Any) -> ResourceConfig:
        return ResourceConfig(device=CpuConfig(), **kwargs)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentConfig:
    """Job environment configuration.

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


def create_environment(
    workspace: str | None = None,
    docker_image: str | None = None,
    pip_packages: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
    extras: Sequence[str] | None = None,
) -> EnvironmentConfig:
    """Create an EnvironmentConfig with sensible defaults.

    Sets HF_DATASETS_TRUST_REMOTE_CODE, TOKENIZERS_PARALLELISM, HF_TOKEN,
    and WANDB_API_KEY from the current environment by default.
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

    merged_env_vars = {k: v for k, v in {**default_env_vars, **(env_vars or {})}.items() if v is not None}

    return EnvironmentConfig(
        workspace=workspace,
        docker_image=docker_image,
        pip_packages=list(pip_packages or []),
        env_vars=merged_env_vars,
        extras=list(extras or []),
    )


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


@dataclass
class JobRequest:
    """Complete job specification for submission.

    Args:
        name: Human-readable job name (no spaces)
        entrypoint: Job entrypoint (command-line or callable)
        resources: Resource requirements per replica
        environment: Environment configuration (dependencies, env vars)
        replicas: Gang-scheduled replicas (e.g. TPU slices for multislice training)
        max_retries_failure: Max retries on failure
        max_retries_preemption: Max retries on preemption
    """

    name: str
    entrypoint: Entrypoint
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    environment: EnvironmentConfig | None = None
    replicas: int | None = None
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    def __post_init__(self):
        if " " in self.name:
            raise ValueError("Job name must not contain spaces")
        if self.replicas is None:
            # Pick up replicas from ResourceConfig (set by e.g. with_tpu slice_count)
            self.replicas = self.resources.replicas


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"

    @staticmethod
    def finished(status: JobStatus) -> bool:
        return status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED)
