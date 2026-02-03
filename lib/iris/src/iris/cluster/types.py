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

"""Core types for the iris cluster layer.

This module provides Python types for the Iris cluster API:
- ResourceSpec: Dataclass for specifying job resources with human-readable values
- EnvironmentSpec: Dataclass for specifying job environment configuration
- Entrypoint: Callable wrapper for job execution
- Namespace: Type-safe namespace identifier

Wire-format types (ResourceSpecProto, JobStatus, etc.) are defined in cluster.proto.
"""

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, NewType

import humanfriendly

from iris.rpc import cluster_pb2

JobId = NewType("JobId", str)


class DeviceType(Enum):
    """Device type for demand routing."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


TaskId = NewType("TaskId", str)
WorkerId = NewType("WorkerId", str)
EndpointId = NewType("EndpointId", str)


@dataclass(frozen=True)
class VmWorkerStatus:
    """Worker status keyed by VM address for autoscaler.

    The VM address is the worker's identity. This enables the autoscaler
    to look up worker status directly by VM address without needing
    to correlate separate worker_id to VM.
    """

    vm_address: str
    running_task_ids: frozenset[str]

    @property
    def is_idle(self) -> bool:
        return len(self.running_task_ids) == 0


# Map of VM address -> worker status, used by autoscaler for idle tracking
VmWorkerStatusMap = dict[str, VmWorkerStatus]


@dataclass(frozen=True)
class AttributeValue:
    """Typed attribute value for worker attributes and constraint matching.

    Used for coscheduling and constraint-based worker filtering.
    Values can be strings, integers, or floats.
    """

    value: str | int | float

    def to_proto(self) -> cluster_pb2.AttributeValue:
        """Convert to protobuf representation."""
        proto = cluster_pb2.AttributeValue()
        if isinstance(self.value, str):
            proto.string_value = self.value
        elif isinstance(self.value, int):
            proto.int_value = self.value
        elif isinstance(self.value, float):
            proto.float_value = self.value
        return proto

    @staticmethod
    def from_proto(proto: cluster_pb2.AttributeValue) -> "AttributeValue":
        """Convert from protobuf representation."""
        if proto.HasField("string_value"):
            return AttributeValue(proto.string_value)
        elif proto.HasField("int_value"):
            return AttributeValue(proto.int_value)
        elif proto.HasField("float_value"):
            return AttributeValue(proto.float_value)
        # Default to empty string if no value set
        return AttributeValue("")


class ConstraintOp(IntEnum):
    """Constraint operators for worker attribute matching.

    Used to define constraints that filter which workers can run a job.
    Each operator compares a worker attribute against a constraint value.

    Example:
        >>> # Match workers where region equals "us-central1"
        >>> Constraint(key="region", op=ConstraintOp.EQ, value="us-central1")
        >>> # Match workers with memory > 32GB
        >>> Constraint(key="memory_gb", op=ConstraintOp.GT, value=32)
        >>> # Match workers that have the "gpu" attribute set
        >>> Constraint(key="gpu", op=ConstraintOp.EXISTS)
    """

    EQ = 0
    NE = 1
    EXISTS = 2
    NOT_EXISTS = 3
    GT = 4
    GE = 5
    LT = 6
    LE = 7

    def to_proto(self) -> cluster_pb2.ConstraintOp:
        """Convert to protobuf ConstraintOp enum value."""
        mapping = {
            ConstraintOp.EQ: cluster_pb2.CONSTRAINT_OP_EQ,
            ConstraintOp.NE: cluster_pb2.CONSTRAINT_OP_NE,
            ConstraintOp.EXISTS: cluster_pb2.CONSTRAINT_OP_EXISTS,
            ConstraintOp.NOT_EXISTS: cluster_pb2.CONSTRAINT_OP_NOT_EXISTS,
            ConstraintOp.GT: cluster_pb2.CONSTRAINT_OP_GT,
            ConstraintOp.GE: cluster_pb2.CONSTRAINT_OP_GE,
            ConstraintOp.LT: cluster_pb2.CONSTRAINT_OP_LT,
            ConstraintOp.LE: cluster_pb2.CONSTRAINT_OP_LE,
        }
        return mapping[self]


@dataclass(frozen=True)
class Constraint:
    """Worker constraint for job scheduling.

    Constraints filter which workers are eligible to run a job based on
    worker attributes. Workers must satisfy all constraints to be considered.

    Example:
        >>> # Require a specific TPU pod
        >>> Constraint(key="tpu-name", op=ConstraintOp.EQ, value="my-tpu-pod")
        >>> # Require workers in a specific zone
        >>> Constraint(key="zone", op=ConstraintOp.EQ, value="us-central1-a")
        >>> # Require workers with at least 64GB memory
        >>> Constraint(key="memory_gb", op=ConstraintOp.GE, value=64)
        >>> # Require workers that have a GPU
        >>> Constraint(key="gpu", op=ConstraintOp.EXISTS)
    """

    key: str
    op: ConstraintOp
    value: str | int | float | None = None

    def to_proto(self) -> cluster_pb2.Constraint:
        """Convert to protobuf representation."""
        proto = cluster_pb2.Constraint(key=self.key, op=self.op.to_proto())
        if self.value is not None:
            proto.value.CopyFrom(AttributeValue(self.value).to_proto())
        return proto


@dataclass(frozen=True)
class CoschedulingConfig:
    """Configuration for coscheduling job tasks together.

    Coscheduling ensures that all tasks of a job are scheduled on workers
    that share a common attribute value. This is essential for multi-host
    TPU jobs where all workers must belong to the same TPU pod.

    Example:
        >>> # Schedule all tasks on workers from the same TPU pod
        >>> CoschedulingConfig(group_by="tpu-name")
    """

    group_by: str

    def to_proto(self) -> cluster_pb2.CoschedulingConfig:
        """Convert to protobuf representation."""
        return cluster_pb2.CoschedulingConfig(group_by=self.group_by)


def tpu_device(variant: str, count: int | None = None) -> cluster_pb2.DeviceConfig:
    """Create a DeviceConfig for a TPU device.

    Args:
        variant: TPU variant string (e.g., "v5litepod-16", "v4-8", "v6e-256").
        count: Number of TPU chips. If None, inferred from topology.

    Returns:
        DeviceConfig with the tpu field set to the specified variant and chip count.

    Example:
        >>> config = tpu_device("v5litepod-16")
        >>> config.tpu.variant
        'v5litepod-16'
        >>> config.tpu.count
        4
    """
    chip_count = count
    if chip_count is None:
        try:
            topo = get_tpu_topology(variant)
            chip_count = topo.chips_per_vm
        except ValueError:
            chip_count = 0
    return cluster_pb2.DeviceConfig(
        tpu=cluster_pb2.TpuDevice(
            variant=variant,
            count=chip_count,
        )
    )


def parse_memory_string(memory_str: str) -> int:
    """Parse human-readable memory string to bytes.

    Supports various formats:
    - "8G", "8GB", "8 GB", "8 gigabytes"
    - "512M", "512MB", "512 megabytes"
    - "1024K", "1024KB", "1024 kilobytes"
    - Plain numbers treated as bytes

    Args:
        memory_str: Memory string (e.g., "8g", "16gb", "512m")

    Returns:
        Memory in bytes

    Raises:
        ValueError: If format is invalid
    """
    if not memory_str:
        return 0

    memory_str = memory_str.strip()
    if not memory_str or memory_str == "0":
        return 0

    try:
        return humanfriendly.parse_size(memory_str, binary=True)
    except humanfriendly.InvalidSize as e:
        raise ValueError(str(e)) from e


@dataclass
class ResourceSpec:
    """Resource specification for jobs.

    Accepts human-readable memory/disk values (e.g., "8g", "512m").
    """

    cpu: int = 0
    memory: str | int = 0  # "8g" or bytes
    disk: str | int = 0
    device: cluster_pb2.DeviceConfig | None = None
    regions: Sequence[str] | None = None

    def to_proto(self) -> cluster_pb2.ResourceSpecProto:
        """Convert to wire format."""
        memory_bytes = self.memory if isinstance(self.memory, int) else parse_memory_string(self.memory)
        disk_bytes = self.disk if isinstance(self.disk, int) else parse_memory_string(self.disk)
        spec = cluster_pb2.ResourceSpecProto(
            cpu=self.cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            regions=list(self.regions or []),
        )
        if self.device is not None:
            spec.device.CopyFrom(self.device)
        return spec


DOCKERFILE_TEMPLATE = """FROM {base_image}

RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

COPY . .

RUN --mount=type=cache,id=iris-uv-global,sharing=shared,target=/opt/uv-cache \\
    --mount=type=cache,id=iris-cargo,sharing=shared,target=/root/.cargo/registry \\
    --mount=type=cache,id=iris-cargo-git,sharing=shared,target=/root/.cargo/git \\
    uv sync {extras_flags}

ENV PATH="/app/.venv/bin:$PATH"

RUN uv pip install cloudpickle
{pip_install_step}"""


def generate_dockerfile(
    python_version: str,
    extras: list[str] | None = None,
    pip_packages: list[str] | None = None,
) -> str:
    """Generate a Dockerfile for an Iris job container.

    Uses the full Python image when pip_packages are specified (native wheels
    often depend on system libraries stripped from slim images).

    Extras can be specified in two formats:
    - "extra_name" -> "--extra extra_name"
    - "package:extra_name" -> "--package package --extra extra_name"
    """
    base_image = f"python:{python_version}" if pip_packages else f"python:{python_version}-slim"

    # Parse extras: support both "extra" and "package:extra" syntax
    extras_parts = []
    for e in extras or []:
        if ":" in e:
            package, extra = e.split(":", 1)
            extras_parts.append(f"--package {package} --extra {extra}")
        else:
            extras_parts.append(f"--extra {e}")
    extras_flags = " ".join(extras_parts)

    pip_install_step = ""
    if pip_packages:
        packages_str = " ".join(f'"{pkg}"' for pkg in pip_packages)
        pip_install_step = f"\nRUN uv pip install {packages_str}\n"

    return DOCKERFILE_TEMPLATE.format(
        base_image=base_image,
        extras_flags=extras_flags,
        pip_install_step=pip_install_step,
    )


@dataclass
class EnvironmentSpec:
    """Environment specification for jobs.

    Default environment variables (automatically set if not overridden):
    - HF_DATASETS_TRUST_REMOTE_CODE: "1" (allows custom dataset code)
    - TOKENIZERS_PARALLELISM: "false" (avoids tokenizer deadlocks)
    - HF_TOKEN: from os.environ (if set)
    - WANDB_API_KEY: from os.environ (if set)

    Note: To specify workspace for bundle creation, use IrisClient.remote(workspace=...).
    """

    pip_packages: Sequence[str] | None = None
    env_vars: dict[str, str] | None = None
    extras: Sequence[str] | None = None
    dockerfile: str | None = None

    def to_proto(self) -> cluster_pb2.EnvironmentConfig:
        """Convert to wire format with sensible defaults applied.

        Generates the dockerfile on the client side so the worker can use it
        directly without regenerating (which would lose extras like package:extra).
        """
        default_env_vars = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }

        merged_env_vars = {k: v for k, v in {**default_env_vars, **(self.env_vars or {})}.items() if v is not None}

        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        if self.dockerfile is not None:
            dockerfile = self.dockerfile
        else:
            dockerfile = generate_dockerfile(
                python_version=py_version,
                extras=list(self.extras or []),
                pip_packages=list(self.pip_packages or []),
            )

        return cluster_pb2.EnvironmentConfig(
            pip_packages=list(self.pip_packages or []),
            env_vars=merged_env_vars,
            extras=list(self.extras or []),
            python_version=py_version,
            dockerfile=dockerfile,
        )


class Namespace(str):
    """Namespace for actor isolation.

    Namespaces provide isolation between different jobs/environments.
    Actors in one namespace cannot discover actors in another namespace.

    The namespace is derived from the root job ID: all jobs in a hierarchy
    share the same namespace. This ensures automatic isolation without
    explicit configuration.
    """

    def __new__(cls, value: str) -> "Namespace":
        if not value:
            raise ValueError("Namespace cannot be empty")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Namespace({super().__repr__()})"

    @classmethod
    def from_job_id(cls, job_id: str) -> "Namespace":
        """Derive namespace from hierarchical job ID.

        The namespace is the first component of the job ID hierarchy.
        For example:
            "abc123" -> Namespace("abc123")
            "abc123/worker-0" -> Namespace("abc123")
            "abc123/worker-0/sub-task" -> Namespace("abc123")

        Args:
            job_id: Hierarchical job ID

        Returns:
            Namespace derived from root job ID

        Raises:
            ValueError: If job_id is empty
        """
        if not job_id:
            raise ValueError("Job ID cannot be empty")
        return cls(job_id.split("/")[0])


PREEMPTIBLE_ATTRIBUTE_KEY = "preemptible"


def preemptible_constraint(preemptible: bool = True) -> Constraint:
    """Constraint requiring workers to be preemptible (or not)."""
    return Constraint(key=PREEMPTIBLE_ATTRIBUTE_KEY, op=ConstraintOp.EQ, value=str(preemptible).lower())


def is_job_finished(state: int) -> bool:
    return state in (
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    )


def is_task_finished(state: int) -> bool:
    """Check if a task state is terminal.

    This is a simple check for whether the state is a terminal state.
    For ControllerTask, use task.is_finished() which also considers retry budgets.
    """
    # Avoid circular import - define inline since this is a stable set
    terminal_states = frozenset(
        {
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
            cluster_pb2.TASK_STATE_UNSCHEDULABLE,
        }
    )
    return state in terminal_states


JobState = cluster_pb2.JobState
TaskState = cluster_pb2.TaskState


@dataclass(frozen=True)
class TpuTopologyInfo:
    """TPU topology configuration."""

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


class Entrypoint:
    """Job entrypoint specification.

    Supports two execution modes:
    1. Callable: A Python function with args/kwargs (cloudpickled)
    2. Command: A command-line invocation (e.g., ["python", "train.py", "--epochs", "10"])

    Callable entrypoints are stored as cloudpickle bytes. The bytes are the
    single source of truth — they pass from client to worker to task container
    without deserialization, avoiding Python version mismatches between the
    client and worker processes.

    Examples:
        # Callable entrypoint
        entrypoint = Entrypoint.from_callable(my_func, arg1, arg2, key=val)

        # Command entrypoint
        entrypoint = Entrypoint.from_command("python", "train.py", "--epochs", "10")
    """

    def __init__(
        self,
        *,
        callable_bytes: bytes | None = None,
        command: list[str] | None = None,
    ):
        has_callable = callable_bytes is not None
        has_command = command is not None
        if has_callable == has_command:
            raise ValueError("Exactly one of 'callable_bytes' or 'command' must be set")
        self._callable_bytes = callable_bytes
        self.command = command

    @property
    def callable_bytes(self) -> bytes | None:
        return self._callable_bytes

    @property
    def is_callable(self) -> bool:
        return self._callable_bytes is not None

    @property
    def is_command(self) -> bool:
        return self.command is not None

    def resolve(self) -> tuple[Callable[..., Any], tuple, dict[str, Any]]:
        """Deserialize the callable, args, kwargs from pickle bytes.

        Only call this when you need to actually invoke the function locally
        (e.g. local_client). Avoid on the worker — use callable_bytes directly
        to pass through to the task container without version-sensitive unpickling.
        """
        if self._callable_bytes is None:
            raise ValueError("Not a callable entrypoint")
        import cloudpickle

        return cloudpickle.loads(self._callable_bytes)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Entrypoint":
        import cloudpickle

        return cls(callable_bytes=cloudpickle.dumps((fn, args, kwargs)))

    @classmethod
    def from_command(cls, *argv: str) -> "Entrypoint":
        """Create a command-line entrypoint.

        Args:
            *argv: Command and arguments (e.g., "python", "train.py", "--epochs", "10")

        Returns:
            Entrypoint configured for command execution
        """
        if not argv:
            raise ValueError("Command must have at least one argument")
        return cls(command=list(argv))

    def to_proto(self) -> cluster_pb2.Entrypoint:
        """Convert to protobuf representation."""
        proto = cluster_pb2.Entrypoint()
        if self._callable_bytes is not None:
            proto.callable = self._callable_bytes
        elif self.command is not None:
            proto.command.argv[:] = self.command
        return proto

    @classmethod
    def from_proto(cls, proto: cluster_pb2.Entrypoint) -> "Entrypoint":
        """Create from protobuf representation.

        For callable entrypoints, stores the raw pickle bytes without
        deserializing. This avoids Python version mismatches when the
        worker runs a different Python than the client.
        """
        kind = proto.WhichOneof("kind")
        if kind == "callable":
            return cls(callable_bytes=proto.callable)
        elif kind == "command":
            return cls(command=list(proto.command.argv))
        else:
            raise ValueError(f"Unknown entrypoint kind: {kind}")
