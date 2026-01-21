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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NewType

import humanfriendly

from iris.rpc import cluster_pb2

JobId = NewType("JobId", str)
TaskId = NewType("TaskId", str)
WorkerId = NewType("WorkerId", str)
EndpointId = NewType("EndpointId", str)


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
    replicas: int = 0
    preemptible: bool = False
    regions: Sequence[str] | None = None

    def to_proto(self) -> cluster_pb2.ResourceSpecProto:
        """Convert to wire format."""
        memory_bytes = self.memory if isinstance(self.memory, int) else parse_memory_string(self.memory)
        disk_bytes = self.disk if isinstance(self.disk, int) else parse_memory_string(self.disk)
        spec = cluster_pb2.ResourceSpecProto(
            cpu=self.cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            replicas=self.replicas,
            preemptible=self.preemptible,
            regions=list(self.regions or []),
        )
        if self.device is not None:
            spec.device.CopyFrom(self.device)
        return spec


@dataclass
class EnvironmentSpec:
    """Environment specification for jobs.

    Default environment variables (automatically set if not overridden):
    - HF_DATASETS_TRUST_REMOTE_CODE: "1" (allows custom dataset code)
    - TOKENIZERS_PARALLELISM: "false" (avoids tokenizer deadlocks)
    - HF_TOKEN: from os.environ (if set)
    - WANDB_API_KEY: from os.environ (if set)
    """

    workspace: str | None = None
    pip_packages: Sequence[str] | None = None
    env_vars: dict[str, str] | None = None
    extras: Sequence[str] | None = None

    def to_proto(self) -> cluster_pb2.EnvironmentConfig:
        """Convert to wire format with sensible defaults applied."""
        workspace = self.workspace if self.workspace is not None else os.getcwd()

        default_env_vars = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }

        merged_env_vars = {k: v for k, v in {**default_env_vars, **(self.env_vars or {})}.items() if v is not None}

        return cluster_pb2.EnvironmentConfig(
            workspace=workspace,
            pip_packages=list(self.pip_packages or []),
            env_vars=merged_env_vars,
            extras=list(self.extras or []),
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


@dataclass
class Entrypoint:
    """Job entrypoint specification.

    A callable with args/kwargs that will be executed by the worker.
    The callable must be picklable (via cloudpickle).

    Example:
        entrypoint = Entrypoint.from_callable(my_func, arg1, arg2, key=val)
    """

    callable: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Entrypoint":
        return cls(callable=fn, args=args, kwargs=kwargs)
