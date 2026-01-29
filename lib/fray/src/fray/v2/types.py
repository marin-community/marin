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

"""Core types for Fray v2 API.

This module provides fresh types aligned with Iris semantics.
These types do not depend on Fray v1.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NewType

import humanfriendly

# Type aliases
JobId = NewType("JobId", str)
Namespace = NewType("Namespace", str)


class JobStatus(StrEnum):
    """Job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"

    @staticmethod
    def is_finished(status: JobStatus) -> bool:
        """Check if a job status is terminal."""
        return status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.KILLED)


def parse_memory_string(memory_str: str | int) -> int:
    """Parse human-readable memory string to bytes.

    Args:
        memory_str: Memory string (e.g., "8g", "512m") or bytes as int

    Returns:
        Memory in bytes
    """
    if isinstance(memory_str, int):
        return memory_str

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
    Aligned with Iris ResourceSpec semantics.
    """

    cpu: int = 0
    memory: str | int = 0  # "8g" or bytes
    disk: str | int = 0
    replicas: int = 1
    preemptible: bool = False
    regions: Sequence[str] | None = None

    # Device configuration
    device_type: str | None = None  # "tpu" or "gpu"
    device_variant: str | None = None  # "v5litepod-16", "H100", etc.
    device_count: int = 0

    @classmethod
    def with_tpu(cls, variant: str, replicas: int = 1, **kw: Any) -> ResourceSpec:
        """Create TPU resource spec."""
        return cls(device_type="tpu", device_variant=variant, replicas=replicas, **kw)

    @classmethod
    def with_gpu(cls, variant: str = "auto", count: int = 1, **kw: Any) -> ResourceSpec:
        """Create GPU resource spec."""
        return cls(device_type="gpu", device_variant=variant, device_count=count, **kw)

    @classmethod
    def with_cpu(cls, cpu: int = 1, memory: str | int = "2g", **kw: Any) -> ResourceSpec:
        """Create CPU-only resource spec."""
        return cls(cpu=cpu, memory=memory, **kw)

    def memory_bytes(self) -> int:
        """Get memory in bytes."""
        return parse_memory_string(self.memory)

    def disk_bytes(self) -> int:
        """Get disk in bytes."""
        return parse_memory_string(self.disk)


@dataclass
class EnvironmentSpec:
    """Environment specification for jobs.

    Default environment variables (automatically set if not overridden):
    - HF_DATASETS_TRUST_REMOTE_CODE: "1"
    - TOKENIZERS_PARALLELISM: "false"
    - HF_TOKEN: from os.environ (if set)
    - WANDB_API_KEY: from os.environ (if set)
    """

    workspace: str | None = None
    pip_packages: Sequence[str] | None = None
    env_vars: dict[str, str] | None = None
    extras: Sequence[str] | None = None

    def effective_env_vars(self) -> dict[str, str]:
        """Get environment variables with defaults applied."""
        default_env_vars = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }
        merged = {k: v for k, v in {**default_env_vars, **(self.env_vars or {})}.items() if v is not None}
        return merged

    def effective_workspace(self) -> str:
        """Get workspace path, defaulting to cwd."""
        return self.workspace if self.workspace is not None else os.getcwd()


@dataclass
class Entrypoint:
    """Job entrypoint specification.

    A callable with args/kwargs that will be executed by the worker.
    The callable must be picklable (via cloudpickle).
    """

    callable: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Entrypoint:
        """Create entrypoint from a callable."""
        return cls(callable=fn, args=args, kwargs=kwargs)


def namespace_from_job_id(job_id: str) -> Namespace:
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
    return Namespace(job_id.split("/")[0])
