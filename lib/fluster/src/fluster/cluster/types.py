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

This module contains:
- Type aliases for IDs (JobId, WorkerId, etc.)
- Helper functions for working with proto types
- TPU topology information for scheduling
- Entrypoint dataclass for job execution

Wire-format types (ResourceSpec, JobStatus, etc.) are defined in cluster.proto.
"""

import os
from dataclasses import dataclass, field
from typing import Any, NewType
from collections.abc import Callable, Sequence

from fluster import cluster_pb2

# Type aliases for clarity
JobId = NewType("JobId", str)
Namespace = NewType("Namespace", str)
WorkerId = NewType("WorkerId", str)
EndpointId = NewType("EndpointId", str)


def is_job_finished(state: int) -> bool:
    """Check if job has reached terminal state."""
    return state in (
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
    )


# Re-export proto enum for convenience
JobState = cluster_pb2.JobState


# TPU Topology Information


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


# Job Entrypoint


@dataclass
class Entrypoint:
    """Job entrypoint specification.

    A callable with args/kwargs that will be executed by the worker.
    The callable must be picklable (via cloudpickle).

    Args:
        callable: Python callable to execute
        args: Positional arguments to pass
        kwargs: Keyword arguments to pass
    """

    callable: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


# Helper functions for creating proto messages


def create_environment(
    workspace: str | None = None,
    pip_packages: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
    extras: Sequence[str] | None = None,
) -> cluster_pb2.EnvironmentConfig:
    """Create an EnvironmentConfig proto with sensible defaults.

    Default environment variables:
    - HF_DATASETS_TRUST_REMOTE_CODE: "1" (allows custom dataset code)
    - TOKENIZERS_PARALLELISM: "false" (avoids tokenizer deadlocks)
    - HF_TOKEN: from os.environ (if set)
    - WANDB_API_KEY: from os.environ (if set)

    Args:
        workspace: Path to workspace root (default: current directory)
        pip_packages: Additional pip packages to install
        env_vars: Custom environment variables (merged with defaults)
        extras: Extra dependency groups for uv

    Returns:
        EnvironmentConfig proto message with defaults applied
    """
    if workspace is None:
        workspace = os.getcwd()

    default_env_vars = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
    }

    # Filter out None values and merge with user-provided vars
    merged_env_vars = {k: v for k, v in {**default_env_vars, **(env_vars or {})}.items() if v is not None}

    config = cluster_pb2.EnvironmentConfig(
        workspace=workspace,
        pip_packages=list(pip_packages or []),
        env_vars=merged_env_vars,
        extras=list(extras or []),
    )

    return config
