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

"""Core types for Fray distributed execution framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Lifetime(str, Enum):
    """Actor lifetime options."""

    EPHEMERAL = "ephemeral"
    """Actor lifetime tied to the creating job. Dies when job completes."""

    DETACHED = "detached"
    """Actor persists beyond the creating job. Lives until explicitly killed or cluster shutdown."""


# Type aliases for backend-specific implementations
# These will be specialized by each backend
ObjectRef = Any
ActorRef = Any

# Entry point for cluster job submission
# Jobs are submitted as shell commands (not Python callables) to support:
# - Script-based execution: "python train.py --config foo.yaml"
# - Docker containers: with custom entry commands
# - Language-agnostic workloads
#
# For in-cluster task execution, use JobContext.create_task() which accepts Python functions
EntryPoint = str


@dataclass
class Resource:
    """
    Resource requirement specification.

    Represents a named resource (CPU, memory, GPU, TPU, etc.) with a quantity.
    Backend implementations translate these to their native resource specifications.

    Examples:
        Resource("CPU", 4.0)  # 4 CPUs
        Resource("memory", 8 * 1024 * 1024 * 1024)  # 8GB RAM
        Resource("TPU", 8)  # 8 TPU chips
    """

    name: str
    quantity: float

    def __repr__(self):
        return f"Resource(name={self.name!r}, quantity={self.quantity})"


@dataclass
class RuntimeEnv:
    """
    Execution environment specification for a job.

    Describes the environment in which tasks should run, including package
    dependencies, resource constraints, and environment variables.

    Attributes:
        package_requirements: List of pip package specifications (e.g., ["numpy>=1.20", "pandas"])
        minimum_resources: Minimum resources required for the job to start
        maximum_resources: Maximum resources the job can use (for autoscaling)
        env: Environment variables to set in the execution environment

    Raises:
        ValueError: If validation fails (e.g., minimum > maximum resources)
        TypeError: If field types are invalid
    """

    package_requirements: list[str] = field(default_factory=list)
    minimum_resources: list[Resource] = field(default_factory=list)
    maximum_resources: list[Resource] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate RuntimeEnv configuration."""
        # Validate package requirements
        if self.package_requirements:
            for pkg in self.package_requirements:
                if not isinstance(pkg, str):
                    raise TypeError(f"package_requirements must be strings, got {type(pkg)}: {pkg}")
                if not pkg.strip():
                    raise ValueError("package_requirements cannot contain empty strings")

        # Validate env vars
        if self.env:
            for key, value in self.env.items():
                if not isinstance(key, str):
                    raise TypeError(f"env keys must be strings, got {type(key)}: {key}")
                if not isinstance(value, str):
                    raise TypeError(f"env values must be strings, got {type(value)}: {value}")

        # Validate resource constraints
        if self.minimum_resources and self.maximum_resources:
            # Build resource maps for comparison
            min_map = {r.name: r.quantity for r in self.minimum_resources}
            max_map = {r.name: r.quantity for r in self.maximum_resources}

            # Check that minimum <= maximum for all resources
            for name, min_qty in min_map.items():
                if name in max_map:
                    max_qty = max_map[name]
                    if min_qty > max_qty:
                        raise ValueError(
                            f"minimum_resources[{name}]={min_qty} exceeds " f"maximum_resources[{name}]={max_qty}"
                        )

        # Validate Resource objects themselves
        for resource in self.minimum_resources + self.maximum_resources:
            if resource.quantity < 0:
                raise ValueError(f"Resource quantities cannot be negative: " f"{resource.name}={resource.quantity}")

    def __repr__(self):
        parts = []
        if self.package_requirements:
            parts.append(f"packages={len(self.package_requirements)}")
        if self.minimum_resources:
            parts.append(f"min_resources={self.minimum_resources}")
        if self.maximum_resources:
            parts.append(f"max_resources={self.maximum_resources}")
        if self.env:
            parts.append(f"env_vars={len(self.env)}")
        return f"RuntimeEnv({', '.join(parts)})"


@dataclass
class ActorOptions:
    """
    Options for actor creation.

    Attributes:
        name: Named actor identifier for retrieval/reuse across calls
        get_if_exists: If True and name exists, return existing actor instead of creating new one
        resources: Resource requirements (e.g., {"CPU": 0, "GPU": 1, "head_node": 0.0001})
        lifetime: Actor lifetime (EPHEMERAL: dies with job, DETACHED: persists beyond job)

    Examples:
        Create a named actor that can be retrieved later:
            options = ActorOptions(name="my_actor")

        Get existing actor or create if it doesn't exist:
            options = ActorOptions(name="my_actor", get_if_exists=True)

        Specify resource requirements:
            options = ActorOptions(resources={"CPU": 0, "GPU": 1})

        Schedule on head node (non-preemptible):
            options = ActorOptions(resources={"CPU": 0, "head_node": 0.0001})

        Create detached actor that persists beyond job:
            options = ActorOptions(name="status_actor", lifetime=Lifetime.DETACHED)
    """

    name: str | None = None
    get_if_exists: bool = False
    resources: dict[str, float] | None = None
    lifetime: Lifetime = Lifetime.EPHEMERAL


@dataclass
class TaskOptions:
    """
    Options for task creation.

    Allows specifying resource requirements and runtime environment per task.
    Currently mirrors job-level RuntimeEnv capabilities; in the future, may be
    restricted to subsets of job-level resources.

    Attributes:
        resources: Resource requirements (e.g., {"CPU": 4, "GPU": 1, "memory": 8*1024**3})
        runtime_env: Execution environment (packages, env vars) for this specific task
        name: Task name for debugging/observability (backend-specific support)
        max_calls: Maximum times this task worker can be reused (None = unlimited).
                   Set to 1 to force process restart after each execution (useful for TPU cleanup).

    Examples:
        Specify memory limit:
            options = TaskOptions(resources={"memory": 8 * 1024**3})

        Per-task packages:
            options = TaskOptions(
                runtime_env=RuntimeEnv(package_requirements=["torch>=2.0"])
            )

        Force process restart (TPU cleanup):
            options = TaskOptions(resources={"TPU": 4}, max_calls=1)

        Combined:
            options = TaskOptions(
                resources={"CPU": 4, "memory": 16*1024**3},
                runtime_env=RuntimeEnv(
                    package_requirements=["transformers"],
                    env={"CUDA_VISIBLE_DEVICES": "0"}
                )
            )
    """

    resources: dict[str, float] | None = None
    runtime_env: RuntimeEnv | None = None
    name: str | None = None
    max_calls: int | None = None


@dataclass
class JobInfo:
    """
    Information about a cluster job.

    Attributes:
        id: Unique job identifier
        status: Job status (PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED)
        submission_time: When the job was submitted (Unix timestamp)
        start_time: When the job started execution (Unix timestamp, None if not started)
        end_time: When the job finished (Unix timestamp, None if not finished)
    """

    id: str
    status: str
    submission_time: float
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class TpuRunConfig:
    """
    Configuration for running a job on TPU slices.

    TPU jobs run on Google Cloud TPU pods with automatic retry on preemption
    and node failures. The function is executed once per TPU VM/host across
    all slices.

    Attributes:
        tpu_type: TPU configuration string (e.g., "v4-32", "v5p-128", "v6e-256")
        num_slices: Number of TPU slices to allocate (default: 1)
        max_retries_preemption: Maximum retries for TPU preemption events.
            TPU VMs can be preempted by Google Cloud at any time. Default 10000
            (effectively unlimited for spot instances).
        max_retries_failure: Maximum retries for application-level failures.
            Default 10 (prevents infinite loops on buggy code).

    Examples:
        Single slice for development:
            config = TpuRunConfig(tpu_type="v4-32")

        Multiple slices for production:
            config = TpuRunConfig(tpu_type="v5p-128", num_slices=4)

        Custom retry limits:
            config = TpuRunConfig(
                tpu_type="v4-32",
                num_slices=2,
                max_retries_preemption=100,
                max_retries_failure=5
            )
    """

    tpu_type: str
    num_slices: int = 1
    max_retries_preemption: int = 10000
    max_retries_failure: int = 10
