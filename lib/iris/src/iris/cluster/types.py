# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from pathlib import Path
from typing import Any, NewType

import cloudpickle
import humanfriendly

from iris.cluster.constraints import Constraint
from iris.rpc import cluster_pb2


@dataclass(frozen=True, slots=True)
class JobName:
    """Structured hierarchical job name.

    Canonical form: /user/root-job/child
    Tasks are job names with numeric suffix: /user/root-job/child/0

    The first path component identifies the submitting user. Job hierarchy starts
    at the second component:
        /alice/root-job
        /alice/root-job/child-1
        /alice/root-job/child-1/grandchild
        /alice/root-job/0
    """

    _parts: tuple[str, ...]

    def __post_init__(self):
        if len(self._parts) < 2:
            raise ValueError("JobName must use canonical '/<user>/<job>[...]' format")
        for part in self._parts:
            if "/" in part:
                raise ValueError(f"JobName component cannot contain '/': {part}")
            if not part or not part.strip():
                raise ValueError("JobName component cannot be empty or whitespace")

    @classmethod
    def from_string(cls, s: str) -> "JobName":
        """Parse a job name string like '/user/root/child/grandchild'.

        Examples:
            JobName.from_string("/alice/my-job") -> JobName(("alice", "my-job"))
            JobName.from_string("/alice/parent/child") -> JobName(("alice", "parent", "child"))
            JobName.from_string("/alice/job/0") -> JobName(("alice", "job", "0"))
        """
        if not s:
            raise ValueError("Job name must use canonical '/<user>/<job>[...]' format")
        if not s.startswith("/"):
            raise ValueError(f"Job name must use canonical '/<user>/<job>[...]' format: {s}")
        parts = tuple(s[1:].split("/"))
        if len(parts) < 2:
            raise ValueError(f"Job name must use canonical '/<user>/<job>[...]' format: {s}")
        if any(not part or not part.strip() for part in parts):
            raise ValueError(f"Job name contains empty or whitespace-only component: {s}")
        return cls(parts)

    @classmethod
    def root(cls, user: str, name: str) -> "JobName":
        """Create a root job name (no parent)."""
        return cls((user, name))

    def child(self, name: str) -> "JobName":
        """Create a child job name."""
        return JobName((*self._parts, name))

    def task(self, index: int) -> "JobName":
        """Create a task name for this job.

        Tasks are job names with a numeric suffix.

        Example:
            JobName.from_string("/alice/my-job").task(0) -> JobName(("alice", "my-job", "0"))
        """
        return JobName((*self._parts, str(index)))

    @property
    def parent(self) -> "JobName | None":
        """Get parent job name, or None if this is a root job."""
        if self.is_root:
            return None
        return JobName(self._parts[:-1])

    @property
    def user(self) -> str:
        """Get the submitting user."""
        return self._parts[0]

    @property
    def root_job(self) -> "JobName":
        """Get the root job for this hierarchy."""
        return JobName(self._parts[:2])

    @property
    def namespace(self) -> str:
        """Get the actor namespace (user/root job) for actor isolation."""
        return "/".join(self.root_job._parts)

    @property
    def name(self) -> str:
        """Get the local name (last component)."""
        return self._parts[-1]

    @property
    def is_root(self) -> bool:
        """True if this is a root job (no parent)."""
        return len(self._parts) == 2

    @property
    def task_index(self) -> int | None:
        """If this is a task (last component is numeric), return the index."""
        if len(self._parts) < 3:
            return None
        try:
            return int(self._parts[-1])
        except ValueError:
            return None

    @property
    def is_task(self) -> bool:
        """True if this is a task (last component is numeric)."""
        return self.task_index is not None

    @property
    def depth(self) -> int:
        """Depth in the job hierarchy. Root jobs have depth 1.

        Tasks inherit their parent job's depth (the task index
        is not counted as a depth level).

        Examples:
            /alice/root -> 1
            /alice/root/child -> 2
            /alice/root/child/grandchild -> 3
            /alice/root/0 (task) -> 1
            /alice/root/child/0 (task) -> 2
        """
        if self.is_task:
            return len(self._parts) - 2
        return len(self._parts) - 1

    def is_ancestor_of(self, other: "JobName", *, include_self: bool = True) -> bool:
        """True if this job name is an ancestor of another job name."""
        if include_self and self == other:
            return True
        if len(self._parts) >= len(other._parts):
            return False
        return other._parts[: len(self._parts)] == self._parts

    def to_safe_token(self) -> str:
        """Return a filesystem/tag-safe token derived from this name."""
        return "job__" + "__".join(self._parts)

    def require_task(self) -> tuple["JobName", int]:
        """Return (parent_job, task_index) for task names.

        Raises:
            ValueError: If this name is not a task or has no parent.
        """
        task_index = self.task_index
        if task_index is None:
            raise ValueError(f"JobName is not a task: {self}")
        if self.parent is None:
            raise ValueError(f"Task has no parent job: {self}")
        return (self.parent, task_index)

    def __str__(self) -> str:
        """Canonical wire format: '/user/root/child/grandchild'."""
        return "/" + "/".join(self._parts)

    def __repr__(self) -> str:
        return f"JobName({str(self)!r})"

    def to_wire(self) -> str:
        """Serialize to wire format for RPC/env vars."""
        return str(self)

    @classmethod
    def from_wire(cls, s: str) -> "JobName":
        """Parse from wire format. Alias for from_string."""
        return cls.from_string(s)


@dataclass(frozen=True, slots=True)
class TaskAttempt:
    """A task identity combining a task-level JobName with an optional attempt qualifier.

    Canonical wire format: /user/job/0:attempt_id
    When attempt_id is None, the wire format omits the suffix: /user/job/0

    The task_id must be a task-level JobName (last component numeric).
    attempt_id is optional — when absent, semantics are per-operation but
    typically "use the latest active attempt" is implied.

    Examples:
        TaskAttempt.from_wire("/alice/job/0")     -> TaskAttempt(task_id=/alice/job/0, attempt_id=None)
        TaskAttempt.from_wire("/alice/job/0:3")   -> TaskAttempt(task_id=/alice/job/0, attempt_id=3)
    """

    task_id: JobName
    attempt_id: int | None = None

    @classmethod
    def from_wire(cls, s: str) -> "TaskAttempt":
        """Parse a wire-format string like '/user/job/0' or '/user/job/0:3'."""
        if not s:
            raise ValueError("TaskAttempt wire format must not be empty")
        colon = s.rfind(":")
        if colon >= 0:
            task_part = s[:colon]
            attempt_str = s[colon + 1 :]
            try:
                attempt_id = int(attempt_str)
            except ValueError as exc:
                raise ValueError(f"Invalid attempt ID in TaskAttempt '{s}': '{attempt_str}' is not an integer") from exc
            return cls(task_id=JobName.from_wire(task_part), attempt_id=attempt_id)
        return cls(task_id=JobName.from_wire(s))

    def to_wire(self) -> str:
        """Serialize to wire format: '/user/job/0' or '/user/job/0:3'."""
        base = self.task_id.to_wire()
        if self.attempt_id is not None:
            return f"{base}:{self.attempt_id}"
        return base

    def require_attempt(self) -> int:
        """Return attempt_id or raise if absent."""
        if self.attempt_id is None:
            raise ValueError(f"TaskAttempt has no attempt_id: {self}")
        return self.attempt_id

    @property
    def job_id(self) -> JobName:
        """Get the parent job name (task_id without the task index)."""
        parent = self.task_id.parent
        if parent is None:
            raise ValueError(f"TaskAttempt task_id has no parent job: {self.task_id}")
        return parent

    @property
    def task_index(self) -> int:
        """Get the task index from the task_id."""
        return self.task_id.require_task()[1]

    def with_attempt(self, attempt_id: int) -> "TaskAttempt":
        """Return a new TaskAttempt with the given attempt_id."""
        return TaskAttempt(task_id=self.task_id, attempt_id=attempt_id)

    def without_attempt(self) -> "TaskAttempt":
        """Return a new TaskAttempt with attempt_id=None."""
        return TaskAttempt(task_id=self.task_id)

    def __str__(self) -> str:
        return self.to_wire()

    def __repr__(self) -> str:
        return f"TaskAttempt({self.to_wire()!r})"


def get_gpu_count(device: cluster_pb2.DeviceConfig) -> int:
    """Extract GPU count from DeviceConfig."""
    if device.HasField("gpu"):
        return device.gpu.count or 1
    return 0


def get_tpu_count(device: cluster_pb2.DeviceConfig) -> int:
    """Extract TPU count from DeviceConfig."""
    if device.HasField("tpu"):
        return device.tpu.count or 0
    return 0


WorkerId = NewType("WorkerId", str)
EndpointId = NewType("EndpointId", str)


@dataclass(frozen=True)
class WorkerStatus:
    """Worker status keyed by worker_id for autoscaler idle tracking."""

    worker_id: str
    running_task_ids: frozenset[str]

    @property
    def is_idle(self) -> bool:
        return len(self.running_task_ids) == 0


WorkerStatusMap = dict[str, WorkerStatus]


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


@dataclass(frozen=True)
class ReservationEntry:
    """A single reservation entry describing one worker's worth of resources.

    Used in the high-level client API. Each entry becomes a demand anchor
    that the autoscaler provisions before the reserving job schedules.

    Example:
        >>> ReservationEntry(resources=ResourceSpec(cpu=2, memory="8g"))
        >>> ReservationEntry(resources=ResourceSpec(cpu=2), constraints=[Constraint("region", value="us-central1")])
    """

    resources: "ResourceSpec"
    constraints: list[Constraint] | None = None

    def to_proto(self) -> cluster_pb2.ReservationEntry:
        """Convert to protobuf representation."""
        constraints_proto = [c.to_proto() for c in self.constraints or []]
        return cluster_pb2.ReservationEntry(
            resources=self.resources.to_proto(),
            constraints=constraints_proto,
        )


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


def gpu_device(variant: str, count: int = 1) -> cluster_pb2.DeviceConfig:
    """Create a DeviceConfig for a GPU device.

    Args:
        variant: GPU variant string (e.g., "H100", "A100").
        count: Number of GPUs per node.

    Returns:
        DeviceConfig with the gpu field set.
    """
    return cluster_pb2.DeviceConfig(
        gpu=cluster_pb2.GpuDevice(
            variant=variant,
            count=count,
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

    cpu: float = 0.0
    memory: str | int = 0  # "8g" or bytes
    disk: str | int = 0
    device: cluster_pb2.DeviceConfig | None = None

    def to_proto(self) -> cluster_pb2.ResourceSpecProto:
        """Convert to wire format."""
        memory_bytes = self.memory if isinstance(self.memory, int) else parse_memory_string(self.memory)
        disk_bytes = self.disk if isinstance(self.disk, int) else parse_memory_string(self.disk)
        spec = cluster_pb2.ResourceSpecProto(
            cpu_millicores=int(self.cpu * 1000),
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
        )
        if self.device is not None:
            spec.device.CopyFrom(self.device)
        return spec


CALLABLE_RUNNER = """\
import cloudpickle
import os
import sys
import traceback
import logging

# Reinitialize logging with the unified Iris format.
# Uses single-letter level prefix: I=INFO, W=WARNING, E=ERROR, D=DEBUG, C=CRITICAL.
# NOTE: This duplicates LevelPrefixFormatter and _LEVEL_PREFIX from iris.logging
# because CALLABLE_RUNNER executes inside an isolated task container that may not
# have the iris package installed (e.g. user-provided Docker images).
_LEVEL_PREFIX = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

class _LevelPrefixFormatter(logging.Formatter):
    def format(self, record):
        record.levelprefix = _LEVEL_PREFIX.get(record.levelname, "?")
        return super().format(record)

_root = logging.getLogger()
_root.handlers.clear()
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(_LevelPrefixFormatter(
    fmt="%(levelprefix)s%(asctime)s %(name)s %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
))
_root.addHandler(_handler)
_root.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

workdir = os.environ["IRIS_WORKDIR"]

try:
    with open(os.path.join(workdir, "_callable.pkl"), "rb") as f:
        fn, args, kwargs = cloudpickle.loads(f.read())
    fn(*args, **kwargs)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


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

    def to_proto(self) -> cluster_pb2.EnvironmentConfig:
        """Convert to wire format with sensible defaults applied."""
        default_env_vars = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }

        merged_env_vars = {k: v for k, v in {**default_env_vars, **(self.env_vars or {})}.items() if v is not None}

        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        return cluster_pb2.EnvironmentConfig(
            pip_packages=list(self.pip_packages or []),
            env_vars=merged_env_vars,
            extras=list(self.extras or []),
            python_version=py_version,
        )


class Namespace(str):
    """Namespace for actor isolation.

    Namespaces provide isolation between different jobs/environments.
    Actors in one namespace cannot discover actors in another namespace.

    The namespace is derived from the user/root job pair: all jobs in a hierarchy
    share the same namespace. This preserves actor isolation between unrelated
    jobs from the same user.
    """

    def __repr__(self) -> str:
        return f"Namespace({super().__repr__()})"

    @classmethod
    def from_job_id(cls, job_id: JobName) -> "Namespace":
        """Derive namespace from hierarchical job ID.

        The namespace is the first component of the job ID hierarchy.
        For example:
            JobName.from_string("/alice/abc123/worker-0") -> Namespace("alice/abc123")

        Args:
            job_id: Hierarchical job ID

        Returns:
            Namespace derived from root job ID

        Raises:
            ValueError: If job_id is empty
        """
        return cls(job_id.namespace)


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


def adjust_tpu_replicas(device: "cluster_pb2.DeviceConfig | None", replicas: int) -> int:
    """Adjust replicas for multi-host TPU topologies.

    Multi-host TPU topologies (e.g. v6e-32 with vm_count=8) require one task
    per VM. When ``replicas`` is 1 (the default), this auto-scales to
    ``vm_count`` so callers don't need to know the topology. For explicitly
    set replicas (>1) that don't align, raises ``ValueError``.

    Returns:
        The (possibly adjusted) replica count.
    """
    if device is None or not device.HasField("tpu"):
        return replicas

    variant = device.tpu.variant
    if not variant:
        return replicas

    try:
        topo = get_tpu_topology(variant)
    except ValueError:
        return replicas

    if topo.vm_count <= 1:
        return replicas

    if replicas == 1:
        return topo.vm_count

    if replicas % topo.vm_count != 0:
        raise ValueError(
            f"TPU type '{variant}' requires {topo.vm_count} VMs per slice, "
            f"so replicas must be a multiple of {topo.vm_count} (got replicas={replicas}). "
            f"For a single slice, use replicas={topo.vm_count}. "
            f"For N slices, use replicas=N*{topo.vm_count}."
        )

    return replicas


class Entrypoint:
    """Job entrypoint specification.

    Every entrypoint has a command (what to run) and optional workdir_files
    that the worker writes to $IRIS_WORKDIR/{name} before executing the command.

    Examples:
        entrypoint = Entrypoint.from_callable(my_func, arg1, arg2, key=val)
        entrypoint = Entrypoint.from_command("python", "train.py", "--epochs", "10")
    """

    def __init__(
        self,
        *,
        command: list[str],
        workdir_files: dict[str, bytes] | None = None,
    ):
        if not command:
            raise ValueError("Command must have at least one argument")
        self.command = command
        self.workdir_files: dict[str, bytes] = workdir_files or {}

    def resolve(self) -> tuple[Callable[..., Any], tuple, dict[str, Any]]:
        """Deserialize the callable, args, kwargs from pickle bytes.

        Only call this when you need to actually invoke the function locally
        (e.g. local_client). Avoid on the worker — use workdir_files directly
        to pass through to the task container without version-sensitive unpickling.
        """
        payload = self.workdir_files.get("_callable.pkl")
        if payload is None:
            raise ValueError("Not a callable entrypoint")

        return cloudpickle.loads(payload)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Entrypoint":
        # mark any testing code as pickle_by_value so we can use it with cloudpickle
        module = sys.modules.get(fn.__module__)
        module_path = Path(module.__file__).parts if module and getattr(module, "__file__", None) else ()
        if module and (module.__package__ is None or module.__spec__ is None or "tests" in module_path):
            cloudpickle.register_pickle_by_value(module)

        # We use bash -c so that $IRIS_WORKDIR and $IRIS_PYTHON are expanded
        # at runtime from the container's environment.  ProcessContainerHandle
        # remaps IRIS_WORKDIR to the host workdir and IRIS_PYTHON to
        # sys.executable for local execution; in Docker containers these are
        # set to "/app" and "python" respectively by task_attempt env setup.
        # `exec` replaces bash with python to avoid an extra parent process.
        return cls(
            command=["bash", "-c", "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py"],
            workdir_files={
                "_callable.pkl": cloudpickle.dumps((fn, args, kwargs)),
                "_callable_runner.py": CALLABLE_RUNNER.encode(),
            },
        )

    @classmethod
    def from_command(cls, *argv: str) -> "Entrypoint":
        """Create a command-line entrypoint.

        Args:
            *argv: Command and arguments (e.g., "python", "train.py", "--epochs", "10")
        """
        if not argv:
            raise ValueError("Command must have at least one argument")
        return cls(command=list(argv), workdir_files={})

    def to_proto(self) -> cluster_pb2.RuntimeEntrypoint:
        """Convert to protobuf representation.

        Produces a RuntimeEntrypoint with no setup_commands (those are added
        by build_runtime_entrypoint when submitting to the cluster).
        """
        proto = cluster_pb2.RuntimeEntrypoint()
        proto.run_command.argv[:] = self.command
        for name, data in self.workdir_files.items():
            proto.workdir_files[name] = data
        return proto

    @classmethod
    def from_proto(cls, proto: cluster_pb2.RuntimeEntrypoint) -> "Entrypoint":
        """Create from protobuf representation."""
        command = list(proto.run_command.argv)
        workdir_files = dict(proto.workdir_files) if proto.workdir_files else None
        return cls(command=command, workdir_files=workdir_files)
