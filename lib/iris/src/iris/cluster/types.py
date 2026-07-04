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

import functools
import hashlib
import os
import sys
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, NewType

import cloudpickle
import humanfriendly
from rigging.timing import Timestamp

from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script, setup_is_quiet, wants_gpu_extra
from iris.cluster.tpu_topology import get_tpu_topology
from iris.rpc import controller_pb2, job_pb2


class AcceleratorType(StrEnum):
    """Device/accelerator type for scale groups."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class CapacityType(StrEnum):
    """Capacity type for provisioning — controls which cloud API is used."""

    PREEMPTIBLE = "preemptible"
    ON_DEMAND = "on_demand"
    RESERVED = "reserved"


class GcpSliceMode(StrEnum):
    """Provisioning mode for GCP slices: a TPU pod or a plain CPU VM."""

    TPU = "tpu"
    VM = "vm"


DEFAULT_BACKEND_ID = "default"
"""Backend id of the implicit single backend synthesized from top-level config.

Shared by the runtime config synthesis (``iris.cluster.config.resolve_backends``)
and the ``0032_backend_id`` migration backfill — the migration has only a raw DB
connection (no config object), so both must agree on this exact literal.
"""


class BackendStatus(IntEnum):
    """Lifecycle state of a task backend, stored as an INTEGER in ``backends``."""

    ACTIVE = 0
    DRAINING = 1
    REMOVED = 2


class WellKnownAttribute(StrEnum):
    """Canonical attribute keys for constraint-based scheduling."""

    DEVICE_TYPE = "device-type"
    DEVICE_VARIANT = "device-variant"
    PREEMPTIBLE = "preemptible"
    REGION = "region"
    ZONE = "zone"
    TPU_NAME = "tpu-name"
    TPU_WORKER_ID = "tpu-worker-id"
    TPU_TOPOLOGY = "tpu-topology"
    TPU_VM_COUNT = "tpu-vm-count"
    GPU_VARIANT = "gpu-variant"
    GPU_COUNT = "gpu-count"


# The reserved cluster name for work this controller owns and runs itself. Every
# ``jobs``/``tasks`` row carries a ``cluster`` column that defaults to
# ``LOCAL_CLUSTER`` and holds a peer's id once the job is handed off, so the
# control plane folds on ``cluster == LOCAL_CLUSTER`` instead of special-casing a
# local-vs-federated boolean. It is a reserved name — a real cluster id may not be
# ``"local"`` (enforced in config validation) — so the sentinel and the global
# cluster-id namespace stay disjoint.
LOCAL_CLUSTER = "local"


def is_federated(cluster: str) -> bool:
    """Whether a job/task ``cluster`` value denotes a peer this controller handed off to.

    Its complement — a locally-owned job — is ``cluster == LOCAL_CLUSTER``; call sites
    that need the fold predicate compare against ``LOCAL_CLUSTER`` directly.
    """
    return cluster != LOCAL_CLUSTER


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

        Parsed names are interned in a process-wide LRU cache (names are
        immutable) so repeated decodes — the TypeDecorator path that fires
        once per row read — collapse to a dict lookup.

        Examples:
            JobName.from_string("/alice/my-job") -> JobName(("alice", "my-job"))
            JobName.from_string("/alice/parent/child") -> JobName(("alice", "parent", "child"))
            JobName.from_string("/alice/job/0") -> JobName(("alice", "job", "0"))
        """
        return _parse_job_name(s)

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
        return "/" + "/".join(self.root_job._parts)

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
        """Return a filesystem/tag-safe token derived from this name.

        Uses ``<user>-<sha256-hex>`` so the token stays short even for deeply
        nested job hierarchies (avoids ``ENAMETOOLONG`` on workdir creation).
        The full canonical name is hashed to preserve uniqueness.
        """
        digest = hashlib.sha256(str(self).encode()).hexdigest()
        return f"{self.user}-{digest}"

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

    def dashboard_url(self, base_url: str) -> str:
        """Public dashboard URL for this job under ``base_url``.

        ``base_url`` is the deployment's dashboard origin (e.g.
        ``https://iris.oa.dev``). The Vue dashboard routes jobs through a hash
        fragment whose path is the percent-encoded wire name, so
        ``/rav/job`` becomes ``…/#/job/%2Frav%2Fjob``. Inverse of
        ``scripts/job_profile_summary.parse_job_id``.
        """
        encoded = urllib.parse.quote(self.to_wire(), safe="")
        return f"{base_url.rstrip('/')}/#/job/{encoded}"

    @classmethod
    def from_wire(cls, s: str) -> "JobName":
        """Parse from wire format. Alias for from_string."""
        return cls.from_string(s)


@functools.lru_cache(maxsize=2**18)
def _parse_job_name(s: str) -> JobName:
    """Cached parser backing JobName.from_string / from_wire.

    Hot SA Core read paths decode the same job_id / task_id strings on every
    row; this collapses repeated decodes to a dict lookup. ``JobName`` is
    frozen+slots so cached instances can be shared without aliasing risk.
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
    return JobName(parts)


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


def get_gpu_count(device: job_pb2.DeviceConfig) -> int:
    """Extract GPU count from DeviceConfig."""
    if device.HasField("gpu"):
        return device.gpu.count or 1
    return 0


def get_tpu_count(device: job_pb2.DeviceConfig) -> int:
    """Extract TPU count from DeviceConfig."""
    if device.HasField("tpu"):
        return device.tpu.count or 0
    return 0


WorkerId = NewType("WorkerId", str)
EndpointId = NewType("EndpointId", str)
AttemptUid = NewType("AttemptUid", str)


@dataclass(frozen=True, slots=True)
class PendingTask:
    """Controller-side scheduling input projected from task, job, and config rows."""

    task_id: JobName
    job_id: JobName
    backend_id: str
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at_ms: Timestamp
    priority_band: int
    priority_neg_depth: int
    priority_root_submitted_ms: int
    priority_insertion: int
    job_state: int
    scheduling_deadline_epoch_ms: int | None
    scheduling_timeout_ms: int | None
    has_coscheduling: bool
    coscheduling_group_by: str | None
    constraints_json: str | None
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None


@dataclass
class UserBudgetDefaults:
    """Budget settings applied when a user has no override row in ``user_budgets``.

    ``budget_limit=0`` means unlimited; positive values cap spend before
    ``compute_effective_band`` downgrades INTERACTIVE work to BATCH.
    """

    budget_limit: int = 1000
    max_band: int = job_pb2.PRIORITY_BAND_INTERACTIVE


class WorkerUsability(StrEnum):
    """How the control loop may use a worker, derived from its liveness.

    Consumers project this verdict rather than re-deriving it from the raw
    ``healthy``/``active``/``consecutive_failures`` fields:

    - scheduling placement targets ``HEALTHY`` only;
    - the reconcile pass targets ``HEALTHY | DEGRADED`` (it keeps probing a
      mid-failure worker so it can recover or cross the teardown threshold);
    - autoscaler idle-spare accounting counts ``HEALTHY`` only, so a ``DEGRADED``
      idle worker is never reclaimed as free capacity.
    """

    HEALTHY = "healthy"
    """Active, healthy, no consecutive failures — a placement target."""

    DEGRADED = "degraded"
    """Active and healthy but accumulating failures — reconciled, NOT placeable,
    and NOT counted as idle spare. Torn down by the health threshold path, not
    by capacity scale-down."""

    DEAD = "dead"
    """Not active or not healthy — excluded from reconcile, scheduling, and
    idle tracking."""


@dataclass(frozen=True)
class WorkerStatus:
    """Worker status keyed by worker_id for autoscaler idle tracking."""

    worker_id: str
    running_task_ids: frozenset[str]
    usability: WorkerUsability = WorkerUsability.HEALTHY

    @property
    def is_idle(self) -> bool:
        return len(self.running_task_ids) == 0

    @property
    def is_idle_spare(self) -> bool:
        """Idle AND schedulable — safe to reclaim via scale-down.

        A ``DEGRADED`` idle worker is not a spare: counting it as reclaimable
        headroom is exactly what let the autoscaler call an unschedulable slice
        "idle — eligible for scale-down" while the scheduler was still waiting
        for that pool.
        """
        return self.is_idle and self.usability is WorkerUsability.HEALTHY


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

    def to_proto(self) -> job_pb2.CoschedulingConfig:
        """Convert to protobuf representation."""
        return job_pb2.CoschedulingConfig(group_by=self.group_by)


def tpu_device(variant: str, count: int | None = None) -> job_pb2.DeviceConfig:
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
    return job_pb2.DeviceConfig(
        tpu=job_pb2.TpuDevice(
            variant=variant,
            count=chip_count,
        )
    )


def gpu_device(variant: str, count: int = 1) -> job_pb2.DeviceConfig:
    """Create a DeviceConfig for a GPU device.

    Args:
        variant: GPU variant string (e.g., "H100", "A100").
        count: Number of GPUs per node.

    Returns:
        DeviceConfig with the gpu field set.

    Raises:
        ValueError: if count is not a positive integer.
    """
    if count < 1:
        raise ValueError(f"GPU count must be a positive integer, got {count}")
    return job_pb2.DeviceConfig(
        gpu=job_pb2.GpuDevice(
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
    device: job_pb2.DeviceConfig | None = None

    # Accelerator tasks default to enough CPU to avoid bottlenecking on data
    # loading, but explicit CPU requests are preserved for quota-constrained
    # queues and diagnostic runs.
    MIN_ACCELERATOR_CPU_MILLICORES = 4_000

    def to_proto(self) -> job_pb2.ResourceSpecProto:
        """Convert to wire format."""
        memory_bytes = self.memory if isinstance(self.memory, int) else parse_memory_string(self.memory)
        disk_bytes = self.disk if isinstance(self.disk, int) else parse_memory_string(self.disk)
        cpu_mc = int(self.cpu * 1000)
        if self.device is not None and cpu_mc < self.MIN_ACCELERATOR_CPU_MILLICORES:
            cpu_mc = self.MIN_ACCELERATOR_CPU_MILLICORES
        spec = job_pb2.ResourceSpecProto(
            cpu_millicores=cpu_mc,
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
# NOTE: This duplicates LevelPrefixFormatter and _LEVEL_PREFIX from rigging.log_setup
# because CALLABLE_RUNNER executes inside an isolated task container that may not
# have the rigging package installed (e.g. user-provided Docker images).
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

    Setup:
    - ``setup_scripts=None`` builds the default uv-sync script. ``sync_packages``
      scopes that sync to specific workspace members (default: all members).
    - ``setup_scripts`` set to a list runs those scripts verbatim before the
      command, with the task's ``IRIS_*`` env available; ``[]`` means no setup (the
      image is used as-is). Build the default and tweak it via
      ``iris.cluster.setup_scripts.default_setup_script``.

    Whenever any setup runs (default or custom), iris appends its own
    ``iris_runtime_setup_script`` so cloudpickle/profiler support is always
    present; it is skipped only for the no-setup (``[]``) case.

    Note: To specify workspace for bundle creation, use IrisClient.remote(workspace=...).
    """

    pip_packages: Sequence[str] | None = None
    env_vars: dict[str, str] | None = None
    extras: Sequence[str] | None = None
    setup_scripts: Sequence[str] | None = None
    sync_packages: Sequence[str] | None = None

    def to_proto(self) -> job_pb2.EnvironmentConfig:
        """Convert to wire format, resolving the user setup scripts.

        ``setup_scripts=None`` builds the default uv-sync script from
        extras/pip/sync_packages; a list is used verbatim; ``[]`` is no setup. The
        wire carries only this user list.
        """
        default_env_vars = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }

        merged_env_vars = {k: v for k, v in {**default_env_vars, **(self.env_vars or {})}.items() if v is not None}

        if self.setup_scripts is None:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            extras = list(self.extras or [])
            setup_scripts = [
                default_setup_script(
                    extras=extras,
                    pip_packages=list(self.pip_packages or []),
                    python_version=py_version,
                    packages=list(self.sync_packages or []) or None,
                    quiet=setup_is_quiet(merged_env_vars),
                )
            ]
            # GPU jobs need the venv's CUDA toolchain (ptxas/nvlink/libdevice)
            # exposed for JAX/Pallas Mosaic; the script no-ops without it.
            if wants_gpu_extra(extras):
                setup_scripts.append(cuda_toolchain_setup_script())
        else:
            setup_scripts = [s for s in self.setup_scripts if s.strip()]

        return job_pb2.EnvironmentConfig(env_vars=merged_env_vars, setup_scripts=setup_scripts)


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


TERMINAL_JOB_STATES: frozenset[int] = frozenset(
    {
        job_pb2.JOB_STATE_SUCCEEDED,
        job_pb2.JOB_STATE_FAILED,
        job_pb2.JOB_STATE_KILLED,
        job_pb2.JOB_STATE_WORKER_FAILED,
        job_pb2.JOB_STATE_UNSCHEDULABLE,
    }
)

TERMINAL_TASK_STATES: frozenset[int] = frozenset(
    {
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_KILLED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    }
)


def is_job_finished(state: int) -> bool:
    return state in TERMINAL_JOB_STATES


def is_task_finished(state: int) -> bool:
    """Check if a task state is terminal.

    This is a simple check for whether the state is a terminal state.
    For ControllerTask, use task.is_finished() which also considers retry budgets.
    """
    return state in TERMINAL_TASK_STATES


JobState = job_pb2.JobState
TaskState = job_pb2.TaskState
EndpointAccess = controller_pb2.Controller.EndpointAccess


# TPU topology table and lookup helpers live in iris.cluster.tpu_topology so
# both this module and iris.cluster.constraints can reference them without an
# import cycle. Re-exported via the top-level import above.


def adjust_tpu_replicas(device: "job_pb2.DeviceConfig | None", replicas: int) -> int:
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
        workdir_file_refs: dict[str, str] | None = None,
    ):
        if not command:
            raise ValueError("Command must have at least one argument")
        self.command = command
        self.workdir_files: dict[str, bytes] = workdir_files or {}
        self.workdir_file_refs: dict[str, str] = workdir_file_refs or {}

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

    def to_proto(self) -> job_pb2.RuntimeEntrypoint:
        """Convert to protobuf representation.

        Produces a RuntimeEntrypoint with no setup_commands (those are added
        by build_runtime_entrypoint when submitting to the cluster).
        """
        proto = job_pb2.RuntimeEntrypoint()
        proto.run_command.argv[:] = self.command
        for name, data in self.workdir_files.items():
            proto.workdir_files[name] = data
        for name, blob_id in self.workdir_file_refs.items():
            proto.workdir_file_refs[name] = blob_id
        return proto

    @classmethod
    def from_proto(cls, proto: job_pb2.RuntimeEntrypoint) -> "Entrypoint":
        """Create from protobuf representation."""
        command = list(proto.run_command.argv)
        workdir_files = dict(proto.workdir_files) if proto.workdir_files else None
        workdir_file_refs = dict(proto.workdir_file_refs) if proto.workdir_file_refs else None
        return cls(command=command, workdir_files=workdir_files, workdir_file_refs=workdir_file_refs)
