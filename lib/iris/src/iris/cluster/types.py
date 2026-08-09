# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core types for the iris cluster layer.

This module contains controller and worker identifiers plus legacy runtime
records that have not yet moved into the resource model.

Generated messages in :mod:`iris.rpc` are serialization types, not the public
Job and Task model. Public resource records live in :mod:`iris.cluster.resources`.
"""

import functools
import hashlib
import urllib.parse
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import NewType

from rigging.timing import Timestamp

from iris.cluster.resources.state import JobState, PriorityBand, TaskState
from iris.rpc import controller_pb2


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


AUTO_DEVICE_VARIANT = "auto"
"""Device-variant sentinel meaning "unspecified — let the platform pick a variant".

A resource spec or scale group carrying this variant emits no ``device-variant``
routing constraint or advertised attribute, so the job matches any variant.
"""


# The reserved cluster name for work this controller owns and runs itself. Every
# ``jobs``/``tasks`` row carries a ``cluster`` column that defaults to
# ``LOCAL_CLUSTER`` and holds a peer's id once the job is handed off, so the
# control plane folds on ``cluster == LOCAL_CLUSTER`` instead of special-casing a
# local-vs-federated boolean. It is a reserved name — a real cluster id may not be
# ``"local"`` (enforced in config validation) — so the sentinel and the global
# cluster-id namespace stay disjoint.
LOCAL_CLUSTER = "local"

LOCAL_ADMIN_SUBMITTER = "local_admin"
"""``submitting_user`` for a job admitted without an authenticated email.

A CIDR/loopback (null-auth) submitter authenticates as the anonymous admin rather
than a person, so its jobs are attributed to this well-known principal. Per-cluster
federation allowlists key on ``submitting_user``, so ``local_admin`` is admitted to
a peer only if that peer's policy names it explicitly."""


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
    max_band: int = PriorityBand.INTERACTIVE


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
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.KILLED,
        JobState.WORKER_FAILED,
        JobState.UNSCHEDULABLE,
    }
)

TERMINAL_TASK_STATES: frozenset[int] = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.KILLED,
        TaskState.UNSCHEDULABLE,
        TaskState.WORKER_FAILED,
        TaskState.PREEMPTED,
        TaskState.COSCHED_FAILED,
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


EndpointAccess = controller_pb2.Controller.EndpointAccess

# Endpoint-metadata key a registrant sets (as a stringified number of seconds) to
# override the controller proxy's per-request upstream timeout for that endpoint —
# e.g. ``marin-serve`` sizing it to long model generations. In the shared types
# module so registry client and controller proxy agree on the key with no client
# dependency on controller code.
PROXY_TIMEOUT_METADATA_KEY = "proxy_timeout_seconds"
