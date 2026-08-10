# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster topology and scheduling value types."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from rigging.timing import Timestamp

from iris.resources.names import JobName as _JobName
from iris.resources.state import JobState, PriorityBand, TaskState


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
class PendingTask:
    """Controller-side scheduling input projected from task, job, and config rows."""

    task_id: _JobName
    job_id: _JobName
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
