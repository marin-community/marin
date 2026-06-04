# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Effect contract: typed per-entity row deltas + cross-aggregate effect categories.

The batch state model has two layers over one event stream:

* ``snapshot`` — immutable truth loaded once at the start of the batch.
* ``effects`` (:class:`ControllerEffects`) — a typed, per-entity accumulator
  that IS the overlay: reads consult it mid-batch for prospective state, and it
  is flushed to SQL once per touched entity at commit. Each touched task / attempt
  / job collapses into ONE delta via the per-field merge rules in
  :mod:`iris.cluster.controller.reconcile.overlay`; the flush issues one bulk
  statement per entity group.

This module is the pure data half of that contract — it imports no ``db`` /
``schema`` / ``projections`` so the state-machine kernel stays import-clean. The
I/O sink that drains a :class:`ControllerEffects` to SQL lives in
:mod:`iris.cluster.controller.reconcile.commit` (``commit_effects``).
"""

from dataclasses import dataclass, field

from rigging.timing import Timestamp

from iris.cluster.types import JobName, WorkerId

# ---------------------------------------------------------------------------
# Per-entity row deltas
# ---------------------------------------------------------------------------


@dataclass
class TaskRowDelta:
    """Merged write to one ``tasks`` row. See ``Overlay.merge_task`` for the
    per-field merge directions; the flush in ``commit_effects`` issues the same
    coalesce expressions the legacy ``TaskMutation.apply`` did."""

    task_id: JobName
    state: int
    error: str | None = None
    exit_code: int | None = None
    started_at: Timestamp | None = None  # first-wins (coalesce(col, ...))
    finished_at: Timestamp | None = None  # last-wins, may be None to clear
    failure_count: int | None = None
    preemption_count: int | None = None
    container_id: str | None = None


@dataclass
class AttemptRowDelta:
    """Merged write to one ``task_attempts`` row, keyed by (task_id, attempt_id)."""

    task_id: JobName
    attempt_id: int
    state: int | None = None
    started_at: Timestamp | None = None  # first-wins (coalesce(col, ...))
    finished_at: Timestamp | None = None  # first-wins (coalesce(col, ...))
    exit_code: int | None = None
    error: str | None = None


@dataclass
class JobRowDelta:
    """Merged write to one ``jobs`` row.

    ``is_cascade_kill`` selects the cascade-kill merge/flush path (guarded
    overwrite to KILLED). ``allow_overwrite_worker_failed`` widens the guard so
    the kill also overwrites ``JOB_STATE_WORKER_FAILED`` rows (cancel uses this).
    """

    job_id: JobName
    state: int
    started_at: Timestamp | None = None  # first-wins
    finished_at: Timestamp | None = None  # recompute: last-wins; kill: first-wins
    error: str | None = None
    is_cascade_kill: bool = False
    allow_overwrite_worker_failed: bool = False


# ---------------------------------------------------------------------------
# Cross-aggregate effect categories
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EndpointDeletion:
    task_id: JobName


@dataclass(slots=True)
class WorkerHealthEffect:
    heartbeat: list[WorkerId] = field(default_factory=list)
    build_failed: list[WorkerId] = field(default_factory=list)
    make_unhealthy: list[WorkerId] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LogEvent:
    action: str
    entity_id: str
    trigger: str | None = None
    details: tuple[tuple[str, object], ...] = ()


# ---------------------------------------------------------------------------
# ControllerEffects
# ---------------------------------------------------------------------------


@dataclass
class ControllerEffects:
    """Pure output of one state-machine call. Caller persists with ``commit_effects``.

    The per-entity dicts are the accumulator/overlay: reads consult them
    mid-batch; the flush drains them to SQL. Cross-aggregate post-commit
    categories stay separate because their application semantics differ
    (post-commit hooks vs in-Tx SQL).
    """

    tasks: dict[JobName, TaskRowDelta] = field(default_factory=dict)
    attempts: dict[tuple[JobName, int], AttemptRowDelta] = field(default_factory=dict)
    jobs: dict[JobName, JobRowDelta] = field(default_factory=dict)

    endpoint_deletions: list[EndpointDeletion] = field(default_factory=list)
    health: WorkerHealthEffect = field(default_factory=WorkerHealthEffect)
    log_events: list[LogEvent] = field(default_factory=list)
