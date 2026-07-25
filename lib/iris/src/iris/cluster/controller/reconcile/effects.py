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
    """Merged write to one ``tasks`` row. ``Overlay.merge_task`` accumulates
    these fields; the per-field merge directions (first-wins vs last-wins) are
    realized by the coalesce expressions in ``commit._flush_tasks``."""

    task_id: JobName
    state: int
    error: str | None = None
    exit_code: int | None = None
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None
    container_id: str | None = None
    # Tri-state, folded last-non-null: None leaves the column unchanged, "" clears it,
    # a string sets it (commit._flush_tasks coalesces None→keep, ""→clear).
    status_message: str | None = None


@dataclass
class AttemptRowDelta:
    """Merged write to one ``task_attempts`` row, keyed by (task_id, attempt_id)."""

    task_id: JobName
    attempt_id: int
    state: int | None = None
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None
    exit_code: int | None = None
    error: str | None = None
    # Backend object identity + terminal cause, folded last-non-null (see
    # snapshot.TaskUpdate). None leaves the column unchanged.
    pod_name: str | None = None
    pod_uid: str | None = None
    node_name: str | None = None
    terminal_reason: str | None = None


@dataclass
class JobRowDelta:
    """Merged write to one ``jobs`` row.

    ``is_cascade_kill`` selects the cascade-kill merge/flush path (guarded
    overwrite to KILLED). ``allow_overwrite_worker_failed`` widens the guard so
    the kill also overwrites ``JOB_STATE_WORKER_FAILED`` rows (cancel uses this).
    """

    job_id: JobName
    state: int
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None
    error: str | None = None
    is_cascade_kill: bool = False
    allow_overwrite_worker_failed: bool = False


# ---------------------------------------------------------------------------
# Cross-aggregate effect categories
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkerHealthEffect:
    """Kernel-derived health signal surfaced to the controller (not applied here).

    The reconcile kernel cannot reach the in-memory health tracker — health is
    folded in exactly one place (``WorkerHealthTracker.apply``). The kernel only
    *derives* build failures (a BUILDING/ASSIGNED→FAILED transition on the worker
    path); the controller reads ``build_failed`` off the returned effects and
    translates it into ``BUILD_FAILED`` events for ``apply``.
    """

    build_failed: list[WorkerId] = field(default_factory=list)


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

    health: WorkerHealthEffect = field(default_factory=WorkerHealthEffect)
    log_events: list[LogEvent] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether this batch records no committable writes.

        ``health.build_failed`` is excluded: it is folded into the liveness
        tracker by the backend, never persisted by ``commit_effects``.
        """
        return not (self.tasks or self.attempts or self.jobs or self.log_events)
