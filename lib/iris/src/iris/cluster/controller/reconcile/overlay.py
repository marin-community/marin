# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mutable scratch space for one pure-function call into the state machine.

A narrow record-bag threaded through the per-update kernel.
"""

from collections.abc import Iterable
from typing import TypeVar

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.effects import (
    AttemptRowDelta,
    ControllerEffects,
    JobRowDelta,
    LogEvent,
    TaskRowDelta,
)
from iris.cluster.controller.reconcile.policy import CANCEL_GUARD_STATES
from iris.cluster.controller.reconcile.snapshot import (
    JobConfigRow,
    JobStateBasis,
    TransitionSnapshot,
    pick_earliest_task_error,
)
from iris.cluster.controller.task_state import ActiveTaskRow
from iris.cluster.types import TERMINAL_JOB_STATES, JobName, WorkerId
from iris.rpc import job_pb2

_T = TypeVar("_T")


def _first(old: Timestamp | None, new: Timestamp | None) -> Timestamp | None:
    """First-non-null fold: ``coalesce(col, new)`` where ``col`` is the earlier write."""
    return old if old is not None else new


def _last_non_null(old: _T | None, new: _T | None) -> _T | None:
    """Last-non-null fold: ``coalesce(new, col)``."""
    return new if new is not None else old


class Overlay:
    """Mutable in-memory view of a :class:`TransitionSnapshot`.

    Wraps a read-only snapshot and a pending :class:`ControllerEffects`. The
    per-entity delta dicts ARE the overlay: reads layer the accumulated deltas
    on top of the snapshot so iteration N+1 sees iteration N's prospective
    state, and the same dicts are flushed to SQL once per entity at commit.

    Mutation is split by a strict prefix rule:

    * ``merge_*`` (``merge_task`` / ``merge_attempt`` / ``merge_job_state`` /
      ``merge_cascade_kill``) folds a typed row delta into the accumulator per
      the per-field rules (see each method). Reads (:meth:`task_state`,
      :meth:`job_basis`, ...) consult the accumulator, so prospective state and
      the persisted SQL cannot drift.
    * ``emit_*`` appends a cross-aggregate effect category (worker health,
      structured audit log events). These fire after commit and so are not row
      deltas.
    """

    def __init__(self, snapshot: TransitionSnapshot) -> None:
        self._snapshot = snapshot
        self._effects = ControllerEffects()
        # Per-job count of this batch's FAILED attempt deltas, built lazily and
        # cached for the recompute pass (see job_basis). merge_attempt invalidates
        # it only when a FAILED attempt is recorded, so the recompute pass's own
        # kill/cascade attempt writes (never FAILED) never force a rebuild.
        self._failed_attempt_deltas_by_job: dict[JobName, int] | None = None

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def effects(self) -> ControllerEffects:
        return self._effects

    @property
    def snapshot(self) -> TransitionSnapshot:
        return self._snapshot

    @property
    def now(self) -> Timestamp:
        return self._snapshot.now

    def job_config(self, job_id: JobName) -> JobConfigRow | None:
        return self._snapshot.job_configs.get(job_id)

    def _job_state(self, job_id: JobName, fallback: int) -> int:
        delta = self._effects.jobs.get(job_id)
        return delta.state if delta is not None else fallback

    def job_basis(self, job_id: JobName) -> JobStateBasis | None:
        """Compose the snapshot basis with the task/job-state accumulator."""
        basis = self._snapshot.job_state_basis.get(job_id)
        if basis is None:
            return None
        current_state = self._job_state(job_id, basis.state)
        if current_state in TERMINAL_JOB_STATES:
            return JobStateBasis(
                job_id=basis.job_id,
                state=current_state,
                started_at=basis.started_at,
                max_task_failures=basis.max_task_failures,
                task_state_counts={},
                total_failures=0,
                first_task_error=None,
            )
        # Single pass: build the accumulator-aware state histogram and each
        # task's effective (state, error, finished_at) for pick_earliest_task_error
        # to rank afterward.
        counts: dict[int, int] = {}
        error_candidates: list[tuple[int, int, Timestamp | None, str | None]] = []
        for row in self._snapshot.all_tasks_by_job.get(job_id, ()):
            delta = self._effects.tasks.get(row.task_id)
            state = delta.state if delta is not None else row.state
            counts[state] = counts.get(state, 0) + 1
            # The accumulator only carries a value when a delta set a non-null
            # one; otherwise fall back to the snapshot row.
            error = delta.error if (delta is not None and delta.error is not None) else row.error
            finished_at = delta.finished_at if (delta is not None and delta.finished_at is not None) else row.finished_at
            error_candidates.append((row.task_index, state, finished_at, error))
        # Cumulative failure budget: the committed-derived per-job failure count the
        # loader already summed into ``basis.total_failures``, plus this batch's
        # not-yet-committed FAILED attempt writes. This equals what the next tick's
        # loader will derive once these attempts commit — no prospective counter is
        # carried, symmetric with preemption.
        total_failures = basis.total_failures + self._failed_attempt_deltas_for_job(job_id)
        return JobStateBasis(
            job_id=basis.job_id,
            state=current_state,
            started_at=basis.started_at,
            max_task_failures=basis.max_task_failures,
            task_state_counts=counts,
            total_failures=total_failures,
            first_task_error=pick_earliest_task_error(error_candidates),
        )

    def _failed_attempt_deltas_for_job(self, job_id: JobName) -> int:
        """This batch's FAILED attempt-delta count for ``job_id``."""
        if self._failed_attempt_deltas_by_job is None:
            tally: dict[JobName, int] = {}
            for (task_id, _), delta in self._effects.attempts.items():
                if delta.state == job_pb2.TASK_STATE_FAILED:
                    parent = task_id.parent
                    if parent is not None:
                        tally[parent] = tally.get(parent, 0) + 1
            self._failed_attempt_deltas_by_job = tally
        return self._failed_attempt_deltas_by_job.get(job_id, 0)

    def task_state(self, task_id: JobName) -> int | None:
        delta = self._effects.tasks.get(task_id)
        if delta is not None:
            return delta.state
        row = self._snapshot.tasks.get(task_id)
        return row.state if row is not None else None

    def attempt_state(self, task_id: JobName, attempt_id: int) -> int | None:
        """Overlay-aware attempt state (a same-batch delta wins over the snapshot)."""
        delta = self._effects.attempts.get((task_id, attempt_id))
        if delta is not None and delta.state is not None:
            return delta.state
        row = self._snapshot.attempts.get((task_id, attempt_id))
        return row.state if row is not None else None

    def attempt_finished_at(self, task_id: JobName, attempt_id: int) -> int | None:
        """Overlay-aware attempt ``finished_at`` in epoch-ms (delta wins over snapshot)."""
        delta = self._effects.attempts.get((task_id, attempt_id))
        if delta is not None and delta.finished_at is not None:
            return delta.finished_at.epoch_ms()
        row = self._snapshot.attempts.get((task_id, attempt_id))
        return row.finished_at_ms if row is not None else None

    def active_tasks_for_job(
        self,
        job_id: JobName,
        *,
        exclude: JobName | None = None,
        states: Iterable[int],
    ) -> list[ActiveTaskRow]:
        """Return non-terminal tasks for ``job_id`` filtered to ``states``, accumulator-aware."""
        state_set = frozenset(int(s) for s in states)
        rows = self._snapshot.active_tasks_by_job.get(job_id, ())
        out: list[ActiveTaskRow] = []
        for row in rows:
            if exclude is not None and row.task_id == exclude:
                continue
            delta = self._effects.tasks.get(row.task_id)
            effective_state = delta.state if delta is not None else row.state
            if effective_state not in state_set:
                continue
            if effective_state != row.state:
                row = ActiveTaskRow(
                    task_id=row.task_id,
                    job_id=row.job_id,
                    state=effective_state,
                    current_attempt_id=row.current_attempt_id,
                    current_worker_id=row.current_worker_id,
                    preemption_count=row.preemption_count,
                    max_retries_failure=row.max_retries_failure,
                    max_retries_preemption=row.max_retries_preemption,
                    has_coscheduling=row.has_coscheduling,
                )
            out.append(row)
        return out

    def job_descendants(self, job_id: JobName) -> list[JobName]:
        desc = self._snapshot.job_descendants.get(job_id)
        if desc is None:
            return []
        return list(desc.descendants)

    def job_preemption_policy(self, job_id: JobName) -> int:
        """Resolve the effective preemption policy.

        Defaults: single-task jobs → ``TERMINATE_CHILDREN``, multi-task →
        ``PRESERVE_CHILDREN``.
        """
        cfg = self._snapshot.job_configs.get(job_id)
        if cfg is None:
            return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
        policy = cfg.preemption_policy
        if policy != job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED:
            return policy
        if cfg.num_tasks <= 1:
            return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
        return job_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN

    # ------------------------------------------------------------------
    # Accumulator merge entry points
    # ------------------------------------------------------------------

    def merge_task(self, delta: TaskRowDelta) -> None:
        """Merge a task delta into the accumulator.

        Per-field fold (earlier accumulated ``old`` then newer ``delta``):
        state last-wins; error/exit_code/container_id/status_message last-non-null;
        started_at first-non-null; finished_at last-wins (may clear to None).
        """
        old = self._effects.tasks.get(delta.task_id)
        if old is None:
            self._effects.tasks[delta.task_id] = delta
            return
        self._effects.tasks[delta.task_id] = TaskRowDelta(
            task_id=delta.task_id,
            state=delta.state,
            error=_last_non_null(old.error, delta.error),
            exit_code=_last_non_null(old.exit_code, delta.exit_code),
            started_at=_first(old.started_at, delta.started_at),
            finished_at=delta.finished_at,
            container_id=_last_non_null(old.container_id, delta.container_id),
            status_message=_last_non_null(old.status_message, delta.status_message),
        )

    def merge_attempt(self, delta: AttemptRowDelta) -> None:
        """Merge an attempt delta into the accumulator.

        Per-field fold: state/exit_code/error last-non-null; started_at and
        finished_at first-non-null (attempts differ from tasks: attempt
        ``finished_at`` is coalesce-with-column / first-wins).
        """
        key = (delta.task_id, delta.attempt_id)
        old = self._effects.attempts.get(key)
        if old is None:
            merged = delta
        else:
            merged = AttemptRowDelta(
                task_id=delta.task_id,
                attempt_id=delta.attempt_id,
                state=_last_non_null(old.state, delta.state),
                started_at=_first(old.started_at, delta.started_at),
                finished_at=_first(old.finished_at, delta.finished_at),
                exit_code=_last_non_null(old.exit_code, delta.exit_code),
                error=_last_non_null(old.error, delta.error),
            )
        self._effects.attempts[key] = merged
        # The job-wide failure budget counts FAILED attempt deltas (job_basis); a
        # newly-FAILED attempt invalidates the cached per-job tally so the next
        # job_basis read rebuilds it.
        if merged.state == job_pb2.TASK_STATE_FAILED:
            self._failed_attempt_deltas_by_job = None

    def merge_job_state(self, delta: JobRowDelta) -> None:
        """Merge a recompute job-state write into the accumulator.

        state last-wins; started_at first-non-null; finished_at last-wins;
        error overwrite-if-given.
        """
        old = self._effects.jobs.get(delta.job_id)
        if old is None:
            self._effects.jobs[delta.job_id] = delta
            return
        # recompute_state early-returns on an already-terminal job, so it never
        # records over a cascade-kill entry. Assert the invariant rather than
        # silently demoting a kill back to a recompute write.
        assert not old.is_cascade_kill, f"recompute write onto cascade-killed job {delta.job_id}"
        self._effects.jobs[delta.job_id] = JobRowDelta(
            job_id=delta.job_id,
            state=delta.state,
            started_at=_first(old.started_at, delta.started_at),
            finished_at=delta.finished_at if delta.finished_at is not None else old.finished_at,
            error=delta.error if delta.error is not None else old.error,
            is_cascade_kill=False,
            allow_overwrite_worker_failed=old.allow_overwrite_worker_failed,
        )

    def merge_cascade_kill(self, delta: JobRowDelta) -> None:
        """Merge a guarded cascade-kill write into the accumulator.

        Guard against the accumulated-or-snapshot state: if that state is in the
        guard set (``CANCEL_GUARD_STATES`` when ``allow_overwrite_worker_failed``
        else ``TERMINAL_JOB_STATES``), the kill is a no-op. Otherwise the job
        moves to KILLED with first-non-null ``finished_at``. This reproduces the
        SQL ``WHERE state NOT IN guard`` for the in-memory overlay.
        """
        old = self._effects.jobs.get(delta.job_id)
        basis = self._snapshot.job_state_basis.get(delta.job_id)
        current = old.state if old is not None else (basis.state if basis is not None else None)
        guard = CANCEL_GUARD_STATES if delta.allow_overwrite_worker_failed else TERMINAL_JOB_STATES
        if current is not None and current in guard:
            return
        prior_finished = old.finished_at if old is not None else None
        self._effects.jobs[delta.job_id] = JobRowDelta(
            job_id=delta.job_id,
            state=job_pb2.JOB_STATE_KILLED,
            started_at=old.started_at if old is not None else None,
            finished_at=_first(prior_finished, delta.finished_at),
            error=delta.error,
            is_cascade_kill=True,
            allow_overwrite_worker_failed=delta.allow_overwrite_worker_failed,
        )

    # ------------------------------------------------------------------
    # Cross-aggregate effect emitters. NOT row deltas — these are
    # post-commit categories, kept separate from the row-delta setters.
    # ------------------------------------------------------------------

    def emit_log_event(self, event: LogEvent) -> None:
        self._effects.log_events.append(event)

    def emit_worker_build_failed(self, worker_id: WorkerId) -> None:
        self._effects.health.build_failed.append(worker_id)
