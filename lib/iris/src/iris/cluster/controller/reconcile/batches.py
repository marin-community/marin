# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch orchestrators: snapshot in → ControllerEffects out.

Cross-aggregate orchestration lives here — each ``apply_*_batch`` builds one
:class:`WorkingState` and threads it through the per-update kernel so cascades
triggered by earlier items in a batch are visible to later items.
"""

import logging
from dataclasses import dataclass, field

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile import job, peers, task, worker
from iris.cluster.controller.reconcile.effects import (
    ControllerEffects,
    JobRowDelta,
    LogEvent,
)
from iris.cluster.controller.reconcile.policy import (
    FAILURE_TASK_STATES,
    NON_TERMINAL_TASK_STATES,
    TERMINAL_STATE_REASONS,
)
from iris.cluster.controller.reconcile.snapshot import (
    TaskUpdate,
    TransitionSnapshot,
)
from iris.cluster.controller.reconcile.working_state import WorkingState
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, ActiveTaskRow
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    JobName,
    WorkerId,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job-aggregate orchestration helpers
# ---------------------------------------------------------------------------
#
# These compose ``job.recompute_state`` with task-killing primitives. They
# live here (not in job.py) because ``_kill_non_terminal_tasks`` and
# ``_cascade_to_children`` need to invoke ``task.mark_task_terminating``, and
# job.py is forbidden from importing task.


def _kill_non_terminal_tasks(
    state: WorkingState,
    job_id: JobName,
    reason: str,
    now_ms: int,
) -> None:
    """Kill all non-terminal tasks for a single job and delete endpoints."""
    rows = state.active_tasks_for_job(job_id, states=NON_TERMINAL_TASK_STATES)
    for row in rows:
        task.mark_task_terminating(
            state,
            row.task_id.to_wire(),
            row.current_attempt_id,
            job_pb2.TASK_STATE_KILLED,
            reason,
            now_ms,
        )


def _cascade_to_children(
    state: WorkingState,
    job_id: JobName,
    now_ms: int,
    reason: str,
    exclude_reservation_holders: bool = False,
) -> None:
    """Kill descendant jobs (not the job itself) on a parent terminal/preempt."""
    descendants = state.job_descendants(job_id, exclude_holders=exclude_reservation_holders)
    for child_job_id in descendants:
        _kill_non_terminal_tasks(state, child_job_id, reason, now_ms)
        state.record_cascade_kill(
            JobRowDelta(
                job_id=child_job_id,
                state=job_pb2.JOB_STATE_KILLED,
                error=reason,
                finished_at=Timestamp.from_ms(now_ms),
                is_cascade_kill=True,
            )
        )


def _finalize_terminal_job(
    state: WorkingState,
    job_id: JobName,
    terminal_state: int,
    now_ms: int,
) -> None:
    """Kill remaining tasks and optionally cascade to children when a job goes terminal."""
    reason = TERMINAL_STATE_REASONS.get(terminal_state, "Job finalized")
    _kill_non_terminal_tasks(state, job_id, reason, now_ms)
    should_cascade = True
    if terminal_state != job_pb2.JOB_STATE_SUCCEEDED:
        policy = state.job_preemption_policy(job_id)
        should_cascade = policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    if should_cascade:
        _cascade_to_children(state, job_id, now_ms, reason)


# ---------------------------------------------------------------------------
# Shared inner kernel
# ---------------------------------------------------------------------------


def _cascade_to_peers(
    state: WorkingState,
    outcome: task.TransitionOutcome,
    now_ms: int,
) -> None:
    """Coscheduled-sibling cascade for one transition. No job recompute."""
    if not outcome.has_coscheduling:
        return
    siblings = peers.find_coscheduled_siblings(state, outcome.job_id, outcome.task_id, True)
    if outcome.new_task_state in FAILURE_TASK_STATES:
        peers.terminate_coscheduled_siblings(state, siblings, outcome.task_id, now_ms)
    else:
        peers.requeue_coscheduled_siblings(state, siblings, outcome.task_id, now_ms)


@dataclass
class _TouchedJobs:
    """Deduped work-list of jobs touched by the apply pass, shared across a batch.

    ``touched_jobs`` is insertion-ordered + deduped: jobs with any state-changing
    task transition. ``_recompute_touched_jobs`` recomputes each once (display
    state + terminal finalize). The whole point is to lift recompute out of the
    per-worker loop so it runs once per batch instead of once per worker.
    """

    touched_jobs: list[JobName] = field(default_factory=list)
    _touched_seen: set[JobName] = field(default_factory=set)

    def note(self, job_id: JobName) -> None:
        if job_id not in self._touched_seen:
            self._touched_seen.add(job_id)
            self.touched_jobs.append(job_id)


def _apply_transitions(
    state: WorkingState,
    snapshot: TransitionSnapshot,
    updates: list[TaskUpdate],
    now_ms: int,
    acc: _TouchedJobs,
    *,
    source: task.TransitionSource = task.TransitionSource.WORKER_RECONCILE,
) -> None:
    """Apply pass: apply task transitions + per-update peer cascade.

    Peer cascades run per update so later updates see requeued/terminated
    siblings. Job recompute + terminal finalize are deferred to the recompute
    pass (``_recompute_touched_jobs``) over the whole batch: ``job.recompute_state``
    rescans the whole job task histogram, so per-update recompute is
    O(tasks_per_job²) on dispatch. Folding it makes the batch order-independent:
    a task that reaches a terminal-success state in the same batch where a
    sibling fails the job stays SUCCEEDED, since finalize only kills NON-terminal
    tasks.

    ``source`` selects caller-specific health side effects: worker reconcile
    reaps build-failing hosts; direct providers manage their own hosts.
    """
    for update in updates:
        outcome = task.apply_one_transition(state, snapshot, update, now_ms, source=source)
        if outcome is None:
            continue
        _cascade_to_peers(state, outcome, now_ms)
        if outcome.new_task_state != outcome.prior_state:
            acc.note(outcome.job_id)


def _recompute_touched_jobs(
    state: WorkingState,
    acc: _TouchedJobs,
    now_ms: int,
) -> set[JobName]:
    """Recompute pass: recompute each touched job once; finalize the ones that go terminal.

    Returns the set of jobs that went terminal in this batch (so the caller can
    emit the per-batch ``job_terminated`` log events). ``recompute_state``
    early-returns on already-terminal jobs and only records a write on change.
    """
    cascaded_jobs: set[JobName] = set()
    for job_id in acc.touched_jobs:
        new_job_state = job.recompute_state(state, job_id)
        if new_job_state in TERMINAL_JOB_STATES:
            _finalize_terminal_job(state, job_id, new_job_state, now_ms)
            cascaded_jobs.add(job_id)
    return cascaded_jobs


def _apply_and_recompute(
    state: WorkingState,
    snapshot: TransitionSnapshot,
    updates: list[TaskUpdate],
    now_ms: int,
    *,
    source: task.TransitionSource = task.TransitionSource.WORKER_RECONCILE,
) -> set[JobName]:
    """Single-call convenience: apply pass over ``updates`` then recompute pass.

    Used by batch entry points that process one task-update list (direct
    provider). Multi-worker entry points (reconcile) run the apply pass across
    all workers first, then a single recompute pass.
    """
    acc = _TouchedJobs()
    _apply_transitions(state, snapshot, updates, now_ms, acc, source=source)
    return _recompute_touched_jobs(state, acc, now_ms)


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------


def apply_reconcile_batch(
    snapshot: TransitionSnapshot,
    plan_results: list[tuple[worker.WorkerReconcilePlan, worker.ReconcileResult]],
    now: Timestamp,
) -> ControllerEffects:
    """Apply many workers' reconcile outcomes against a single snapshot.

    The apply pass applies every worker's task transitions (and per-update peer
    cascades) into the shared touched-jobs work-list; the recompute pass then
    recomputes/finalizes every touched job once for the whole batch.
    """
    now_ms = now.epoch_ms()
    state = WorkingState(snapshot=snapshot)
    acc = _TouchedJobs()

    heartbeat_workers = tuple(
        plan.worker_id
        for plan, result in plan_results
        if result.error is None and plan.worker_id in snapshot.active_workers
    )
    if heartbeat_workers:
        state.record_worker_heartbeat(heartbeat_workers)

    for plan, result in plan_results:
        worker_id = plan.worker_id

        if result.error is not None:
            state.record_log_event(
                LogEvent(
                    action="reconcile_rpc_failed",
                    entity_id=str(worker_id),
                    details=(("error", result.error),),
                )
            )
            candidates: list[tuple[JobName, int]] = []
            for desired in plan.request.desired:
                if not desired.HasField("run") or not desired.run.HasField("request"):
                    continue
                req_proto = desired.run.request
                candidates.append((JobName.from_wire(req_proto.task_id), req_proto.attempt_id))
            if not candidates:
                continue
            assigned_updates = worker.assigned_updates_from_plan(snapshot, candidates, result.error)
            if not assigned_updates:
                continue
            _apply_transitions(state, snapshot, assigned_updates, now_ms, acc)
            continue

        if worker_id not in snapshot.active_workers:
            logger.warning(
                "apply_reconcile_batch: worker %s no longer present; dropping %d observations",
                worker_id,
                len(result.observations),
            )
            continue

        observations = worker.filter_observations_to_plan(plan, result.observations, worker_id)
        if not observations:
            continue
        all_updates = worker.observations_to_updates(snapshot, observations)
        if not all_updates:
            continue
        _apply_transitions(state, snapshot, all_updates, now_ms, acc)

    _recompute_touched_jobs(state, acc, now_ms)
    return state.effects


def apply_direct_provider_updates_batch(
    snapshot: TransitionSnapshot,
    updates: list[TaskUpdate],
) -> ControllerEffects:
    """Apply a batch of task-state updates from a direct (e.g. Kubernetes) provider."""
    now_ms = snapshot.now.epoch_ms()
    state = WorkingState(snapshot=snapshot)
    # Direct providers manage their own hosts -> no build-failed reaping.
    cascaded_jobs = _apply_and_recompute(state, snapshot, updates, now_ms, source=task.TransitionSource.DIRECT_PROVIDER)

    if cascaded_jobs:
        state.record_log_event(LogEvent(action="direct_provider_updates_applied", entity_id="direct"))
        for job_id in cascaded_jobs:
            state.record_log_event(
                LogEvent(
                    action="job_terminated",
                    entity_id=job_id.to_wire(),
                    trigger="direct_provider_updates_applied",
                )
            )
    return state.effects


def apply_worker_failures_batch(
    snapshot: TransitionSnapshot,
    failures: list[tuple[WorkerId, str | None, str]],
) -> ControllerEffects:
    """Cascade a batch of worker failures against a single shared overlay.

    Active tasks on the failed workers are derived from the snapshot — the
    loader (``load_closed_snapshot`` seeded by worker) closes them, so the
    batch reads only the snapshot.
    """
    state = WorkingState(snapshot=snapshot)
    now_ms = snapshot.now.epoch_ms()

    # Group the snapshot's active task rows by their failed ``current_worker_id``.
    # Only ACTIVE rows carry a worker id; PENDING rows are unassigned (NULL worker)
    # and so are naturally excluded. Per-worker order follows the snapshot's
    # ``active_tasks_by_job`` ordering.
    failed_worker_ids = {wid for wid, _, _ in failures}
    task_rows_by_worker: dict[WorkerId, list[ActiveTaskRow]] = {wid: [] for wid in failed_worker_ids}
    for rows in snapshot.active_tasks_by_job.values():
        for row in rows:
            if row.current_worker_id in failed_worker_ids:
                task_rows_by_worker[row.current_worker_id].append(row)

    for worker_id, worker_address, error in failures:
        task_rows = task_rows_by_worker.get(worker_id, [])
        for task_row in task_rows:
            task_id = task_row.task_id
            # Overlay-aware prior state: an earlier worker failure (or its peer
            # cascade) in this same batch may have already finalized this task —
            # e.g. a coscheduled sibling spanning two failed workers, or a task
            # both directly held and cascade-targeted. Re-applying from the stale
            # snapshot row would overwrite that mutation, so skip rows the overlay
            # shows are no longer in an active state.
            effective_state = state.task_state(task_id)
            if effective_state is None or effective_state not in ACTIVE_TASK_STATES:
                continue
            prior_state = effective_state
            is_reservation_holder = task_row.is_reservation_holder
            if is_reservation_holder:
                new_task_state = job_pb2.TASK_STATE_PENDING
                preemption_count = task_row.preemption_count
            else:
                new_task_state, preemption_count = task.resolve_task_failure_state(
                    prior_state,
                    task_row.preemption_count,
                    task_row.max_retries_preemption,
                    job_pb2.TASK_STATE_WORKER_FAILED,
                )
            holder_preemption_count = 0 if is_reservation_holder else preemption_count
            task.finalize_attempt(
                state,
                task_id.to_wire(),
                task_row.current_attempt_id,
                new_task_state,
                f"Worker {worker_id} failed: {error}",
                now_ms,
                attempt_state=job_pb2.TASK_STATE_WORKER_FAILED,
                preemption_count=holder_preemption_count,
            )
            parent_job_id, _ = task_id.require_task()
            new_job_state = job.recompute_state(state, parent_job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                reason = f"Worker {worker_id} failed"
                _kill_non_terminal_tasks(state, parent_job_id, reason, now_ms)
                _cascade_to_children(state, parent_job_id, now_ms, reason)
            elif new_task_state == job_pb2.TASK_STATE_PENDING:
                policy = state.job_preemption_policy(parent_job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    _cascade_to_children(
                        state,
                        parent_job_id,
                        now_ms,
                        "Parent task preempted",
                        exclude_reservation_holders=True,
                    )
            # Coscheduled siblings on surviving slice workers.
            if not is_reservation_holder and task_row.has_coscheduling:
                siblings = peers.find_coscheduled_siblings(state, parent_job_id, task_id, True)
                if new_task_state in FAILURE_TASK_STATES:
                    peers.terminate_coscheduled_siblings(state, siblings, task_id, now_ms)
                else:
                    peers.requeue_coscheduled_siblings(state, siblings, task_id, now_ms)

        state.record_worker_make_unhealthy(worker_id)
        state.record_log_event(
            LogEvent(
                action="worker_failed",
                entity_id=str(worker_id),
                details=(
                    ("address", worker_address or "-"),
                    ("error", error),
                ),
            )
        )
    return state.effects


def apply_terminal_decisions_batch(
    snapshot: TransitionSnapshot,
    decisions: list[task.TerminalDecision],
) -> ControllerEffects:
    """Batched terminal-state assertions: preempt / timeout / unschedulable."""
    state = WorkingState(snapshot=snapshot)
    if not decisions:
        return state.effects
    now_ms = snapshot.now.epoch_ms()

    seen_tasks: set[JobName] = set()
    ordered: list[task.TerminalDecision] = []
    for decision in decisions:
        if decision.task_id in seen_tasks:
            continue
        seen_tasks.add(decision.task_id)
        ordered.append(decision)

    # Batch timeout decisions together so the two-phase sibling dedup
    # operates on the full set at once.
    timeout_rows: list[ActiveTaskRow] = []
    timeout_reason: str | None = None
    for decision in ordered:
        if decision.kind is not task.TerminalKind.TIMEOUT:
            continue
        row = task.active_row_from_snapshot(snapshot, decision.task_id)
        if row is None:
            continue
        timeout_rows.append(row)
        if timeout_reason is None:
            timeout_reason = decision.reason
    if timeout_rows and timeout_reason is not None:
        _cascade_timeouts(state, timeout_rows, timeout_reason, now_ms)

    for decision in ordered:
        if decision.kind is task.TerminalKind.PREEMPT:
            row = task.active_row_from_snapshot(snapshot, decision.task_id)
            outcome = task.preempt_one(state, snapshot, decision.task_id, decision.reason, row=row)
            if outcome is None:
                continue
            # Coscheduled retry: bounce siblings back to PENDING so the
            # job re-coschedules atomically.
            if outcome.new_state == job_pb2.TASK_STATE_PENDING and outcome.has_coscheduling:
                siblings = peers.find_coscheduled_siblings(state, outcome.job_id, decision.task_id, True)
                peers.requeue_coscheduled_siblings(state, siblings, decision.task_id, now_ms)
            new_job_state = job.recompute_state(state, outcome.job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                _finalize_terminal_job(state, outcome.job_id, new_job_state, now_ms)
            elif outcome.new_state == job_pb2.TASK_STATE_PENDING:
                policy = state.job_preemption_policy(outcome.job_id)
                if policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
                    _cascade_to_children(
                        state,
                        outcome.job_id,
                        now_ms,
                        decision.reason,
                        exclude_reservation_holders=True,
                    )
            state.record_log_event(
                LogEvent(
                    action="task_preempted",
                    entity_id=decision.task_id.to_wire(),
                    details=(("reason", decision.reason),),
                )
            )
        elif decision.kind is task.TerminalKind.UNSCHEDULABLE:
            job_id = task.unschedulable_one(state, snapshot, decision.task_id, decision.reason)
            if job_id is None:
                continue
            new_job_state = job.recompute_state(state, job_id)
            if new_job_state is not None and new_job_state in TERMINAL_JOB_STATES:
                _finalize_terminal_job(state, job_id, new_job_state, now_ms)
            state.record_log_event(
                LogEvent(
                    action="task_unschedulable",
                    entity_id=decision.task_id.to_wire(),
                    details=(("reason", decision.reason),),
                )
            )
        # TIMEOUT handled above.
    return state.effects


def _cascade_timeouts(
    state: WorkingState,
    rows: list[ActiveTaskRow],
    reason: str,
    now_ms: int,
) -> None:
    """Two-phase timeout cascade with sibling dedup."""
    if not rows:
        return
    direct_task_wires: set[str] = set()
    siblings_by_job: dict[str, list[ActiveTaskRow]] = {}

    for row in rows:
        task_id_wire = row.task_id.to_wire()
        direct_task_wires.add(task_id_wire)
        job_id_wire = row.job_id.to_wire()
        siblings = peers.find_coscheduled_siblings(state, row.job_id, row.task_id, row.has_coscheduling)
        if siblings:
            existing = siblings_by_job.get(job_id_wire, [])
            existing.extend(siblings)
            siblings_by_job[job_id_wire] = existing

    # Deduplicate: drop siblings that will already be terminated as direct
    # victims; dedupe across multiple trigger tasks within the same job.
    for job_id_wire, siblings in siblings_by_job.items():
        seen: set[str] = set()
        deduped: list[ActiveTaskRow] = []
        for sib in siblings:
            sib_wire = sib.task_id.to_wire()
            if sib_wire not in direct_task_wires and sib_wire not in seen:
                seen.add(sib_wire)
                deduped.append(sib)
        siblings_by_job[job_id_wire] = deduped

    jobs_to_update: set[str] = set()
    task_ids_to_log: set[JobName] = set()

    for row in rows:
        tid = row.task_id
        task_ids_to_log.add(tid)
        task.timeout_one(state, row, reason, now_ms)
        jobs_to_update.add(row.job_id.to_wire())

    for job_id_wire, siblings in siblings_by_job.items():
        if not siblings:
            continue
        cause_tid = next(r.task_id for r in rows if r.job_id.to_wire() == job_id_wire)
        peers.terminate_coscheduled_siblings(state, siblings, cause_tid, now_ms)
        for sib in siblings:
            task_ids_to_log.add(sib.task_id)
        jobs_to_update.add(job_id_wire)

    for job_wire in jobs_to_update:
        new_job_state = job.recompute_state(state, JobName.from_wire(job_wire))
        if new_job_state in TERMINAL_JOB_STATES:
            _finalize_terminal_job(state, JobName.from_wire(job_wire), new_job_state, now_ms)

    for tid in task_ids_to_log:
        state.record_log_event(
            LogEvent(
                action="task_timeout",
                entity_id=tid.to_wire(),
                details=(("reason", reason),),
            )
        )


def apply_cancel_job_batch(
    snapshot: TransitionSnapshot,
    job_id: JobName,
    reason: str,
    now: Timestamp,
) -> ControllerEffects:
    """Cancel ``job_id`` and its full transitive descendant subtree.

    The subtree is derived from the snapshot's ``job_descendants`` — the
    loader closes it, so the batch reads only the snapshot.
    """
    state = WorkingState(snapshot=snapshot)
    descendants = snapshot.job_descendants.get(job_id)
    if descendants is None:
        return state.effects
    subtree = [job_id, *descendants.descendants_full]
    now_ms = now.epoch_ms()
    finished_at = Timestamp.from_ms(now_ms)

    for jid in subtree:
        _kill_non_terminal_tasks(state, jid, reason, now_ms)

        state.record_cascade_kill(
            JobRowDelta(
                job_id=jid,
                state=job_pb2.JOB_STATE_KILLED,
                error=reason,
                finished_at=finished_at,
                is_cascade_kill=True,
                allow_overwrite_worker_failed=True,
            )
        )

        # Coscheduled cascade for tasks just killed in this job.
        jc = state.job_config(jid)
        if jc is None or not jc.has_coscheduling:
            continue
        for row in snapshot.active_tasks_by_job.get(jid, ()):
            siblings = peers.find_coscheduled_siblings(state, row.job_id, row.task_id, row.has_coscheduling)
            if siblings:
                peers.terminate_coscheduled_siblings(state, siblings, row.task_id, now_ms)

    state.record_log_event(
        LogEvent(
            action="job_cancelled",
            entity_id=job_id.to_wire(),
            details=(("reason", reason),),
        )
    )
    return state.effects
