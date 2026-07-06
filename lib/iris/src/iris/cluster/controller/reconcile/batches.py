# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reconcile kernel facade: snapshot in → ``ControllerEffects`` out.

:class:`ReconcileState` is one batch session over a closed
:class:`TransitionSnapshot`. Its public methods are the controller's state
operations — ``reconcile``, ``fail_workers``, ``record_updates``,
``finalize_tasks``, ``cancel_job`` — each of which runs the same
two-pass contract and returns the accumulated effects:

* **apply pass** — per task-update or controller-asserted outcome: apply the
  per-task mutation, cascade to coscheduled peers (so later items in the batch
  observe the terminate/requeue), and record the job on a touched-jobs work-list
  (plus any deferred PENDING-rollback child cascade).
* **recompute pass** (``_recompute_and_finalize``) — recompute each touched job
  once, finalize the ones that go terminal, then drain the deferred child
  cascades. Folding recompute out of the per-update loop makes a batch
  order-independent and keeps ``job.recompute_state`` (which rescans a job's
  whole task histogram) off the O(tasks_per_job²) per-dispatch path.

The cross-aggregate primitives below the facade (``_kill_non_terminal_tasks``,
``_cascade_to_children``, ``_finalize_terminal_job``, ``_cascade_to_peers``) stay
free functions over an :class:`Overlay` so they remain unit-testable in isolation.
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
from iris.cluster.controller.reconcile.overlay import Overlay
from iris.cluster.controller.reconcile.policy import (
    FAILURE_TASK_STATES,
    NON_TERMINAL_TASK_STATES,
    TERMINAL_STATE_REASONS,
)
from iris.cluster.controller.reconcile.snapshot import (
    TaskUpdate,
    TransitionSnapshot,
)
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, ActiveTaskRow
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
    WorkerId,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-aggregate primitives (free functions over an Overlay)
# ---------------------------------------------------------------------------
#
# Job-aggregate cascades live here (not in job.py) because they invoke
# ``task.merge_task_termination``, and job.py is forbidden from importing task.


def _kill_non_terminal_tasks(overlay: Overlay, job_id: JobName, reason: str, now_ms: int) -> None:
    """Kill all non-terminal tasks for a single job and delete endpoints."""
    for row in overlay.active_tasks_for_job(job_id, states=NON_TERMINAL_TASK_STATES):
        task.merge_task_termination(
            overlay,
            row.task_id.to_wire(),
            row.current_attempt_id,
            job_pb2.TASK_STATE_KILLED,
            reason,
            now_ms,
            stamp_attempt_finished=False,
        )


def _cascade_to_children(
    overlay: Overlay,
    job_id: JobName,
    now_ms: int,
    reason: str,
) -> None:
    """Kill descendant jobs (not the job itself) on a parent terminal/preempt."""
    descendants = overlay.job_descendants(job_id)
    for child_job_id in descendants:
        _kill_non_terminal_tasks(overlay, child_job_id, reason, now_ms)
        overlay.merge_cascade_kill(
            JobRowDelta(
                job_id=child_job_id,
                state=job_pb2.JOB_STATE_KILLED,
                error=reason,
                finished_at=Timestamp.from_ms(now_ms),
                is_cascade_kill=True,
            )
        )


def _finalize_terminal_job(overlay: Overlay, job_id: JobName, terminal_state: int, now_ms: int) -> None:
    """Kill remaining tasks and optionally cascade to children when a job goes terminal."""
    reason = TERMINAL_STATE_REASONS.get(terminal_state, "Job finalized")
    _kill_non_terminal_tasks(overlay, job_id, reason, now_ms)
    should_cascade = True
    if terminal_state != job_pb2.JOB_STATE_SUCCEEDED:
        policy = overlay.job_preemption_policy(job_id)
        should_cascade = policy == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    if should_cascade:
        _cascade_to_children(overlay, job_id, now_ms, reason)


def _cascade_to_peers(overlay: Overlay, outcome: task.TransitionOutcome, now_ms: int) -> None:
    """Coscheduled-sibling cascade for one transition. No job recompute."""
    if not outcome.cascade_to_peers:
        return
    siblings = peers.find_coscheduled_siblings(overlay, outcome.job_id, outcome.task_id)
    if outcome.new_task_state in FAILURE_TASK_STATES:
        peers.terminate_coscheduled_siblings(overlay, siblings, outcome.task_id, now_ms)
    else:
        peers.requeue_coscheduled_siblings(overlay, siblings, outcome.task_id, now_ms)


# ---------------------------------------------------------------------------
# The batch facade
# ---------------------------------------------------------------------------


@dataclass
class ReconcileState:
    """One batch session over a closed snapshot.

    Holds the mutable :class:`Overlay` plus the two cross-batch accumulators the
    two-pass contract threads through every operation:

    * ``touched`` — insertion-ordered, deduped work-list of jobs with a
      state-changing task transition. The recompute pass recomputes each once.
    * ``pending_child_cascades`` — jobs whose parent task rolled back to PENDING
      under a ``TERMINATE_CHILDREN`` policy; their descendant cascade is deferred
      (deduped per job) so it runs once after every sibling in the batch settled.
    """

    overlay: Overlay
    touched: list[JobName] = field(default_factory=list)
    pending_child_cascades: dict[JobName, str] = field(default_factory=dict)
    _touched_seen: set[JobName] = field(default_factory=set)

    @classmethod
    def open(cls, snapshot: TransitionSnapshot) -> "ReconcileState":
        return cls(overlay=Overlay(snapshot))

    @property
    def _snapshot(self) -> TransitionSnapshot:
        return self.overlay.snapshot

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def reconcile(
        self,
        plan_results: list[tuple[worker.WorkerReconcilePlan, worker.WorkerReconcileResult]],
        now: Timestamp,
    ) -> ControllerEffects:
        """Apply many workers' reconcile outcomes against the shared overlay.

        Each worker's task updates are applied (with their per-update peer
        cascades) in turn, so a sibling requeued/terminated by an earlier worker
        is visible to a later worker's overlay-aware guards; the recompute pass
        then recomputes/finalizes every touched job once for the whole batch.
        """
        now_ms = now.epoch_ms()
        # Liveness (REACHED/UNREACHABLE) is observed by the backend from its own
        # RPC outcomes and folded by the controller through
        # ``WorkerHealthTracker.apply``; the kernel only derives build failures.
        for plan, result in plan_results:
            for update in self._reconcile_updates_for_plan(plan, result):
                self._apply_update(update, now_ms, source=task.TransitionSource.WORKER_RECONCILE)

        self._recompute_and_finalize(now_ms)
        return self.overlay.effects

    def record_updates(self, updates: list[TaskUpdate]) -> ControllerEffects:
        """Apply a batch of task-state updates from a direct (e.g. Kubernetes) provider."""
        now_ms = self._snapshot.now.epoch_ms()
        # Direct providers manage their own hosts -> no build-failed reaping.
        for update in updates:
            self._apply_update(update, now_ms, source=task.TransitionSource.DISPATCH)
        cascaded_jobs = self._recompute_and_finalize(now_ms)

        if cascaded_jobs:
            self.overlay.emit_log_event(LogEvent(action="dispatch_updates_applied", entity_id="direct"))
            for job_id in cascaded_jobs:
                self.overlay.emit_log_event(
                    LogEvent(
                        action="job_terminated",
                        entity_id=job_id.to_wire(),
                        trigger="dispatch_updates_applied",
                    )
                )
        return self.overlay.effects

    def fail_workers(self, failures: list[tuple[WorkerId, str | None, str]]) -> ControllerEffects:
        """Cascade a batch of worker failures against the shared overlay.

        Active tasks on the failed workers are derived from the snapshot — the
        loader (``load_closed_snapshot`` seeded by worker) closes them, so the
        batch reads only the snapshot.
        """
        now_ms = self._snapshot.now.epoch_ms()
        rows_by_worker = self._active_rows_by_failed_worker({wid for wid, _, _ in failures})

        for worker_id, worker_address, error in failures:
            for task_row in rows_by_worker.get(worker_id, []):
                outcome = self._fail_one_task(task_row, worker_id, error, now_ms)
                if outcome is not None:
                    self._fan_out(outcome, child_reason="Parent task preempted", now_ms=now_ms)
            # No health mutation here: the controller has already decided this
            # worker is dead and forgets it once removal commits.
            self.overlay.emit_log_event(
                LogEvent(
                    action="worker_failed",
                    entity_id=str(worker_id),
                    details=(("address", worker_address or "-"), ("error", error)),
                )
            )

        self._recompute_and_finalize(now_ms)
        return self.overlay.effects

    def finalize_tasks(self, decisions: list[task.TerminalDecision]) -> ControllerEffects:
        """Batched terminal-state assertions: preempt / timeout / unschedulable."""
        if not decisions:
            return self.overlay.effects
        now_ms = self._snapshot.now.epoch_ms()

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
            row = task.active_row_from_snapshot(self._snapshot, decision.task_id)
            if row is None:
                continue
            timeout_rows.append(row)
            if timeout_reason is None:
                timeout_reason = decision.reason
        if timeout_rows and timeout_reason is not None:
            self._cascade_timeouts(timeout_rows, timeout_reason, now_ms)

        for decision in ordered:
            if decision.kind is task.TerminalKind.PREEMPT:
                self._apply_preempt_decision(decision, now_ms)
            elif decision.kind is task.TerminalKind.UNSCHEDULABLE:
                self._apply_unschedulable_decision(decision)
            # TIMEOUT handled above.

        self._recompute_and_finalize(now_ms)
        return self.overlay.effects

    def cancel_job(self, job_id: JobName, reason: str, now: Timestamp) -> ControllerEffects:
        """Cancel ``job_id`` and its full transitive descendant subtree.

        The subtree is derived from the snapshot's ``job_descendants`` — the
        loader closes it, so the batch reads only the snapshot. Killing every
        job's tasks covers all coscheduled siblings too (siblings always live in
        the same job), so no separate peer cascade is needed: by the time a job's
        tasks are killed, ``find_coscheduled_siblings`` would find none active.
        """
        descendants = self._snapshot.job_descendants.get(job_id)
        if descendants is None:
            return self.overlay.effects
        subtree = [job_id, *descendants.descendants]
        now_ms = now.epoch_ms()
        finished_at = Timestamp.from_ms(now_ms)

        for jid in subtree:
            _kill_non_terminal_tasks(self.overlay, jid, reason, now_ms)
            self.overlay.merge_cascade_kill(
                JobRowDelta(
                    job_id=jid,
                    state=job_pb2.JOB_STATE_KILLED,
                    error=reason,
                    finished_at=finished_at,
                    is_cascade_kill=True,
                    allow_overwrite_worker_failed=True,
                )
            )

        self.overlay.emit_log_event(
            LogEvent(action="job_cancelled", entity_id=job_id.to_wire(), details=(("reason", reason),))
        )
        return self.overlay.effects

    # ------------------------------------------------------------------
    # Two-pass contract (shared by every operation)
    # ------------------------------------------------------------------

    def _apply_update(self, update: TaskUpdate, now_ms: int, *, source: task.TransitionSource) -> None:
        """Apply pass for one worker/provider task update.

        Applies the per-update transition, runs the peer cascade unconditionally
        (so later updates see requeued/terminated siblings), but gates the
        recompute work-list note + deferred child cascade on an actual state
        change: ``apply_one_transition`` emits no-op outcomes (new data, unchanged
        state) that must not touch the work-list.
        """
        outcome = task.apply_one_transition(self.overlay, self._snapshot, update, now_ms, source=source)
        if outcome is None:
            return
        _cascade_to_peers(self.overlay, outcome, now_ms)
        if outcome.new_task_state != outcome.prior_state:
            self._note(outcome.job_id)
            self._defer_pending_child_cascade(outcome, "Parent task retried")

    def _fan_out(self, outcome: task.TransitionOutcome, *, child_reason: str, now_ms: int) -> None:
        """Apply pass for one controller-asserted transition (failure / preempt / timeout).

        Same three steps as :meth:`_apply_update`, but the note/defer is
        unconditional: controller-asserted callers only build an outcome for a
        real transition, so there is no no-op outcome to gate against.
        """
        _cascade_to_peers(self.overlay, outcome, now_ms)
        self._note(outcome.job_id)
        self._defer_pending_child_cascade(outcome, child_reason)

    def _recompute_and_finalize(self, now_ms: int) -> set[JobName]:
        """Recompute pass: recompute each touched job once; finalize terminals; drain cascades.

        Returns the set of jobs that went terminal in this batch. PENDING
        rollbacks that did not take the job terminal still get their
        ``TERMINATE_CHILDREN`` descendant cascade here, skipping any job already
        finalized above so children never cascade twice.
        """
        cascaded_jobs: set[JobName] = set()
        for job_id in self.touched:
            new_job_state = job.recompute_state(self.overlay, job_id)
            if new_job_state in TERMINAL_JOB_STATES:
                _finalize_terminal_job(self.overlay, job_id, new_job_state, now_ms)
                cascaded_jobs.add(job_id)
        for job_id, reason in self.pending_child_cascades.items():
            if job_id in cascaded_jobs:
                continue
            _cascade_to_children(self.overlay, job_id, now_ms, reason)
        return cascaded_jobs

    def _note(self, job_id: JobName) -> None:
        """Add ``job_id`` to the deduped, insertion-ordered recompute work-list."""
        if job_id not in self._touched_seen:
            self._touched_seen.add(job_id)
            self.touched.append(job_id)

    def _defer_pending_child_cascade(self, outcome: task.TransitionOutcome, reason: str) -> None:
        """Queue the descendant cascade for a PENDING rollback under TERMINATE_CHILDREN.

        Drained by :meth:`_recompute_and_finalize` so it runs once per job after
        every sibling in the batch has settled.
        """
        if outcome.new_task_state != job_pb2.TASK_STATE_PENDING:
            return
        if self.overlay.job_preemption_policy(outcome.job_id) == job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN:
            self.pending_child_cascades.setdefault(outcome.job_id, reason)

    # ------------------------------------------------------------------
    # reconcile() helpers
    # ------------------------------------------------------------------

    def _reconcile_updates_for_plan(
        self,
        plan: worker.WorkerReconcilePlan,
        result: worker.WorkerReconcileResult,
    ) -> list[TaskUpdate]:
        """Derive the task updates one worker's reconcile result contributes."""
        worker_id = plan.worker_id

        if result.error is not None:
            self.overlay.emit_log_event(
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
                cand_task_id = JobName.from_wire(req_proto.task_id)
                # Overlay-aware gate: a sibling already requeued to PENDING earlier
                # in this same batch is no longer ASSIGNED, so it must not be
                # fabricated into a synthetic WORKER_FAILED (split-slice corruption).
                # ``assigned_updates_from_plan`` re-checks the snapshot, but that
                # read is blind to same-batch overlay mutations.
                if self.overlay.task_state(cand_task_id) != job_pb2.TASK_STATE_ASSIGNED:
                    continue
                candidates.append((cand_task_id, req_proto.attempt_id))
            if not candidates:
                return []
            return worker.assigned_updates_from_plan(self._snapshot, candidates, result.error)

        if worker_id not in self._snapshot.active_workers:
            logger.warning(
                "reconcile: worker %s no longer present; dropping %d observations",
                worker_id,
                len(result.observations),
            )
            return []

        observations = worker.filter_observations_to_plan(plan, result.observations, worker_id)
        if not observations:
            return []
        return worker.observations_to_updates(self._snapshot, observations)

    # ------------------------------------------------------------------
    # fail_workers() helpers
    # ------------------------------------------------------------------

    def _active_rows_by_failed_worker(self, failed_worker_ids: set[WorkerId]) -> dict[WorkerId, list[ActiveTaskRow]]:
        """Group the snapshot's active task rows by their failed ``current_worker_id``.

        Only ACTIVE rows carry a worker id; PENDING rows are unassigned (NULL
        worker) and so are naturally excluded. Per-worker order follows the
        snapshot's ``active_tasks_by_job`` ordering.
        """
        rows_by_worker: dict[WorkerId, list[ActiveTaskRow]] = {wid: [] for wid in failed_worker_ids}
        for rows in self._snapshot.active_tasks_by_job.values():
            for row in rows:
                wid = row.current_worker_id
                if wid is not None and wid in failed_worker_ids:
                    rows_by_worker[wid].append(row)
        return rows_by_worker

    def _fail_one_task(
        self,
        task_row: ActiveTaskRow,
        worker_id: WorkerId,
        error: str,
        now_ms: int,
    ) -> task.TransitionOutcome | None:
        """Finalize one task whose worker failed; return its cross-aggregate outcome.

        Returns ``None`` when the overlay shows the task is no longer active —
        an earlier worker failure (or its peer cascade) in this same batch may
        have already finalized it (a coscheduled sibling spanning two failed
        workers, or a task both directly held and cascade-targeted). Re-applying
        from the stale snapshot row would overwrite that mutation.
        """
        task_id = task_row.task_id
        effective_state = self.overlay.task_state(task_id)
        if effective_state is None or effective_state not in ACTIVE_TASK_STATES:
            return None
        prior_state = effective_state
        new_task_state = task.resolve_task_failure_state(
            prior_state,
            task_row.preemption_count,
            task_row.max_retries_preemption,
            job_pb2.TASK_STATE_WORKER_FAILED,
        )
        # The worker is gone, so the attempt is truly done: finalize it (stamp
        # finished_at) rather than leaving it for a status update that will never
        # arrive.
        task.merge_task_termination(
            self.overlay,
            task_id.to_wire(),
            task_row.current_attempt_id,
            new_task_state,
            f"Worker {worker_id} failed: {error}",
            now_ms,
            stamp_attempt_finished=True,
            attempt_state=job_pb2.TASK_STATE_WORKER_FAILED,
        )
        parent_job_id, _ = task_id.require_task()
        return task.TransitionOutcome(
            task_id=task_id,
            job_id=parent_job_id,
            prior_state=prior_state,
            new_task_state=new_task_state,
            cascade_to_peers=task_row.has_coscheduling,
        )

    # ------------------------------------------------------------------
    # finalize_tasks() helpers
    # ------------------------------------------------------------------

    def _apply_preempt_decision(self, decision: task.TerminalDecision, now_ms: int) -> None:
        # Overlay-aware: skip a task an earlier same-batch decision (e.g. a
        # timeout sibling cascade) already moved out of an active state.
        effective_state = self.overlay.task_state(decision.task_id)
        if effective_state is None or effective_state not in ACTIVE_TASK_STATES:
            return
        row = task.active_row_from_snapshot(self._snapshot, decision.task_id)
        outcome = task.preempt_one(self.overlay, self._snapshot, decision.task_id, decision.reason, row=row)
        if outcome is None:
            return
        self._fan_out(
            task.TransitionOutcome(
                task_id=decision.task_id,
                job_id=outcome.job_id,
                prior_state=effective_state,
                new_task_state=outcome.new_task_state,
                cascade_to_peers=outcome.cascade_to_peers,
            ),
            child_reason=decision.reason,
            now_ms=now_ms,
        )
        self.overlay.emit_log_event(
            LogEvent(
                action="task_preempted",
                entity_id=decision.task_id.to_wire(),
                details=(("reason", decision.reason),),
            )
        )

    def _apply_unschedulable_decision(self, decision: task.TerminalDecision) -> None:
        # A still-PENDING task is the normal unschedulable target, so (unlike
        # preempt) don't require an active state — only skip a task an earlier
        # same-batch decision already drove terminal.
        effective_state = self.overlay.task_state(decision.task_id)
        if effective_state is not None and effective_state in TERMINAL_TASK_STATES:
            return
        job_id = task.unschedulable_one(self.overlay, self._snapshot, decision.task_id, decision.reason)
        if job_id is None:
            return
        self._note(job_id)
        self.overlay.emit_log_event(
            LogEvent(
                action="task_unschedulable",
                entity_id=decision.task_id.to_wire(),
                details=(("reason", decision.reason),),
            )
        )

    def _cascade_timeouts(self, rows: list[ActiveTaskRow], reason: str, now_ms: int) -> None:
        """Two-phase timeout cascade with sibling dedup.

        Notes touched jobs; the recompute pass finalizes any that go terminal.
        """
        if not rows:
            return
        direct_task_wires: set[str] = set()
        siblings_by_job: dict[str, list[ActiveTaskRow]] = {}

        for row in rows:
            direct_task_wires.add(row.task_id.to_wire())
            job_id_wire = row.job_id.to_wire()
            if not row.has_coscheduling:
                continue
            siblings = peers.find_coscheduled_siblings(self.overlay, row.job_id, row.task_id)
            if siblings:
                siblings_by_job.setdefault(job_id_wire, []).extend(siblings)

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

        task_ids_to_log: set[JobName] = set()

        for row in rows:
            task_ids_to_log.add(row.task_id)
            task.timeout_one(self.overlay, row, reason, now_ms)
            self._note(row.job_id)

        for job_id_wire, siblings in siblings_by_job.items():
            if not siblings:
                continue
            cause_tid = next(r.task_id for r in rows if r.job_id.to_wire() == job_id_wire)
            peers.terminate_coscheduled_siblings(self.overlay, siblings, cause_tid, now_ms)
            for sib in siblings:
                task_ids_to_log.add(sib.task_id)
            self._note(JobName.from_wire(job_id_wire))

        for tid in task_ids_to_log:
            self.overlay.emit_log_event(
                LogEvent(action="task_timeout", entity_id=tid.to_wire(), details=(("reason", reason),))
            )
