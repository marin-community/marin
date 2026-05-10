# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for issue #5470: TPU placement collision after preemption.

Verify the scheduler does not reassign a TPU slice while sibling kill RPCs
for the slice are still in flight. Worker capacity is derived from
unfinished worker-bound attempts (``finished_at_ms IS NULL``), so
producer-side cancel/preempt does not free the slice — only the heartbeat
path's terminal finalization does.

Production incident timeline (Incident B, v5p-256):
  09:14:31  lr0.5  assigned -> slice 389585fe
  09:14:34  lr0.67 assigned -> slice 2e06c8f1  (separate slice, correct)
  18:43:32  lr0.5  reassigned -> slice 8996b868 (after preemption)
  18:43:57  lr0.5  TERMINATED -- task 28 hit 502 during uv sync
  18:43:58  lr0.67 reassigned -> slice 8996b868 (COLLISION -- 1s after decommit)
"""

import pytest
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.codec import constraints_from_json, resource_spec_from_scalars
from iris.cluster.controller.controller import SchedulingOutcome
from iris.cluster.controller.scheduler import JobRequirements, Scheduler, worker_snapshot_from_row
from iris.cluster.controller.stores import WorkerResourceUsage
from iris.cluster.controller.transitions import (
    Assignment,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2

from .conftest import (
    building_counts as _building_counts,
)
from .conftest import (
    check_task_can_be_scheduled,
    healthy_active_workers,
    make_test_entrypoint,
    make_worker_metadata,
    query_tasks_for_job,
    register_worker,
    submit_job,
)
from .conftest import query_job as _query_job
from .conftest import query_task as _query_task

CHIPS_PER_VM = 4
VMS_PER_SLICE = 8


def _make_v5p_worker(tpu_name: str, worker_idx: int):
    meta = make_worker_metadata(cpu=208, memory_bytes=448 * 1024**3, tpu_name="v5p-64")
    meta.device.tpu.count = CHIPS_PER_VM
    meta.attributes[WellKnownAttribute.TPU_NAME].string_value = tpu_name
    meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = worker_idx
    return meta


def _register_slice(state, name, num_vms=VMS_PER_SLICE):
    wids = []
    for i in range(num_vms):
        wid = f"{name}-w{i}"
        register_worker(state, wid, f"10.0.{hash(name) % 256}.{i}", _make_v5p_worker(name, i))
        wids.append(WorkerId(wid))
    return wids


def _make_gang_request(name):
    req = controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=32_000,
            memory_bytes=128 * 1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-64", count=CHIPS_PER_VM)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=VMS_PER_SLICE,
        max_retries_preemption=1000,
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    return req


def _job_requirements_from_job(job):
    return JobRequirements(
        resources=resource_spec_from_scalars(
            job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json
        ),
        constraints=constraints_from_json(job.constraints_json),
        is_coscheduled=job.has_coscheduling,
        coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
    )


def _read_usage_by_worker(state) -> dict[WorkerId, WorkerResourceUsage]:
    """Snapshot the derived resource-usage map (replaces workers.committed_*)."""
    with state._db.read_snapshot() as snap:
        return state._store.attempts.resource_usage_by_worker(snap)


def _tpu_used(usage_map: dict[WorkerId, WorkerResourceUsage], wid: WorkerId) -> int:
    """Look up tpu chips currently reserved for ``wid``; absent worker means 0."""
    entry = usage_map.get(wid)
    return entry.tpu_count if entry is not None else 0


def _build_context(scheduler, state):
    pending = _schedulable_tasks_for_test(state)
    workers = list(healthy_active_workers(state))
    bc = _building_counts(state)
    task_ids = []
    jobs = {}
    for task in pending:
        if not check_task_can_be_scheduled(task):
            continue
        task_ids.append(task.task_id)
        if task.job_id not in jobs:
            job = _query_job(state, task.job_id)
            if job:
                jobs[task.job_id] = _job_requirements_from_job(job)
    usage = _read_usage_by_worker(state)
    snapshots = [worker_snapshot_from_row(w, usage.get(w.worker_id)) for w in workers]
    return scheduler.create_scheduling_context(snapshots, building_counts=bc, pending_tasks=task_ids, jobs=jobs)


def _schedulable_tasks_for_test(state):
    from .conftest import schedulable_tasks as _schedulable_tasks

    return _schedulable_tasks(state)


def _schedule_and_commit(scheduler, state):
    ctx = _build_context(scheduler, state)
    result = scheduler.find_assignments(ctx)
    for tid, wid in result.assignments:
        task = _query_task(state, tid)
        if task:
            with state._store.transaction() as cur:
                state.queue_assignments(cur, [Assignment(task_id=tid, worker_id=wid)])
    return result


def _transition_to_running(state, task):
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=task.current_worker_id,
                updates=[
                    TaskUpdate(
                        task_id=task.task_id, attempt_id=task.current_attempt_id, new_state=job_pb2.TASK_STATE_RUNNING
                    )
                ],
            ),
        )


def _heartbeat_killed(state, task):
    """Synthesize a terminal heartbeat for ``task``: that is what releases capacity now."""
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=task.current_worker_id,
                updates=[
                    TaskUpdate(
                        task_id=task.task_id,
                        attempt_id=task.current_attempt_id,
                        new_state=job_pb2.TASK_STATE_KILLED,
                    )
                ],
            ),
        )


def _worker_fail_one_task(state, task):
    """Send WORKER_FAILED for one task. For coscheduled jobs this triggers
    _requeue_coscheduled_siblings which bounces all siblings to PENDING."""
    with state._store.transaction() as cur:
        result = state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=task.current_worker_id,
                updates=[
                    TaskUpdate(
                        task_id=task.task_id,
                        attempt_id=task.current_attempt_id,
                        new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                        error="502 Bad Gateway downloading Python (simulated GCP preemption)",
                    )
                ],
            ),
        )
    return result


def _assigned_workers_by_job(assignments):
    """Group assigned worker IDs by parent job."""
    by_job: dict[JobName, set[WorkerId]] = {}
    for tid, wid in assignments:
        by_job.setdefault(tid.parent, set()).add(wid)
    return by_job


def _mark_slice_unhealthy(state, prefix):
    for i in range(VMS_PER_SLICE):
        wid = WorkerId(f"{prefix}-w{i}")
        state._store.workers.set_health_for_test(wid, healthy=False)


@pytest.fixture
def scheduler():
    return Scheduler()


class TestPreemptionReassignment:
    """Capacity stays held by unfinished worker-bound attempts until heartbeats finalize them."""

    def _setup_two_gangs_running(self, scheduler_or_ctrl, state):
        """Common setup: two slices, two gangs, all running. Returns (job_a_id, job_b_id)."""
        _register_slice(state, "slice-1")
        _register_slice(state, "slice-2")

        submit_job(state, "job-a", _make_gang_request("train-a"))
        submit_job(state, "job-b", _make_gang_request("train-b"))

        job_a_id = JobName.root("test-user", "job-a")
        job_b_id = JobName.root("test-user", "job-b")

        if isinstance(scheduler_or_ctrl, Scheduler):
            result = _schedule_and_commit(scheduler_or_ctrl, state)
            assert len(result.assignments) == VMS_PER_SLICE * 2
            by_job = _assigned_workers_by_job(result.assignments)
            assert len(by_job) == 2
            assert not (by_job[job_a_id] & by_job[job_b_id]), "gangs must start on separate slices"
        else:
            outcome = scheduler_or_ctrl._run_scheduling()
            assert outcome == SchedulingOutcome.ASSIGNMENTS_MADE

        for task in query_tasks_for_job(state, job_a_id):
            _transition_to_running(state, task)
        for task in query_tasks_for_job(state, job_b_id):
            _transition_to_running(state, task)

        return job_a_id, job_b_id

    def _preempt_and_new_slice(self, state, job_a_id, job_b_id):
        """Preempt both gangs, mark old slices unhealthy, register slice-3.

        Under the new contract the trigger task IS finalized via the
        heartbeat path that delivered WORKER_FAILED, but the siblings
        bounced by ``_requeue_coscheduled_siblings`` use
        ``finalize_attempt=False``. So one worker in each slice has its
        capacity released; the other 7 hold ``CHIPS_PER_VM`` until their
        own terminal heartbeats arrive.
        """
        tasks_a = query_tasks_for_job(state, job_a_id)
        tasks_b = query_tasks_for_job(state, job_b_id)
        trigger_a = tasks_a[0]
        trigger_b = tasks_b[0]
        _worker_fail_one_task(state, trigger_a)
        _worker_fail_one_task(state, trigger_b)

        usage = _read_usage_by_worker(state)
        # Trigger workers were finalized by their WORKER_FAILED heartbeat
        # (heartbeat path is the only finalizer).
        assert _tpu_used(usage, trigger_a.current_worker_id) == 0
        assert _tpu_used(usage, trigger_b.current_worker_id) == 0
        # Sibling workers still hold their reservations because the producer-
        # side requeue path leaves the attempt unfinished.
        for i in range(VMS_PER_SLICE):
            wid = WorkerId(f"slice-1-w{i}")
            if wid == trigger_a.current_worker_id:
                continue
            assert _tpu_used(usage, wid) == CHIPS_PER_VM, f"slice-1-w{i} should still be reserved"
        for i in range(VMS_PER_SLICE):
            wid = WorkerId(f"slice-2-w{i}")
            if wid == trigger_b.current_worker_id:
                continue
            assert _tpu_used(usage, wid) == CHIPS_PER_VM, f"slice-2-w{i} should still be reserved"

        _mark_slice_unhealthy(state, "slice-1")
        _mark_slice_unhealthy(state, "slice-2")
        _register_slice(state, "slice-3")

    def test_scheduler_does_not_reassign_until_heartbeats_finalize_old_slice(self, scheduler, state):
        """Under the new derived-usage contract, the scheduler cannot place a
        gang onto a slice whose attempts are still unfinished. Even after
        ``_requeue_coscheduled_siblings`` writes ``tasks.state=PENDING``, the
        attempts on the old slice keep their workers' chips reserved until
        terminal heartbeats arrive — so the second gang slots into slice-3 and
        cannot collide with slice-1's still-running processes.
        """
        job_a_id, _job_b_id = self._setup_two_gangs_running(scheduler, state)
        self._preempt_and_new_slice(state, job_a_id, _job_b_id)

        # Scheduler assigns the first gang back to slice-3 (the only fully-free slice).
        result1 = _schedule_and_commit(scheduler, state)
        assert len(result1.assignments) == VMS_PER_SLICE
        by_job1 = _assigned_workers_by_job(result1.assignments)
        assert len(by_job1) == 1
        first_job = next(iter(by_job1))
        slice3_workers = {WorkerId(f"slice-3-w{i}") for i in range(VMS_PER_SLICE)}
        assert by_job1[first_job] == slice3_workers, "first reassignment must land on slice-3"

        # First gang fails during BUILDING -> producer-side requeue marks
        # tasks.state PENDING but does NOT stamp finished_at_ms on slice-3
        # attempts (siblings) — so slice-3's chips remain reserved.
        first_job_tasks = query_tasks_for_job(state, first_job)
        fail_result = _worker_fail_one_task(state, first_job_tasks[0])
        assert fail_result.tasks_to_kill, "Coscheduled requeue should produce kill targets"

        usage_after_fail = _read_usage_by_worker(state)
        # The trigger task's worker was finalized by its terminal heartbeat.
        trigger_wid = first_job_tasks[0].current_worker_id
        assert _tpu_used(usage_after_fail, trigger_wid) == 0
        # The other slice-3 workers still hold the bounced siblings' chips.
        for wid in slice3_workers:
            if wid == trigger_wid:
                continue
            assert (
                _tpu_used(usage_after_fail, wid) == CHIPS_PER_VM
            ), f"slice-3 sibling on {wid} must keep its reservation until heartbeat finalizes the attempt"

        # Without the old decommit, the scheduler does not see free capacity:
        # only one worker on slice-3 has freed up, but a coscheduled gang needs
        # all VMS_PER_SLICE in the same group, so it stays pending.
        result2 = _schedule_and_commit(scheduler, state)
        assert result2.assignments == [], "Scheduler must not reassign while sibling attempts are still unfinished"

        # Synthesize terminal heartbeats for the remaining slice-3 attempts —
        # this is what the worker would emit after auto-killing the stale
        # processes via reconcile. The bounced tasks went back to PENDING
        # with current_worker_id=NULL on the tasks row, but their attempt
        # rows still reference the old worker. We finalize each unfinished
        # attempt on slice-3 directly.
        with state._db.read_snapshot() as snap:
            placeholders = ",".join("?" for _ in slice3_workers)
            unfinished = snap.fetchall(
                f"SELECT task_id, attempt_id FROM task_attempts "
                f"WHERE worker_id IN ({placeholders}) AND finished_at_ms IS NULL",
                tuple(str(w) for w in slice3_workers),
            )
        with state._store.transaction() as cur:
            for row in unfinished:
                state._store.attempts.mark_finished(
                    cur,
                    JobName.from_wire(row["task_id"]),
                    int(row["attempt_id"]),
                    job_pb2.TASK_STATE_KILLED,
                    finished_at_ms=1,
                    error="terminal heartbeat (test)",
                )

        usage_after_drain = _read_usage_by_worker(state)
        for wid in slice3_workers:
            assert _tpu_used(usage_after_drain, wid) == 0, f"{wid} must be released after terminal heartbeat"

        # Now the scheduler can reuse slice-3 — the old kill RPCs have landed.
        result3 = _schedule_and_commit(scheduler, state)
        assert len(result3.assignments) == VMS_PER_SLICE
        assert {wid for _tid, wid in result3.assignments} == slice3_workers

    def test_requeue_keeps_slice_reserved_until_heartbeats(self, make_controller):
        """Cascaded requeue does not free the slice until terminal heartbeats land.

        Pre-Jumbo, this scenario relied on a separate ``_workers_pending_kill``
        in-memory guard while StopTasks RPCs were in flight. Post-Jumbo, the
        worker auto-kills via the polling reconcile loop and the only thing
        keeping the slice reserved is the unfinished attempt rows. This test
        asserts that a scheduling tick run *immediately* after a sibling
        requeue (no heartbeats yet) cannot place gang B on the still-busy
        slice — the conservative-state property described in design §6.1.
        """
        ctrl = make_controller(remote_state_dir="file:///tmp/iris-5470-test")
        state = ctrl._transitions

        job_a_id, job_b_id = self._setup_two_gangs_running(ctrl, state)
        self._preempt_and_new_slice(state, job_a_id, job_b_id)

        # Drain the trigger workers' reserved chips on slice-1 & slice-2 so the
        # next scheduling tick has a clean slice-3 to look at, then synthesize
        # terminal heartbeats for the remaining (still-reserved) sibling
        # attempts so slice-3 reassignment can proceed.
        for jid in (job_a_id, job_b_id):
            for task in query_tasks_for_job(state, jid):
                if task.current_worker_id is not None:
                    _heartbeat_killed(state, task)

        # Schedule gang A onto slice-3
        ctrl._run_scheduling()
        tasks_a = query_tasks_for_job(state, job_a_id)
        assigned_a = [t for t in tasks_a if t.current_worker_id is not None]
        assert len(assigned_a) == VMS_PER_SLICE, "Gang A should be assigned to slice-3"

        # Transition gang A to RUNNING so the heartbeat failure triggers
        # coscheduled requeue (only fires from EXECUTING states).
        for t in assigned_a:
            _transition_to_running(state, t)
        tasks_a = query_tasks_for_job(state, job_a_id)
        trigger_task = tasks_a[0]
        fail_request = HeartbeatApplyRequest(
            worker_id=trigger_task.current_worker_id,
            updates=[
                TaskUpdate(
                    task_id=trigger_task.task_id,
                    attempt_id=trigger_task.current_attempt_id,
                    new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                    error="502 Bad Gateway (simulated)",
                )
            ],
        )

        # Drive the production code path: heartbeat-induced cascade requeues
        # gang A's siblings without finalizing their attempts. Then run a
        # scheduling tick before any worker confirms termination.
        ctrl._process_heartbeat_updates([fail_request])
        ctrl._run_scheduling()

        # Gang B must not land on slice-3 — the still-unfinished sibling
        # attempts hold the slice's chips in the derived-usage ledger.
        tasks_b_during = query_tasks_for_job(state, job_b_id)
        b_on_slice3 = [
            t
            for t in tasks_b_during
            if t.current_worker_id is not None and str(t.current_worker_id).startswith("slice-3")
        ]
        assert len(b_on_slice3) == 0, (
            f"Gang B must not be assigned to slice-3 while siblings are unfinished: "
            f"{[str(t.current_worker_id) for t in b_on_slice3]}"
        )

    def test_derived_usage_correct_through_full_cycle(self, scheduler, state):
        """Track the derived per-worker tpu usage at every step.

        Producer transitions (cancel, requeue) leave attempts unfinished and
        the worker keeps its reservation; only terminal heartbeats release it.
        """
        slice1_wids = _register_slice(state, "slice-1")

        submit_job(state, "job-a", _make_gang_request("train-a"))

        # Assign: each worker now hosts an attempt that holds CHIPS_PER_VM.
        _schedule_and_commit(scheduler, state)
        usage = _read_usage_by_worker(state)
        for wid in slice1_wids:
            assert _tpu_used(usage, wid) == CHIPS_PER_VM, f"after assign: {wid}={_tpu_used(usage, wid)}"

        # Transition to RUNNING — attempt state changes but it still holds the chips.
        job_a_id = JobName.root("test-user", "job-a")
        for task in query_tasks_for_job(state, job_a_id):
            _transition_to_running(state, task)
        usage = _read_usage_by_worker(state)
        for wid in slice1_wids:
            assert _tpu_used(usage, wid) == CHIPS_PER_VM, f"after running: {wid}={_tpu_used(usage, wid)}"

        # Fail one task -> requeue all siblings. Trigger task gets finalized by
        # the heartbeat path that delivered WORKER_FAILED; siblings are bounced
        # to PENDING but their attempts are NOT finalized — so 7/8 workers still
        # show their reservation.
        tasks = query_tasks_for_job(state, job_a_id)
        trigger = tasks[0]
        _worker_fail_one_task(state, trigger)
        usage = _read_usage_by_worker(state)
        assert (
            _tpu_used(usage, trigger.current_worker_id) == 0
        ), f"trigger worker {trigger.current_worker_id} must be released by its terminal heartbeat"
        for wid in slice1_wids:
            if wid == trigger.current_worker_id:
                continue
            assert (
                _tpu_used(usage, wid) == CHIPS_PER_VM
            ), f"sibling worker {wid} must keep its reservation until its own terminal heartbeat lands"

        # Synthesize terminal heartbeats for the bounced sibling attempts —
        # what the worker would emit after reconcile auto-kills the stale
        # processes. The bounced tasks were sent back to PENDING and the
        # attempt rows still reference the old worker; finalising them is
        # what the heartbeat path would normally do.
        for wid in slice1_wids:
            if wid == trigger.current_worker_id:
                continue
            with state._db.read_snapshot() as snap:
                rows = snap.fetchall(
                    "SELECT task_id, attempt_id FROM task_attempts " "WHERE worker_id = ? AND finished_at_ms IS NULL",
                    (str(wid),),
                )
            with state._store.transaction() as cur:
                for row in rows:
                    state._store.attempts.mark_finished(
                        cur,
                        JobName.from_wire(row["task_id"]),
                        int(row["attempt_id"]),
                        job_pb2.TASK_STATE_KILLED,
                        finished_at_ms=1,
                        error="terminal heartbeat (test)",
                    )

        usage = _read_usage_by_worker(state)
        for wid in slice1_wids:
            assert _tpu_used(usage, wid) == 0, f"after terminal heartbeats: {wid}={_tpu_used(usage, wid)}"

        # Reassign — capacity is now genuinely available.
        _schedule_and_commit(scheduler, state)
        usage = _read_usage_by_worker(state)
        for wid in slice1_wids:
            assert _tpu_used(usage, wid) == CHIPS_PER_VM, f"after reassign: {wid}={_tpu_used(usage, wid)}"
