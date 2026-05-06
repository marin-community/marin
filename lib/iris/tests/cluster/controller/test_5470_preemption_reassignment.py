# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduction test for issue #5470: TPU placement collision after preemption.

Root cause (from production controller logs): when a coscheduled TPU gang fails
during BUILDING, `_requeue_coscheduled_siblings` decommits committed_tpu on all
workers in the same DB transaction. The kill RPCs for the still-running sibling
processes are sent AFTER the transaction commits (async). The scheduling thread
can run a tick between the decommit and the kills, see the freed capacity, and
assign a second gang to workers that still have stale processes.

The fix populates `_workers_pending_kill` inside the decommit transaction (before
commit) so the scheduling thread cannot observe freed capacity without also
seeing the pending-kill marker.

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
from iris.cluster.controller.scheduler import JobRequirements, Scheduler
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
from .conftest import query_worker as _query_worker
from .conftest import schedulable_tasks as _schedulable_tasks

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


def _build_context(scheduler, state):
    pending = _schedulable_tasks(state)
    workers = [w for w in healthy_active_workers(state) if w.healthy]
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
    return scheduler.create_scheduling_context(workers, building_counts=bc, pending_tasks=task_ids, jobs=jobs)


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


def _worker_fail_one_task(state, task):
    """Send WORKER_FAILED for one task. For coscheduled jobs this triggers
    _requeue_coscheduled_siblings which bounces all siblings to PENDING and
    decommits their resources."""
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
        with state._store.transaction() as cur:
            state._store.workers.set_health_for_test(cur, wid, healthy=False)


@pytest.fixture
def scheduler():
    return Scheduler()


class TestPreemptionReassignmentRace:
    """Reproduce the #5470 collision via the decommit-before-kill race."""

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
        """Preempt both gangs, mark old slices unhealthy, register slice-3."""
        tasks_a = query_tasks_for_job(state, job_a_id)
        tasks_b = query_tasks_for_job(state, job_b_id)
        _worker_fail_one_task(state, tasks_a[0])
        _worker_fail_one_task(state, tasks_b[0])

        for wid in [WorkerId(f"slice-1-w{i}") for i in range(VMS_PER_SLICE)]:
            assert _query_worker(state, wid).committed_tpu == 0
        for wid in [WorkerId(f"slice-2-w{i}") for i in range(VMS_PER_SLICE)]:
            assert _query_worker(state, wid).committed_tpu == 0

        _mark_slice_unhealthy(state, "slice-1")
        _mark_slice_unhealthy(state, "slice-2")
        _register_slice(state, "slice-3")

    def test_scheduler_assigns_to_decommitted_workers(self, scheduler, state):
        """Without the pending-kill guard, the scheduler assigns a gang to
        workers whose resources were decommitted but whose kill RPCs haven't
        landed. This demonstrates the race that #5470 fixes."""
        job_a_id, job_b_id = self._setup_two_gangs_running(scheduler, state)
        self._preempt_and_new_slice(state, job_a_id, job_b_id)

        # Scheduler assigns first gang to slice-3
        result1 = _schedule_and_commit(scheduler, state)
        assert len(result1.assignments) == VMS_PER_SLICE
        by_job1 = _assigned_workers_by_job(result1.assignments)
        assert len(by_job1) == 1
        first_job = next(iter(by_job1))

        # First gang fails during BUILDING -> decommit happens in DB.
        # In production, kill RPCs are queued but NOT yet sent.
        first_job_tasks = query_tasks_for_job(state, first_job)
        fail_result = _worker_fail_one_task(state, first_job_tasks[0])
        assert fail_result.tasks_to_kill, "Coscheduled requeue should produce kill targets"

        for i in range(VMS_PER_SLICE):
            w = _query_worker(state, WorkerId(f"slice-3-w{i}"))
            assert w.committed_tpu == 0, f"slice-3-w{i} should be decommitted"

        # Without the guard: scheduler sees free capacity and reassigns to slice-3.
        # This is the race -- in production the old processes are still running.
        result2 = _schedule_and_commit(scheduler, state)
        assert (
            len(result2.assignments) == VMS_PER_SLICE
        ), "Scheduler should assign to the decommitted workers (no guard at scheduler level)"

        # Verify the assignment lands on slice-3 -- the same workers where old
        # processes are still being killed. In production this causes port 8476
        # collision between old and new JAX coordinators.
        slice3_workers = {WorkerId(f"slice-3-w{i}") for i in range(VMS_PER_SLICE)}
        assigned_workers = set()
        for _, wid in result2.assignments:
            assigned_workers.add(wid)
        assert assigned_workers == slice3_workers, "Assignment should land on slice-3 (the only healthy slice)"

    def test_pending_kill_guard_prevents_reassignment(self, make_controller):
        """WITH the fix: _process_heartbeat_updates populates _workers_pending_kill
        inside the decommit transaction, and the scheduler excludes those workers.

        Mocks _stop_tasks_direct (the RPC boundary) to run a scheduling tick
        mid-kill, verifying the guard blocks reassignment while kills are in
        flight.
        """
        ctrl = make_controller(remote_state_dir="file:///tmp/iris-5470-test")
        state = ctrl._transitions

        job_a_id, job_b_id = self._setup_two_gangs_running(ctrl, state)
        self._preempt_and_new_slice(state, job_a_id, job_b_id)

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

        # Mock _stop_tasks_direct to intercept the kill RPCs and run scheduling
        # mid-kill -- this is the exact interleaving that causes #5470.
        pending_kill_during_rpc = None

        def intercepting_stop(task_ids, task_kill_workers=None):
            nonlocal pending_kill_during_rpc
            with ctrl._workers_pending_kill_lock:
                pending_kill_during_rpc = set(ctrl._workers_pending_kill)
            ctrl._run_scheduling()

        ctrl._stop_tasks_direct = intercepting_stop

        # Drive the production code path
        ctrl._process_heartbeat_updates([fail_request])

        # Verify pending-kill was populated during the RPC window
        assert pending_kill_during_rpc, "pending-kill set should be non-empty during kill RPCs"
        slice3_workers = {WorkerId(f"slice-3-w{i}") for i in range(VMS_PER_SLICE)}
        assert pending_kill_during_rpc & slice3_workers, "slice-3 workers should be in pending-kill during kill RPCs"

        # The mid-kill scheduling tick should NOT have placed gang B on slice-3
        # because 7 of 8 workers were excluded (the gang needs all 8).
        tasks_b_during = query_tasks_for_job(state, job_b_id)
        b_on_slice3 = [
            t
            for t in tasks_b_during
            if t.current_worker_id is not None and str(t.current_worker_id).startswith("slice-3")
        ]
        assert len(b_on_slice3) == 0, (
            f"Gang B must not be assigned to slice-3 while kills are pending: "
            f"{[str(t.current_worker_id) for t in b_on_slice3]}"
        )

        # After kills complete, pending-kill should be cleared
        with ctrl._workers_pending_kill_lock:
            assert len(ctrl._workers_pending_kill) == 0, "pending-kill should be empty after kills"

    def test_committed_tpu_correct_through_full_cycle(self, scheduler, state):
        """Track committed_tpu at every step to verify accounting."""
        slice1_wids = _register_slice(state, "slice-1")

        submit_job(state, "job-a", _make_gang_request("train-a"))

        # Assign
        _schedule_and_commit(scheduler, state)
        for wid in slice1_wids:
            w = _query_worker(state, wid)
            assert w.committed_tpu == CHIPS_PER_VM, f"after assign: {wid}={w.committed_tpu}"

        # Run
        job_a_id = JobName.root("test-user", "job-a")
        for task in query_tasks_for_job(state, job_a_id):
            _transition_to_running(state, task)
        for wid in slice1_wids:
            w = _query_worker(state, wid)
            assert w.committed_tpu == CHIPS_PER_VM, f"after running: {wid}={w.committed_tpu}"

        # Fail one task -> requeue all siblings
        tasks = query_tasks_for_job(state, job_a_id)
        _worker_fail_one_task(state, tasks[0])
        for wid in slice1_wids:
            w = _query_worker(state, wid)
            assert w.committed_tpu == 0, f"after requeue: {wid}={w.committed_tpu}"

        # Reassign
        _schedule_and_commit(scheduler, state)
        for wid in slice1_wids:
            w = _query_worker(state, wid)
            assert w.committed_tpu == CHIPS_PER_VM, f"after reassign: {wid}={w.committed_tpu}"
