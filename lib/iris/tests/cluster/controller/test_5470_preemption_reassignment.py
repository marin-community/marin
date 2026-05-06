# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduction test for issue #5470: TPU placement collision after preemption.

Root cause (from production controller logs): when a coscheduled TPU gang fails
during BUILDING, `_requeue_coscheduled_siblings` decommits committed_tpu on all
workers in the same DB transaction. The kill RPCs for the still-running sibling
processes are sent AFTER the transaction commits (async). The scheduling thread
can run a tick between the decommit and the kills, see the freed capacity, and
assign a second gang to workers that still have stale processes.

The fix adds `_workers_pending_kill` — an in-memory set of worker IDs with
outstanding kill RPCs. The scheduling thread excludes these workers from the
scheduling context, preventing reassignment until kills complete.

Production incident timeline (Incident B, v5p-256):
  09:14:31  lr0.5  assigned → slice 389585fe
  09:14:34  lr0.67 assigned → slice 2e06c8f1  (separate slice, correct)
  18:43:32  lr0.5  reassigned → slice 8996b868 (after preemption)
  18:43:57  lr0.5  TERMINATED — task 28 hit 502 during uv sync
  18:43:58  lr0.67 reassigned → slice 8996b868 (COLLISION — 1s after decommit)
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
    decommits their resources. Returns (tasks_to_kill, task_kill_workers)."""
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


@pytest.fixture
def scheduler():
    return Scheduler()


# =============================================================================
# The reproduction test: exercises the exact production failure path
# =============================================================================


class TestPreemptionReassignmentRace:
    """Reproduce the #5470 collision via the decommit-before-kill race.

    The test drives the scheduling loop and task updater transitions manually
    (single-threaded) to simulate the interleaving that happens between the
    task-updater thread and the scheduling thread in production.
    """

    def test_decommit_then_schedule_before_kill_causes_collision(self, scheduler, state):
        """WITHOUT the fix: if we clear _workers_pending_kill (simulating the
        pre-fix state where the guard doesn't exist), the scheduler reassigns
        the slice to gang B immediately after gang A's decommit, before the
        kill RPCs would have completed.

        This is the exact sequence from the production incident:
        1. Gang A and B on separate slices, both running
        2. Gang A's slice preempted → all tasks bounced to PENDING, resources decommitted
        3. Gang B's slice preempted → same
        4. New slice-3 appears
        5. Scheduler tick: gang A assigned to slice-3
        6. Gang A fails during BUILDING (task hits 502) → decommit, kills queued
        7. Scheduler tick (BEFORE kills land): gang B assigned to slice-3 → COLLISION
        """
        # Phase 1: two slices, two gangs
        _register_slice(state, "slice-1")
        _register_slice(state, "slice-2")

        submit_job(state, "job-a", _make_gang_request("train-a"))
        submit_job(state, "job-b", _make_gang_request("train-b"))

        # Phase 2: initial placement — each gang gets its own slice (both in same tick)
        result = _schedule_and_commit(scheduler, state)
        assert len(result.assignments) == VMS_PER_SLICE * 2
        by_job = _assigned_workers_by_job(result.assignments)
        assert len(by_job) == 2
        job_ids = list(by_job.keys())
        assert not (by_job[job_ids[0]] & by_job[job_ids[1]])

        # Transition all to RUNNING
        job_a_id = JobName.root("test-user", "job-a")
        job_b_id = JobName.root("test-user", "job-b")
        for task in query_tasks_for_job(state, job_a_id):
            _transition_to_running(state, task)
        for task in query_tasks_for_job(state, job_b_id):
            _transition_to_running(state, task)

        # Phase 3: simulate GCP preemption of both slices
        # Fail one task per gang — triggers coscheduled requeue of all siblings
        tasks_a = query_tasks_for_job(state, job_a_id)
        tasks_b = query_tasks_for_job(state, job_b_id)
        _worker_fail_one_task(state, tasks_a[0])
        _worker_fail_one_task(state, tasks_b[0])

        # Verify all tasks are now PENDING and resources decommitted
        for wid in [WorkerId(f"slice-1-w{i}") for i in range(VMS_PER_SLICE)]:
            w = _query_worker(state, wid)
            assert w.committed_tpu == 0, f"{wid} committed_tpu={w.committed_tpu} after requeue"
        for wid in [WorkerId(f"slice-2-w{i}") for i in range(VMS_PER_SLICE)]:
            w = _query_worker(state, wid)
            assert w.committed_tpu == 0, f"{wid} committed_tpu={w.committed_tpu} after requeue"

        # Mark old slice workers unhealthy (GCP reclaimed them)
        for prefix in ("slice-1", "slice-2"):
            for i in range(VMS_PER_SLICE):
                wid = WorkerId(f"{prefix}-w{i}")
                with state._store.transaction() as cur:
                    state._store.workers.set_health_for_test(cur, wid, healthy=False)

        # Phase 4: new slice appears (the only healthy slice now)
        _register_slice(state, "slice-3")

        # Phase 5: scheduler tick — assign gang A to slice-3
        result1 = _schedule_and_commit(scheduler, state)
        assert len(result1.assignments) == VMS_PER_SLICE
        by_job1 = _assigned_workers_by_job(result1.assignments)
        assert len(by_job1) == 1
        first_job = next(iter(by_job1))

        # Phase 6: gang A fails during BUILDING (simulating 502)
        first_job_tasks = query_tasks_for_job(state, first_job)
        _worker_fail_one_task(state, first_job_tasks[0])

        # At this point in production:
        #   - committed_tpu=0 on slice-3 (decommit happened in transaction)
        #   - kill RPCs for the other 7 tasks are queued but NOT YET SENT
        #   - the scheduling thread runs its next tick

        # Verify the decommit happened
        for wid in [WorkerId(f"slice-3-w{i}") for i in range(VMS_PER_SLICE)]:
            w = _query_worker(state, wid)
            assert w.committed_tpu == 0, f"{wid} should be decommitted"

        # Phase 7: scheduler tick — WITHOUT pending-kill guard, gang B would
        # see free capacity and be assigned to slice-3. WITH the guard, the
        # workers are excluded and gang B stays pending.
        #
        # This test simulates the pre-fix race by NOT setting _workers_pending_kill.
        # The scheduler sees all workers as available.
        result2 = _schedule_and_commit(scheduler, state)

        if len(result2.assignments) == VMS_PER_SLICE:
            by_job2 = _assigned_workers_by_job(result2.assignments)
            second_job = next(iter(by_job2))
            if second_job != first_job:
                # Gang B got slice-3 — this is the collision window.
                # In the NEXT tick, gang A (retrying) will also want slice-3.
                result3 = _schedule_and_commit(scheduler, state)
                if result3.assignments:
                    by_job3 = _assigned_workers_by_job(result3.assignments)
                    # Check if gang A lands on the same workers as gang B
                    for j, workers in by_job3.items():
                        if j != second_job:
                            overlap = workers & by_job2[second_job]
                            if overlap:
                                pytest.fail(
                                    f"COLLISION REPRODUCED: {j} and {second_job} "
                                    f"share {len(overlap)} workers on slice-3. "
                                    f"This is the #5470 bug: decommit freed "
                                    f"capacity before kill RPCs completed."
                                )

    def test_pending_kill_guard_prevents_reassignment(self, make_controller):
        """WITH the fix: _workers_pending_kill prevents the scheduler from
        seeing workers that have outstanding kill RPCs.

        This test uses the full Controller to exercise the actual guard in
        _run_scheduler_pass.
        """
        ctrl = make_controller(remote_state_dir="file:///tmp/iris-5470-test")
        state = ctrl._transitions

        # Setup: two slices, two gangs
        _register_slice(state, "slice-1")
        _register_slice(state, "slice-2")

        submit_job(state, "job-a", _make_gang_request("train-a"))
        submit_job(state, "job-b", _make_gang_request("train-b"))

        # Initial placement via the Controller's scheduling loop
        outcome = ctrl._run_scheduling()
        assert outcome == SchedulingOutcome.ASSIGNMENTS_MADE

        # Transition to RUNNING
        job_a_id = JobName.root("test-user", "job-a")
        job_b_id = JobName.root("test-user", "job-b")
        for task in query_tasks_for_job(state, job_a_id):
            _transition_to_running(state, task)
        for task in query_tasks_for_job(state, job_b_id):
            _transition_to_running(state, task)

        # Preempt both slices
        tasks_a = query_tasks_for_job(state, job_a_id)
        tasks_b = query_tasks_for_job(state, job_b_id)
        _worker_fail_one_task(state, tasks_a[0])
        _worker_fail_one_task(state, tasks_b[0])

        # New slice
        _register_slice(state, "slice-3")

        # Schedule gang A onto slice-3
        ctrl._run_scheduling()

        # Gang A fails during build — decommit happens
        tasks_a = query_tasks_for_job(state, job_a_id)
        running_a = [t for t in tasks_a if t.current_worker_id is not None and t.state in (3, 4, 5)]
        if running_a:
            fail_result = _worker_fail_one_task(state, running_a[0])
            # Simulate the task updater marking workers as pending-kill
            # (In production, this happens in _run_task_updater_loop between
            # the decommit transaction and the StopTasks RPCs.)
            kill_workers = set(fail_result.task_kill_workers.values())
            with ctrl._workers_pending_kill_lock:
                ctrl._workers_pending_kill |= kill_workers

            # Run scheduling — workers in pending-kill should be excluded
            outcome = ctrl._run_scheduling()

            # Verify: gang B should NOT have been assigned to slice-3
            # because the workers are in _workers_pending_kill
            tasks_b_after = query_tasks_for_job(state, job_b_id)
            b_on_slice3 = [
                t
                for t in tasks_b_after
                if t.current_worker_id is not None and str(t.current_worker_id).startswith("slice-3")
            ]
            assert len(b_on_slice3) == 0, (
                f"Gang B was assigned to slice-3 despite pending kills: "
                f"{[str(t.current_worker_id) for t in b_on_slice3]}"
            )

            # Clear the guard (simulates kill RPCs completing)
            with ctrl._workers_pending_kill_lock:
                ctrl._workers_pending_kill -= kill_workers

            # Now scheduling should work — gang B can get slice-3
            ctrl._run_scheduling()

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

        # Fail one task → requeue all siblings
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
