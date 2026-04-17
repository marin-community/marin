# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task scheduler.

The scheduler is a shallow interface that takes inputs (pending tasks, workers,
job requirements) and returns outputs (assignments). It does not dispatch tasks,
modify state, or run threads.
"""

import pytest

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.codec import constraints_from_json, resource_spec_from_scalars
from iris.cluster.controller.store import _decode_attribute_rows
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingResult,
)
from iris.cluster.controller.transitions import Assignment, ControllerTransitions, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.types import JobName, WorkerId
from iris.cluster.controller.autoscaler.status import PendingHint, build_job_pending_hints
from iris.rpc import config_pb2, vm_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, Timestamp

from tests.cluster.conftest import eq_constraint, in_constraint
from ._testing import set_worker_health as _set_worker_health
from .conftest import (
    building_counts as _building_counts,
    check_task_can_be_scheduled,
    healthy_active_workers,
    make_job_request,
    make_test_entrypoint as _make_test_entrypoint,
    make_worker_metadata,
    query_job as _query_job,
    query_task as _query_task,
    query_task_with_attempts,
    query_tasks_for_job as _query_tasks_for_job,
    query_worker as _query_worker,
    register_worker,
    schedulable_tasks as _schedulable_tasks,
    submit_job,
)


def _job_requirements_from_job(job) -> JobRequirements:
    return JobRequirements(
        resources=resource_spec_from_scalars(
            job.resources.cpu_millicores, job.resources.memory_bytes, job.resources.disk_bytes, job.resources.device_json
        ),
        constraints=constraints_from_json(job.constraints_json),
        is_coscheduled=job.has_coscheduling,
        coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
    )


def _worker_attr(state: ControllerTransitions, worker_id: WorkerId, key: str):
    with state._db.snapshot() as q:
        rows = q.raw(
            "SELECT worker_id, key, value_type, str_value, int_value, float_value"
            " FROM worker_attributes WHERE worker_id = ? AND key = ?",
            (str(worker_id), key),
        )
    if not rows:
        return None
    attrs = _decode_attribute_rows(rows)
    return attrs.get(worker_id, {}).get(key)


def assign_task_to_worker(state: ControllerTransitions, task, worker_id: WorkerId) -> None:
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])


def transition_task_to_running(state: ControllerTransitions, task) -> None:
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=task.current_worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=task.current_attempt_id,
                    new_state=job_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )


def transition_task_to_state(state: ControllerTransitions, task, new_state: int) -> None:
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=task.current_worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=task.current_attempt_id,
                    new_state=new_state,
                )
            ],
        )
    )


def _build_context(scheduler, state):
    pending_tasks = _schedulable_tasks(state)
    workers = [w for w in healthy_active_workers(state) if w.healthy]
    building_counts = _building_counts(state)

    task_ids = []
    jobs = {}
    for task in pending_tasks:
        if not check_task_can_be_scheduled(task):
            continue
        task_ids.append(task.task_id)
        if task.job_id not in jobs:
            job = _query_job(state, task.job_id)
            if job:
                jobs[task.job_id] = _job_requirements_from_job(job)

    return scheduler.create_scheduling_context(
        workers,
        building_counts=building_counts,
        pending_tasks=task_ids,
        jobs=jobs,
    )


def schedule_until_done(
    scheduler: Scheduler,
    state: ControllerTransitions,
    max_cycles: int = 100,
) -> SchedulingResult:
    """Drive the scheduler until no more tasks can be assigned.

    Runs scheduling cycles, applying assignments to state between cycles,
    until no progress is made. Returns aggregated results.
    """
    all_assignments: list[tuple[JobName, WorkerId]] = []

    for _ in range(max_cycles):
        context = _build_context(scheduler, state)

        if not context.pending_tasks:
            break

        result = scheduler.find_assignments(context)

        if not result.assignments:
            break

        all_assignments.extend(result.assignments)

        for task_id, worker_id in result.assignments:
            task = _query_task(state, task_id)
            if task:
                assign_task_to_worker(state, task, worker_id)

    return SchedulingResult(assignments=all_assignments)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scheduler():
    """Create a Scheduler instance."""
    return Scheduler()


def test_scheduler_finds_assignment_for_task(scheduler, state):
    """Verify scheduler assigns task to available worker."""
    register_worker(state, "w1", "addr", make_worker_metadata())

    tasks = submit_job(state, "j1", make_job_request())
    task = tasks[0]

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == task.task_id
    assert result.assignments[0][1] == WorkerId("w1")


def test_scheduler_returns_empty_when_no_workers(scheduler, state):
    """Verify scheduler returns empty result when no workers available."""
    submit_job(state, "j1", make_job_request())

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 0


def test_scheduler_round_robins_tasks_across_workers(scheduler, state):
    """Verify scheduler distributes tasks across workers instead of packing one worker."""
    register_worker(state, "w1", "addr1", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))
    register_worker(state, "w2", "addr2", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))
    register_worker(state, "w3", "addr3", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    submit_job(state, "j1", make_job_request(cpu=2))
    submit_job(state, "j2", make_job_request(cpu=2))
    submit_job(state, "j3", make_job_request(cpu=2))

    result = schedule_until_done(scheduler, state)

    # All 3 tasks assigned, each to a different worker (round-robin)
    assert len(result.assignments) == 3
    assigned_worker_ids = {worker_id for _, worker_id in result.assignments}
    assert len(assigned_worker_ids) == 3


def test_scheduler_assigns_multiple_tasks_to_single_worker(scheduler, state):
    """Verify scheduler assigns multiple tasks to one worker when it's the only option."""
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    tasks1 = submit_job(state, "j1", make_job_request(cpu=2))
    tasks2 = submit_job(state, "j2", make_job_request(cpu=2))
    tasks3 = submit_job(state, "j3", make_job_request(cpu=2))

    result = schedule_until_done(scheduler, state)

    # All 3 tasks eventually assigned to the single worker
    assert len(result.assignments) == 3
    assigned_task_ids = {task_id for task_id, _ in result.assignments}
    assert assigned_task_ids == {tasks1[0].task_id, tasks2[0].task_id, tasks3[0].task_id}
    # All assigned to the same worker
    assert all(worker_id == WorkerId("w1") for _, worker_id in result.assignments)


def test_scheduler_skips_tasks_that_dont_fit(scheduler, state):
    """Verify scheduler skips tasks that don't fit and continues to next."""
    # Worker with 4 CPUs
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=4, memory_bytes=16 * 1024**3))

    # Job 1: needs 8 CPUs (won't fit on 4 CPU worker)
    submit_job(state, "j1", make_job_request(cpu=8))
    # Job 2: needs 2 CPUs (will fit)
    tasks2 = submit_job(state, "j2", make_job_request(cpu=2))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Only job2's task should be assigned
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks2[0].task_id


def test_scheduler_detects_timed_out_tasks(state):
    """Verify timed-out tasks are handled by the controller (not the scheduler).

    The scheduler no longer handles timeouts -- the controller filters them out
    before calling find_assignments. This test verifies the overall behavior
    by testing the controller-level flow.
    """
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=2))

    # Job that requires 100 CPUs (will never fit) with 1 second timeout
    request = controller_pb2.Controller.LaunchJobRequest(
        name="impossible-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=100000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    request.scheduling_timeout.CopyFrom(duration_to_proto(Duration.from_seconds(1)))
    tasks = submit_job(state, "j1", request)

    # Manually set deadline epoch to past timestamp in DB.
    state._db.execute(
        "UPDATE jobs SET scheduling_deadline_epoch_ms = ? WHERE job_id = ?",
        (Timestamp.now().epoch_ms() - 2000, JobName.root("test-user", "j1").to_wire()),
    )

    # When building context, the timed-out task should be filtered out
    pending_tasks = _schedulable_tasks(state)

    # Simulate controller-level timeout filtering
    schedulable_task_ids = []
    jobs = {}
    timed_out_tasks = []
    for task in pending_tasks:
        if not check_task_can_be_scheduled(task):
            continue
        j = _query_job(state, task.job_id)
        if (
            j
            and j.scheduling_deadline_epoch_ms is not None
            and j.scheduling_deadline_epoch_ms <= Timestamp.now().epoch_ms()
        ):
            timed_out_tasks.append(task)
            continue
        schedulable_task_ids.append(task.task_id)
        if task.job_id not in jobs:
            jobs[task.job_id] = _job_requirements_from_job(j)

    # The task is timed out, so no schedulable tasks
    assert len(timed_out_tasks) == 1
    assert timed_out_tasks[0] == tasks[0]
    assert len(schedulable_task_ids) == 0


def test_scheduler_no_timeout_when_zero(scheduler, state):
    """Verify task with scheduling_timeout=0 never times out."""
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=2))

    # Job that can't fit but has no timeout (0)
    request = controller_pb2.Controller.LaunchJobRequest(
        name="no-timeout-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=100000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    # No timeout set (field not present)
    submit_job(state, "j1", request, timestamp_ms=Timestamp.now().epoch_ms() - 10000)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Task should not be assigned (just skipped, no assignment)
    assert len(result.assignments) == 0


def test_scheduler_respects_worker_capacity_across_assignments(scheduler, state):
    """Verify scheduler tracks capacity used by earlier assignments across cycles."""
    # Worker with 4 CPUs
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=4))

    # Submit 3 jobs, each requiring 2 CPUs (only 2 will fit)
    for i in range(3):
        submit_job(state, f"j{i}", make_job_request(cpu=2))

    result = schedule_until_done(scheduler, state)

    # Only 2 tasks assigned (4 CPUs / 2 CPUs each = 2 tasks max)
    assert len(result.assignments) == 2

    # Third task still pending
    pending = _schedulable_tasks(state)
    assert len(pending) == 1


def test_scheduler_skips_unhealthy_workers(scheduler, state):
    """Verify scheduler ignores unhealthy workers."""
    register_worker(state, "w1", "addr1", make_worker_metadata())
    register_worker(state, "w2", "addr2", make_worker_metadata())
    # Mark second worker as unhealthy
    _set_worker_health(state._db, WorkerId("w2"), healthy=False)

    submit_job(state, "j1", make_job_request())

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w1")


def test_scheduler_considers_running_tasks_for_capacity(scheduler, state):
    """Verify scheduler accounts for tasks already running on workers."""
    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "addr", make_worker_metadata(cpu=4))

    # Submit a job that uses 3 CPUs, assign it to the worker, and mark it running
    running_tasks = submit_job(state, "running", make_job_request(cpu=3))
    assign_task_to_worker(state, running_tasks[0], worker_id)
    transition_task_to_running(state, running_tasks[0])

    # Try to schedule a job that needs 2 CPUs (won't fit, only 1 CPU available)
    submit_job(state, "j1", make_job_request(cpu=2))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 0


def test_scheduler_reports_task_too_large_for_cluster(scheduler, state):
    """Verify scheduler reports when a task requires more resources than any worker can provide."""
    # Worker with only 2 CPUs
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=2))

    # Job that needs 4 CPUs
    submit_job(state, "j1", make_job_request(cpu=4))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Primary observable behavior: task cannot be assigned
    assert len(result.assignments) == 0


# =============================================================================
# Constraint Filtering Tests
# =============================================================================


def test_constraint_filters_workers_by_attribute(scheduler, state):
    """Job with constraint only schedules on workers with matching attribute."""
    # Worker 1 with tpu-name attribute
    meta1 = make_worker_metadata()
    meta1.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2 with different tpu-name
    meta2 = make_worker_metadata()
    meta2.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
    register_worker(state, "w2", "addr2", meta2)

    # Job with constraint requiring tpu-name = "tpu-a"
    req = make_job_request()
    req.constraints.append(eq_constraint(WellKnownAttribute.TPU_NAME, "tpu-a").to_proto())
    tasks = submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks[0].task_id
    assert result.assignments[0][1] == WorkerId("w1")


@pytest.mark.parametrize(
    "op,worker_value,constraint_value,should_match",
    [
        # EQ operator tests
        (job_pb2.CONSTRAINT_OP_EQ, "us-west", "us-west", True),
        (job_pb2.CONSTRAINT_OP_EQ, "us-east", "us-west", False),
        # NE operator tests
        (job_pb2.CONSTRAINT_OP_NE, "us-east", "us-west", True),
        (job_pb2.CONSTRAINT_OP_NE, "us-west", "us-west", False),
    ],
    ids=[
        "EQ-match",
        "EQ-no-match",
        "NE-match",
        "NE-no-match",
    ],
)
def test_constraint_string_operators(scheduler, state, op, worker_value, constraint_value, should_match):
    """String equality operators (EQ, NE) filter workers by attribute value."""
    meta = make_worker_metadata()
    meta.attributes[WellKnownAttribute.REGION].string_value = worker_value
    register_worker(state, "w1", "addr", meta)

    req = make_job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.REGION
    constraint.op = op
    constraint.value.string_value = constraint_value
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    if should_match:
        assert len(result.assignments) == 1
        assert result.assignments[0][1] == WorkerId("w1")
    else:
        assert len(result.assignments) == 0


@pytest.mark.parametrize(
    "op,worker_has_attribute,should_match",
    [
        (job_pb2.CONSTRAINT_OP_EXISTS, True, True),
        (job_pb2.CONSTRAINT_OP_EXISTS, False, False),
        (job_pb2.CONSTRAINT_OP_NOT_EXISTS, True, False),
        (job_pb2.CONSTRAINT_OP_NOT_EXISTS, False, True),
    ],
    ids=[
        "EXISTS-present",
        "EXISTS-absent",
        "NOT_EXISTS-present",
        "NOT_EXISTS-absent",
    ],
)
def test_constraint_existence_operators(scheduler, state, op, worker_has_attribute, should_match):
    """Existence operators (EXISTS, NOT_EXISTS) check for attribute presence."""
    meta = make_worker_metadata()
    if worker_has_attribute:
        meta.attributes["gpu-model"].string_value = "A100"
    register_worker(state, "w1", "addr", meta)

    req = make_job_request()
    constraint = req.constraints.add()
    constraint.key = "gpu-model"
    constraint.op = op
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    if should_match:
        assert len(result.assignments) == 1
        assert result.assignments[0][1] == WorkerId("w1")
    else:
        assert len(result.assignments) == 0


@pytest.mark.parametrize(
    "op,worker_value,constraint_value,should_match",
    [
        # GT: worker > constraint
        (job_pb2.CONSTRAINT_OP_GT, 10, 5, True),
        (job_pb2.CONSTRAINT_OP_GT, 5, 5, False),
        (job_pb2.CONSTRAINT_OP_GT, 3, 5, False),
        # GE: worker >= constraint
        (job_pb2.CONSTRAINT_OP_GE, 10, 5, True),
        (job_pb2.CONSTRAINT_OP_GE, 5, 5, True),
        (job_pb2.CONSTRAINT_OP_GE, 3, 5, False),
        # LT: worker < constraint
        (job_pb2.CONSTRAINT_OP_LT, 3, 5, True),
        (job_pb2.CONSTRAINT_OP_LT, 5, 5, False),
        (job_pb2.CONSTRAINT_OP_LT, 10, 5, False),
        # LE: worker <= constraint
        (job_pb2.CONSTRAINT_OP_LE, 3, 5, True),
        (job_pb2.CONSTRAINT_OP_LE, 5, 5, True),
        (job_pb2.CONSTRAINT_OP_LE, 10, 5, False),
    ],
    ids=[
        "GT-greater",
        "GT-equal",
        "GT-less",
        "GE-greater",
        "GE-equal",
        "GE-less",
        "LT-less",
        "LT-equal",
        "LT-greater",
        "LE-less",
        "LE-equal",
        "LE-greater",
    ],
)
def test_constraint_numeric_operators(scheduler, state, op, worker_value, constraint_value, should_match):
    """Numeric comparison operators (GT, GE, LT, LE) compare attribute values."""
    meta = make_worker_metadata()
    meta.attributes["priority"].int_value = worker_value
    register_worker(state, "w1", "addr", meta)

    req = make_job_request()
    constraint = req.constraints.add()
    constraint.key = "priority"
    constraint.op = op
    constraint.value.int_value = constraint_value
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    if should_match:
        assert len(result.assignments) == 1
        assert result.assignments[0][1] == WorkerId("w1")
    else:
        assert len(result.assignments) == 0


def test_constraint_numeric_operators_with_floats(scheduler, state):
    """Numeric comparison operators work with float values."""
    meta = make_worker_metadata()
    meta.attributes["load"].float_value = 0.3
    register_worker(state, "w1", "addr", meta)

    req = make_job_request()
    constraint = req.constraints.add()
    constraint.key = "load"
    constraint.op = job_pb2.CONSTRAINT_OP_LT
    constraint.value.float_value = 0.5
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w1")


def test_constraint_in_operator_matches_any_value(scheduler, state):
    """IN constraint matches workers whose attribute value is in the provided set."""
    meta1 = make_worker_metadata()
    meta1.attributes[WellKnownAttribute.REGION].string_value = "us-central1"
    register_worker(state, "w1", "addr1", meta1)

    meta2 = make_worker_metadata()
    meta2.attributes[WellKnownAttribute.REGION].string_value = "us-central2"
    register_worker(state, "w2", "addr2", meta2)

    meta3 = make_worker_metadata()
    meta3.attributes[WellKnownAttribute.REGION].string_value = "eu-west4"
    register_worker(state, "w3", "addr3", meta3)

    # Job with IN constraint: region IN (us-central1, us-central2)
    req = make_job_request()
    req.constraints.append(in_constraint(WellKnownAttribute.REGION, ["us-central1", "us-central2"]).to_proto())

    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Only w1 and w2 match the IN constraint (not w3 in eu-west4)
    assert len(result.assignments) == 1
    assert result.assignments[0][1] in {WorkerId("w1"), WorkerId("w2")}


def test_constraint_in_operator_no_match(scheduler, state):
    """IN constraint with no matching workers produces no assignments."""
    meta = make_worker_metadata()
    meta.attributes[WellKnownAttribute.REGION].string_value = "eu-west4"
    register_worker(state, "w1", "addr1", meta)

    req = make_job_request()
    req.constraints.append(in_constraint(WellKnownAttribute.REGION, ["us-central1", "us-central2"]).to_proto())
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 0


def test_multiple_constraints_all_must_match(scheduler, state):
    """Multiple constraints are ANDed together."""
    # Worker 1: tpu-name=tpu-a, tpu-worker-id=0
    meta1 = make_worker_metadata()
    meta1.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    meta1.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = 0
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2: tpu-name=tpu-a, tpu-worker-id=1
    meta2 = make_worker_metadata()
    meta2.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    meta2.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = 1
    register_worker(state, "w2", "addr2", meta2)

    # Worker 3: tpu-name=tpu-b, tpu-worker-id=0
    meta3 = make_worker_metadata()
    meta3.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
    meta3.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = 0
    register_worker(state, "w3", "addr3", meta3)

    # Job requiring tpu-name=tpu-a AND tpu-worker-id=0
    req = make_job_request()
    req.constraints.append(eq_constraint(WellKnownAttribute.TPU_NAME, "tpu-a").to_proto())
    c2 = req.constraints.add()
    c2.key = WellKnownAttribute.TPU_WORKER_ID
    c2.op = job_pb2.CONSTRAINT_OP_EQ
    c2.value.int_value = 0
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Only w1 matches both constraints
    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w1")


def test_constraint_with_missing_attribute_fails(scheduler, state):
    """Constraint on missing attribute fails for EQ/NE/GT/etc (except NOT_EXISTS)."""
    # Worker without the required attribute
    meta = make_worker_metadata()
    register_worker(state, "w1", "addr", meta)

    # Job requiring tpu-name = "tpu-a"
    req = make_job_request()
    req.constraints.append(eq_constraint(WellKnownAttribute.TPU_NAME, "tpu-a").to_proto())
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Worker doesn't have tpu-name attribute, so constraint fails
    assert len(result.assignments) == 0


def test_job_without_constraints_schedules_anywhere(scheduler, state):
    """Job without constraints can be scheduled on any worker."""
    # Worker 1 with attribute
    meta1 = make_worker_metadata()
    meta1.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2 without attribute
    meta2 = make_worker_metadata()
    register_worker(state, "w2", "addr2", meta2)

    # Job without constraints
    req = make_job_request()
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should be assigned to either worker
    assert len(result.assignments) == 1


# =============================================================================
# Coscheduling Tests
# =============================================================================


def test_coscheduled_job_assigns_all_tasks_atomically(scheduler, state):
    """Coscheduled job assigns all tasks to workers in the same group."""
    # Create 4 workers on tpu-a
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should be assigned
    assert len(result.assignments) == 4

    # All assigned to workers with same tpu-name
    assigned_worker_ids = {worker_id for _, worker_id in result.assignments}
    # Verify all workers are in the tpu-a group
    for worker_id in assigned_worker_ids:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-a"

    # Tasks assigned in order: task-0 -> worker-0, task-1 -> worker-1, etc.
    for task_id, worker_id in result.assignments:
        task = _query_task(state, task_id)
        expected_worker_id = f"w{int(task.task_id.to_wire().rsplit('/', 1)[-1])}"
        assert worker_id == WorkerId(expected_worker_id)


def test_coscheduled_job_waits_when_insufficient_workers(scheduler, state):
    """Coscheduled job stays pending when not enough workers in any group."""
    # Only 2 workers on tpu-a
    for i in range(2):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Job requires 4 replicas
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # No assignments - job stays pending
    assert len(result.assignments) == 0


def test_coscheduled_job_chooses_group_with_capacity(scheduler, state):
    """Coscheduled job chooses the group that has capacity."""
    # tpu-a: 4 workers, 2 are busy (low capacity)
    for i in range(4):
        meta = make_worker_metadata(cpu=2)  # Each worker has 2 CPUs
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Consume capacity on first 2 workers of tpu-a by submitting a job
    busy_req = controller_pb2.Controller.LaunchJobRequest(
        name="busy-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=2,
        environment=job_pb2.EnvironmentConfig(),
    )
    submit_job(state, "busy", busy_req)

    # Assign the busy job's tasks to wa0 and wa1
    busy_tasks = _query_tasks_for_job(state, JobName.root("test-user", "busy"))
    assign_task_to_worker(state, busy_tasks[0], WorkerId("wa0"))
    assign_task_to_worker(state, busy_tasks[1], WorkerId("wa1"))
    transition_task_to_running(state, busy_tasks[0])
    transition_task_to_running(state, busy_tasks[1])

    # tpu-b: 4 workers, all free
    for i in range(4):
        meta = make_worker_metadata(cpu=2)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Job should be assigned to tpu-b (has 4 free workers)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-b"


def test_coscheduled_job_assigns_tasks_in_order(scheduler, state):
    """Task indices map to worker IDs in sorted order."""
    # Create workers with non-sequential IDs to verify sorting
    worker_ids = [3, 1, 0, 2]  # Deliberately out of order
    for i, wid in enumerate(worker_ids):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = wid
        register_worker(state, f"w{wid}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 4

    # Verify task-0 -> worker with tpu-worker-id=0, task-1 -> worker with tpu-worker-id=1, etc.
    for task_id, worker_id in result.assignments:
        task = _query_task(state, task_id)
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_WORKER_ID)
        assert attr is not None
        worker_tpu_id = attr.value
        task_idx = int(task.task_id.to_wire().rsplit("/", 1)[-1])
        assert task_idx == worker_tpu_id, f"Task {task_idx} assigned to worker with tpu-worker-id={worker_tpu_id}"


def test_coscheduled_job_with_constraints(scheduler, state):
    """Coscheduled job respects additional constraints."""
    # tpu-a: 4 workers with region=us-west
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.REGION].string_value = "us-west"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # tpu-b: 4 workers with region=us-east
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.REGION].string_value = "us-east"
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job requiring region=us-east
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    req.constraints.append(eq_constraint(WellKnownAttribute.REGION, "us-east").to_proto())
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should be assigned to tpu-b (only group matching region=us-east)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-b"


def test_coscheduled_job_with_partial_capacity(scheduler, state):
    """Coscheduled job waits when some workers in group lack capacity, then schedules when capacity is added."""
    # Create 4 workers, but 2 have insufficient CPU
    for i in range(4):
        cpu = 2 if i < 2 else 1  # First 2 have 2 CPU, last 2 have only 1
        meta = make_worker_metadata(cpu=cpu)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # No assignments - only 2 workers have sufficient capacity
    assert len(result.assignments) == 0

    # Now add a new TPU group with 4 workers, all with sufficient capacity
    for i in range(4):
        meta = make_worker_metadata(cpu=2)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Re-run the scheduler - job should now be assigned to the new group
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should now be assigned to tpu-b
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-b"


# =============================================================================
# Taint Constraint Tests
# =============================================================================


def test_tainted_worker_not_used_for_coscheduled_job(scheduler, state):
    """Coscheduled job skips groups containing tainted workers."""
    # Create TPU group "tpu-a" with 4 workers, one tainted
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        if i == 0:
            meta.attributes["taint:maintenance"].string_value = "true"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Create TPU group "tpu-b" with 4 workers, none tainted
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job with 4 replicas + NOT_EXISTS taint constraint
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    c = req.constraints.add()
    c.key = "taint:maintenance"
    c.op = job_pb2.CONSTRAINT_OP_NOT_EXISTS
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should be assigned to tpu-b (tpu-a has a tainted worker)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-b"


# =============================================================================
# TPU Chip Count Tracking Tests
# =============================================================================


def test_tpu_chip_count_deducted_from_capacity(scheduler, state):
    """TPU chip count is deducted when task is scheduled."""
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # First job requires 4 TPU chips
    req1 = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks1 = submit_job(state, "j1", req1)

    # First scheduling cycle - task should be assigned
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks1[0].task_id

    # Commit the assignment
    assign_task_to_worker(state, tasks1[0], WorkerId("w1"))
    transition_task_to_running(state, tasks1[0])

    # Submit second job that also requires 4 TPU chips
    req2 = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j2", req2)

    # Second scheduling cycle - no TPU chips available
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0


def test_tpu_job_rejected_when_insufficient_chips(scheduler, state):
    """TPU job is not scheduled when worker has fewer chips than required."""
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # Job requires 8 TPU chips - more than worker has
    req = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=8)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Task should not be scheduled - not enough TPU chips
    assert len(result.assignments) == 0


def test_tpu_count_released_after_task_completion(scheduler, state):
    """TPU chips are released when task completes, allowing new tasks to schedule."""
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # First job uses all 4 TPU chips
    req1 = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks1 = submit_job(state, "j1", req1)
    assign_task_to_worker(state, tasks1[0], WorkerId("w1"))
    transition_task_to_running(state, tasks1[0])

    # Submit second job
    req2 = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j2", req2)

    # Second job can't be scheduled yet
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0

    # Complete first task
    transition_task_to_state(state, query_task_with_attempts(state, tasks1[0].task_id), job_pb2.TASK_STATE_SUCCEEDED)

    # Now second job can be scheduled
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0].parent == JobName.root("test-user", "j2")


# =============================================================================
# Preemptible Constraint Tests
# =============================================================================


def test_preemptible_constraint_routes_to_matching_worker(scheduler, state):
    """Job constrained to non-preemptible workers is only scheduled on a matching worker."""
    # Preemptible worker
    meta_preemptible = make_worker_metadata()
    meta_preemptible.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "true"
    register_worker(state, "w-preemptible", "addr1", meta_preemptible)

    # On-demand worker
    meta_ondemand = make_worker_metadata()
    meta_ondemand.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
    register_worker(state, "w-ondemand", "addr2", meta_ondemand)

    # Job requiring non-preemptible worker
    req = make_job_request()
    req.constraints.append(eq_constraint(WellKnownAttribute.PREEMPTIBLE, "false").to_proto())
    tasks = submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks[0].task_id
    assert result.assignments[0][1] == WorkerId("w-ondemand")


def test_soft_preemptible_constraint_prefers_matching_but_allows_fallback(scheduler, state):
    """Job with soft preemptible constraint schedules on preemptible worker
    when available, but falls back to non-preemptible when it is the only option."""
    # Preemptible worker
    meta_preemptible = make_worker_metadata()
    meta_preemptible.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "true"
    register_worker(state, "w-preemptible", "addr1", meta_preemptible)

    # On-demand worker
    meta_ondemand = make_worker_metadata()
    meta_ondemand.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
    register_worker(state, "w-ondemand", "addr2", meta_ondemand)

    # Job with soft preemptible constraint
    req = make_job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.PREEMPTIBLE
    constraint.op = job_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "true"
    constraint.mode = job_pb2.CONSTRAINT_MODE_PREFERRED
    tasks = submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should prefer the preemptible worker
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks[0].task_id
    assert result.assignments[0][1] == WorkerId("w-preemptible")


def test_soft_constraint_falls_back_when_preferred_worker_at_capacity(scheduler, state):
    """When soft-preferred worker is at capacity, soft constraint allows fallback to non-matching worker."""
    # Preemptible worker with minimal resources (can only fit 1 task)
    meta_preemptible = make_worker_metadata(cpu=1, memory_bytes=1024**3)
    meta_preemptible.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "true"
    register_worker(state, "w-preemptible", "addr1", meta_preemptible)

    # On-demand worker with plenty of resources
    meta_ondemand = make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3)
    meta_ondemand.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
    register_worker(state, "w-ondemand", "addr2", meta_ondemand)

    # Submit 2 jobs with soft preemptible constraint
    for i in range(2):
        req = make_job_request(f"j{i}", cpu=1)
        constraint = req.constraints.add()
        constraint.key = WellKnownAttribute.PREEMPTIBLE
        constraint.op = job_pb2.CONSTRAINT_OP_EQ
        constraint.value.string_value = "true"
        constraint.mode = job_pb2.CONSTRAINT_MODE_PREFERRED
        submit_job(state, f"j{i}", req)

    result = schedule_until_done(scheduler, state)

    # Both should be assigned - one to preemptible, one to on-demand fallback
    assert len(result.assignments) == 2
    assigned_workers = {a[1] for a in result.assignments}
    assert WorkerId("w-preemptible") in assigned_workers
    assert WorkerId("w-ondemand") in assigned_workers


def test_soft_constraint_only_non_matching_workers_available(scheduler, state):
    """When no worker matches the soft constraint, job still schedules (unlike hard constraint)."""
    # Only on-demand worker available
    meta_ondemand = make_worker_metadata()
    meta_ondemand.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
    register_worker(state, "w-ondemand", "addr1", meta_ondemand)

    # Job with soft preemptible=true (no preemptible worker exists)
    req = make_job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.PREEMPTIBLE
    constraint.op = job_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "true"
    constraint.mode = job_pb2.CONSTRAINT_MODE_PREFERRED
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should still schedule on the non-preemptible worker
    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w-ondemand")


def test_coscheduled_soft_preemptible_constraint_does_not_block_scheduling(scheduler, state):
    """Coscheduled job with soft preemptible=true schedules on on-demand workers.

    Regression test: the coscheduled path used to pass all constraints (including
    soft) to matching_workers(), making soft constraints act as hard filters for
    multi-host jobs. A 4-task TPU pod job with soft preemptible=true should
    schedule on 4 on-demand workers in the same tpu-name group rather than
    staying pending because no preemptible workers exist.
    """
    # 4 on-demand workers in one tpu-name group — no preemptible workers at all
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled 4-replica job with soft preemptible=true
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-soft-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.PREEMPTIBLE
    constraint.op = job_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "true"
    constraint.mode = job_pb2.CONSTRAINT_MODE_PREFERRED
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should be assigned despite no preemptible workers
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-a"


def test_coscheduled_soft_constraint_prefers_matching_group(scheduler, state):
    """When multiple groups exist, coscheduled job prefers the group that
    satisfies the most soft constraints."""
    # Group tpu-a: 4 on-demand workers
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Group tpu-b: 4 preemptible workers
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "true"
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job with soft preemptible=true
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-soft-prefer",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.PREEMPTIBLE
    constraint.op = job_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "true"
    constraint.mode = job_pb2.CONSTRAINT_MODE_PREFERRED
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should prefer tpu-b (preemptible group matching the soft constraint)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-b"


def test_coscheduled_soft_constraint_schedules_on_non_matching_group(scheduler, state):
    """Coscheduled job with soft preemptible=true schedules on on-demand workers
    even when no preemptible group exists — verifies the coscheduled path treats
    soft constraints as ranking hints, not hard filters."""
    # 4 on-demand workers in one tpu-name group — none satisfy preemptible=true
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
        register_worker(state, f"w{i}", f"addr{i}", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-diag-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.PREEMPTIBLE
    constraint.op = job_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "true"
    constraint.mode = job_pb2.CONSTRAINT_MODE_PREFERRED
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should be assigned to the on-demand group despite soft
    # preemptible=true not matching — soft constraints must not block scheduling
    assert len(result.assignments) == 4
    assigned_workers = {worker_id for _, worker_id in result.assignments}
    assert assigned_workers == {"w0", "w1", "w2", "w3"}


# =============================================================================
# Depth-First Scheduling Priority Assignment Tests
# =============================================================================


def test_scheduler_assigns_deeper_job_before_shallow(scheduler, state):
    """Scheduler assigns deeper jobs before shallow ones when both fit."""
    # Worker with enough resources for both jobs
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit root job and child job (both with 1 CPU)
    submit_job(state, "root", make_job_request("root", cpu=1))
    submit_job(state, "/test-user/root/child", make_job_request("child", cpu=1))

    # Run scheduler
    result = schedule_until_done(scheduler, state)

    # Both tasks assigned, child first
    assert len(result.assignments) == 2
    assert result.assignments[0][0].parent == JobName.from_string("/test-user/root/child")
    assert result.assignments[1][0].parent == JobName.root("test-user", "root")


def test_scheduler_assigns_older_root_tree_first(scheduler, state):
    """At same depth, scheduler assigns older root tree first."""
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit two root jobs
    submit_job(state, "user-a-job", make_job_request("user-a-job", cpu=1))
    submit_job(state, "user-b-job", make_job_request("user-b-job", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 2
    # user-a-job submitted first
    assert result.assignments[0][0].parent == JobName.root("test-user", "user-a-job")
    assert result.assignments[1][0].parent == JobName.root("test-user", "user-b-job")


def test_scheduler_child_of_older_tree_beats_newer_root(scheduler, state):
    """Child of older tree is assigned before root of newer tree."""
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit old tree
    submit_job(state, "old-tree", make_job_request("old-tree", cpu=1))

    # Submit new tree
    submit_job(state, "new-tree", make_job_request("new-tree", cpu=1))

    # Submit child of old tree
    submit_job(state, "/test-user/old-tree/child", make_job_request("child", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3
    # Order: child (depth 2), old-tree (depth 1, older), new-tree (depth 1, newer)
    assert result.assignments[0][0].parent == JobName.from_string("/test-user/old-tree/child")
    assert result.assignments[1][0].parent == JobName.root("test-user", "old-tree")
    assert result.assignments[2][0].parent == JobName.root("test-user", "new-tree")


# =============================================================================
# Error Message Tests
# =============================================================================


def test_scheduler_reports_device_variant_mismatch(scheduler, state):
    """Scheduler reports constraint failure when no worker matches device variant."""
    # Worker with v5litepod-16
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    register_worker(state, "w1", "addr", meta)

    # Job requesting v5litepod-32
    req = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-32", count=4)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(healthy_active_workers(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if check_task_can_be_scheduled(t)), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    # Constraint-based matching: the device-variant constraint key is reported
    assert "device-variant" in diagnostics
    assert "constraints" in diagnostics.lower()


def test_scheduler_reports_tpu_count_exceeded(scheduler, state):
    """Scheduler reports TPU count exceeded in error message."""
    # Worker with 4 TPU chips -- use fixture so device attributes are populated
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # Job requesting 8 TPU chips
    req = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=8)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(healthy_active_workers(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if check_task_can_be_scheduled(t)), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    assert "tpu" in diagnostics.lower()
    assert "8" in diagnostics
    assert "4" in diagnostics


def test_scheduler_reports_device_type_mismatch(scheduler, state):
    """Scheduler reports constraint failure when worker device type doesn't match."""
    # CPU-only worker
    meta = make_worker_metadata()
    register_worker(state, "w1", "addr", meta)

    # Job requesting TPU
    req = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(healthy_active_workers(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if check_task_can_be_scheduled(t)), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    # Constraint-based matching: the device-type constraint is in the diagnostic
    assert "device-type" in diagnostics
    assert "constraints" in diagnostics.lower()


def test_scheduler_reports_coscheduling_capacity_details(scheduler, state):
    """Scheduler reports detailed coscheduling capacity issues."""
    # Create 4 workers but only 2 have sufficient CPU
    for i in range(4):
        cpu = 4 if i < 2 else 1  # First 2 have 4 CPU, last 2 have only 1
        meta = make_worker_metadata(cpu=cpu)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(healthy_active_workers(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if check_task_can_be_scheduled(t)), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    # Should mention it's a coscheduling issue with capacity details
    assert "coscheduling" in diagnostics.lower() or "group" in diagnostics.lower()
    # Should indicate how many workers have capacity vs needed
    assert "2" in diagnostics or "4" in diagnostics


def test_diagnostics_for_schedulable_job_does_not_say_unknown_failure(scheduler, state):
    """When a job can be scheduled, diagnostics should not say 'Unknown scheduling failure'."""
    register_worker(state, "w1", "addr1", make_worker_metadata())
    tasks = submit_job(state, "j1", make_job_request())

    context = scheduler.create_scheduling_context(healthy_active_workers(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if check_task_can_be_scheduled(t)), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    assert "unknown" not in diagnostics.lower()
    assert "schedulable" in diagnostics.lower()


def test_coscheduled_tpu_jobs_cannot_double_book_group(scheduler, state):
    """Two coscheduled TPU jobs cannot use the same TPU group simultaneously."""
    # Create 4 workers in tpu-group "tpu-a", each with 4 TPU chips
    for i in range(4):
        meta = make_worker_metadata(tpu_name="v5litepod-16")
        meta.device.tpu.count = 4
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    tpu_resource = job_pb2.ResourceSpecProto(
        cpu_millicores=1000,
        memory_bytes=1024**3,
        device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16", count=4)),
    )

    # Job 1: coscheduled across all 4 workers
    req1 = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=tpu_resource,
        environment=job_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req1.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks1 = submit_job(state, "j1", req1)

    # Schedule and commit job 1
    result1 = schedule_until_done(scheduler, state)
    assert len(result1.assignments) == 4
    for task in tasks1:
        transition_task_to_running(state, task)

    # Job 2: same shape, should be blocked because TPU chips are exhausted
    req2 = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        entrypoint=_make_test_entrypoint(),
        resources=tpu_resource,
        environment=job_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req2.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j2", req2)

    context = _build_context(scheduler, state)
    result2 = scheduler.find_assignments(context)
    assert len(result2.assignments) == 0

    # Complete all job 1 tasks
    for task in tasks1:
        transition_task_to_state(state, query_task_with_attempts(state, task.task_id), job_pb2.TASK_STATE_SUCCEEDED)

    # Job 2 should now be schedulable
    result3 = schedule_until_done(scheduler, state)
    assert len(result3.assignments) == 4
    assigned_jobs = {task_id.parent for task_id, _ in result3.assignments}
    assert assigned_jobs == {JobName.root("test-user", "j2")}


def test_scheduler_fifo_within_same_depth_and_tree(scheduler, state):
    """Scheduler respects FIFO within same depth and tree."""
    register_worker(state, "w1", "addr", make_worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit parent
    submit_job(state, "tree", make_job_request("tree", cpu=1))

    # Submit two children
    submit_job(state, "/test-user/tree/child-a", make_job_request("child-a", cpu=1))
    submit_job(state, "/test-user/tree/child-b", make_job_request("child-b", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3

    # Find child assignments
    child_assignments = [
        (task_id, worker_id)
        for task_id, worker_id in result.assignments
        if task_id.parent.parent == JobName.root("test-user", "tree")
    ]
    assert len(child_assignments) == 2
    # child-a submitted first
    assert child_assignments[0][0].parent == JobName.from_string("/test-user/tree/child-a")
    assert child_assignments[1][0].parent == JobName.from_string("/test-user/tree/child-b")


# =============================================================================
# Device Index / Variant Scheduling Tests
# =============================================================================


def test_mixed_variant_cluster_schedules_all_matching_jobs(scheduler, state):
    """Jobs targeting different TPU variants each land on the correct worker."""
    variants = ["v5litepod-4", "v5litepod-16", "v5litepod-32"]
    for i, variant in enumerate(variants):
        meta = make_worker_metadata(tpu_name=variant)
        meta.device.tpu.variant = variant
        meta.device.tpu.count = 4
        register_worker(state, f"w-{variant}", f"addr{i}", meta)

    for variant in variants:
        req = controller_pb2.Controller.LaunchJobRequest(
            name=f"job-{variant}",
            entrypoint=_make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(
                cpu_millicores=1000,
                memory_bytes=1024**3,
                device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant=variant, count=2)),
            ),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        submit_job(state, f"job-{variant}", req)

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3
    for task_id, worker_id in result.assignments:
        # The job name encodes the variant it targets; the worker should match
        expected_variant = str(task_id.parent).split("job-")[1]
        worker = _query_worker(state, worker_id)
        assert worker.device_variant == expected_variant


def test_variant_none_job_schedules_on_any_tpu_worker(scheduler, state):
    """A TPU job with no specific variant schedules on any TPU worker."""
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    meta.device.tpu.count = 4
    register_worker(state, "w-tpu", "addr1", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="auto", count=2)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w-tpu")


def test_cpu_job_schedules_on_tpu_worker(scheduler, state):
    """A CPU job can run on a TPU worker since every host has a CPU."""
    meta = make_worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    meta.device.tpu.count = 4
    register_worker(state, "w-tpu", "addr1", meta)

    submit_job(state, "j1", make_job_request(cpu=1))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w-tpu")


def test_multiple_jobs_across_variants_in_single_cycle(scheduler, state):
    """Multiple jobs targeting different variants are all assigned in a single find_assignments call."""
    for variant in ["v5litepod-4", "v5litepod-16", "v5litepod-32"]:
        meta = make_worker_metadata(tpu_name=variant)
        meta.device.tpu.variant = variant
        meta.device.tpu.count = 4
        register_worker(state, f"w-{variant}", f"addr-{variant}", meta)

    for variant in ["v5litepod-4", "v5litepod-16", "v5litepod-32"]:
        req = controller_pb2.Controller.LaunchJobRequest(
            name=f"job-{variant}",
            entrypoint=_make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(
                cpu_millicores=1000,
                memory_bytes=1024**3,
                device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant=variant, count=2)),
            ),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        submit_job(state, f"job-{variant}", req)

    # Single call, not schedule_until_done
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 3 assigned in one cycle (round-robin gives each its own worker)
    assert len(result.assignments) == 3
    assigned_variants = set()
    for _, worker_id in result.assignments:
        worker = _query_worker(state, worker_id)
        assigned_variants.add(worker.device_variant)
    assert assigned_variants == {"v5litepod-4", "v5litepod-16", "v5litepod-32"}


def test_scheduler_tries_all_workers_before_rejecting(scheduler, state):
    """Scheduler must try all matching workers, not give up on first rejection."""
    # Register many workers with the wrong variant
    for i in range(10):
        meta = make_worker_metadata(tpu_name="v5litepod-32")
        meta.device.tpu.variant = "v5litepod-32"
        meta.device.tpu.count = 4
        register_worker(state, f"wrong-{i}", f"addr-wrong-{i}", meta)

    # Register one worker with the correct variant
    meta = make_worker_metadata(tpu_name="v5litepod-4")
    meta.device.tpu.variant = "v5litepod-4"
    meta.device.tpu.count = 4
    register_worker(state, "correct", "addr-correct", meta)

    # Job requesting v5litepod-4
    req = controller_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-4", count=2)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1, (
        f"Job should be scheduled on the v5litepod-4 worker regardless of iteration order, "
        f"got {len(result.assignments)} assignments"
    )
    assert result.assignments[0][1] == WorkerId("correct")


def test_many_jobs_on_single_variant_all_scheduled(state):
    """25 jobs targeting 8 workers of the same variant all get scheduled across cycles."""
    # High building limit so back-pressure doesn't interfere with the test
    sched = Scheduler(max_building_tasks_per_worker=1000)
    num_workers = 8
    num_jobs = 25

    for i in range(num_workers):
        meta = make_worker_metadata(cpu=100, memory_bytes=100 * 1024**3, tpu_name="v5litepod-8")
        meta.device.tpu.variant = "v5litepod-8"
        meta.device.tpu.count = 4
        register_worker(state, f"w{i}", f"addr{i}", meta)

    for i in range(num_jobs):
        req = controller_pb2.Controller.LaunchJobRequest(
            name=f"job-{i}",
            entrypoint=_make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        submit_job(state, f"job-{i}", req)

    result = schedule_until_done(sched, state)

    assert len(result.assignments) == num_jobs, (
        f"Expected all {num_jobs} jobs scheduled, got {len(result.assignments)}. "
        f"Remaining pending: {len(_schedulable_tasks(state))}"
    )
    assert len(_schedulable_tasks(state)) == 0


def test_mixed_variant_cluster_many_jobs_all_scheduled(state):
    """Mixed-variant cluster schedules all jobs to the correct device variant across cycles."""
    sched = Scheduler(max_building_tasks_per_worker=1000)
    # 10 v5litepod-4, 8 v5litepod-8, 20 v5litepod-16
    variant_workers = [
        ("v5litepod-4", 10),
        ("v5litepod-8", 8),
        ("v5litepod-16", 20),
    ]
    for variant, count in variant_workers:
        for i in range(count):
            meta = make_worker_metadata(cpu=100, memory_bytes=100 * 1024**3, tpu_name=variant)
            meta.device.tpu.variant = variant
            meta.device.tpu.count = 100
            register_worker(state, f"w-{variant}-{i}", f"addr-{variant}-{i}", meta)

    variant_jobs = [
        ("v5litepod-4", 60),
        ("v5litepod-8", 25),
        ("v5litepod-16", 40),
    ]
    total_jobs = 0
    for variant, count in variant_jobs:
        for i in range(count):
            req = controller_pb2.Controller.LaunchJobRequest(
                name=f"job-{variant}-{i}",
                entrypoint=_make_test_entrypoint(),
                resources=job_pb2.ResourceSpecProto(
                    cpu_millicores=1000,
                    memory_bytes=1024**3,
                    device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant=variant, count=1)),
                ),
                environment=job_pb2.EnvironmentConfig(),
                replicas=1,
            )
            submit_job(state, f"job-{variant}-{i}", req)
            total_jobs += 1

    result = schedule_until_done(sched, state)

    assert len(result.assignments) == total_jobs, (
        f"Expected all {total_jobs} jobs scheduled, got {len(result.assignments)}. "
        f"Remaining pending: {len(_schedulable_tasks(state))}"
    )
    assert len(_schedulable_tasks(state)) == 0

    # Verify each job landed on a worker with the correct variant
    for task_id, worker_id in result.assignments:
        job_name = str(task_id.parent)
        worker = _query_worker(state, worker_id)
        if "v5litepod-4" in job_name:
            assert (
                worker.device_variant == "v5litepod-4"
            ), f"Job {job_name} assigned to {worker.device_variant}, expected v5litepod-4"
        elif "v5litepod-8" in job_name:
            assert (
                worker.device_variant == "v5litepod-8"
            ), f"Job {job_name} assigned to {worker.device_variant}, expected v5litepod-8"
        elif "v5litepod-16" in job_name:
            assert (
                worker.device_variant == "v5litepod-16"
            ), f"Job {job_name} assigned to {worker.device_variant}, expected v5litepod-16"


def test_gpu_job_matches_worker_with_config_variant(scheduler, state):
    """A GPU job requesting variant="H100" matches a worker with device-variant="H100".

    In production, the worker's device-variant attribute comes from the scale
    group config (e.g. "H100"), not the nvidia-smi probe string. Both job and
    worker use the same canonical name, matched via EQ constraint.
    """
    meta = make_worker_metadata(gpu_count=8, gpu_name="H100")
    register_worker(state, "gpu-w1", "addr", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100", count=8)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    context = scheduler.create_scheduling_context(
        healthy_active_workers(state),
        pending_tasks=[t.task_id for t in tasks],
        jobs={tasks[0].job_id: _job_requirements_from_job(_query_job(state, tasks[0].job_id))},
    )
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1, f"Expected 1 assignment, got {len(result.assignments)}"
    assert result.assignments[0][1] == WorkerId("gpu-w1")


def _register_worker_with_probed_attributes(state, worker_id, address, metadata):
    """Register a worker, populating attributes via _build_worker_attributes (as real workers do)."""
    from iris.cluster.worker.env_probe import _build_worker_attributes

    # Determine accelerator_type and variant from the device config on metadata,
    # mirroring what the autoscaler would set on WorkerConfig.
    if metadata.device.HasField("tpu"):
        accel_type = config_pb2.ACCELERATOR_TYPE_TPU
        accel_variant = metadata.device.tpu.variant
    elif metadata.device.HasField("gpu"):
        accel_type = config_pb2.ACCELERATOR_TYPE_GPU
        accel_variant = metadata.device.gpu.variant
    else:
        accel_type = config_pb2.ACCELERATOR_TYPE_CPU
        accel_variant = ""

    attrs = _build_worker_attributes(
        accelerator_type=accel_type,
        accelerator_variant=accel_variant,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
        tpu_name=metadata.tpu_name,
        tpu_worker_id=str(0),
        device=metadata.device,
        extra_attributes={},
    )
    for key, val in attrs.items():
        metadata.attributes[key].CopyFrom(val)
    return register_worker(state, worker_id, address, metadata)


def test_device_variant_in_constraint_matches_probed_workers(scheduler, state):
    """device_variant_constraint matches workers whose attributes come from _build_worker_attributes.

    This is the end-to-end test: worker attributes are built the same way real
    workers build them, and the scheduler's IN constraint finds a match.

    Uses v5litepod-8 and v4-8 as the flexible alternatives (both vm_count=1)
    so the constraint represents a realistic flexible request.
    """
    meta1 = make_worker_metadata(tpu_name="v5litepod-8")
    _register_worker_with_probed_attributes(state, "w1", "addr1", meta1)

    meta2 = make_worker_metadata(tpu_name="v4-8")
    _register_worker_with_probed_attributes(state, "w2", "addr2", meta2)

    meta3 = make_worker_metadata(tpu_name="v5litepod-16")
    _register_worker_with_probed_attributes(state, "w3", "addr3", meta3)

    req = make_job_request()
    req.constraints.append(in_constraint(WellKnownAttribute.DEVICE_VARIANT, ["v5litepod-8", "v4-8"]).to_proto())

    submit_job(state, "flex-job", req)
    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 1
    assigned_worker = result.assignments[0][1]
    assert assigned_worker in {WorkerId("w1"), WorkerId("w2")}


# --- Pending diagnostics tests (merged from test_pending_diagnostics.py) ---


def _pending_task(job: str, idx: int) -> str:
    return JobName.root("test-user", job).task(idx).to_wire()


def test_build_job_pending_hints_reports_scale_up_group() -> None:
    routing = vm_pb2.RoutingDecision(
        group_to_launch={"tpu_v5e_32": 1},
        routed_entries={
            "tpu_v5e_32": vm_pb2.DemandEntryStatusList(
                entries=[vm_pb2.DemandEntryStatus(task_ids=[_pending_task("job-a", 0)])]
            )
        },
    )

    hints = build_job_pending_hints(routing)

    assert hints[JobName.root("test-user", "job-a").to_wire()] == PendingHint(
        message="Waiting for worker scale-up in scale group 'tpu_v5e_32' (1 slice(s) requested)",
        is_scaling_up=True,
    )


def test_build_job_pending_hints_reports_waiting_ready_when_no_launch() -> None:
    routing = vm_pb2.RoutingDecision(
        group_to_launch={"tpu_v5e_32": 0},
        routed_entries={
            "tpu_v5e_32": vm_pb2.DemandEntryStatusList(
                entries=[vm_pb2.DemandEntryStatus(task_ids=[_pending_task("job-b", 0), _pending_task("job-b", 1)])]
            )
        },
    )

    hints = build_job_pending_hints(routing)

    assert hints[JobName.root("test-user", "job-b").to_wire()] == PendingHint(
        message="Waiting for workers in scale group 'tpu_v5e_32' to become ready",
        is_scaling_up=False,
    )


def test_build_job_pending_hints_reports_unmet_when_not_routed() -> None:
    routing = vm_pb2.RoutingDecision(
        unmet_entries=[
            vm_pb2.UnmetDemand(
                entry=vm_pb2.DemandEntryStatus(task_ids=[_pending_task("job-c", 0)]),
                reason="no_matching_group: need device=tpu:v5p-8",
            )
        ]
    )

    hints = build_job_pending_hints(routing)

    assert hints[JobName.root("test-user", "job-c").to_wire()] == PendingHint(
        message="Unsatisfied autoscaler demand: no_matching_group: need device=tpu:v5p-8",
        is_scaling_up=False,
    )
