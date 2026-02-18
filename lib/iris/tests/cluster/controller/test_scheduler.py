# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task scheduler.

The scheduler is a shallow interface that takes inputs (pending tasks, workers,
job requirements) and returns outputs (assignments). It does not dispatch tasks,
modify state, or run threads.
"""

import pytest
from iris.cluster.controller.events import (
    JobSubmittedEvent,
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import JobRequirements, Scheduler, SchedulingResult
from iris.cluster.controller.state import ControllerState, ControllerTask
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def _job_requirements_from_job(job) -> JobRequirements:
    """Convert a ControllerJob to JobRequirements for testing."""
    return JobRequirements(
        resources=job.request.resources,
        constraints=list(job.request.constraints),
        is_coscheduled=job.is_coscheduled,
        coscheduling_group_by=job.coscheduling_group_by,
    )


# =============================================================================
# Event-Based Test Helpers
# =============================================================================


def register_worker(
    state: ControllerState,
    worker_id: str,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
) -> WorkerId:
    """Register a worker via event."""
    wid = WorkerId(worker_id)
    state.handle_event(
        WorkerRegisteredEvent(
            worker_id=wid,
            address=address,
            metadata=metadata,
            timestamp=Timestamp.now(),
        )
    )
    return wid


def submit_job(
    state: ControllerState,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
    timestamp_ms: int | None = None,
) -> list[ControllerTask]:
    """Submit a job via event and return created tasks."""
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root(job_id)
    request.name = jid.to_wire()
    state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp=Timestamp.from_ms(timestamp_ms) if timestamp_ms is not None else Timestamp.now(),
        )
    )
    return state.get_job_tasks(jid)


def assign_task_to_worker(state: ControllerState, task: ControllerTask, worker_id: WorkerId) -> None:
    """Assign a task to a worker via event."""
    state.handle_event(
        TaskAssignedEvent(
            task_id=task.task_id,
            worker_id=worker_id,
        )
    )


def transition_task_to_running(state: ControllerState, task: ControllerTask) -> None:
    """Transition a task to RUNNING state via event."""
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task.task_id,
            new_state=cluster_pb2.TASK_STATE_RUNNING,
            attempt_id=task.current_attempt_id,
        )
    )


def _build_context(scheduler, state):
    """Build a SchedulingContext from state, including pending tasks and job requirements."""
    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()
    building_counts = state.snapshot_building_counts()

    # Build task IDs and job requirements from pending tasks
    task_ids = []
    jobs = {}
    for task in pending_tasks:
        if not task.can_be_scheduled():
            continue
        task_ids.append(task.task_id)
        if task.job_id not in jobs:
            job = state.get_job(task.job_id)
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
    state: ControllerState,
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
            task = state.get_task(task_id)
            if task:
                assign_task_to_worker(state, task, worker_id)

    return SchedulingResult(assignments=all_assignments)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        cpu: int = 1,
        memory_bytes: int = 1024**3,
        scheduling_timeout_seconds: int = 0,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        from iris.time_utils import Duration

        job_name = JobName.from_string(name) if name.startswith("/") else JobName.root(name)
        request = cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )
        if scheduling_timeout_seconds > 0:
            request.scheduling_timeout.CopyFrom(Duration.from_seconds(scheduling_timeout_seconds).to_proto())
        return request

    return _make


@pytest.fixture
def coscheduled_job_request():
    """Create a coscheduled LaunchJobRequest for TPU-style multi-replica jobs."""

    def _make(
        name: str = "coscheduled-job",
        cpu: int = 1,
        memory_bytes: int = 1024**3,
        replicas: int = 4,
        group_by: str = "tpu-name",
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        job_name = JobName.from_string(name) if name.startswith("/") else JobName.root(name)
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=replicas,
        )
        req.coscheduling.group_by = group_by
        return req

    return _make


@pytest.fixture
def resource_spec():
    """Create a ResourceSpec for testing with enough capacity for multiple jobs."""

    def _make(cpu: int = 10, memory_bytes: int = 10 * 1024**3) -> cluster_pb2.ResourceSpecProto:
        return cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes, disk_bytes=10 * 1024**3)

    return _make


@pytest.fixture
def worker_metadata():
    """Create WorkerMetadata for testing."""

    def _make(
        cpu: int = 10,
        memory_bytes: int = 10 * 1024**3,
        disk_bytes: int = 10 * 1024**3,
        gpu_count: int = 0,
        gpu_name: str = "",
        tpu_name: str = "",
    ) -> cluster_pb2.WorkerMetadata:
        device = cluster_pb2.DeviceConfig()
        if tpu_name:
            device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant=tpu_name))
        elif gpu_count > 0:
            device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant=gpu_name or "auto", count=gpu_count))
        else:
            device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        return cluster_pb2.WorkerMetadata(
            hostname="test-worker",
            ip_address="127.0.0.1",
            cpu_count=cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            tpu_name=tpu_name,
            device=device,
        )

    return _make


@pytest.fixture
def state():
    """Create a fresh ControllerState for each test."""
    return ControllerState()


@pytest.fixture
def scheduler():
    """Create a Scheduler instance."""
    return Scheduler()


def test_scheduler_finds_assignment_for_task(scheduler, state, job_request, worker_metadata):
    """Verify scheduler assigns task to available worker."""
    register_worker(state, "w1", "addr", worker_metadata())

    tasks = submit_job(state, "j1", job_request())
    task = tasks[0]

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == task.task_id
    assert result.assignments[0][1] == WorkerId("w1")


def test_scheduler_returns_empty_when_no_workers(scheduler, state, job_request):
    """Verify scheduler returns empty result when no workers available."""
    submit_job(state, "j1", job_request())

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 0


def test_scheduler_round_robins_tasks_across_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler distributes tasks across workers instead of packing one worker."""
    register_worker(state, "w1", "addr1", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))
    register_worker(state, "w2", "addr2", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))
    register_worker(state, "w3", "addr3", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    submit_job(state, "j1", job_request(cpu=2))
    submit_job(state, "j2", job_request(cpu=2))
    submit_job(state, "j3", job_request(cpu=2))

    result = schedule_until_done(scheduler, state)

    # All 3 tasks assigned, each to a different worker (round-robin)
    assert len(result.assignments) == 3
    assigned_worker_ids = {worker_id for _, worker_id in result.assignments}
    assert len(assigned_worker_ids) == 3


def test_scheduler_assigns_multiple_tasks_to_single_worker(scheduler, state, job_request, worker_metadata):
    """Verify scheduler assigns multiple tasks to one worker when it's the only option."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    tasks1 = submit_job(state, "j1", job_request(cpu=2))
    tasks2 = submit_job(state, "j2", job_request(cpu=2))
    tasks3 = submit_job(state, "j3", job_request(cpu=2))

    result = schedule_until_done(scheduler, state)

    # All 3 tasks eventually assigned to the single worker
    assert len(result.assignments) == 3
    assigned_task_ids = {task_id for task_id, _ in result.assignments}
    assert assigned_task_ids == {tasks1[0].task_id, tasks2[0].task_id, tasks3[0].task_id}
    # All assigned to the same worker
    assert all(worker_id == WorkerId("w1") for _, worker_id in result.assignments)


def test_scheduler_skips_tasks_that_dont_fit(scheduler, state, job_request, worker_metadata):
    """Verify scheduler skips tasks that don't fit and continues to next."""
    # Worker with 4 CPUs
    register_worker(state, "w1", "addr", worker_metadata(cpu=4, memory_bytes=16 * 1024**3))

    # Job 1: needs 8 CPUs (won't fit on 4 CPU worker)
    submit_job(state, "j1", job_request(cpu=8))
    # Job 2: needs 2 CPUs (will fit)
    tasks2 = submit_job(state, "j2", job_request(cpu=2))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Only job2's task should be assigned
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks2[0].task_id


def test_scheduler_detects_timed_out_tasks(state, worker_metadata):
    """Verify timed-out tasks are handled by the controller (not the scheduler).

    The scheduler no longer handles timeouts -- the controller filters them out
    before calling find_assignments. This test verifies the overall behavior
    by testing the controller-level flow.
    """
    import time

    from iris.time_utils import Deadline, Duration

    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that requires 100 CPUs (will never fit) with 1 second timeout
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="impossible-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=100, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    request.scheduling_timeout.CopyFrom(Duration.from_seconds(1).to_proto())
    tasks = submit_job(state, "j1", request)

    # Manually set the deadline to 2 seconds ago (using monotonic time)
    job = state.get_job(JobName.root("j1"))
    job.scheduling_deadline = Deadline(time.monotonic() - 2.0)

    # When building context, the timed-out task should be filtered out
    pending_tasks = state.peek_pending_tasks()

    # Simulate controller-level timeout filtering
    schedulable_task_ids = []
    jobs = {}
    timed_out_tasks = []
    for task in pending_tasks:
        if not task.can_be_scheduled():
            continue
        j = state.get_job(task.job_id)
        if j and j.scheduling_deadline is not None and j.scheduling_deadline.expired():
            timed_out_tasks.append(task)
            continue
        schedulable_task_ids.append(task.task_id)
        if task.job_id not in jobs:
            jobs[task.job_id] = _job_requirements_from_job(j)

    # The task is timed out, so no schedulable tasks
    assert len(timed_out_tasks) == 1
    assert timed_out_tasks[0] == tasks[0]
    assert len(schedulable_task_ids) == 0


def test_scheduler_no_timeout_when_zero(scheduler, state, worker_metadata):
    """Verify task with scheduling_timeout=0 never times out."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that can't fit but has no timeout (0)
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="no-timeout-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=100, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    # No timeout set (field not present)
    submit_job(state, "j1", request, timestamp_ms=Timestamp.now().epoch_ms() - 10000)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Task should not be assigned (just skipped, no assignment)
    assert len(result.assignments) == 0


def test_scheduler_respects_worker_capacity_across_assignments(scheduler, state, job_request, worker_metadata):
    """Verify scheduler tracks capacity used by earlier assignments across cycles."""
    # Worker with 4 CPUs
    register_worker(state, "w1", "addr", worker_metadata(cpu=4))

    # Submit 3 jobs, each requiring 2 CPUs (only 2 will fit)
    for i in range(3):
        submit_job(state, f"j{i}", job_request(cpu=2))

    result = schedule_until_done(scheduler, state)

    # Only 2 tasks assigned (4 CPUs / 2 CPUs each = 2 tasks max)
    assert len(result.assignments) == 2

    # Third task still pending
    pending = state.peek_pending_tasks()
    assert len(pending) == 1


def test_scheduler_skips_unhealthy_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler ignores unhealthy workers."""
    register_worker(state, "w1", "addr1", worker_metadata())
    register_worker(state, "w2", "addr2", worker_metadata())
    # Mark second worker as unhealthy
    unhealthy_worker = state.get_worker(WorkerId("w2"))
    unhealthy_worker.healthy = False

    submit_job(state, "j1", job_request())

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w1")


def test_scheduler_considers_running_tasks_for_capacity(scheduler, state, job_request, worker_metadata):
    """Verify scheduler accounts for tasks already running on workers."""
    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "addr", worker_metadata(cpu=4))

    # Submit a job that uses 3 CPUs, assign it to the worker, and mark it running
    running_tasks = submit_job(state, "running", job_request(cpu=3))
    assign_task_to_worker(state, running_tasks[0], worker_id)
    transition_task_to_running(state, running_tasks[0])

    # Try to schedule a job that needs 2 CPUs (won't fit, only 1 CPU available)
    submit_job(state, "j1", job_request(cpu=2))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 0


def test_scheduler_reports_task_too_large_for_cluster(scheduler, state, job_request, worker_metadata):
    """Verify scheduler reports when a task requires more resources than any worker can provide."""
    # Worker with only 2 CPUs
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that needs 4 CPUs
    submit_job(state, "j1", job_request(cpu=4))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Primary observable behavior: task cannot be assigned
    assert len(result.assignments) == 0


# =============================================================================
# Constraint Filtering Tests
# =============================================================================


def test_constraint_filters_workers_by_attribute(scheduler, state, job_request, worker_metadata):
    """Job with constraint only schedules on workers with matching attribute."""
    # Worker 1 with tpu-name attribute
    meta1 = worker_metadata()
    meta1.attributes["tpu-name"].string_value = "tpu-a"
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2 with different tpu-name
    meta2 = worker_metadata()
    meta2.attributes["tpu-name"].string_value = "tpu-b"
    register_worker(state, "w2", "addr2", meta2)

    # Job with constraint requiring tpu-name = "tpu-a"
    req = job_request()
    constraint = req.constraints.add()
    constraint.key = "tpu-name"
    constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "tpu-a"
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
        (cluster_pb2.CONSTRAINT_OP_EQ, "us-west", "us-west", True),
        (cluster_pb2.CONSTRAINT_OP_EQ, "us-east", "us-west", False),
        # NE operator tests
        (cluster_pb2.CONSTRAINT_OP_NE, "us-east", "us-west", True),
        (cluster_pb2.CONSTRAINT_OP_NE, "us-west", "us-west", False),
    ],
    ids=[
        "EQ-match",
        "EQ-no-match",
        "NE-match",
        "NE-no-match",
    ],
)
def test_constraint_string_operators(
    scheduler, state, job_request, worker_metadata, op, worker_value, constraint_value, should_match
):
    """String equality operators (EQ, NE) filter workers by attribute value."""
    meta = worker_metadata()
    meta.attributes["region"].string_value = worker_value
    register_worker(state, "w1", "addr", meta)

    req = job_request()
    constraint = req.constraints.add()
    constraint.key = "region"
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
        (cluster_pb2.CONSTRAINT_OP_EXISTS, True, True),
        (cluster_pb2.CONSTRAINT_OP_EXISTS, False, False),
        (cluster_pb2.CONSTRAINT_OP_NOT_EXISTS, True, False),
        (cluster_pb2.CONSTRAINT_OP_NOT_EXISTS, False, True),
    ],
    ids=[
        "EXISTS-present",
        "EXISTS-absent",
        "NOT_EXISTS-present",
        "NOT_EXISTS-absent",
    ],
)
def test_constraint_existence_operators(
    scheduler, state, job_request, worker_metadata, op, worker_has_attribute, should_match
):
    """Existence operators (EXISTS, NOT_EXISTS) check for attribute presence."""
    meta = worker_metadata()
    if worker_has_attribute:
        meta.attributes["gpu-model"].string_value = "A100"
    register_worker(state, "w1", "addr", meta)

    req = job_request()
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
        (cluster_pb2.CONSTRAINT_OP_GT, 10, 5, True),
        (cluster_pb2.CONSTRAINT_OP_GT, 5, 5, False),
        (cluster_pb2.CONSTRAINT_OP_GT, 3, 5, False),
        # GE: worker >= constraint
        (cluster_pb2.CONSTRAINT_OP_GE, 10, 5, True),
        (cluster_pb2.CONSTRAINT_OP_GE, 5, 5, True),
        (cluster_pb2.CONSTRAINT_OP_GE, 3, 5, False),
        # LT: worker < constraint
        (cluster_pb2.CONSTRAINT_OP_LT, 3, 5, True),
        (cluster_pb2.CONSTRAINT_OP_LT, 5, 5, False),
        (cluster_pb2.CONSTRAINT_OP_LT, 10, 5, False),
        # LE: worker <= constraint
        (cluster_pb2.CONSTRAINT_OP_LE, 3, 5, True),
        (cluster_pb2.CONSTRAINT_OP_LE, 5, 5, True),
        (cluster_pb2.CONSTRAINT_OP_LE, 10, 5, False),
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
def test_constraint_numeric_operators(
    scheduler, state, job_request, worker_metadata, op, worker_value, constraint_value, should_match
):
    """Numeric comparison operators (GT, GE, LT, LE) compare attribute values."""
    meta = worker_metadata()
    meta.attributes["priority"].int_value = worker_value
    register_worker(state, "w1", "addr", meta)

    req = job_request()
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


def test_constraint_numeric_operators_with_floats(scheduler, state, job_request, worker_metadata):
    """Numeric comparison operators work with float values."""
    meta = worker_metadata()
    meta.attributes["load"].float_value = 0.3
    register_worker(state, "w1", "addr", meta)

    req = job_request()
    constraint = req.constraints.add()
    constraint.key = "load"
    constraint.op = cluster_pb2.CONSTRAINT_OP_LT
    constraint.value.float_value = 0.5
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w1")


def test_multiple_constraints_all_must_match(scheduler, state, job_request, worker_metadata):
    """Multiple constraints are ANDed together."""
    # Worker 1: tpu-name=tpu-a, tpu-worker-id=0
    meta1 = worker_metadata()
    meta1.attributes["tpu-name"].string_value = "tpu-a"
    meta1.attributes["tpu-worker-id"].int_value = 0
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2: tpu-name=tpu-a, tpu-worker-id=1
    meta2 = worker_metadata()
    meta2.attributes["tpu-name"].string_value = "tpu-a"
    meta2.attributes["tpu-worker-id"].int_value = 1
    register_worker(state, "w2", "addr2", meta2)

    # Worker 3: tpu-name=tpu-b, tpu-worker-id=0
    meta3 = worker_metadata()
    meta3.attributes["tpu-name"].string_value = "tpu-b"
    meta3.attributes["tpu-worker-id"].int_value = 0
    register_worker(state, "w3", "addr3", meta3)

    # Job requiring tpu-name=tpu-a AND tpu-worker-id=0
    req = job_request()
    c1 = req.constraints.add()
    c1.key = "tpu-name"
    c1.op = cluster_pb2.CONSTRAINT_OP_EQ
    c1.value.string_value = "tpu-a"
    c2 = req.constraints.add()
    c2.key = "tpu-worker-id"
    c2.op = cluster_pb2.CONSTRAINT_OP_EQ
    c2.value.int_value = 0
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Only w1 matches both constraints
    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w1")


def test_constraint_with_missing_attribute_fails(scheduler, state, job_request, worker_metadata):
    """Constraint on missing attribute fails for EQ/NE/GT/etc (except NOT_EXISTS)."""
    # Worker without the required attribute
    meta = worker_metadata()
    register_worker(state, "w1", "addr", meta)

    # Job requiring tpu-name = "tpu-a"
    req = job_request()
    constraint = req.constraints.add()
    constraint.key = "tpu-name"
    constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "tpu-a"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Worker doesn't have tpu-name attribute, so constraint fails
    assert len(result.assignments) == 0


def test_job_without_constraints_schedules_anywhere(scheduler, state, job_request, worker_metadata):
    """Job without constraints can be scheduled on any worker."""
    # Worker 1 with attribute
    meta1 = worker_metadata()
    meta1.attributes["tpu-name"].string_value = "tpu-a"
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2 without attribute
    meta2 = worker_metadata()
    register_worker(state, "w2", "addr2", meta2)

    # Job without constraints
    req = job_request()
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should be assigned to either worker
    assert len(result.assignments) == 1


# =============================================================================
# Coscheduling Tests
# =============================================================================


def test_coscheduled_job_assigns_all_tasks_atomically(scheduler, state, worker_metadata):
    """Coscheduled job assigns all tasks to workers in the same group."""
    # Create 4 workers on tpu-a
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should be assigned
    assert len(result.assignments) == 4

    # All assigned to workers with same tpu-name
    assigned_worker_ids = {worker_id for _, worker_id in result.assignments}
    # Verify all workers are in the tpu-a group
    for worker_id in assigned_worker_ids:
        worker = state.get_worker(worker_id)
        assert worker.attributes["tpu-name"].value == "tpu-a"

    # Tasks assigned in order: task-0 -> worker-0, task-1 -> worker-1, etc.
    for task_id, worker_id in result.assignments:
        task = state.get_task(task_id)
        expected_worker_id = f"w{task.task_index}"
        assert worker_id == WorkerId(expected_worker_id)


def test_coscheduled_job_waits_when_insufficient_workers(scheduler, state, worker_metadata):
    """Coscheduled job stays pending when not enough workers in any group."""
    # Only 2 workers on tpu-a
    for i in range(2):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Job requires 4 replicas
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # No assignments - job stays pending
    assert len(result.assignments) == 0


def test_coscheduled_job_chooses_group_with_capacity(scheduler, state, worker_metadata):
    """Coscheduled job chooses the group that has capacity."""
    # tpu-a: 4 workers, 2 are busy (low capacity)
    for i in range(4):
        meta = worker_metadata(cpu=2)  # Each worker has 2 CPUs
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Consume capacity on first 2 workers of tpu-a by submitting a job
    busy_req = cluster_pb2.Controller.LaunchJobRequest(
        name="busy-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3),
        replicas=2,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    submit_job(state, "busy", busy_req)

    # Assign the busy job's tasks to wa0 and wa1
    busy_tasks = state.get_job_tasks(JobName.root("busy"))
    assign_task_to_worker(state, busy_tasks[0], WorkerId("wa0"))
    assign_task_to_worker(state, busy_tasks[1], WorkerId("wa1"))
    transition_task_to_running(state, busy_tasks[0])
    transition_task_to_running(state, busy_tasks[1])

    # tpu-b: 4 workers, all free
    for i in range(4):
        meta = worker_metadata(cpu=2)
        meta.attributes["tpu-name"].string_value = "tpu-b"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Job should be assigned to tpu-b (has 4 free workers)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        worker = state.get_worker(worker_id)
        assert worker.attributes["tpu-name"].value == "tpu-b"


def test_coscheduled_job_assigns_tasks_in_order(scheduler, state, worker_metadata):
    """Task indices map to worker IDs in sorted order."""
    # Create workers with non-sequential IDs to verify sorting
    worker_ids = [3, 1, 0, 2]  # Deliberately out of order
    for i, wid in enumerate(worker_ids):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = wid
        register_worker(state, f"w{wid}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 4

    # Verify task-0 -> worker with tpu-worker-id=0, task-1 -> worker with tpu-worker-id=1, etc.
    for task_id, worker_id in result.assignments:
        task = state.get_task(task_id)
        worker = state.get_worker(worker_id)
        worker_tpu_id = worker.attributes["tpu-worker-id"].value
        assert (
            task.task_index == worker_tpu_id
        ), f"Task {task.task_index} assigned to worker with tpu-worker-id={worker_tpu_id}"


def test_coscheduled_job_with_constraints(scheduler, state, worker_metadata):
    """Coscheduled job respects additional constraints."""
    # tpu-a: 4 workers with region=us-west
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        meta.attributes["region"].string_value = "us-west"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # tpu-b: 4 workers with region=us-east
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-b"
        meta.attributes["tpu-worker-id"].int_value = i
        meta.attributes["region"].string_value = "us-east"
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job requiring region=us-east
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    constraint = req.constraints.add()
    constraint.key = "region"
    constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "us-east"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should be assigned to tpu-b (only group matching region=us-east)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        worker = state.get_worker(worker_id)
        assert worker.attributes["tpu-name"].value == "tpu-b"


def test_coscheduled_job_with_partial_capacity(scheduler, state, worker_metadata):
    """Coscheduled job waits when some workers in group lack capacity, then schedules when capacity is added."""
    # Create 4 workers, but 2 have insufficient CPU
    for i in range(4):
        cpu = 2 if i < 2 else 1  # First 2 have 2 CPU, last 2 have only 1
        meta = worker_metadata(cpu=cpu)
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # No assignments - only 2 workers have sufficient capacity
    assert len(result.assignments) == 0

    # Now add a new TPU group with 4 workers, all with sufficient capacity
    for i in range(4):
        meta = worker_metadata(cpu=2)
        meta.attributes["tpu-name"].string_value = "tpu-b"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Re-run the scheduler - job should now be assigned to the new group
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should now be assigned to tpu-b
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        worker = state.get_worker(worker_id)
        assert worker.attributes["tpu-name"].value == "tpu-b"


# =============================================================================
# Taint Constraint Tests
# =============================================================================


def test_tainted_worker_not_used_for_coscheduled_job(scheduler, state, worker_metadata):
    """Coscheduled job skips groups containing tainted workers."""
    # Create TPU group "tpu-a" with 4 workers, one tainted
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        if i == 0:
            meta.attributes["taint:maintenance"].string_value = "true"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Create TPU group "tpu-b" with 4 workers, none tainted
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-b"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job with 4 replicas + NOT_EXISTS taint constraint
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    c = req.constraints.add()
    c.key = "taint:maintenance"
    c.op = cluster_pb2.CONSTRAINT_OP_NOT_EXISTS
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # All 4 tasks should be assigned to tpu-b (tpu-a has a tainted worker)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        worker = state.get_worker(worker_id)
        assert worker.attributes["tpu-name"].value == "tpu-b"


# =============================================================================
# TPU Chip Count Tracking Tests
# =============================================================================


def test_tpu_chip_count_deducted_from_capacity(scheduler, state):
    """TPU chip count is deducted when task is scheduled."""
    # Worker with 4 TPU chips (simulating v5litepod-16 per-VM)
    meta = cluster_pb2.WorkerMetadata(
        hostname="tpu-worker",
        ip_address="127.0.0.1",
        cpu_count=10,
        memory_bytes=10 * 1024**3,
        disk_bytes=10 * 1024**3,
        tpu_name="v5litepod-16",
    )
    device = cluster_pb2.DeviceConfig()
    device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))
    meta.device.CopyFrom(device)
    register_worker(state, "w1", "addr1", meta)

    # First job requires 4 TPU chips
    req1 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
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
    req2 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j2", req2)

    # Second scheduling cycle - no TPU chips available
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0


def test_tpu_job_rejected_when_insufficient_chips(scheduler, state):
    """TPU job is not scheduled when worker has fewer chips than required."""
    # Worker with 4 TPU chips
    meta = cluster_pb2.WorkerMetadata(
        hostname="tpu-worker",
        ip_address="127.0.0.1",
        cpu_count=10,
        memory_bytes=10 * 1024**3,
        disk_bytes=10 * 1024**3,
        tpu_name="v5litepod-16",
    )
    device = cluster_pb2.DeviceConfig()
    device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))
    meta.device.CopyFrom(device)
    register_worker(state, "w1", "addr1", meta)

    # Job requires 8 TPU chips - more than worker has
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=8)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Task should not be scheduled - not enough TPU chips
    assert len(result.assignments) == 0


def test_tpu_count_released_after_task_completion(scheduler, state):
    """TPU chips are released when task completes, allowing new tasks to schedule."""
    # Worker with 4 TPU chips
    meta = cluster_pb2.WorkerMetadata(
        hostname="tpu-worker",
        ip_address="127.0.0.1",
        cpu_count=10,
        memory_bytes=10 * 1024**3,
        disk_bytes=10 * 1024**3,
        tpu_name="v5litepod-16",
    )
    device = cluster_pb2.DeviceConfig()
    device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))
    meta.device.CopyFrom(device)
    register_worker(state, "w1", "addr1", meta)

    # First job uses all 4 TPU chips
    req1 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks1 = submit_job(state, "j1", req1)
    assign_task_to_worker(state, tasks1[0], WorkerId("w1"))
    transition_task_to_running(state, tasks1[0])

    # Submit second job
    req2 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j2", req2)

    # Second job can't be scheduled yet
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0

    # Complete first task
    state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks1[0].task_id,
            new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
            attempt_id=tasks1[0].current_attempt_id,
        )
    )

    # Now second job can be scheduled
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0].parent == JobName.root("j2")


# =============================================================================
# Preemptible Constraint Tests
# =============================================================================


def test_preemptible_constraint_routes_to_matching_worker(scheduler, state, job_request, worker_metadata):
    """Job constrained to non-preemptible workers is only scheduled on a matching worker."""
    # Preemptible worker
    meta_preemptible = worker_metadata()
    meta_preemptible.attributes["preemptible"].string_value = "true"
    register_worker(state, "w-preemptible", "addr1", meta_preemptible)

    # On-demand worker
    meta_ondemand = worker_metadata()
    meta_ondemand.attributes["preemptible"].string_value = "false"
    register_worker(state, "w-ondemand", "addr2", meta_ondemand)

    # Job requiring non-preemptible worker
    req = job_request()
    constraint = req.constraints.add()
    constraint.key = "preemptible"
    constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "false"
    tasks = submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks[0].task_id
    assert result.assignments[0][1] == WorkerId("w-ondemand")


# =============================================================================
# Depth-First Scheduling Priority Assignment Tests
# =============================================================================


def test_scheduler_assigns_deeper_job_before_shallow(scheduler, state, job_request, worker_metadata):
    """Scheduler assigns deeper jobs before shallow ones when both fit."""
    # Worker with enough resources for both jobs
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit root job and child job (both with 1 CPU)
    submit_job(state, "root", job_request("root", cpu=1))
    submit_job(state, "/root/child", job_request("child", cpu=1))

    # Run scheduler
    result = schedule_until_done(scheduler, state)

    # Both tasks assigned, child first
    assert len(result.assignments) == 2
    assert result.assignments[0][0].parent == JobName.from_string("/root/child")
    assert result.assignments[1][0].parent == JobName.root("root")


def test_scheduler_assigns_older_root_tree_first(scheduler, state, job_request, worker_metadata):
    """At same depth, scheduler assigns older root tree first."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit two root jobs
    submit_job(state, "user-a-job", job_request("user-a-job", cpu=1))
    submit_job(state, "user-b-job", job_request("user-b-job", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 2
    # user-a-job submitted first
    assert result.assignments[0][0].parent == JobName.root("user-a-job")
    assert result.assignments[1][0].parent == JobName.root("user-b-job")


def test_scheduler_child_of_older_tree_beats_newer_root(scheduler, state, job_request, worker_metadata):
    """Child of older tree is assigned before root of newer tree."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit old tree
    submit_job(state, "old-tree", job_request("old-tree", cpu=1))

    # Submit new tree
    submit_job(state, "new-tree", job_request("new-tree", cpu=1))

    # Submit child of old tree
    submit_job(state, "/old-tree/child", job_request("child", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3
    # Order: child (depth 2), old-tree (depth 1, older), new-tree (depth 1, newer)
    assert result.assignments[0][0].parent == JobName.from_string("/old-tree/child")
    assert result.assignments[1][0].parent == JobName.root("old-tree")
    assert result.assignments[2][0].parent == JobName.root("new-tree")


# =============================================================================
# Error Message Tests
# =============================================================================


def test_scheduler_reports_device_variant_mismatch(scheduler, state, worker_metadata):
    """Scheduler reports device variant mismatch in error message."""
    # Worker with v5litepod-16
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    register_worker(state, "w1", "addr", meta)

    # Job requesting v5litepod-32
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-32", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(state.get_available_workers())
    job = state.get_job(tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next((t.task_id for t in state.get_job_tasks(job.job_id) if t.can_be_scheduled()), None)
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(state.get_job_tasks(job.job_id))
    )

    assert "variant" in diagnostics.lower()
    assert "v5litepod-32" in diagnostics
    assert "v5litepod-16" in diagnostics


def test_scheduler_reports_tpu_count_exceeded(scheduler, state, worker_metadata):
    """Scheduler reports TPU count exceeded in error message."""
    # Worker with 4 TPU chips
    meta = cluster_pb2.WorkerMetadata(
        hostname="tpu-worker",
        ip_address="127.0.0.1",
        cpu_count=10,
        memory_bytes=10 * 1024**3,
        disk_bytes=10 * 1024**3,
        tpu_name="v5litepod-16",
    )
    device = cluster_pb2.DeviceConfig()
    device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))
    meta.device.CopyFrom(device)
    register_worker(state, "w1", "addr1", meta)

    # Job requesting 8 TPU chips
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=8)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(state.get_available_workers())
    job = state.get_job(tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next((t.task_id for t in state.get_job_tasks(job.job_id) if t.can_be_scheduled()), None)
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(state.get_job_tasks(job.job_id))
    )

    assert "tpu" in diagnostics.lower()
    assert "8" in diagnostics
    assert "4" in diagnostics


def test_scheduler_reports_device_type_mismatch(scheduler, state, worker_metadata):
    """Scheduler reports device type mismatch in error message."""
    # CPU-only worker
    meta = worker_metadata()
    register_worker(state, "w1", "addr", meta)

    # Job requesting TPU
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(state.get_available_workers())
    job = state.get_job(tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next((t.task_id for t in state.get_job_tasks(job.job_id) if t.can_be_scheduled()), None)
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(state.get_job_tasks(job.job_id))
    )

    assert "device" in diagnostics.lower()
    assert "tpu" in diagnostics.lower()


def test_scheduler_reports_coscheduling_capacity_details(scheduler, state, worker_metadata):
    """Scheduler reports detailed coscheduling capacity issues."""
    # Create 4 workers but only 2 have sufficient CPU
    for i in range(4):
        cpu = 4 if i < 2 else 1  # First 2 have 4 CPU, last 2 have only 1
        meta = worker_metadata(cpu=cpu)
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(state.get_available_workers())
    job = state.get_job(tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next((t.task_id for t in state.get_job_tasks(job.job_id) if t.can_be_scheduled()), None)
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(state.get_job_tasks(job.job_id))
    )

    # Should mention it's a coscheduling issue with capacity details
    assert "coscheduling" in diagnostics.lower() or "group" in diagnostics.lower()
    # Should indicate how many workers have capacity vs needed
    assert "2" in diagnostics or "4" in diagnostics


def test_diagnostics_for_schedulable_job_does_not_say_unknown_failure(scheduler, state, job_request, worker_metadata):
    """When a job can be scheduled, diagnostics should not say 'Unknown scheduling failure'."""
    register_worker(state, "w1", "addr1", worker_metadata())
    tasks = submit_job(state, "j1", job_request())

    context = scheduler.create_scheduling_context(state.get_available_workers())
    job = state.get_job(tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next((t.task_id for t in state.get_job_tasks(job.job_id) if t.can_be_scheduled()), None)
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(state.get_job_tasks(job.job_id))
    )

    assert "unknown" not in diagnostics.lower()
    assert "schedulable" in diagnostics.lower()


def test_coscheduled_tpu_jobs_cannot_double_book_group(scheduler, state):
    """Two coscheduled TPU jobs cannot use the same TPU group simultaneously."""
    # Create 4 workers in tpu-group "tpu-a", each with 4 TPU chips
    for i in range(4):
        meta = cluster_pb2.WorkerMetadata(
            hostname=f"tpu-worker-{i}",
            ip_address="127.0.0.1",
            cpu_count=10,
            memory_bytes=10 * 1024**3,
            disk_bytes=10 * 1024**3,
            tpu_name="v5litepod-16",
        )
        device = cluster_pb2.DeviceConfig()
        device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))
        meta.device.CopyFrom(device)
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    tpu_resource = cluster_pb2.ResourceSpecProto(
        cpu=1,
        memory_bytes=1024**3,
        device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
    )

    # Job 1: coscheduled across all 4 workers
    req1 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=tpu_resource,
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req1.coscheduling.group_by = "tpu-name"
    tasks1 = submit_job(state, "j1", req1)

    # Schedule and commit job 1
    result1 = schedule_until_done(scheduler, state)
    assert len(result1.assignments) == 4
    for task in tasks1:
        transition_task_to_running(state, task)

    # Job 2: same shape, should be blocked because TPU chips are exhausted
    req2 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        entrypoint=_make_test_entrypoint(),
        resources=tpu_resource,
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req2.coscheduling.group_by = "tpu-name"
    submit_job(state, "j2", req2)

    context = _build_context(scheduler, state)
    result2 = scheduler.find_assignments(context)
    assert len(result2.assignments) == 0

    # Complete all job 1 tasks
    for task in tasks1:
        state.handle_event(
            TaskStateChangedEvent(
                task_id=task.task_id,
                new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
                attempt_id=task.current_attempt_id,
            )
        )

    # Job 2 should now be schedulable
    result3 = schedule_until_done(scheduler, state)
    assert len(result3.assignments) == 4
    assigned_jobs = {task_id.parent for task_id, _ in result3.assignments}
    assert assigned_jobs == {JobName.root("j2")}


def test_scheduler_fifo_within_same_depth_and_tree(scheduler, state, job_request, worker_metadata):
    """Scheduler respects FIFO within same depth and tree."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit parent
    submit_job(state, "tree", job_request("tree", cpu=1))

    # Submit two children
    submit_job(state, "/tree/child-a", job_request("child-a", cpu=1))
    submit_job(state, "/tree/child-b", job_request("child-b", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3

    # Find child assignments
    child_assignments = [
        (task_id, worker_id)
        for task_id, worker_id in result.assignments
        if task_id.parent.parent == JobName.root("tree")
    ]
    assert len(child_assignments) == 2
    # child-a submitted first
    assert child_assignments[0][0].parent == JobName.from_string("/tree/child-a")
    assert child_assignments[1][0].parent == JobName.from_string("/tree/child-b")


# =============================================================================
# Device Index / Variant Scheduling Tests
# =============================================================================


def test_mixed_variant_cluster_schedules_all_matching_jobs(scheduler, state, worker_metadata):
    """Jobs targeting different TPU variants each land on the correct worker."""
    variants = ["v5litepod-4", "v5litepod-16", "v5litepod-32"]
    for i, variant in enumerate(variants):
        meta = worker_metadata(tpu_name=variant)
        meta.device.tpu.variant = variant
        meta.device.tpu.count = 4
        register_worker(state, f"w-{variant}", f"addr{i}", meta)

    for variant in variants:
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=f"job-{variant}",
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(
                cpu=1,
                memory_bytes=1024**3,
                device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant=variant, count=2)),
            ),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )
        submit_job(state, f"job-{variant}", req)

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3
    for task_id, worker_id in result.assignments:
        # The job name encodes the variant it targets; the worker should match
        expected_variant = str(task_id.parent).split("job-")[1]
        worker = state.get_worker(worker_id)
        assert worker.device_variant == expected_variant


def test_variant_none_job_schedules_on_any_tpu_worker(scheduler, state, worker_metadata):
    """A TPU job with no specific variant schedules on any TPU worker."""
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    meta.device.tpu.count = 4
    register_worker(state, "w-tpu", "addr1", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="auto", count=2)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w-tpu")


def test_cpu_job_schedules_on_tpu_worker(scheduler, state, job_request, worker_metadata):
    """A CPU job can run on a TPU worker since every host has a CPU."""
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    meta.device.tpu.count = 4
    register_worker(state, "w-tpu", "addr1", meta)

    submit_job(state, "j1", job_request(cpu=1))

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == WorkerId("w-tpu")


def test_multiple_jobs_across_variants_in_single_cycle(scheduler, state, worker_metadata):
    """Multiple jobs targeting different variants are all assigned in a single find_assignments call."""
    for variant in ["v5litepod-4", "v5litepod-16", "v5litepod-32"]:
        meta = worker_metadata(tpu_name=variant)
        meta.device.tpu.variant = variant
        meta.device.tpu.count = 4
        register_worker(state, f"w-{variant}", f"addr-{variant}", meta)

    for variant in ["v5litepod-4", "v5litepod-16", "v5litepod-32"]:
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=f"job-{variant}",
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(
                cpu=1,
                memory_bytes=1024**3,
                device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant=variant, count=2)),
            ),
            environment=cluster_pb2.EnvironmentConfig(),
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
        worker = state.get_worker(worker_id)
        assigned_variants.add(worker.device_variant)
    assert assigned_variants == {"v5litepod-4", "v5litepod-16", "v5litepod-32"}


def test_scheduler_tries_all_workers_before_rejecting(scheduler, state, worker_metadata):
    """Scheduler must try all matching workers, not give up on first rejection."""
    # Register many workers with the wrong variant
    for i in range(10):
        meta = worker_metadata(tpu_name="v5litepod-32")
        meta.device.tpu.variant = "v5litepod-32"
        meta.device.tpu.count = 4
        register_worker(state, f"wrong-{i}", f"addr-wrong-{i}", meta)

    # Register one worker with the correct variant
    meta = worker_metadata(tpu_name="v5litepod-4")
    meta.device.tpu.variant = "v5litepod-4"
    meta.device.tpu.count = 4
    register_worker(state, "correct", "addr-correct", meta)

    # Job requesting v5litepod-4
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-4", count=2)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
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


def test_many_jobs_on_single_variant_all_scheduled(state, worker_metadata):
    """25 jobs targeting 8 workers of the same variant all get scheduled across cycles."""
    # High building limit so back-pressure doesn't interfere with the test
    sched = Scheduler(max_building_tasks_per_worker=1000)
    num_workers = 8
    num_jobs = 25

    for i in range(num_workers):
        meta = worker_metadata(cpu=100, memory_bytes=100 * 1024**3, tpu_name="v5litepod-8")
        meta.device.tpu.variant = "v5litepod-8"
        meta.device.tpu.count = 4
        register_worker(state, f"w{i}", f"addr{i}", meta)

    for i in range(num_jobs):
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=f"job-{i}",
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )
        submit_job(state, f"job-{i}", req)

    result = schedule_until_done(sched, state)

    assert len(result.assignments) == num_jobs, (
        f"Expected all {num_jobs} jobs scheduled, got {len(result.assignments)}. "
        f"Remaining pending: {len(state.peek_pending_tasks())}"
    )
    assert len(state.peek_pending_tasks()) == 0


def test_mixed_variant_cluster_many_jobs_all_scheduled(state, worker_metadata):
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
            meta = worker_metadata(cpu=100, memory_bytes=100 * 1024**3, tpu_name=variant)
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
            req = cluster_pb2.Controller.LaunchJobRequest(
                name=f"job-{variant}-{i}",
                entrypoint=_make_test_entrypoint(),
                resources=cluster_pb2.ResourceSpecProto(
                    cpu=1,
                    memory_bytes=1024**3,
                    device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant=variant, count=1)),
                ),
                environment=cluster_pb2.EnvironmentConfig(),
                replicas=1,
            )
            submit_job(state, f"job-{variant}-{i}", req)
            total_jobs += 1

    result = schedule_until_done(sched, state)

    assert len(result.assignments) == total_jobs, (
        f"Expected all {total_jobs} jobs scheduled, got {len(result.assignments)}. "
        f"Remaining pending: {len(state.peek_pending_tasks())}"
    )
    assert len(state.peek_pending_tasks()) == 0

    # Verify each job landed on a worker with the correct variant
    for task_id, worker_id in result.assignments:
        job_name = str(task_id.parent)
        worker = state.get_worker(worker_id)
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
