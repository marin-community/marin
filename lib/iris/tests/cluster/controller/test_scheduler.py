# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task scheduler.

The scheduler is a shallow interface that takes inputs (pending tasks, workers,
job requirements) and returns outputs (assignments). It does not dispatch tasks,
modify state, or run threads.
"""

import pytest
from iris.cluster.constraints import WellKnownAttribute, constraints_from_resources
from iris.cluster.controller.scheduler import (
    JobRequirements,
    Scheduler,
    SchedulingResult,
)

from iris.cluster.controller.db import (
    ATTEMPTS,
    ControllerDB,
    JOBS,
    TASKS,
    TERMINAL_TASK_STATES,
    WORKERS,
    WORKER_ATTRIBUTES,
    Job,
    Task,
    Worker,
    _decode_attribute_rows,
    _tasks_with_attempts,
)
from iris.cluster.log_store import LogStore
from iris.cluster.controller.transitions import Assignment, ControllerTransitions, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.constraints import Constraint, merge_constraints
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2, config_pb2
from iris.time_utils import Timestamp


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def _job_requirements_from_job(job) -> JobRequirements:
    """Convert a job row to JobRequirements for testing."""
    return JobRequirements(
        resources=job.request.resources,
        constraints=list(job.request.constraints),
        is_coscheduled=job.is_coscheduled,
        coscheduling_group_by=job.coscheduling_group_by,
    )


def _schedulable_tasks(state: ControllerTransitions):
    with state._db.snapshot() as q:
        tasks = q.select(
            TASKS,
            where=TASKS.c.state.not_null() & ~TASKS.c.state.in_(list(TERMINAL_TASK_STATES)),
            order_by=(
                TASKS.c.priority_neg_depth.asc(),
                TASKS.c.priority_root_submitted_ms.asc(),
                TASKS.c.submitted_at_ms.asc(),
                TASKS.c.task_id.asc(),
            ),
        )
    return [t for t in tasks if t.can_be_scheduled()]


def _worker_capacities(state: ControllerTransitions):
    from dataclasses import replace as _replace

    with state._db.snapshot() as q:
        workers = q.select(WORKERS, where=(WORKERS.c.healthy == 1) & (WORKERS.c.active == 1))
        if not workers:
            return []
        attrs = q.select(
            WORKER_ATTRIBUTES,
            columns=(
                WORKER_ATTRIBUTES.c.worker_id,
                WORKER_ATTRIBUTES.c.key,
                WORKER_ATTRIBUTES.c.value_type,
                WORKER_ATTRIBUTES.c.str_value,
                WORKER_ATTRIBUTES.c.int_value,
                WORKER_ATTRIBUTES.c.float_value,
            ),
            where=WORKER_ATTRIBUTES.c.worker_id.in_([str(w.worker_id) for w in workers]),
        )
    attrs_by_worker = _decode_attribute_rows(attrs)
    return [_replace(w, attributes=attrs_by_worker.get(w.worker_id, {})) for w in workers]


def _building_counts(state: ControllerTransitions) -> dict[WorkerId, int]:
    with state._db.snapshot() as snapshot:
        rows = snapshot.raw(
            "SELECT a.worker_id, COUNT(*) as c FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "JOIN jobs j ON t.job_id = j.job_id "
            "WHERE t.state IN (?, ?) AND j.is_reservation_holder = 0 "
            "GROUP BY a.worker_id ORDER BY a.worker_id ASC",
            (
                cluster_pb2.TASK_STATE_BUILDING,
                cluster_pb2.TASK_STATE_ASSIGNED,
            ),
            decoders={"worker_id": WorkerId, "c": int},
        )
    return {row.worker_id: row.c for row in rows}


def _task_by_id_with_attempts(state: ControllerTransitions, task_id: JobName) -> Task | None:
    wire = task_id.to_wire()
    with state._db.snapshot() as q:
        tasks = q.select(TASKS, where=TASKS.c.task_id == wire)
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id == wire,
            order_by=(ATTEMPTS.c.attempt_id.asc(),),
        )
    hydrated = _tasks_with_attempts(tasks, attempts)
    return hydrated[0] if hydrated else None


def _query_job(state: ControllerTransitions, job_id: JobName) -> Job | None:
    with state._db.snapshot() as q:
        return q.one(JOBS, where=JOBS.c.job_id == job_id.to_wire())


def _query_task(state: ControllerTransitions, task_id: JobName) -> Task | None:
    with state._db.snapshot() as q:
        return q.one(TASKS, where=TASKS.c.task_id == task_id.to_wire())


def _query_worker(state: ControllerTransitions, worker_id: WorkerId) -> Worker | None:
    with state._db.snapshot() as q:
        return q.one(WORKERS, where=WORKERS.c.worker_id == str(worker_id))


def _query_tasks_for_job(state: ControllerTransitions, job_id: JobName) -> list[Task]:
    with state._db.snapshot() as q:
        return q.select(TASKS, where=TASKS.c.job_id == job_id.to_wire())


def _worker_attr(state: ControllerTransitions, worker_id: WorkerId, key: str):
    with state._db.snapshot() as q:
        rows = q.select(
            WORKER_ATTRIBUTES,
            columns=(
                WORKER_ATTRIBUTES.c.worker_id,
                WORKER_ATTRIBUTES.c.key,
                WORKER_ATTRIBUTES.c.value_type,
                WORKER_ATTRIBUTES.c.str_value,
                WORKER_ATTRIBUTES.c.int_value,
                WORKER_ATTRIBUTES.c.float_value,
            ),
            where=(WORKER_ATTRIBUTES.c.worker_id == str(worker_id)) & (WORKER_ATTRIBUTES.c.key == key),
        )
    if not rows:
        return None
    attrs = _decode_attribute_rows(rows)
    return attrs.get(worker_id, {}).get(key)


# =============================================================================
# Command/Query Test Helpers
# =============================================================================


def register_worker(
    state: ControllerTransitions,
    worker_id: str,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
) -> WorkerId:
    """Register a worker via state command API."""
    wid = WorkerId(worker_id)
    state.register_or_refresh_worker(
        worker_id=wid,
        address=address,
        metadata=metadata,
        ts=Timestamp.now(),
    )
    return wid


def _inject_device_constraints(request: cluster_pb2.Controller.LaunchJobRequest) -> None:
    """Auto-inject device constraints from the resource spec, mirroring service.py.

    In production, the service layer merges auto-generated device constraints
    into the request before storing the job. Tests bypass the service layer,
    so we replicate that logic here.
    """
    auto = constraints_from_resources(request.resources)
    if not auto:
        return

    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)

    del request.constraints[:]
    for c in merged:
        request.constraints.append(c.to_proto())


def submit_job(
    state: ControllerTransitions,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
    timestamp_ms: int | None = None,
) -> list:
    """Submit a job and return created task rows."""
    _inject_device_constraints(request)
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    state.submit_job(
        jid,
        request,
        Timestamp.from_ms(timestamp_ms) if timestamp_ms is not None else Timestamp.now(),
    )
    return _query_tasks_for_job(state, jid)


def assign_task_to_worker(state: ControllerTransitions, task, worker_id: WorkerId) -> None:
    """Assign a task to a worker via command API."""
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])


def transition_task_to_running(state: ControllerTransitions, task) -> None:
    """Transition a task to RUNNING state via heartbeat update command."""
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=task.worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=task.current_attempt_id,
                    new_state=cluster_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )


def transition_task_to_state(state: ControllerTransitions, task, new_state: int) -> None:
    """Transition a task attempt to an arbitrary state via heartbeat update command."""
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=task.worker_id,
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
    """Build a SchedulingContext from state, including pending tasks and job requirements."""
    pending_tasks = _schedulable_tasks(state)
    workers = [w for w in _worker_capacities(state) if w.healthy]
    building_counts = _building_counts(state)

    # Build task IDs and job requirements from pending tasks
    task_ids = []
    jobs = {}
    for task in pending_tasks:
        if not task.can_be_scheduled():
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
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        cpu: int = 1,
        memory_bytes: int = 1024**3,
        scheduling_timeout_seconds: int = 0,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        from iris.time_utils import Duration

        job_name = JobName.from_string(name) if name.startswith("/") else JobName.root("test-user", name)
        request = cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=cpu * 1000, memory_bytes=memory_bytes),
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
        group_by: str = WellKnownAttribute.TPU_NAME,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        job_name = JobName.from_string(name) if name.startswith("/") else JobName.root("test-user", name)
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=cpu * 1000, memory_bytes=memory_bytes),
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
        return cluster_pb2.ResourceSpecProto(
            cpu_millicores=cpu * 1000, memory_bytes=memory_bytes, disk_bytes=10 * 1024**3
        )

    return _make


@pytest.fixture
def worker_metadata():
    """Create WorkerMetadata for testing.

    Automatically populates device-type and device-variant attributes so
    constraint-based scheduling works the same way as production (where
    _build_worker_attributes sets these from WorkerConfig).
    """

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

        meta = cluster_pb2.WorkerMetadata(
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

        # Populate well-known attributes matching what _build_worker_attributes does
        if tpu_name:
            meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "tpu"
            meta.attributes[WellKnownAttribute.DEVICE_VARIANT].string_value = tpu_name.lower()
        elif gpu_count > 0:
            meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "gpu"
            if gpu_name:
                meta.attributes[WellKnownAttribute.DEVICE_VARIANT].string_value = gpu_name.lower()
        else:
            meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "cpu"

        return meta

    return _make


@pytest.fixture
def state(tmp_path):
    """Create a fresh ControllerTransitions for each test."""
    db_path = tmp_path / "controller.sqlite3"
    db = ControllerDB(db_path=db_path)
    log_store = LogStore(db_path=db_path)
    s = ControllerTransitions(db=db, log_store=log_store)
    yield s
    log_store.close()
    db.close()


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
    from iris.time_utils import Duration

    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that requires 100 CPUs (will never fit) with 1 second timeout
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="impossible-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=100000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    request.scheduling_timeout.CopyFrom(Duration.from_seconds(1).to_proto())
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
        if not task.can_be_scheduled():
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


def test_scheduler_no_timeout_when_zero(scheduler, state, worker_metadata):
    """Verify task with scheduling_timeout=0 never times out."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that can't fit but has no timeout (0)
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="no-timeout-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=100000, memory_bytes=1024**3),
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
    pending = _schedulable_tasks(state)
    assert len(pending) == 1


def test_scheduler_skips_unhealthy_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler ignores unhealthy workers."""
    register_worker(state, "w1", "addr1", worker_metadata())
    register_worker(state, "w2", "addr2", worker_metadata())
    # Mark second worker as unhealthy
    state.set_worker_health_for_test(WorkerId("w2"), False)

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
    meta1.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2 with different tpu-name
    meta2 = worker_metadata()
    meta2.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
    register_worker(state, "w2", "addr2", meta2)

    # Job with constraint requiring tpu-name = "tpu-a"
    req = job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.TPU_NAME
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
    meta.attributes[WellKnownAttribute.REGION].string_value = worker_value
    register_worker(state, "w1", "addr", meta)

    req = job_request()
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


def test_constraint_in_operator_matches_any_value(scheduler, state, job_request, worker_metadata):
    """IN constraint matches workers whose attribute value is in the provided set."""
    meta1 = worker_metadata()
    meta1.attributes[WellKnownAttribute.REGION].string_value = "us-central1"
    register_worker(state, "w1", "addr1", meta1)

    meta2 = worker_metadata()
    meta2.attributes[WellKnownAttribute.REGION].string_value = "us-central2"
    register_worker(state, "w2", "addr2", meta2)

    meta3 = worker_metadata()
    meta3.attributes[WellKnownAttribute.REGION].string_value = "eu-west4"
    register_worker(state, "w3", "addr3", meta3)

    # Job with IN constraint: region IN (us-central1, us-central2)
    req = job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.REGION
    constraint.op = cluster_pb2.CONSTRAINT_OP_IN
    constraint.values.append(cluster_pb2.AttributeValue(string_value="us-central1"))
    constraint.values.append(cluster_pb2.AttributeValue(string_value="us-central2"))

    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Only w1 and w2 match the IN constraint (not w3 in eu-west4)
    assert len(result.assignments) == 1
    assert result.assignments[0][1] in {WorkerId("w1"), WorkerId("w2")}


def test_constraint_in_operator_no_match(scheduler, state, job_request, worker_metadata):
    """IN constraint with no matching workers produces no assignments."""
    meta = worker_metadata()
    meta.attributes[WellKnownAttribute.REGION].string_value = "eu-west4"
    register_worker(state, "w1", "addr1", meta)

    req = job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.REGION
    constraint.op = cluster_pb2.CONSTRAINT_OP_IN
    constraint.values.append(cluster_pb2.AttributeValue(string_value="us-central1"))
    constraint.values.append(cluster_pb2.AttributeValue(string_value="us-central2"))
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    assert len(result.assignments) == 0


def test_multiple_constraints_all_must_match(scheduler, state, job_request, worker_metadata):
    """Multiple constraints are ANDed together."""
    # Worker 1: tpu-name=tpu-a, tpu-worker-id=0
    meta1 = worker_metadata()
    meta1.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    meta1.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = 0
    register_worker(state, "w1", "addr1", meta1)

    # Worker 2: tpu-name=tpu-a, tpu-worker-id=1
    meta2 = worker_metadata()
    meta2.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
    meta2.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = 1
    register_worker(state, "w2", "addr2", meta2)

    # Worker 3: tpu-name=tpu-b, tpu-worker-id=0
    meta3 = worker_metadata()
    meta3.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
    meta3.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = 0
    register_worker(state, "w3", "addr3", meta3)

    # Job requiring tpu-name=tpu-a AND tpu-worker-id=0
    req = job_request()
    c1 = req.constraints.add()
    c1.key = WellKnownAttribute.TPU_NAME
    c1.op = cluster_pb2.CONSTRAINT_OP_EQ
    c1.value.string_value = "tpu-a"
    c2 = req.constraints.add()
    c2.key = WellKnownAttribute.TPU_WORKER_ID
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
    constraint.key = WellKnownAttribute.TPU_NAME
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
    meta1.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
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
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
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
        expected_worker_id = f"w{task.task_index}"
        assert worker_id == WorkerId(expected_worker_id)


def test_coscheduled_job_waits_when_insufficient_workers(scheduler, state, worker_metadata):
    """Coscheduled job stays pending when not enough workers in any group."""
    # Only 2 workers on tpu-a
    for i in range(2):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Job requires 4 replicas
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
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
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Consume capacity on first 2 workers of tpu-a by submitting a job
    busy_req = cluster_pb2.Controller.LaunchJobRequest(
        name="busy-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=2,
        environment=cluster_pb2.EnvironmentConfig(),
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
        meta = worker_metadata(cpu=2)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
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


def test_coscheduled_job_assigns_tasks_in_order(scheduler, state, worker_metadata):
    """Task indices map to worker IDs in sorted order."""
    # Create workers with non-sequential IDs to verify sorting
    worker_ids = [3, 1, 0, 2]  # Deliberately out of order
    for i, wid in enumerate(worker_ids):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = wid
        register_worker(state, f"w{wid}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
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
        assert (
            task.task_index == worker_tpu_id
        ), f"Task {task.task_index} assigned to worker with tpu-worker-id={worker_tpu_id}"


def test_coscheduled_job_with_constraints(scheduler, state, worker_metadata):
    """Coscheduled job respects additional constraints."""
    # tpu-a: 4 workers with region=us-west
    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.REGION].string_value = "us-west"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # tpu-b: 4 workers with region=us-east
    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        meta.attributes[WellKnownAttribute.REGION].string_value = "us-east"
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job requiring region=us-east
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.REGION
    constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "us-east"
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # Should be assigned to tpu-b (only group matching region=us-east)
    assert len(result.assignments) == 4
    for _, worker_id in result.assignments:
        attr = _worker_attr(state, worker_id, WellKnownAttribute.TPU_NAME)
        assert attr is not None and attr.value == "tpu-b"


def test_coscheduled_job_with_partial_capacity(scheduler, state, worker_metadata):
    """Coscheduled job waits when some workers in group lack capacity, then schedules when capacity is added."""
    # Create 4 workers, but 2 have insufficient CPU
    for i in range(4):
        cpu = 2 if i < 2 else 1  # First 2 have 2 CPU, last 2 have only 1
        meta = worker_metadata(cpu=cpu)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # No assignments - only 2 workers have sufficient capacity
    assert len(result.assignments) == 0

    # Now add a new TPU group with 4 workers, all with sufficient capacity
    for i in range(4):
        meta = worker_metadata(cpu=2)
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


def test_tainted_worker_not_used_for_coscheduled_job(scheduler, state, worker_metadata):
    """Coscheduled job skips groups containing tainted workers."""
    # Create TPU group "tpu-a" with 4 workers, one tainted
    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        if i == 0:
            meta.attributes["taint:maintenance"].string_value = "true"
        register_worker(state, f"wa{i}", f"addra{i}", meta)

    # Create TPU group "tpu-b" with 4 workers, none tainted
    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Coscheduled job with 4 replicas + NOT_EXISTS taint constraint
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    c = req.constraints.add()
    c.key = "taint:maintenance"
    c.op = cluster_pb2.CONSTRAINT_OP_NOT_EXISTS
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


def test_tpu_chip_count_deducted_from_capacity(scheduler, state, worker_metadata):
    """TPU chip count is deducted when task is scheduled."""
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # First job requires 4 TPU chips
    req1 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
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
            cpu_millicores=1000,
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


def test_tpu_job_rejected_when_insufficient_chips(scheduler, state, worker_metadata):
    """TPU job is not scheduled when worker has fewer chips than required."""
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # Job requires 8 TPU chips - more than worker has
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
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


def test_tpu_count_released_after_task_completion(scheduler, state, worker_metadata):
    """TPU chips are released when task completes, allowing new tasks to schedule."""
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # First job uses all 4 TPU chips
    req1 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-1",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
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
            cpu_millicores=1000,
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
    transition_task_to_state(
        state, _task_by_id_with_attempts(state, tasks1[0].task_id), cluster_pb2.TASK_STATE_SUCCEEDED
    )

    # Now second job can be scheduled
    context = _build_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0].parent == JobName.root("test-user", "j2")


# =============================================================================
# Preemptible Constraint Tests
# =============================================================================


def test_preemptible_constraint_routes_to_matching_worker(scheduler, state, job_request, worker_metadata):
    """Job constrained to non-preemptible workers is only scheduled on a matching worker."""
    # Preemptible worker
    meta_preemptible = worker_metadata()
    meta_preemptible.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "true"
    register_worker(state, "w-preemptible", "addr1", meta_preemptible)

    # On-demand worker
    meta_ondemand = worker_metadata()
    meta_ondemand.attributes[WellKnownAttribute.PREEMPTIBLE].string_value = "false"
    register_worker(state, "w-ondemand", "addr2", meta_ondemand)

    # Job requiring non-preemptible worker
    req = job_request()
    constraint = req.constraints.add()
    constraint.key = WellKnownAttribute.PREEMPTIBLE
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
    submit_job(state, "/test-user/root/child", job_request("child", cpu=1))

    # Run scheduler
    result = schedule_until_done(scheduler, state)

    # Both tasks assigned, child first
    assert len(result.assignments) == 2
    assert result.assignments[0][0].parent == JobName.from_string("/test-user/root/child")
    assert result.assignments[1][0].parent == JobName.root("test-user", "root")


def test_scheduler_assigns_older_root_tree_first(scheduler, state, job_request, worker_metadata):
    """At same depth, scheduler assigns older root tree first."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit two root jobs
    submit_job(state, "user-a-job", job_request("user-a-job", cpu=1))
    submit_job(state, "user-b-job", job_request("user-b-job", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 2
    # user-a-job submitted first
    assert result.assignments[0][0].parent == JobName.root("test-user", "user-a-job")
    assert result.assignments[1][0].parent == JobName.root("test-user", "user-b-job")


def test_scheduler_child_of_older_tree_beats_newer_root(scheduler, state, job_request, worker_metadata):
    """Child of older tree is assigned before root of newer tree."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit old tree
    submit_job(state, "old-tree", job_request("old-tree", cpu=1))

    # Submit new tree
    submit_job(state, "new-tree", job_request("new-tree", cpu=1))

    # Submit child of old tree
    submit_job(state, "/test-user/old-tree/child", job_request("child", cpu=1))

    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 3
    # Order: child (depth 2), old-tree (depth 1, older), new-tree (depth 1, newer)
    assert result.assignments[0][0].parent == JobName.from_string("/test-user/old-tree/child")
    assert result.assignments[1][0].parent == JobName.root("test-user", "old-tree")
    assert result.assignments[2][0].parent == JobName.root("test-user", "new-tree")


# =============================================================================
# Error Message Tests
# =============================================================================


def test_scheduler_reports_device_variant_mismatch(scheduler, state, worker_metadata):
    """Scheduler reports constraint failure when no worker matches device variant."""
    # Worker with v5litepod-16
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.variant = "v5litepod-16"
    register_worker(state, "w1", "addr", meta)

    # Job requesting v5litepod-32
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-32", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(_worker_capacities(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if t.can_be_scheduled()), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    # Constraint-based matching: the device-variant constraint key is reported
    assert "device-variant" in diagnostics
    assert "constraints" in diagnostics.lower()


def test_scheduler_reports_tpu_count_exceeded(scheduler, state, worker_metadata):
    """Scheduler reports TPU count exceeded in error message."""
    # Worker with 4 TPU chips -- use fixture so device attributes are populated
    meta = worker_metadata(tpu_name="v5litepod-16")
    meta.device.tpu.count = 4
    register_worker(state, "w1", "addr1", meta)

    # Job requesting 8 TPU chips
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=8)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(_worker_capacities(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if t.can_be_scheduled()), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    assert "tpu" in diagnostics.lower()
    assert "8" in diagnostics
    assert "4" in diagnostics


def test_scheduler_reports_device_type_mismatch(scheduler, state, worker_metadata):
    """Scheduler reports constraint failure when worker device type doesn't match."""
    # CPU-only worker
    meta = worker_metadata()
    register_worker(state, "w1", "addr", meta)

    # Job requesting TPU
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(_worker_capacities(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if t.can_be_scheduled()), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    # Constraint-based matching: the device-type constraint is in the diagnostic
    assert "device-type" in diagnostics
    assert "constraints" in diagnostics.lower()


def test_scheduler_reports_coscheduling_capacity_details(scheduler, state, worker_metadata):
    """Scheduler reports detailed coscheduling capacity issues."""
    # Create 4 workers but only 2 have sufficient CPU
    for i in range(4):
        cpu = 4 if i < 2 else 1  # First 2 have 4 CPU, last 2 have only 1
        meta = worker_metadata(cpu=cpu)
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Coscheduled job requiring 4 replicas, 2 CPUs each
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    # Get job-level scheduling diagnostics
    context = scheduler.create_scheduling_context(_worker_capacities(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if t.can_be_scheduled()), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    # Should mention it's a coscheduling issue with capacity details
    assert "coscheduling" in diagnostics.lower() or "group" in diagnostics.lower()
    # Should indicate how many workers have capacity vs needed
    assert "2" in diagnostics or "4" in diagnostics


def test_diagnostics_for_schedulable_job_does_not_say_unknown_failure(scheduler, state, job_request, worker_metadata):
    """When a job can be scheduled, diagnostics should not say 'Unknown scheduling failure'."""
    register_worker(state, "w1", "addr1", worker_metadata())
    tasks = submit_job(state, "j1", job_request())

    context = scheduler.create_scheduling_context(_worker_capacities(state))
    job = _query_job(state, tasks[0].job_id)
    job_req = _job_requirements_from_job(job)
    schedulable_task_id = next(
        (t.task_id for t in _query_tasks_for_job(state, job.job_id) if t.can_be_scheduled()), None
    )
    diagnostics = scheduler.get_job_scheduling_diagnostics(
        job_req, context, schedulable_task_id, num_tasks=len(_query_tasks_for_job(state, job.job_id))
    )

    assert "unknown" not in diagnostics.lower()
    assert "schedulable" in diagnostics.lower()


def test_coscheduled_tpu_jobs_cannot_double_book_group(scheduler, state, worker_metadata):
    """Two coscheduled TPU jobs cannot use the same TPU group simultaneously."""
    # Create 4 workers in tpu-group "tpu-a", each with 4 TPU chips
    for i in range(4):
        meta = worker_metadata(tpu_name="v5litepod-16")
        meta.device.tpu.count = 4
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    tpu_resource = cluster_pb2.ResourceSpecProto(
        cpu_millicores=1000,
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
    req1.coscheduling.group_by = WellKnownAttribute.TPU_NAME
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
    req2.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j2", req2)

    context = _build_context(scheduler, state)
    result2 = scheduler.find_assignments(context)
    assert len(result2.assignments) == 0

    # Complete all job 1 tasks
    for task in tasks1:
        transition_task_to_state(state, _task_by_id_with_attempts(state, task.task_id), cluster_pb2.TASK_STATE_SUCCEEDED)

    # Job 2 should now be schedulable
    result3 = schedule_until_done(scheduler, state)
    assert len(result3.assignments) == 4
    assigned_jobs = {task_id.parent for task_id, _ in result3.assignments}
    assert assigned_jobs == {JobName.root("test-user", "j2")}


def test_scheduler_fifo_within_same_depth_and_tree(scheduler, state, job_request, worker_metadata):
    """Scheduler respects FIFO within same depth and tree."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=10, memory_bytes=10 * 1024**3))

    # Submit parent
    submit_job(state, "tree", job_request("tree", cpu=1))

    # Submit two children
    submit_job(state, "/test-user/tree/child-a", job_request("child-a", cpu=1))
    submit_job(state, "/test-user/tree/child-b", job_request("child-b", cpu=1))

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
                cpu_millicores=1000,
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
        worker = _query_worker(state, worker_id)
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
            cpu_millicores=1000,
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
                cpu_millicores=1000,
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
        worker = _query_worker(state, worker_id)
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
            cpu_millicores=1000,
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
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )
        submit_job(state, f"job-{i}", req)

    result = schedule_until_done(sched, state)

    assert len(result.assignments) == num_jobs, (
        f"Expected all {num_jobs} jobs scheduled, got {len(result.assignments)}. "
        f"Remaining pending: {len(_schedulable_tasks(state))}"
    )
    assert len(_schedulable_tasks(state)) == 0


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
                    cpu_millicores=1000,
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


def test_gpu_job_matches_worker_with_config_variant(scheduler, state, worker_metadata):
    """A GPU job requesting variant="H100" matches a worker with device-variant="H100".

    In production, the worker's device-variant attribute comes from the scale
    group config (e.g. "H100"), not the nvidia-smi probe string. Both job and
    worker use the same canonical name, matched via EQ constraint.
    """
    meta = worker_metadata(gpu_count=8, gpu_name="H100")
    register_worker(state, "gpu-w1", "addr", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="H100", count=8)),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    tasks = submit_job(state, "j1", req)

    context = scheduler.create_scheduling_context(
        _worker_capacities(state),
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
        preemptible=False,
        tpu_name=metadata.tpu_name,
        tpu_worker_id=str(0),
        device=metadata.device,
        extra_attributes={},
    )
    for key, val in attrs.items():
        metadata.attributes[key].CopyFrom(val)
    return register_worker(state, worker_id, address, metadata)


def test_device_variant_in_constraint_matches_probed_workers(scheduler, state, job_request, worker_metadata):
    """device_variant_constraint matches workers whose attributes come from _build_worker_attributes.

    This is the end-to-end test: worker attributes are built the same way real
    workers build them, and the scheduler's IN constraint finds a match.

    Uses v5litepod-8 and v4-8 as the flexible alternatives (both vm_count=1)
    so the constraint represents a realistic flexible request.
    """
    meta1 = worker_metadata(tpu_name="v5litepod-8")
    _register_worker_with_probed_attributes(state, "w1", "addr1", meta1)

    meta2 = worker_metadata(tpu_name="v4-8")
    _register_worker_with_probed_attributes(state, "w2", "addr2", meta2)

    meta3 = worker_metadata(tpu_name="v5litepod-16")
    _register_worker_with_probed_attributes(state, "w3", "addr3", meta3)

    req = job_request()
    constraint = cluster_pb2.Constraint(
        key=WellKnownAttribute.DEVICE_VARIANT,
        op=cluster_pb2.CONSTRAINT_OP_IN,
    )
    for v in ["v5litepod-8", "v4-8"]:
        constraint.values.append(cluster_pb2.AttributeValue(string_value=v))
    req.constraints.append(constraint)

    submit_job(state, "flex-job", req)
    result = schedule_until_done(scheduler, state)

    assert len(result.assignments) == 1
    assigned_worker = result.assignments[0][1]
    assert assigned_worker in {WorkerId("w1"), WorkerId("w2")}
