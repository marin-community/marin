# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for task scheduler.

The scheduler is a shallow interface that takes inputs (pending tasks, workers,
current time) and returns outputs (assignments, timed-out tasks). It does not
dispatch tasks, modify state, or run threads.
"""

import pytest
from iris.cluster.controller.events import (
    JobSubmittedEvent,
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import Scheduler, SchedulingResult
from iris.cluster.controller.state import ControllerState, ControllerTask, ControllerWorker
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

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
            timestamp_ms=now_ms(),
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
    jid = JobId(job_id)
    state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else now_ms(),
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
        )
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
    all_assignments: list[tuple[ControllerTask, ControllerWorker]] = []
    all_timed_out: list[ControllerTask] = []

    for _ in range(max_cycles):
        pending = state.peek_pending_tasks()
        workers = state.get_available_workers()
        result = scheduler.find_assignments(pending, workers)

        if not result.assignments and not result.timed_out_tasks:
            break

        all_assignments.extend(result.assignments)
        all_timed_out.extend(result.timed_out_tasks)

        for task, worker in result.assignments:
            assign_task_to_worker(state, task, worker.worker_id)

    return SchedulingResult(assignments=all_assignments, timed_out_tasks=all_timed_out)


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
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
            scheduling_timeout_seconds=scheduling_timeout_seconds,
        )

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
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes, replicas=replicas),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
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
def scheduler(state):
    """Create a Scheduler instance."""
    return Scheduler(state)


def test_scheduler_finds_assignment_for_task(scheduler, state, job_request, worker_metadata):
    """Verify scheduler assigns task to available worker."""
    register_worker(state, "w1", "addr", worker_metadata())

    tasks = submit_job(state, "j1", job_request())
    task = tasks[0]

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == task
    assert result.assignments[0][1].worker_id == WorkerId("w1")
    assert len(result.timed_out_tasks) == 0


def test_scheduler_returns_empty_when_no_workers(scheduler, state, job_request):
    """Verify scheduler returns empty result when no workers available."""
    submit_job(state, "j1", job_request())

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()  # Empty

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 0
    assert len(result.timed_out_tasks) == 0


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
    assigned_worker_ids = {worker.worker_id for _, worker in result.assignments}
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
    assigned_task_ids = {task.task_id for task, _ in result.assignments}
    assert assigned_task_ids == {tasks1[0].task_id, tasks2[0].task_id, tasks3[0].task_id}
    # All assigned to the same worker
    assert all(worker.worker_id == WorkerId("w1") for _, worker in result.assignments)


def test_scheduler_skips_tasks_that_dont_fit(scheduler, state, job_request, worker_metadata):
    """Verify scheduler skips tasks that don't fit and continues to next."""
    # Worker with 4 CPUs
    register_worker(state, "w1", "addr", worker_metadata(cpu=4, memory_bytes=16 * 1024**3))

    # Job 1: needs 8 CPUs (won't fit on 4 CPU worker)
    submit_job(state, "j1", job_request(cpu=8))
    # Job 2: needs 2 CPUs (will fit)
    tasks2 = submit_job(state, "j2", job_request(cpu=2))

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Only job2's task should be assigned
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks2[0]


def test_scheduler_detects_timed_out_tasks(scheduler, state, worker_metadata):
    """Verify scheduler identifies tasks that exceeded scheduling timeout and logs the event."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that requires 100 CPUs (will never fit) with 1 second timeout
    # Submitted 2 seconds ago, so it should be timed out
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="impossible-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=100, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        scheduling_timeout_seconds=1,
    )
    tasks = submit_job(state, "j1", request, timestamp_ms=now_ms() - 2000)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Primary observable behavior: task is marked as timed out
    assert len(result.assignments) == 0
    assert len(result.timed_out_tasks) == 1
    assert result.timed_out_tasks[0] == tasks[0]


def test_scheduler_no_timeout_when_zero(scheduler, state, worker_metadata):
    """Verify task with scheduling_timeout_seconds=0 never times out."""
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that can't fit but has no timeout (0)
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="no-timeout-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=100, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        scheduling_timeout_seconds=0,  # No timeout
    )
    submit_job(state, "j1", request, timestamp_ms=now_ms() - 10000)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Task should not be in timed_out_tasks (just skipped, no assignment)
    assert len(result.assignments) == 0
    assert len(result.timed_out_tasks) == 0


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

    pending_tasks = state.peek_pending_tasks()
    # get_available_workers() already filters unhealthy workers
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 1
    assert result.assignments[0][1].worker_id == WorkerId("w1")


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

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 0


def test_scheduler_reports_task_too_large_for_cluster(scheduler, state, job_request, worker_metadata):
    """Verify scheduler reports when a task requires more resources than any worker can provide.

    This is distinct from temporary capacity unavailability - the task will *never* be
    schedulable on the current cluster configuration.
    """
    # Worker with only 2 CPUs - this is the largest worker in our "cluster"
    register_worker(state, "w1", "addr", worker_metadata(cpu=2))

    # Job that needs 4 CPUs - exceeds the capacity of any single worker
    submit_job(state, "j1", job_request(cpu=4))

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

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

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks[0]
    assert result.assignments[0][1].worker_id == WorkerId("w1")


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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

    if should_match:
        assert len(result.assignments) == 1
        assert result.assignments[0][1].worker_id == WorkerId("w1")
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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

    if should_match:
        assert len(result.assignments) == 1
        assert result.assignments[0][1].worker_id == WorkerId("w1")
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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

    if should_match:
        assert len(result.assignments) == 1
        assert result.assignments[0][1].worker_id == WorkerId("w1")
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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

    assert len(result.assignments) == 1
    assert result.assignments[0][1].worker_id == WorkerId("w1")


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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

    # Only w1 matches both constraints
    assert len(result.assignments) == 1
    assert result.assignments[0][1].worker_id == WorkerId("w1")


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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

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

    result = scheduler.find_assignments(state.peek_pending_tasks(), state.get_available_workers())

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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # All 4 tasks should be assigned
    assert len(result.assignments) == 4

    # All assigned to workers with same tpu-name
    assigned_tpu_names = {w.attributes["tpu-name"].value for _, w in result.assignments}
    assert assigned_tpu_names == {"tpu-a"}

    # Tasks assigned in order: task-0 -> worker-0, task-1 -> worker-1, etc.
    for task, worker in result.assignments:
        expected_worker_id = f"w{task.task_index}"
        assert worker.worker_id == WorkerId(expected_worker_id)


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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3, replicas=2),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "busy", busy_req)

    # Assign the busy job's tasks to wa0 and wa1
    busy_tasks = state.get_job_tasks(JobId("busy"))
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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # Job should be assigned to tpu-b (has 4 free workers)
    assert len(result.assignments) == 4
    assigned_tpu_names = {w.attributes["tpu-name"].value for _, w in result.assignments}
    assert assigned_tpu_names == {"tpu-b"}


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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    assert len(result.assignments) == 4

    # Verify task-0 -> worker with tpu-worker-id=0, task-1 -> worker with tpu-worker-id=1, etc.
    for task, worker in result.assignments:
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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    constraint = req.constraints.add()
    constraint.key = "region"
    constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
    constraint.value.string_value = "us-east"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # Should be assigned to tpu-b (only group matching region=us-east)
    assert len(result.assignments) == 4
    assigned_tpu_names = {w.attributes["tpu-name"].value for _, w in result.assignments}
    assert assigned_tpu_names == {"tpu-b"}


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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # No assignments - only 2 workers have sufficient capacity
    assert len(result.assignments) == 0

    # Now add a new TPU group with 4 workers, all with sufficient capacity
    for i in range(4):
        meta = worker_metadata(cpu=2)
        meta.attributes["tpu-name"].string_value = "tpu-b"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"wb{i}", f"addrb{i}", meta)

    # Re-run the scheduler - job should now be assigned to the new group
    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # All 4 tasks should now be assigned to tpu-b
    assert len(result.assignments) == 4
    assigned_tpu_names = {w.attributes["tpu-name"].value for _, w in result.assignments}
    assert assigned_tpu_names == {"tpu-b"}


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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    c = req.constraints.add()
    c.key = "taint:maintenance"
    c.op = cluster_pb2.CONSTRAINT_OP_NOT_EXISTS
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # All 4 tasks should be assigned to tpu-b (tpu-a has a tainted worker)
    assert len(result.assignments) == 4
    assigned_tpu_names = {w.attributes["tpu-name"].value for _, w in result.assignments}
    assert assigned_tpu_names == {"tpu-b"}


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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    tasks1 = submit_job(state, "j1", req1)

    # First scheduling cycle - task should be assigned
    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks1[0]

    # Commit the assignment
    assign_task_to_worker(state, tasks1[0], WorkerId("w1"))
    transition_task_to_running(state, tasks1[0])

    # Submit second job that also requires 4 TPU chips
    req2 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "j2", req2)

    # Second scheduling cycle - no TPU chips available
    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )
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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=8)),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # Task should not be scheduled - not enough TPU chips
    assert len(result.assignments) == 0


def test_tpu_chips_released_after_task_completion(scheduler, state):
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
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    tasks1 = submit_job(state, "j1", req1)
    assign_task_to_worker(state, tasks1[0], WorkerId("w1"))
    transition_task_to_running(state, tasks1[0])

    # Submit second job
    req2 = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job-2",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16", count=4)),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "j2", req2)

    # Second job can't be scheduled yet
    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )
    assert len(result.assignments) == 0

    # Complete first task
    state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks1[0].task_id,
            new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
        )
    )

    # Now second job can be scheduled
    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )
    assert len(result.assignments) == 1
    assert result.assignments[0][0].job_id == JobId("j2")
