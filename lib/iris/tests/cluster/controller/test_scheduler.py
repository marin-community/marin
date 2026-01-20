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

from iris.cluster.controller.scheduler import Scheduler
from iris.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker, expand_job_to_tasks
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms


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


def _add_job(state, job):
    """Add a job with auto-expanded tasks."""
    return state.add_job(job)


def test_scheduler_finds_assignment_for_task(scheduler, state, job_request, worker_metadata):
    """Verify scheduler assigns task to available worker."""
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata())
    state.add_worker(worker)

    job = ControllerJob(JobId("j1"), request=job_request())
    tasks = _add_job(state, job)
    task = tasks[0]

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 1
    assert result.assignments[0] == (task, worker)
    assert len(result.timed_out_tasks) == 0


def test_scheduler_returns_empty_when_no_workers(scheduler, state, job_request):
    """Verify scheduler returns empty result when no workers available."""
    job = ControllerJob(JobId("j1"), request=job_request())
    _add_job(state, job)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()  # Empty

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 0
    assert len(result.timed_out_tasks) == 0


def test_scheduler_assigns_multiple_tasks_to_worker(scheduler, state, job_request, worker_metadata):
    """Verify scheduler can assign multiple tasks to one worker."""
    worker = ControllerWorker(
        WorkerId("w1"),
        "addr",
        worker_metadata(cpu=10, memory_bytes=10 * 1024**3),
    )
    state.add_worker(worker)

    job1 = ControllerJob(JobId("j1"), request=job_request(cpu=2))
    job2 = ControllerJob(JobId("j2"), request=job_request(cpu=2))
    job3 = ControllerJob(JobId("j3"), request=job_request(cpu=2))
    tasks1 = _add_job(state, job1)
    tasks2 = _add_job(state, job2)
    tasks3 = _add_job(state, job3)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 3
    assigned_task_ids = {task.task_id for task, _ in result.assignments}
    assert assigned_task_ids == {tasks1[0].task_id, tasks2[0].task_id, tasks3[0].task_id}


def test_scheduler_skips_tasks_that_dont_fit(scheduler, state, job_request, worker_metadata):
    """Verify scheduler skips tasks that don't fit and continues to next."""
    # Worker with 4 CPUs
    worker = ControllerWorker(
        WorkerId("w1"),
        "addr",
        worker_metadata(cpu=4, memory_bytes=16 * 1024**3),
    )
    state.add_worker(worker)

    # Job 1: needs 8 CPUs (won't fit on 4 CPU worker)
    job1 = ControllerJob(JobId("j1"), request=job_request(cpu=8))
    # Job 2: needs 2 CPUs (will fit)
    job2 = ControllerJob(JobId("j2"), request=job_request(cpu=2))
    _add_job(state, job1)
    tasks2 = _add_job(state, job2)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Only job2's task should be assigned
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks2[0]


def test_scheduler_detects_timed_out_tasks(scheduler, state, worker_metadata):
    """Verify scheduler identifies tasks that exceeded scheduling timeout and logs the event."""
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=2))
    state.add_worker(worker)

    # Job that requires 100 CPUs (will never fit) with 1 second timeout
    # Submitted 2 seconds ago, so it should be timed out
    job_request = cluster_pb2.Controller.LaunchJobRequest(
        name="impossible-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=100, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        scheduling_timeout_seconds=1,
    )
    job = ControllerJob(
        JobId("j1"),
        request=job_request,
        submitted_at_ms=now_ms() - 2000,  # Submitted 2s ago
    )
    tasks = expand_job_to_tasks(job)
    state.add_job(job, tasks)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Primary observable behavior: task is marked as timed out
    assert len(result.assignments) == 0
    assert len(result.timed_out_tasks) == 1
    assert result.timed_out_tasks[0] == tasks[0]


def test_scheduler_no_timeout_when_zero(scheduler, state, worker_metadata):
    """Verify task with scheduling_timeout_seconds=0 never times out."""
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=2))
    state.add_worker(worker)

    # Job that can't fit but has no timeout (0)
    job_request = cluster_pb2.Controller.LaunchJobRequest(
        name="no-timeout-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=100, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        scheduling_timeout_seconds=0,  # No timeout
    )
    job = ControllerJob(
        JobId("j1"),
        request=job_request,
        submitted_at_ms=now_ms() - 10000,  # Submitted 10s ago
    )
    tasks = expand_job_to_tasks(job)
    state.add_job(job, tasks)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Task should not be in timed_out_tasks (just skipped, no assignment)
    assert len(result.assignments) == 0
    assert len(result.timed_out_tasks) == 0


def test_scheduler_respects_worker_capacity_across_assignments(scheduler, state, job_request, worker_metadata):
    """Verify scheduler tracks capacity used by earlier assignments in same cycle."""
    # Worker with 4 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=4))
    state.add_worker(worker)

    # Submit 4 jobs, each requiring 2 CPUs
    # Only 2 should fit at a time
    for i in range(4):
        job = ControllerJob(JobId(f"j{i}"), request=job_request(cpu=2))
        _add_job(state, job)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Only first 2 tasks should be assigned (using all 4 CPUs)
    assert len(result.assignments) == 2


def test_scheduler_skips_unhealthy_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler ignores unhealthy workers."""
    healthy_worker = ControllerWorker(WorkerId("w1"), "addr1", worker_metadata())
    unhealthy_worker = ControllerWorker(WorkerId("w2"), "addr2", worker_metadata())
    unhealthy_worker.healthy = False

    state.add_worker(healthy_worker)
    state.add_worker(unhealthy_worker)

    job = ControllerJob(JobId("j1"), request=job_request())
    _add_job(state, job)

    pending_tasks = state.peek_pending_tasks()
    # get_available_workers() already filters unhealthy workers
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == healthy_worker


def test_scheduler_considers_running_tasks_for_capacity(scheduler, state, job_request, worker_metadata):
    """Verify scheduler accounts for tasks already running on workers."""
    # Worker with 4 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=4))
    state.add_worker(worker)

    # Add a running job with a running task that uses 3 CPUs
    running_job = ControllerJob(
        JobId("running"),
        request=job_request(cpu=3),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    running_tasks = expand_job_to_tasks(running_job)
    # Mark task as running on worker (creates attempt with worker_id)
    running_tasks[0].create_attempt(worker.worker_id)
    state.add_job(running_job, running_tasks)
    # Assign task to worker using the proper method to track committed resources
    worker.assign_task(running_tasks[0].task_id, running_job.request.resources)

    # Try to schedule a job that needs 2 CPUs (won't fit, only 1 CPU available)
    job = ControllerJob(JobId("j1"), request=job_request(cpu=2))
    _add_job(state, job)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 0


def test_scheduler_assigns_to_multiple_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler can assign tasks across multiple workers."""
    # Two workers with 2 CPUs each
    worker1 = ControllerWorker(WorkerId("w1"), "addr1", worker_metadata(cpu=2))
    worker2 = ControllerWorker(WorkerId("w2"), "addr2", worker_metadata(cpu=2))
    state.add_worker(worker1)
    state.add_worker(worker2)

    # Three jobs needing 2 CPUs each
    # Two should fit (one on each worker), third won't fit
    for i in range(3):
        job = ControllerJob(JobId(f"j{i}"), request=job_request(cpu=2))
        _add_job(state, job)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    assert len(result.assignments) == 2
    assigned_workers = {w.worker_id for _, w in result.assignments}
    assert assigned_workers == {"w1", "w2"}


def test_scheduler_reports_task_too_large_for_cluster(scheduler, state, job_request, worker_metadata):
    """Verify scheduler reports when a task requires more resources than any worker can provide.

    This is distinct from temporary capacity unavailability - the task will *never* be
    schedulable on the current cluster configuration.
    """
    # Worker with only 2 CPUs - this is the largest worker in our "cluster"
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=2))
    state.add_worker(worker)

    # Job that needs 4 CPUs - exceeds the capacity of any single worker
    job = ControllerJob(JobId("j1"), request=job_request(cpu=4))
    _add_job(state, job)

    pending_tasks = state.peek_pending_tasks()
    workers = state.get_available_workers()

    result = scheduler.find_assignments(pending_tasks, workers)

    # Primary observable behavior: task cannot be assigned
    assert len(result.assignments) == 0
