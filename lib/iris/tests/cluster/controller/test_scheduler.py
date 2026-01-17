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

"""Tests for job scheduler.

The scheduler is a shallow interface that takes inputs (pending jobs, workers,
current time) and returns outputs (assignments, timed-out jobs). It does not
dispatch jobs, modify state, or run threads.
"""

import time

import pytest

from iris.cluster.controller.scheduler import Scheduler
from iris.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2


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


def test_scheduler_finds_assignment_for_job(scheduler, state, job_request, worker_metadata):
    """Verify scheduler assigns job to available worker."""
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata())
    state.add_worker(worker)

    job = ControllerJob(JobId("j1"), request=job_request())
    state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 1
    assert result.assignments[0] == (job, worker)
    assert len(result.timed_out_jobs) == 0


def test_scheduler_returns_empty_when_no_workers(scheduler, state, job_request):
    """Verify scheduler returns empty result when no workers available."""
    job = ControllerJob(JobId("j1"), request=job_request())
    state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()  # Empty
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 0
    assert len(result.timed_out_jobs) == 0


def test_scheduler_assigns_multiple_jobs_to_worker(scheduler, state, job_request, worker_metadata):
    """Verify scheduler can assign multiple jobs to one worker."""
    worker = ControllerWorker(
        WorkerId("w1"),
        "addr",
        worker_metadata(cpu=10, memory_bytes=10 * 1024**3),
    )
    state.add_worker(worker)

    job1 = ControllerJob(JobId("j1"), request=job_request(cpu=2))
    job2 = ControllerJob(JobId("j2"), request=job_request(cpu=2))
    job3 = ControllerJob(JobId("j3"), request=job_request(cpu=2))
    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 3
    assigned_job_ids = {job.job_id for job, _ in result.assignments}
    assert assigned_job_ids == {job1.job_id, job2.job_id, job3.job_id}


def test_scheduler_skips_jobs_that_dont_fit(scheduler, state, job_request, worker_metadata):
    """Verify scheduler skips jobs that don't fit and continues to next."""
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
    state.add_job(job1)
    state.add_job(job2)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    # Only job2 should be assigned
    assert len(result.assignments) == 1
    assert result.assignments[0][0] == job2


def test_scheduler_detects_timed_out_jobs(scheduler, state, worker_metadata):
    """Verify scheduler identifies jobs that exceeded scheduling timeout."""
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
        submitted_at_ms=int(time.time() * 1000) - 2000,  # Submitted 2s ago
    )
    state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 0
    assert len(result.timed_out_jobs) == 1
    assert result.timed_out_jobs[0] == job


def test_scheduler_no_timeout_when_zero(scheduler, state, worker_metadata):
    """Verify job with scheduling_timeout_seconds=0 never times out."""
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
        submitted_at_ms=int(time.time() * 1000) - 10000,  # Submitted 10s ago
    )
    state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    # Job should not be in timed_out_jobs (just skipped, no assignment)
    assert len(result.assignments) == 0
    assert len(result.timed_out_jobs) == 0


def test_scheduler_respects_worker_capacity_across_assignments(scheduler, state, job_request, worker_metadata):
    """Verify scheduler tracks capacity used by earlier assignments in same cycle."""
    # Worker with 4 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=4))
    state.add_worker(worker)

    # Submit 4 jobs, each requiring 2 CPUs
    # Only 2 should fit at a time
    for i in range(4):
        job = ControllerJob(JobId(f"j{i}"), request=job_request(cpu=2))
        state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    # Only first 2 jobs should be assigned (using all 4 CPUs)
    assert len(result.assignments) == 2


def test_scheduler_skips_unhealthy_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler ignores unhealthy workers."""
    healthy_worker = ControllerWorker(WorkerId("w1"), "addr1", worker_metadata())
    unhealthy_worker = ControllerWorker(WorkerId("w2"), "addr2", worker_metadata())
    unhealthy_worker.healthy = False

    state.add_worker(healthy_worker)
    state.add_worker(unhealthy_worker)

    job = ControllerJob(JobId("j1"), request=job_request())
    state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    # get_available_workers() already filters unhealthy workers
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 1
    assert result.assignments[0][1] == healthy_worker


def test_scheduler_considers_running_jobs_for_capacity(scheduler, state, job_request, worker_metadata):
    """Verify scheduler accounts for jobs already running on workers."""
    # Worker with 4 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", worker_metadata(cpu=4))
    state.add_worker(worker)

    # Add a running job that uses 3 CPUs
    running_job = ControllerJob(
        JobId("running"),
        request=job_request(cpu=3),
        state=cluster_pb2.JOB_STATE_RUNNING,
        worker_id=worker.worker_id,
    )
    state._jobs[running_job.job_id] = running_job
    worker.running_jobs.add(running_job.job_id)

    # Try to schedule a job that needs 2 CPUs (won't fit, only 1 CPU available)
    job = ControllerJob(JobId("j1"), request=job_request(cpu=2))
    state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 0


def test_scheduler_assigns_to_multiple_workers(scheduler, state, job_request, worker_metadata):
    """Verify scheduler can assign jobs across multiple workers."""
    # Two workers with 2 CPUs each
    worker1 = ControllerWorker(WorkerId("w1"), "addr1", worker_metadata(cpu=2))
    worker2 = ControllerWorker(WorkerId("w2"), "addr2", worker_metadata(cpu=2))
    state.add_worker(worker1)
    state.add_worker(worker2)

    # Three jobs needing 2 CPUs each
    # Two should fit (one on each worker), third won't fit
    for i in range(3):
        job = ControllerJob(JobId(f"j{i}"), request=job_request(cpu=2))
        state.add_job(job)

    pending_jobs = state.peek_pending_jobs()
    workers = state.get_available_workers()
    now_ms = int(time.time() * 1000)

    result = scheduler.find_assignments(pending_jobs, workers, now_ms)

    assert len(result.assignments) == 2
    assigned_workers = {w.worker_id for _, w in result.assignments}
    assert assigned_workers == {"w1", "w2"}
