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

"""Tests for job scheduler background thread."""

import time

import pytest

from fluster import cluster_pb2
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.LaunchJobRequest:
        return cluster_pb2.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def make_resource_spec():
    """Create a ResourceSpec for testing with enough capacity for multiple jobs."""

    def _make(cpu: int = 10, memory: str = "10g") -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=cpu, memory=memory, disk="10g")

    return _make


def test_scheduler_dispatches_jobs(make_job_request, make_resource_spec):
    """Verify scheduler dispatches jobs to workers."""
    state = ControllerState()
    dispatched = []

    def mock_dispatch(job, worker):
        dispatched.append((job.job_id, worker.worker_id))
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add worker and job
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)
    job = ControllerJob(JobId("j1"), request=make_job_request())
    state.add_job(job)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Verify dispatch_fn was called with correct args
    assert dispatched == [("j1", "w1")]


def test_scheduler_updates_job_state_on_dispatch(make_job_request, make_resource_spec):
    """Verify scheduler updates job state when dispatch succeeds."""
    state = ControllerState()

    def mock_dispatch(job, worker):
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add worker and job
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)
    job = ControllerJob(JobId("j1"), request=make_job_request())
    state.add_job(job)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Verify job state was updated
    assert job.state == cluster_pb2.JOB_STATE_RUNNING
    assert job.worker_id == "w1"
    assert job.started_at_ms is not None
    assert job.started_at_ms > 0

    # Verify worker has job in running_jobs
    assert job.job_id in worker.running_jobs


def test_scheduler_requeues_when_no_workers(make_job_request):
    """Verify scheduler re-queues job when no workers are available."""
    state = ControllerState()

    def mock_dispatch(job, worker):
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add job but no workers
    job = ControllerJob(JobId("j1"), request=make_job_request())
    state.add_job(job)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Job should still be PENDING (re-queued)
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.worker_id is None
    assert job.started_at_ms is None


def test_scheduler_wake_triggers_immediate_scheduling(make_job_request, make_resource_spec):
    """Verify wake() causes scheduler to run immediately."""
    state = ControllerState()
    dispatched = []

    def mock_dispatch(job, worker):
        dispatched.append((job.job_id, worker.worker_id))
        return True

    # Use a long interval so timer won't fire during test
    scheduler = Scheduler(state, mock_dispatch, interval_seconds=10.0)

    # Add worker and job
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)
    job = ControllerJob(JobId("j1"), request=make_job_request())

    scheduler.start()

    # Add job after starting scheduler
    state.add_job(job)

    # Wake immediately - should dispatch before 10s timer expires
    scheduler.wake()
    time.sleep(0.1)
    scheduler.stop()

    # Should have dispatched despite long interval
    assert dispatched == [("j1", "w1")]


def test_scheduler_marks_worker_unhealthy_on_dispatch_failure(make_job_request, make_resource_spec):
    """Verify scheduler marks worker unhealthy when dispatch fails."""
    state = ControllerState()

    def mock_dispatch_fail(job, worker):
        return False

    scheduler = Scheduler(state, mock_dispatch_fail, interval_seconds=0.1)

    # Add worker and job
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)
    job = ControllerJob(JobId("j1"), request=make_job_request())
    state.add_job(job)

    assert worker.healthy is True

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Worker should be marked unhealthy
    assert worker.healthy is False

    # Job should still be PENDING (re-queued)
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.worker_id is None


def test_scheduler_stops_cleanly(make_job_request, make_resource_spec):
    """Verify stop() returns without hanging."""
    state = ControllerState()

    def mock_dispatch(job, worker):
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    scheduler.start()
    time.sleep(0.2)

    # Stop should return quickly
    start_time = time.time()
    scheduler.stop()
    stop_time = time.time()

    # Should complete in well under 5 seconds
    assert stop_time - start_time < 2.0


def test_scheduler_dispatches_multiple_jobs(make_job_request, make_resource_spec):
    """Verify scheduler can dispatch multiple jobs in one cycle."""
    state = ControllerState()
    dispatched = []

    def mock_dispatch(job, worker):
        dispatched.append((job.job_id, worker.worker_id))
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add two workers and three jobs
    worker1 = ControllerWorker(WorkerId("w1"), "addr1", make_resource_spec())
    worker2 = ControllerWorker(WorkerId("w2"), "addr2", make_resource_spec())
    state.add_worker(worker1)
    state.add_worker(worker2)

    job1 = ControllerJob(JobId("j1"), request=make_job_request("job1"))
    job2 = ControllerJob(JobId("j2"), request=make_job_request("job2"))
    job3 = ControllerJob(JobId("j3"), request=make_job_request("job3"))
    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # All jobs should be dispatched (first-fit will use w1 for all)
    assert len(dispatched) == 3
    assert set(j for j, w in dispatched) == {"j1", "j2", "j3"}


def test_scheduler_stops_dispatching_when_workers_exhausted(make_job_request, make_resource_spec):
    """Verify scheduler stops trying when it runs out of workers."""
    state = ControllerState()
    dispatch_attempts = []

    def mock_dispatch(job, worker):
        dispatch_attempts.append((job.job_id, worker.worker_id))
        # First dispatch succeeds, marking worker unhealthy on second
        return len(dispatch_attempts) == 1

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add one worker and two jobs
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)

    job1 = ControllerJob(JobId("j1"), request=make_job_request("job1"))
    job2 = ControllerJob(JobId("j2"), request=make_job_request("job2"))
    state.add_job(job1)
    state.add_job(job2)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Should have attempted both dispatches
    assert len(dispatch_attempts) == 2

    # First job succeeds
    assert job1.state == cluster_pb2.JOB_STATE_RUNNING

    # Second job fails, worker marked unhealthy, job re-queued
    assert job2.state == cluster_pb2.JOB_STATE_PENDING
    assert worker.healthy is False


def test_scheduler_periodic_scheduling(make_job_request, make_resource_spec):
    """Verify scheduler runs periodically even without wake() calls."""
    state = ControllerState()
    dispatched = []

    def mock_dispatch(job, worker):
        dispatched.append(time.time())
        return True

    # Short interval for quick test
    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.15)

    # Add worker
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)

    scheduler.start()

    # Add jobs at different times
    job1 = ControllerJob(JobId("j1"), request=make_job_request("job1"))
    state.add_job(job1)
    time.sleep(0.2)  # Wait for periodic tick

    job2 = ControllerJob(JobId("j2"), request=make_job_request("job2"))
    state.add_job(job2)
    time.sleep(0.2)  # Wait for another periodic tick

    scheduler.stop()

    # Should have dispatched both jobs via periodic timer
    assert len(dispatched) >= 2


# =============================================================================
# Resource-Aware Scheduling Tests
# =============================================================================


def test_scheduler_skips_jobs_that_dont_fit(make_job_request, make_resource_spec):
    """Verify scheduler skips jobs that don't fit and continues to next."""
    state = ControllerState()
    dispatched = []

    def mock_dispatch(job, worker):
        dispatched.append(job.job_id)
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)

    # Add worker with 4 CPUs total
    worker = ControllerWorker(
        WorkerId("w1"),
        "addr",
        cluster_pb2.ResourceSpec(cpu=4, memory="16g"),
    )
    state.add_worker(worker)

    # Job 1: needs 8 CPUs (won't fit on 4 CPU worker)
    big_job_request = cluster_pb2.LaunchJobRequest(
        name="big-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpec(cpu=8, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    job1 = ControllerJob(JobId("j1"), request=big_job_request)

    # Job 2: needs 2 CPUs (will fit)
    small_job_request = cluster_pb2.LaunchJobRequest(
        name="small-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpec(cpu=2, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    job2 = ControllerJob(JobId("j2"), request=small_job_request)

    state.add_job(job1)
    state.add_job(job2)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Job 2 should be dispatched, job 1 skipped (still pending)
    assert "j2" in dispatched
    assert "j1" not in dispatched
    assert job1.state == cluster_pb2.JOB_STATE_PENDING
    assert job2.state == cluster_pb2.JOB_STATE_RUNNING


def test_scheduler_marks_job_unschedulable_on_timeout(make_resource_spec):
    """Verify job is marked UNSCHEDULABLE after scheduling timeout."""
    state = ControllerState()

    def mock_dispatch(job, worker):
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.05)

    # Worker with 2 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec(cpu=2))
    state.add_worker(worker)

    # Job that requires 100 CPUs (will never fit) with 1 second timeout
    # Set submitted_at_ms to 2 seconds ago so it's already timed out
    job_request = cluster_pb2.LaunchJobRequest(
        name="impossible-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpec(cpu=100, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        scheduling_timeout_seconds=1,
    )
    job = ControllerJob(
        JobId("j1"),
        request=job_request,
        submitted_at_ms=int(time.time() * 1000) - 2000,  # Submitted 2s ago
    )
    state.add_job(job)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    assert job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
    assert "timeout" in job.error.lower()


def test_scheduler_no_timeout_when_zero(make_job_request, make_resource_spec):
    """Verify job with scheduling_timeout_seconds=0 never times out."""
    state = ControllerState()

    def mock_dispatch(job, worker):
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.05)

    # Worker with 2 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec(cpu=2))
    state.add_worker(worker)

    # Job that can't fit but has no timeout (0)
    job_request = cluster_pb2.LaunchJobRequest(
        name="no-timeout-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpec(cpu=100, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        scheduling_timeout_seconds=0,  # No timeout
    )
    job = ControllerJob(
        JobId("j1"),
        request=job_request,
        submitted_at_ms=int(time.time() * 1000) - 10000,  # Submitted 10s ago
    )
    state.add_job(job)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # Should still be pending, not unschedulable
    assert job.state == cluster_pb2.JOB_STATE_PENDING


def test_scheduler_serializes_jobs_by_resources(make_resource_spec):
    """Verify jobs serialize when they exceed worker capacity."""
    state = ControllerState()
    dispatch_order = []

    def mock_dispatch(job, worker):
        dispatch_order.append(job.job_id)
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.05)

    # Worker with 4 CPUs
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec(cpu=4))
    state.add_worker(worker)

    # Submit 4 jobs, each requiring 2 CPUs
    # Only 2 should fit at a time, so they should serialize in pairs
    for i in range(4):
        job_request = cluster_pb2.LaunchJobRequest(
            name=f"job-{i}",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=2, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )
        job = ControllerJob(JobId(f"j{i}"), request=job_request)
        state.add_job(job)

    scheduler.start()
    scheduler.wake()
    time.sleep(0.2)
    scheduler.stop()

    # First 2 jobs should be dispatched (using all 4 CPUs)
    # Remaining 2 jobs should still be pending (no capacity)
    assert len(dispatch_order) == 2
    pending_jobs = [j for j in state.list_all_jobs() if j.state == cluster_pb2.JOB_STATE_PENDING]
    assert len(pending_jobs) == 2
