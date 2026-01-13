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

"""Tests for controller core data structures."""

import threading

import pytest

from fluster import cluster_pb2
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
    """Create a minimal ResourceSpec for testing."""

    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=1, memory="1g", disk="10g")

    return _make


def test_controller_state_fifo_order(make_job_request):
    """Verify jobs are returned in FIFO order."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"), submitted_at_ms=100)
    job2 = ControllerJob(job_id=JobId("j2"), request=make_job_request("job2"), submitted_at_ms=200)
    state.add_job(job1)
    state.add_job(job2)

    # Jobs should be popped in the order they were added
    popped1 = state.pop_next_pending()
    assert popped1 is not None
    assert popped1.job_id == "j1"

    popped2 = state.pop_next_pending()
    assert popped2 is not None
    assert popped2.job_id == "j2"

    # Queue should be empty now
    assert state.pop_next_pending() is None


def test_controller_state_skip_non_pending(make_job_request):
    """Verify pop_next_pending skips jobs that are not in PENDING state."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))
    job1.state = cluster_pb2.JOB_STATE_RUNNING  # Already started
    job2 = ControllerJob(job_id=JobId("j2"), request=make_job_request("job2"))
    state.add_job(job1)
    state.add_job(job2)

    # Should skip j1 since it's not PENDING
    popped = state.pop_next_pending()
    assert popped is not None
    assert popped.job_id == "j2"

    # Queue should be empty now
    assert state.pop_next_pending() is None


def test_controller_state_worker_operations(make_resource_spec):
    """Test add/get/list workers."""
    state = ControllerState()
    worker1 = ControllerWorker(worker_id=WorkerId("w1"), address="host1:8080", resources=make_resource_spec())
    worker2 = ControllerWorker(worker_id=WorkerId("w2"), address="host2:8080", resources=make_resource_spec())

    # Add workers
    state.add_worker(worker1)
    state.add_worker(worker2)

    # Get individual worker
    retrieved = state.get_worker(WorkerId("w1"))
    assert retrieved is not None
    assert retrieved.address == "host1:8080"
    assert retrieved.healthy is True

    # Get all available workers
    available = state.get_available_workers()
    assert len(available) == 2
    assert {w.worker_id for w in available} == {"w1", "w2"}

    # Mark one worker unhealthy
    worker1.healthy = False
    available = state.get_available_workers()
    assert len(available) == 1
    assert available[0].worker_id == "w2"


def test_controller_state_gang_tracking(make_job_request):
    """Verify gang jobs are tracked correctly."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"), gang_id="gang1")
    job2 = ControllerJob(job_id=JobId("j2"), request=make_job_request("job2"), gang_id="gang1")
    job3 = ControllerJob(job_id=JobId("j3"), request=make_job_request("job3"), gang_id="gang2")

    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    # Get jobs in gang1
    gang1_jobs = state.get_gang_jobs("gang1")
    assert len(gang1_jobs) == 2
    assert {j.job_id for j in gang1_jobs} == {"j1", "j2"}

    # Get jobs in gang2
    gang2_jobs = state.get_gang_jobs("gang2")
    assert len(gang2_jobs) == 1
    assert gang2_jobs[0].job_id == "j3"

    # Non-existent gang returns empty list
    assert state.get_gang_jobs("nonexistent") == []


def test_controller_state_thread_safety(make_job_request):
    """Verify concurrent access doesn't corrupt state."""
    state = ControllerState()
    num_threads = 10
    jobs_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors = []

    def add_jobs(thread_id: int):
        try:
            # Wait for all threads to be ready
            barrier.wait()

            # Add jobs
            for i in range(jobs_per_thread):
                job_id = f"t{thread_id}_j{i}"
                job = ControllerJob(job_id=JobId(job_id), request=make_job_request(f"job-{job_id}"))
                state.add_job(job)
        except Exception as e:
            errors.append(e)

    # Start threads
    threads = [threading.Thread(target=add_jobs, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check no errors occurred
    assert not errors, f"Errors during concurrent execution: {errors}"

    # Verify all jobs were added
    expected_count = num_threads * jobs_per_thread
    popped_count = 0
    while state.pop_next_pending() is not None:
        popped_count += 1

    assert popped_count == expected_count, f"Expected {expected_count} jobs, got {popped_count}"


def test_controller_state_job_retrieval(make_job_request):
    """Test job retrieval by ID."""
    state = ControllerState()
    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"), submitted_at_ms=12345)
    state.add_job(job)

    # Retrieve by ID
    retrieved = state.get_job(JobId("j1"))
    assert retrieved is not None
    assert retrieved.job_id == "j1"
    assert retrieved.submitted_at_ms == 12345

    # Non-existent job returns None
    assert state.get_job(JobId("nonexistent")) is None


def test_controller_state_worker_retrieval_nonexistent():
    """Test getting non-existent worker returns None."""
    state = ControllerState()
    assert state.get_worker(WorkerId("nonexistent")) is None


def test_controller_state_multiple_gangs(make_job_request):
    """Test tracking multiple gangs simultaneously."""
    state = ControllerState()

    # Create multiple gangs with different sizes
    for gang_num in range(5):
        gang_id = f"gang{gang_num}"
        for job_num in range(gang_num + 1):  # gang0 has 1 job, gang1 has 2, etc.
            job_id = JobId(f"g{gang_num}_j{job_num}")
            job = ControllerJob(job_id=job_id, request=make_job_request(f"job-{job_id}"), gang_id=gang_id)
            state.add_job(job)

    # Verify each gang has correct number of jobs
    for gang_num in range(5):
        gang_id = f"gang{gang_num}"
        gang_jobs = state.get_gang_jobs(gang_id)
        expected_count = gang_num + 1
        assert (
            len(gang_jobs) == expected_count
        ), f"Gang {gang_id} should have {expected_count} jobs, got {len(gang_jobs)}"


def test_controller_job_defaults(make_job_request):
    """Test ControllerJob default values."""
    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.worker_id is None
    assert job.failure_count == 0
    assert job.preemption_count == 0
    assert job.max_retries_failure == 0
    assert job.max_retries_preemption == 100
    assert job.gang_id is None
    assert job.submitted_at_ms == 0
    assert job.started_at_ms is None
    assert job.finished_at_ms is None
    assert job.error is None
    assert job.exit_code is None


def test_controller_worker_defaults(make_resource_spec):
    """Test ControllerWorker default values."""
    worker = ControllerWorker(worker_id=WorkerId("w1"), address="host:8080", resources=make_resource_spec())

    assert worker.healthy is True
    assert worker.consecutive_failures == 0
    assert worker.last_heartbeat_ms == 0
    assert len(worker.running_jobs) == 0


def test_controller_state_requeue_job(make_job_request):
    """Test that jobs can be re-queued by calling add_job again."""
    state = ControllerState()
    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # Add job
    state.add_job(job)

    # Pop it
    popped = state.pop_next_pending()
    assert popped is not None
    assert popped.job_id == "j1"

    # Queue should be empty
    assert state.pop_next_pending() is None

    # Re-queue the same job
    state.add_job(job)

    # Should be available again
    popped = state.pop_next_pending()
    assert popped is not None
    assert popped.job_id == "j1"
