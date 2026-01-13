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

"""Tests for job failure and retry logic."""

import pytest

from fluster import cluster_pb2
from fluster.cluster.controller.retry import handle_gang_failure, handle_job_failure
from fluster.cluster.controller.state import ControllerJob, ControllerState
from fluster.cluster.types import JobId


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


def test_job_retry_on_worker_failure(make_job_request):
    """Worker failure increments preemption_count and retries if under limit."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_preemption=2,
    )
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    state.add_job(job)

    # First worker failure - should retry
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.preemption_count == 1
    assert job.failure_count == 0  # Should not increment job failure count


def test_job_retry_on_job_failure(make_job_request):
    """Job failure increments failure_count and retries if under limit."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_failure=2,
    )
    job.state = cluster_pb2.JOB_STATE_FAILED
    state.add_job(job)

    # First job failure - should retry
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=False)
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.failure_count == 1
    assert job.preemption_count == 0  # Should not increment preemption count


def test_job_exceeds_worker_failure_limit(make_job_request):
    """Stop retrying when preemption_count exceeds max_retries_preemption."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_preemption=2,
    )
    state.add_job(job)

    # First failure - should retry (count 1 <= limit 2)
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.preemption_count == 1

    # Second failure - should retry (count 2 <= limit 2)
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.preemption_count == 2

    # Third failure - should NOT retry (count 3 > limit 2)
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    assert not handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.preemption_count == 3
    assert job.state == cluster_pb2.JOB_STATE_WORKER_FAILED  # State unchanged


def test_job_exceeds_job_failure_limit(make_job_request):
    """Stop retrying when failure_count exceeds max_retries_failure."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_failure=1,
    )
    state.add_job(job)

    # First failure - should retry (count 1 <= limit 1)
    job.state = cluster_pb2.JOB_STATE_FAILED
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=False)
    assert job.failure_count == 1

    # Second failure - should NOT retry (count 2 > limit 1)
    job.state = cluster_pb2.JOB_STATE_FAILED
    assert not handle_job_failure(state, JobId("j1"), is_worker_failure=False)
    assert job.failure_count == 2
    assert job.state == cluster_pb2.JOB_STATE_FAILED  # State unchanged


def test_job_retry_resets_state(make_job_request):
    """Verify job state/worker_id/timestamps cleared on retry."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_failure=1,
    )
    job.state = cluster_pb2.JOB_STATE_FAILED
    job.worker_id = "w1"
    job.started_at_ms = 12345
    job.finished_at_ms = 67890
    job.error = "Something went wrong"
    state.add_job(job)

    # Retry should reset all state
    assert handle_job_failure(state, JobId("j1"), is_worker_failure=False)
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.worker_id is None
    assert job.started_at_ms is None
    assert job.finished_at_ms is None
    assert job.error is None
    # exit_code is not cleared (intentional - keeps historical info)


def test_handle_job_failure_nonexistent_job():
    """Returns False for unknown job_id."""
    state = ControllerState()
    assert not handle_job_failure(state, JobId("nonexistent"), is_worker_failure=True)


def test_gang_all_or_nothing_retry(make_job_request):
    """Gang retry fails if any job has no retries left."""
    state = ControllerState()
    job1 = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=1,
    )
    job2 = ControllerJob(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=0,  # No retries
    )

    state.add_job(job1)
    state.add_job(job2)

    # Mark both as running
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails - j2 has 0 retries, so entire gang cannot retry
    retried = handle_gang_failure(state, "g1", is_worker_failure=False)
    assert retried == []

    # Both jobs should be marked KILLED
    assert job1.state == cluster_pb2.JOB_STATE_KILLED
    assert job2.state == cluster_pb2.JOB_STATE_KILLED
    assert "Gang g1 failed" in job1.error
    assert "Gang g1 failed" in job2.error

    # Failure counts should not be incremented since gang couldn't retry
    assert job1.failure_count == 0
    assert job2.failure_count == 0


def test_gang_retry_success(make_job_request):
    """All jobs in gang are retried together."""
    state = ControllerState()
    job1 = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_preemption=2,
    )
    job2 = ControllerJob(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_preemption=2,
    )

    state.add_job(job1)
    state.add_job(job2)

    # Mark both as running
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails due to worker failure - both have retries left
    retried = handle_gang_failure(state, "g1", is_worker_failure=True)

    assert set(retried) == {"j1", "j2"}

    # Both jobs should be reset to PENDING
    assert job1.state == cluster_pb2.JOB_STATE_PENDING
    assert job2.state == cluster_pb2.JOB_STATE_PENDING

    # Both jobs should have preemption_count incremented
    assert job1.preemption_count == 1
    assert job2.preemption_count == 1

    # State should be cleared
    assert job1.worker_id is None
    assert job2.worker_id is None
    assert job1.error is None
    assert job2.error is None


def test_gang_marks_running_jobs_as_killed(make_job_request):
    """Running jobs in gang marked KILLED on failure."""
    state = ControllerState()
    job1 = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=0,  # No retries
    )
    job2 = ControllerJob(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=0,  # No retries
    )
    job3 = ControllerJob(
        job_id=JobId("j3"),
        request=make_job_request("job3"),
        gang_id="g1",
        max_retries_failure=0,  # No retries
    )

    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    # Mix of states
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_PENDING  # Not started yet
    job3.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails - no retries available
    retried = handle_gang_failure(state, "g1", is_worker_failure=False)
    assert retried == []

    # Only running jobs should be marked KILLED
    assert job1.state == cluster_pb2.JOB_STATE_KILLED
    assert job2.state == cluster_pb2.JOB_STATE_PENDING  # Not running, so not killed
    assert job3.state == cluster_pb2.JOB_STATE_KILLED

    # All running jobs should have error message
    assert job1.error == "Gang g1 failed"
    assert job3.error == "Gang g1 failed"


def test_gang_failure_with_nonexistent_gang():
    """Returns empty list for unknown gang_id."""
    state = ControllerState()
    retried = handle_gang_failure(state, "nonexistent", is_worker_failure=True)
    assert retried == []


def test_gang_retry_tracks_correct_failure_type(make_job_request):
    """Gang retry increments correct counter based on failure type."""
    state = ControllerState()
    job1 = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=2,
        max_retries_preemption=2,
    )
    job2 = ControllerJob(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=2,
        max_retries_preemption=2,
    )

    state.add_job(job1)
    state.add_job(job2)

    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Test worker failure (preemption)
    retried = handle_gang_failure(state, "g1", is_worker_failure=True)
    assert len(retried) == 2
    assert job1.preemption_count == 1
    assert job2.preemption_count == 1
    assert job1.failure_count == 0
    assert job2.failure_count == 0

    # Reset for next test
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Test job failure (internal)
    retried = handle_gang_failure(state, "g1", is_worker_failure=False)
    assert len(retried) == 2
    assert job1.preemption_count == 1  # Unchanged
    assert job2.preemption_count == 1  # Unchanged
    assert job1.failure_count == 1
    assert job2.failure_count == 1


def test_job_with_zero_retries_default(make_job_request):
    """Default max_retries_failure=0 means one try, no retries."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        # max_retries_failure defaults to 0
    )
    job.state = cluster_pb2.JOB_STATE_FAILED
    state.add_job(job)

    # First failure should NOT retry (count 1 > limit 0)
    assert not handle_job_failure(state, JobId("j1"), is_worker_failure=False)
    assert job.failure_count == 1
    assert job.state == cluster_pb2.JOB_STATE_FAILED


def test_job_with_high_preemption_retries_default(make_job_request):
    """Default max_retries_preemption=100 allows many retries."""
    state = ControllerState()
    job = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        # max_retries_preemption defaults to 100
    )
    state.add_job(job)

    # Should be able to retry many times
    for i in range(100):
        job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
        assert handle_job_failure(state, JobId("j1"), is_worker_failure=True)
        assert job.preemption_count == i + 1

    # 101st failure should not retry
    job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
    assert not handle_job_failure(state, JobId("j1"), is_worker_failure=True)
    assert job.preemption_count == 101


def test_gang_retry_checks_all_jobs_for_retries(make_job_request):
    """Gang retry requires ALL jobs to have retries, not just majority."""
    state = ControllerState()
    # Create gang with 3 jobs, only 1 has no retries
    job1 = ControllerJob(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=5,
    )
    job2 = ControllerJob(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=5,
    )
    job3 = ControllerJob(
        job_id=JobId("j3"),
        request=make_job_request("job3"),
        gang_id="g1",
        max_retries_failure=0,  # Only this one has no retries
    )

    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING
    job3.state = cluster_pb2.JOB_STATE_RUNNING

    # Should not retry because j3 has no retries (even though 2/3 do)
    retried = handle_gang_failure(state, "g1", is_worker_failure=False)
    assert retried == []
