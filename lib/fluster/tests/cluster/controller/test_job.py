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

"""Tests for Job state transitions, retries, and gang scheduling."""

import pytest

from fluster.cluster.controller.job import Job, TransitionResult, handle_gang_failure
from fluster.cluster.types import JobId, WorkerId
from fluster.rpc import cluster_pb2


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


# --- Job State Transitions ---


@pytest.mark.parametrize(
    "target_state,exit_code,error",
    [
        (cluster_pb2.JOB_STATE_SUCCEEDED, 0, None),
        (cluster_pb2.JOB_STATE_KILLED, None, "Terminated by user"),
        (cluster_pb2.JOB_STATE_UNSCHEDULABLE, None, None),
    ],
)
def test_job_terminal_transitions(make_job_request, target_state, exit_code, error):
    """Job transitions to terminal states (SUCCEEDED, KILLED, UNSCHEDULABLE) with appropriate metadata."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)

    result = job.transition(target_state, now_ms=2000, exit_code=exit_code, error=error)

    assert result == TransitionResult.COMPLETE
    assert job.state == target_state
    assert job.finished_at_ms == 2000
    assert job.is_finished()

    if exit_code is not None:
        assert job.exit_code == exit_code
    if error is not None:
        assert job.error == error


def test_unschedulable_includes_timeout_in_error(make_job_request):
    """UNSCHEDULABLE state includes scheduling timeout in error message."""
    request = make_job_request()
    request.scheduling_timeout_seconds = 300
    job = Job(job_id=JobId("test"), request=request)

    result = job.transition(cluster_pb2.JOB_STATE_UNSCHEDULABLE, now_ms=2000)

    assert result == TransitionResult.COMPLETE
    assert job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
    assert job.error is not None
    assert "300s" in job.error


def test_job_dispatch_and_revert(make_job_request):
    """Job can be dispatched to worker and reverted back to PENDING."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    assert job.state == cluster_pb2.JOB_STATE_PENDING

    job.mark_dispatched(WorkerId("w1"), now_ms=1000)

    assert job.state == cluster_pb2.JOB_STATE_RUNNING
    assert job.worker_id == WorkerId("w1")
    assert job.started_at_ms == 1000

    job.revert_dispatch()

    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.worker_id is None
    assert job.started_at_ms is None


# --- Job Retry Behavior ---


def test_failure_with_retries_available(make_job_request):
    """Job failure returns SHOULD_RETRY when retries available and resets to PENDING."""
    job = Job(
        job_id=JobId("test"),
        request=make_job_request(),
        max_retries_failure=2,
    )
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)

    result = job.transition(
        cluster_pb2.JOB_STATE_FAILED,
        now_ms=2000,
        error="oops",
    )

    assert result == TransitionResult.SHOULD_RETRY
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.failure_count == 1
    assert job.worker_id is None
    assert job.started_at_ms is None
    assert job.error is None
    assert not job.is_finished()


def test_failure_exceeds_retry_limit(make_job_request):
    """Job failure returns EXCEEDED_RETRY_LIMIT when retry limit exceeded."""
    job = Job(
        job_id=JobId("test"),
        request=make_job_request(),
        max_retries_failure=1,
    )
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)

    # First failure - retry
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
    assert result == TransitionResult.SHOULD_RETRY
    assert job.failure_count == 1

    # Dispatch again
    job.mark_dispatched(WorkerId("w2"), now_ms=3000)

    # Second failure - no more retries
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=4000, error="final error")
    assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
    assert job.state == cluster_pb2.JOB_STATE_FAILED
    assert job.failure_count == 2
    assert job.error == "final error"
    assert job.finished_at_ms == 4000
    assert job.is_finished()


def test_worker_failure_uses_separate_retry_counter(make_job_request):
    """Worker failure increments preemption_count, not failure_count."""
    job = Job(
        job_id=JobId("test"),
        request=make_job_request(),
        max_retries_preemption=1,
    )
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)

    # First preemption - retry
    result = job.transition(
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        now_ms=2000,
        is_worker_failure=True,
    )
    assert result == TransitionResult.SHOULD_RETRY
    assert job.preemption_count == 1
    assert job.failure_count == 0

    # Dispatch again
    job.mark_dispatched(WorkerId("w2"), now_ms=3000)

    # Second preemption - no more retries
    result = job.transition(
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        now_ms=4000,
        is_worker_failure=True,
    )
    assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
    assert job.state == cluster_pb2.JOB_STATE_WORKER_FAILED
    assert job.preemption_count == 2


@pytest.mark.parametrize(
    "failure_type,max_retries,expected_attempts",
    [
        ("job_failure", 0, 1),  # Default: no retries means one attempt
        ("job_failure", 1, 2),  # 1 retry = 2 attempts
        ("job_failure", 3, 4),  # 3 retries = 4 attempts
        ("worker_failure", 100, 101),  # Default preemption: 100 retries
    ],
)
def test_retry_count_limits(make_job_request, failure_type, max_retries, expected_attempts):
    """Job respects retry limits for both failure types."""
    if failure_type == "job_failure":
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_failure=max_retries,
        )
        state = cluster_pb2.JOB_STATE_FAILED
        is_worker_failure = False
    else:
        job = Job(
            job_id=JobId("test"),
            request=make_job_request(),
            max_retries_preemption=max_retries,
        )
        state = cluster_pb2.JOB_STATE_WORKER_FAILED
        is_worker_failure = True

    # Attempt up to the limit
    for attempt in range(max_retries):
        job.mark_dispatched(WorkerId(f"w{attempt}"), now_ms=attempt * 1000)
        result = job.transition(state, now_ms=attempt * 1000 + 500, is_worker_failure=is_worker_failure)
        assert result == TransitionResult.SHOULD_RETRY

    # Final attempt should fail
    job.mark_dispatched(WorkerId(f"w{max_retries}"), now_ms=max_retries * 1000)
    result = job.transition(state, now_ms=max_retries * 1000 + 500, is_worker_failure=is_worker_failure)
    assert result == TransitionResult.EXCEEDED_RETRY_LIMIT
    assert job.is_finished()
    assert job.total_attempts == expected_attempts


# --- Gang Scheduling ---


def test_gang_all_or_nothing_retry(make_job_request):
    """Gang retry fails if any job has no retries left."""
    job1 = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=1,
    )
    job2 = Job(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=0,  # No retries
    )

    # Mark both as running
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails - j2 has 0 retries, so entire gang cannot retry
    retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=False, error="Gang g1 failed")
    assert retried == []

    # Both jobs should be marked KILLED
    assert job1.state == cluster_pb2.JOB_STATE_KILLED
    assert job2.state == cluster_pb2.JOB_STATE_KILLED

    # Failure counts should not be incremented since gang couldn't retry
    assert job1.failure_count == 0
    assert job2.failure_count == 0


def test_gang_retry_success(make_job_request):
    """All jobs in gang are retried together when all have retries available."""
    job1 = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_preemption=2,
    )
    job2 = Job(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_preemption=2,
    )

    # Mark both as running
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails due to worker failure - both have retries left
    retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=True, error="Worker died")

    assert len(retried) == 2
    assert job1 in retried
    assert job2 in retried

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


def test_gang_only_running_jobs_killed(make_job_request):
    """Only RUNNING jobs in gang are marked KILLED when gang fails without retry."""
    job1 = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=0,
    )
    job2 = Job(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=0,
    )
    job3 = Job(
        job_id=JobId("j3"),
        request=make_job_request("job3"),
        gang_id="g1",
        max_retries_failure=0,
    )

    # Mix of states
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_PENDING  # Not started yet
    job3.state = cluster_pb2.JOB_STATE_RUNNING

    # Gang fails - no retries available
    retried = handle_gang_failure([job1, job2, job3], now_ms=1000, is_worker_failure=False, error="Gang failed")
    assert retried == []

    # Only running jobs should be marked KILLED
    assert job1.state == cluster_pb2.JOB_STATE_KILLED
    assert job2.state == cluster_pb2.JOB_STATE_PENDING  # Not running, so not killed
    assert job3.state == cluster_pb2.JOB_STATE_KILLED


def test_gang_tracks_correct_failure_type(make_job_request):
    """Gang retry increments correct counter based on failure type."""
    job1 = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=2,
        max_retries_preemption=2,
    )
    job2 = Job(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=2,
        max_retries_preemption=2,
    )

    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Test worker failure (preemption)
    retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=True, error="Worker died")
    assert len(retried) == 2
    assert job1.preemption_count == 1
    assert job2.preemption_count == 1
    assert job1.failure_count == 0
    assert job2.failure_count == 0

    # Reset for next test
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Test job failure (internal)
    retried = handle_gang_failure([job1, job2], now_ms=2000, is_worker_failure=False, error="Job crashed")
    assert len(retried) == 2
    assert job1.preemption_count == 1  # Unchanged
    assert job2.preemption_count == 1  # Unchanged
    assert job1.failure_count == 1
    assert job2.failure_count == 1


def test_gang_empty_list():
    """Empty job list returns empty list."""
    retried = handle_gang_failure([], now_ms=1000, is_worker_failure=True, error="error")
    assert retried == []
