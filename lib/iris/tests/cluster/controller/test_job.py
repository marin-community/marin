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

from iris.cluster.controller.job import Job, TransitionResult, expand_job_to_tasks, handle_gang_failure
from iris.cluster.types import JobId
from iris.rpc import cluster_pb2


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
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
    job.mark_dispatched(now_ms=1000)

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

    job.mark_dispatched(now_ms=1000)

    assert job.state == cluster_pb2.JOB_STATE_RUNNING
    assert job.started_at_ms == 1000

    job.revert_dispatch()

    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.started_at_ms is None


# --- Job Retry Behavior ---


def test_failure_with_retries_available(make_job_request):
    """Job failure returns SHOULD_RETRY when retries available and resets to PENDING."""
    job = Job(
        job_id=JobId("test"),
        request=make_job_request(),
        max_retries_failure=2,
    )
    job.mark_dispatched(now_ms=1000)

    result = job.transition(
        cluster_pb2.JOB_STATE_FAILED,
        now_ms=2000,
        error="oops",
    )

    assert result == TransitionResult.SHOULD_RETRY
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.failure_count == 1
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
    job.mark_dispatched(now_ms=1000)

    # First failure - retry
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
    assert result == TransitionResult.SHOULD_RETRY
    assert job.failure_count == 1

    # Dispatch again
    job.mark_dispatched(now_ms=3000)

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
    job.mark_dispatched(now_ms=1000)

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
    job.mark_dispatched(now_ms=3000)

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
        job.mark_dispatched(now_ms=attempt * 1000)
        result = job.transition(state, now_ms=attempt * 1000 + 500, is_worker_failure=is_worker_failure)
        assert result == TransitionResult.SHOULD_RETRY

    # Final attempt should fail
    job.mark_dispatched(now_ms=max_retries * 1000)
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
    """Gang retry uses correct retry budget based on failure type."""
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

    # After a worker failure, preemption retry budget should be consumed
    assert job1.can_retry_preemption()
    assert job2.can_retry_preemption()
    retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=True, error="Worker died")
    assert len(retried) == 2

    # Failure retry budget should be untouched
    assert job1.can_retry_failure()
    assert job2.can_retry_failure()

    # Reset for next test
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # After a job failure, failure retry budget should be consumed
    retried = handle_gang_failure([job1, job2], now_ms=2000, is_worker_failure=False, error="Job crashed")
    assert len(retried) == 2

    # Still have one more retry in each budget
    assert job1.can_retry_failure()
    assert job1.can_retry_preemption()


# --- Task State Tracking ---


def test_job_compute_job_state_all_succeeded(make_job_request):
    """Job state becomes SUCCEEDED when all tasks succeed."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    job.num_tasks = 2

    # Start with pending tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 2

    # First task succeeds - job should stay pending
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_SUCCEEDED, now_ms=1000)
    assert new_state is None

    # Second task succeeds - job should become SUCCEEDED
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_SUCCEEDED, now_ms=2000)
    assert new_state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_compute_job_state_failed(make_job_request):
    """Job state becomes FAILED when task failures exceed threshold."""
    request = make_job_request()
    request.max_task_failures = 0
    job = Job(job_id=JobId("test"), request=request)
    job.num_tasks = 2

    # Start with running tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] = 2
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # First task fails - should trigger job failure (0 allowed, 1 failed)
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED, now_ms=1000)
    assert new_state == cluster_pb2.JOB_STATE_FAILED


def test_job_compute_job_state_tolerates_failures(make_job_request):
    """Job state stays RUNNING when failures are within threshold."""
    request = make_job_request()
    request.max_task_failures = 1
    job = Job(job_id=JobId("test"), request=request)
    job.num_tasks = 3

    # Start with running tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] = 3
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # First task fails - job stays running (1 allowed, 1 failed)
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED, now_ms=1000)
    assert new_state is None  # No state change

    # Second task fails - job should fail (1 allowed, 2 failed)
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED, now_ms=2000)
    assert new_state == cluster_pb2.JOB_STATE_FAILED


def test_job_finished_task_count(make_job_request):
    """finished_task_count returns count of tasks in terminal states."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    job.num_tasks = 5

    # Start with 5 pending tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 5

    # Move 2 tasks through running to succeeded
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING, now_ms=1000)
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING, now_ms=1001)
    job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_SUCCEEDED, now_ms=2000)
    job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_SUCCEEDED, now_ms=2001)

    # Move 1 task through running to failed
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING, now_ms=1002)
    job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED, now_ms=2002)

    # Keep 2 tasks running
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING, now_ms=1003)
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING, now_ms=1004)

    assert job.finished_task_count == 3  # 2 succeeded + 1 failed


def test_job_on_task_transition_sets_running_on_first_dispatch(make_job_request):
    """Job state becomes RUNNING when first task starts running."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    job.num_tasks = 2
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 2

    # First task starts running
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING, now_ms=1000)

    assert new_state == cluster_pb2.JOB_STATE_RUNNING
    assert job.started_at_ms == 1000


# --- Job Expansion ---


def test_job_expands_to_correct_number_of_tasks(make_job_request):
    """expand_job_to_tasks creates correct number of tasks based on replicas."""
    request = make_job_request()
    request.resources.replicas = 3
    job = Job(job_id=JobId("test-job"), request=request)

    tasks = expand_job_to_tasks(job, now_ms=1000)

    assert len(tasks) == 3
    for i, task in enumerate(tasks):
        assert task.task_index == i
        assert task.job_id == job.job_id
        assert str(task.task_id) == f"{job.job_id}/task-{i}"
        assert task.max_retries_failure == job.max_retries_failure
        assert task.max_retries_preemption == job.max_retries_preemption


def test_job_expands_single_replica_by_default(make_job_request):
    """expand_job_to_tasks creates single task when replicas not specified."""
    request = make_job_request()
    # replicas defaults to 0 in proto, which should be treated as 1
    job = Job(job_id=JobId("test-job"), request=request)

    tasks = expand_job_to_tasks(job, now_ms=1000)

    assert len(tasks) == 1
    assert tasks[0].task_index == 0


def test_job_becomes_unschedulable_when_task_unschedulable(make_job_request):
    """Job transitions to UNSCHEDULABLE when any task becomes unschedulable."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    job.num_tasks = 3
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 3

    # One task becomes unschedulable
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_UNSCHEDULABLE, now_ms=1000)

    assert new_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE


def test_job_becomes_killed_when_task_killed(make_job_request):
    """Job transitions to KILLED when any task is killed."""
    job = Job(job_id=JobId("test"), request=make_job_request())
    job.num_tasks = 3
    job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] = 3
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # One task is killed
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_KILLED, now_ms=1000)

    assert new_state == cluster_pb2.JOB_STATE_KILLED


def test_gang_failure_respects_minimum_retry_budget(make_job_request):
    """Gang fails when any member has no retries, even if others have budget."""
    job1 = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        gang_id="g1",
        max_retries_failure=5,  # Plenty of retries
    )
    job2 = Job(
        job_id=JobId("j2"),
        request=make_job_request("job2"),
        gang_id="g1",
        max_retries_failure=0,  # No retries
    )

    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job2.state = cluster_pb2.JOB_STATE_RUNNING

    # Job2 cannot retry, so entire gang fails
    assert job1.can_retry_failure()
    assert not job2.can_retry_failure()

    retried = handle_gang_failure([job1, job2], now_ms=1000, is_worker_failure=False, error="Gang failed")

    # No jobs should be retried
    assert retried == []

    # Both should be killed
    assert job1.state == cluster_pb2.JOB_STATE_KILLED
    assert job2.state == cluster_pb2.JOB_STATE_KILLED


def test_job_total_attempts_reflects_retry_history(make_job_request):
    """total_attempts property correctly reflects the job's execution history."""
    job = Job(
        job_id=JobId("test"),
        request=make_job_request(),
        max_retries_failure=3,
        max_retries_preemption=3,
    )

    # Fresh job has 1 attempt (the initial one)
    assert job.total_attempts == 1

    # After a failure retry
    job.mark_dispatched(now_ms=1000)
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
    assert result == TransitionResult.SHOULD_RETRY
    assert job.total_attempts == 1  # 1 failure

    # After a preemption retry
    job.mark_dispatched(now_ms=3000)
    result = job.transition(cluster_pb2.JOB_STATE_WORKER_FAILED, now_ms=4000, is_worker_failure=True)
    assert result == TransitionResult.SHOULD_RETRY
    assert job.total_attempts == 2  # 1 failure + 1 preemption

    # After another failure
    job.mark_dispatched(now_ms=5000)
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=6000)
    assert result == TransitionResult.SHOULD_RETRY
    assert job.total_attempts == 3  # 2 failures + 1 preemption
