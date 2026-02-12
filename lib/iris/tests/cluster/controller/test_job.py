# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Job state transitions and retries."""

import pytest

from iris.cluster.controller.state import ControllerJob, JobTransitionResult, expand_job_to_tasks
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
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
    job = ControllerJob(job_id=JobName.root("test"), request=make_job_request())
    job.mark_dispatched()

    result = job.transition(target_state, exit_code=exit_code, error=error)

    assert result == JobTransitionResult.COMPLETE
    assert job.state == target_state
    assert job.finished_at is not None and job.finished_at.epoch_ms() > 0
    assert job.is_finished()


def test_unschedulable_includes_timeout_in_error(make_job_request):
    """UNSCHEDULABLE state includes scheduling timeout in error message."""
    from iris.time_utils import Duration

    request = make_job_request()
    request.scheduling_timeout.CopyFrom(Duration.from_seconds(300).to_proto())
    job = ControllerJob(job_id=JobName.root("test"), request=request)

    result = job.transition(cluster_pb2.JOB_STATE_UNSCHEDULABLE)

    assert result == JobTransitionResult.COMPLETE
    assert job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
    assert job.error is not None
    assert "300" in job.error


# --- Job Retry Behavior ---


def test_failure_with_retries_available(make_job_request):
    """Job failure returns SHOULD_RETRY when retries available and resets to PENDING."""
    job = ControllerJob(
        job_id=JobName.root("test"),
        request=make_job_request(),
        max_retries_failure=2,
    )
    job.mark_dispatched()

    result = job.transition(
        cluster_pb2.JOB_STATE_FAILED,
        error="oops",
    )

    assert result == JobTransitionResult.SHOULD_RETRY
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.failure_count == 1
    assert job.started_at is None
    assert not job.is_finished()


def test_failure_exceeds_retry_limit(make_job_request):
    """Job failure returns EXCEEDED_RETRY_LIMIT when retry limit exceeded."""
    job = ControllerJob(
        job_id=JobName.root("test"),
        request=make_job_request(),
        max_retries_failure=1,
    )
    job.mark_dispatched()

    # First failure - retry
    result = job.transition(cluster_pb2.JOB_STATE_FAILED)
    assert result == JobTransitionResult.SHOULD_RETRY
    assert job.failure_count == 1

    # Dispatch again
    job.mark_dispatched()

    # Second failure - no more retries
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, error="final error")
    assert result == JobTransitionResult.EXCEEDED_RETRY_LIMIT
    assert job.state == cluster_pb2.JOB_STATE_FAILED
    assert job.failure_count == 2
    assert job.finished_at is not None and job.finished_at.epoch_ms() > 0
    assert job.is_finished()


def test_worker_failure_uses_separate_retry_counter(make_job_request):
    """Worker failure increments preemption_count, not failure_count."""
    job = ControllerJob(
        job_id=JobName.root("test"),
        request=make_job_request(),
        max_retries_preemption=1,
    )
    job.mark_dispatched()

    # First preemption - retry
    result = job.transition(
        cluster_pb2.JOB_STATE_FAILED,
        is_worker_failure=True,
    )
    assert result == JobTransitionResult.SHOULD_RETRY
    assert job.preemption_count == 1
    assert job.failure_count == 0

    # Dispatch again
    job.mark_dispatched()

    # Second preemption - no more retries
    result = job.transition(
        cluster_pb2.JOB_STATE_FAILED,
        is_worker_failure=True,
    )
    assert result == JobTransitionResult.EXCEEDED_RETRY_LIMIT
    assert job.state == cluster_pb2.JOB_STATE_FAILED
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
        job = ControllerJob(
            job_id=JobName.root("test"),
            request=make_job_request(),
            max_retries_failure=max_retries,
        )
        state = cluster_pb2.JOB_STATE_FAILED
        is_worker_failure = False
    else:
        job = ControllerJob(
            job_id=JobName.root("test"),
            request=make_job_request(),
            max_retries_preemption=max_retries,
        )
        state = cluster_pb2.JOB_STATE_FAILED
        is_worker_failure = True

    # Attempt up to the limit
    for _ in range(max_retries):
        job.mark_dispatched()
        result = job.transition(state, is_worker_failure=is_worker_failure)
        assert result == JobTransitionResult.SHOULD_RETRY

    # Final attempt should fail
    job.mark_dispatched()
    result = job.transition(state, is_worker_failure=is_worker_failure)
    assert result == JobTransitionResult.EXCEEDED_RETRY_LIMIT
    assert job.is_finished()
    # expected_attempts = max_retries + 1 (original attempt + retries)
    if failure_type == "job_failure":
        assert job.failure_count == expected_attempts
    else:
        assert job.preemption_count == expected_attempts


# --- Task State Tracking ---


def test_job_compute_job_state_all_succeeded(make_job_request):
    """Job state becomes SUCCEEDED when all tasks succeed."""
    job = ControllerJob(job_id=JobName.root("test"), request=make_job_request())
    job.num_tasks = 2

    # Start with pending tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 2

    # First task succeeds - job should stay pending
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_SUCCEEDED)
    assert new_state is None

    # Second task succeeds - job should become SUCCEEDED
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_SUCCEEDED)
    assert new_state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_compute_job_state_failed(make_job_request):
    """Job state becomes FAILED when task failures exceed threshold."""
    request = make_job_request()
    request.max_task_failures = 0
    job = ControllerJob(job_id=JobName.root("test"), request=request)
    job.num_tasks = 2

    # Start with running tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] = 2
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # First task fails - should trigger job failure (0 allowed, 1 failed)
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED)
    assert new_state == cluster_pb2.JOB_STATE_FAILED


def test_job_compute_job_state_tolerates_failures(make_job_request):
    """Job state stays RUNNING when failures are within threshold."""
    request = make_job_request()
    request.max_task_failures = 1
    job = ControllerJob(job_id=JobName.root("test"), request=request)
    job.num_tasks = 3

    # Start with running tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] = 3
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # First task fails - job stays running (1 allowed, 1 failed)
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED)
    assert new_state is None  # No state change

    # Second task fails - job should fail (1 allowed, 2 failed)
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED)
    assert new_state == cluster_pb2.JOB_STATE_FAILED


def test_job_finished_task_count(make_job_request):
    """finished_task_count returns count of tasks in terminal states."""
    job = ControllerJob(job_id=JobName.root("test"), request=make_job_request())
    job.num_tasks = 5

    # Start with 5 pending tasks
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 5

    # Move 2 tasks through running to succeeded
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING)
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING)
    job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_SUCCEEDED)
    job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Move 1 task through running to failed
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING)
    job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_FAILED)

    # Keep 2 tasks running
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING)
    job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING)

    assert job.finished_task_count == 3  # 2 succeeded + 1 failed


def test_job_on_task_transition_sets_running_on_first_dispatch(make_job_request):
    """Job state becomes RUNNING when first task starts running."""
    job = ControllerJob(job_id=JobName.root("test"), request=make_job_request())
    job.num_tasks = 2
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 2

    # First task starts running
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_RUNNING)

    assert new_state == cluster_pb2.JOB_STATE_RUNNING
    assert job.started_at is not None and job.started_at.epoch_ms() > 0


# --- Job Expansion ---


def test_job_expands_to_correct_number_of_tasks(make_job_request):
    """expand_job_to_tasks creates correct number of tasks based on replicas."""
    request = make_job_request()
    request.replicas = 3
    job = ControllerJob(job_id=JobName.root("test-job"), request=request)

    tasks = expand_job_to_tasks(job)

    assert len(tasks) == 3
    for i, task in enumerate(tasks):
        assert task.task_index == i
        assert task.job_id == job.job_id


def test_job_becomes_unschedulable_when_task_unschedulable(make_job_request):
    """Job transitions to UNSCHEDULABLE when any task becomes unschedulable."""
    job = ControllerJob(job_id=JobName.root("test"), request=make_job_request())
    job.num_tasks = 3
    job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 3

    # One task becomes unschedulable
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_PENDING, cluster_pb2.TASK_STATE_UNSCHEDULABLE)

    assert new_state == cluster_pb2.JOB_STATE_UNSCHEDULABLE


def test_job_becomes_killed_when_task_killed(make_job_request):
    """Job transitions to KILLED when any task is killed."""
    job = ControllerJob(job_id=JobName.root("test"), request=make_job_request())
    job.num_tasks = 3
    job.task_state_counts[cluster_pb2.TASK_STATE_RUNNING] = 3
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # One task is killed
    new_state = job.on_task_transition(cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_KILLED)

    assert new_state == cluster_pb2.JOB_STATE_KILLED
