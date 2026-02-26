# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for job state derivation and job-to-task expansion."""

import pytest

from iris.cluster.controller.state import ControllerJob, expand_job_to_tasks
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
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )

    return _make


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


def test_job_expands_tasks_with_retry_limits_from_request(make_job_request):
    """expand_job_to_tasks reads per-task retry limits from LaunchJobRequest."""
    request = make_job_request()
    request.replicas = 2
    request.max_retries_failure = 3
    request.max_retries_preemption = 7
    job = ControllerJob(job_id=JobName.root("test-job"), request=request)

    tasks = expand_job_to_tasks(job)

    assert len(tasks) == 2
    for task in tasks:
        assert task.max_retries_failure == 3
        assert task.max_retries_preemption == 7


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
