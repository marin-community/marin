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

"""Integration tests for job lifecycle and state invariant maintenance.

These tests verify that state invariants (endpoint cleanup, worker assignment)
are properly maintained throughout the job lifecycle.
"""

import pytest

from fluster.cluster.controller.job import Job, TransitionResult
from fluster.cluster.controller.state import (
    ControllerEndpoint,
    ControllerState,
    ControllerWorker,
)
from fluster.cluster.types import JobId, WorkerId
from fluster.rpc import cluster_pb2


@pytest.fixture
def make_job_request():
    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def make_resource_spec():
    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=8, memory="32g")

    return _make


def test_succeeded_job_cleans_up_endpoints_and_worker(make_job_request, make_resource_spec):
    """Full lifecycle: submit -> dispatch -> succeed -> cleanup."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
    )
    state.add_worker(worker)

    job = Job(job_id=JobId("j1"), request=make_job_request("job1"))
    state.add_job(job)

    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # Verify endpoint visible while running
    results = state.lookup_endpoints("j1/my-actor")
    assert len(results) == 1

    # Job succeeds
    job.transition(cluster_pb2.JOB_STATE_SUCCEEDED, now_ms=2000)
    state.finalize_job(JobId("j1"))

    # Verify cleanup
    assert state.lookup_endpoints("j1/my-actor") == []
    assert JobId("j1") not in worker.running_jobs


def test_failed_job_with_retry_unassigns_worker(make_job_request, make_resource_spec):
    """Failed job with retries unassigns worker but keeps endpoints until final failure."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
    )
    state.add_worker(worker)

    job = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_failure=1,
    )
    state.add_job(job)

    # First attempt
    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # First failure - should retry
    result = job.transition(cluster_pb2.JOB_STATE_FAILED, now_ms=2000)
    assert result == TransitionResult.SHOULD_RETRY

    # Unassign from worker (simulating what controller does)
    state.unassign_job_from_worker(WorkerId("w1"), JobId("j1"))

    # Worker unassigned but endpoint still exists
    assert JobId("j1") not in worker.running_jobs
    assert job.state == cluster_pb2.JOB_STATE_PENDING


def test_killed_job_cleans_up_endpoints(make_job_request):
    """Killed job has endpoints cleaned up."""
    state = ControllerState()

    job = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # Kill the job
    job.transition(cluster_pb2.JOB_STATE_KILLED, now_ms=1000, error="User terminated")
    state.finalize_job(JobId("j1"))

    assert state.lookup_endpoints("j1/actor") == []


def test_worker_timeout_cleans_up_job_endpoints(make_job_request, make_resource_spec):
    """Worker timeout triggers proper job cleanup including endpoints."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
        last_heartbeat_ms=0,
    )
    state.add_worker(worker)

    job = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        max_retries_preemption=0,
    )
    state.add_job(job)

    job.mark_dispatched(WorkerId("w1"), now_ms=1000)
    state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep)

    # Simulate worker timeout - job transitions to WORKER_FAILED
    result = job.transition(
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        now_ms=60000,
        is_worker_failure=True,
        error="Worker timed out",
    )
    assert result == TransitionResult.EXCEEDED_RETRY_LIMIT

    # Finalize the job
    state.finalize_job(JobId("j1"))

    assert state.lookup_endpoints("j1/actor") == []
    assert JobId("j1") not in worker.running_jobs


def test_finalize_job_removes_multiple_endpoints(make_job_request):
    """Finalize removes all endpoints for a job."""
    state = ControllerState()

    job = Job(
        job_id=JobId("j1"),
        request=make_job_request("job1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    for i in range(5):
        ep = ControllerEndpoint(
            endpoint_id=f"ep{i}",
            name=f"j1/actor-{i}",
            address=f"a:{i}",
            job_id=JobId("j1"),
        )
        state.add_endpoint(ep)

    # Verify all endpoints exist
    for i in range(5):
        assert len(state.lookup_endpoints(f"j1/actor-{i}")) == 1

    # Finalize
    job.transition(cluster_pb2.JOB_STATE_SUCCEEDED, now_ms=1000)
    found_job, removed = state.finalize_job(JobId("j1"))

    assert found_job is job
    assert len(removed) == 5

    # All endpoints gone
    for i in range(5):
        assert state.lookup_endpoints(f"j1/actor-{i}") == []


def test_finalize_job_nonexistent_returns_none():
    """Finalize job returns None for missing job."""
    state = ControllerState()

    job, removed = state.finalize_job(JobId("missing"))

    assert job is None
    assert removed == []


def test_assign_job_to_worker(make_resource_spec):
    """Assign job to worker adds job to running_jobs."""
    state = ControllerState()
    worker = ControllerWorker(worker_id=WorkerId("w1"), address="host:8080", resources=make_resource_spec())
    state.add_worker(worker)

    result = state.assign_job_to_worker(WorkerId("w1"), JobId("j1"))

    assert result is True
    assert JobId("j1") in worker.running_jobs


def test_assign_job_to_nonexistent_worker():
    """Assign job to nonexistent worker returns False."""
    state = ControllerState()

    result = state.assign_job_to_worker(WorkerId("missing"), JobId("j1"))

    assert result is False


def test_unassign_job_from_worker(make_resource_spec):
    """Unassign job from worker removes job from running_jobs."""
    state = ControllerState()
    worker = ControllerWorker(worker_id=WorkerId("w1"), address="host:8080", resources=make_resource_spec())
    worker.running_jobs.add(JobId("j1"))
    state.add_worker(worker)

    result = state.unassign_job_from_worker(WorkerId("w1"), JobId("j1"))

    assert result is True
    assert JobId("j1") not in worker.running_jobs


def test_mark_worker_unhealthy(make_resource_spec):
    """Mark worker unhealthy sets healthy=False."""
    state = ControllerState()
    worker = ControllerWorker(worker_id=WorkerId("w1"), address="host:8080", resources=make_resource_spec())
    state.add_worker(worker)

    result = state.mark_worker_unhealthy(WorkerId("w1"))

    assert result is worker
    assert worker.healthy is False


def test_update_worker_heartbeat(make_resource_spec):
    """Update worker heartbeat updates timestamp and healthy."""
    state = ControllerState()
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=make_resource_spec(),
        healthy=False,
        last_heartbeat_ms=0,
    )
    state.add_worker(worker)

    result = state.update_worker_heartbeat(WorkerId("w1"), now_ms=5000)

    assert result is True
    assert worker.healthy is True
    assert worker.last_heartbeat_ms == 5000


def test_update_worker_heartbeat_updates_resources(make_resource_spec):
    """Update worker heartbeat can update resources."""
    state = ControllerState()
    original_resources = make_resource_spec()
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        resources=original_resources,
    )
    state.add_worker(worker)

    new_resources = cluster_pb2.ResourceSpec(cpu=16, memory="64g")
    result = state.update_worker_heartbeat(WorkerId("w1"), now_ms=5000, resources=new_resources)

    assert result is True
    assert worker.resources == new_resources
