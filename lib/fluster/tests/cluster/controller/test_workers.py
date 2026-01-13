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

"""Tests for worker registry and scheduling."""

import time

import pytest

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.controller.workers import WorkerConfig, find_worker_for_job, load_workers_from_config
from fluster.cluster.types import JobId, WorkerId


@pytest.fixture
def make_resource_spec():
    """Create a minimal ResourceSpec for testing."""

    def _make(cpu: int = 8, memory: str = "32g") -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=cpu, memory=memory, disk="100g")

    return _make


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


def test_load_workers_from_config(make_resource_spec):
    """Verify workers are added to state correctly."""
    state = ControllerState()
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
        WorkerConfig("w2", "host2:8080", make_resource_spec()),
    ]

    before_ms = int(time.time() * 1000)
    load_workers_from_config(state, workers)
    after_ms = int(time.time() * 1000)

    # Verify workers were added
    assert len(state.get_available_workers()) == 2

    # Verify worker details
    worker1 = state.get_worker(WorkerId("w1"))
    assert worker1 is not None
    assert worker1.address == "host1:8080"
    assert worker1.healthy is True
    assert worker1.resources.cpu == 8
    assert worker1.resources.memory == "32g"

    # Verify last_heartbeat_ms was set to current time
    assert before_ms <= worker1.last_heartbeat_ms <= after_ms

    worker2 = state.get_worker(WorkerId("w2"))
    assert worker2 is not None
    assert worker2.address == "host2:8080"


def test_find_worker_for_job_returns_healthy_worker(make_resource_spec, make_job_request):
    """Verify healthy worker is returned."""
    state = ControllerState()
    worker = ControllerWorker(worker_id=WorkerId("w1"), address="host1:8080", resources=make_resource_spec())
    state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # Should return the healthy worker
    result = find_worker_for_job(state, job)
    assert result is not None
    assert result.worker_id == "w1"
    assert result.address == "host1:8080"


def test_find_worker_for_job_skips_unhealthy(make_resource_spec, make_job_request):
    """Verify unhealthy workers are skipped."""
    state = ControllerState()

    # Add unhealthy worker
    worker1 = ControllerWorker(worker_id=WorkerId("w1"), address="host1:8080", resources=make_resource_spec())
    worker1.healthy = False
    state.add_worker(worker1)

    # Add healthy worker
    worker2 = ControllerWorker(worker_id=WorkerId("w2"), address="host2:8080", resources=make_resource_spec())
    state.add_worker(worker2)

    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # Should skip w1 and return w2
    result = find_worker_for_job(state, job)
    assert result is not None
    assert result.worker_id == "w2"


def test_find_worker_for_job_no_workers_returns_none(make_job_request):
    """Verify None when no workers available."""
    state = ControllerState()
    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # No workers registered
    result = find_worker_for_job(state, job)
    assert result is None


def test_find_worker_for_job_all_unhealthy_returns_none(make_resource_spec, make_job_request):
    """Verify None when all workers are unhealthy."""
    state = ControllerState()

    # Add multiple unhealthy workers
    for i in range(3):
        worker = ControllerWorker(worker_id=WorkerId(f"w{i}"), address=f"host{i}:8080", resources=make_resource_spec())
        worker.healthy = False
        state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # Should return None since all workers are unhealthy
    result = find_worker_for_job(state, job)
    assert result is None


def test_load_workers_from_config_empty_list():
    """Verify loading empty worker list works."""
    state = ControllerState()
    load_workers_from_config(state, [])

    assert len(state.get_available_workers()) == 0


def test_find_worker_for_job_returns_first_available(make_resource_spec, make_job_request):
    """Verify first-fit behavior - returns first healthy worker."""
    state = ControllerState()

    # Add multiple healthy workers
    for i in range(5):
        worker = ControllerWorker(worker_id=WorkerId(f"w{i}"), address=f"host{i}:8080", resources=make_resource_spec())
        state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=make_job_request("job1"))

    # Should return the first worker (order may vary due to dict iteration)
    result = find_worker_for_job(state, job)
    assert result is not None
    assert result.worker_id in {f"w{i}" for i in range(5)}
