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

"""Integration tests for the Controller.

These tests exercise the complete job lifecycle through the Controller,
using a mock WorkerStubFactory to simulate worker responses.

Tests call Controller methods directly (no background threads) for
deterministic, synchronous testing.
"""

import pytest

from fluster import cluster_pb2
from fluster.cluster.controller.controller import Controller, ControllerConfig
from fluster.cluster.controller.retry import handle_job_failure
from fluster.cluster.controller.workers import WorkerConfig, load_workers_from_config
from fluster.cluster.types import JobId


class MockWorkerStub:
    """Mock worker stub that returns configured responses."""

    def __init__(self):
        self.job_statuses: dict[str, cluster_pb2.JobStatus] = {}
        self.run_job_calls: list[cluster_pb2.RunJobRequest] = []
        self.healthy = True

    def run_job(self, request: cluster_pb2.RunJobRequest) -> cluster_pb2.RunJobResponse:
        self.run_job_calls.append(request)
        return cluster_pb2.RunJobResponse(job_id=request.job_id, state=cluster_pb2.JOB_STATE_RUNNING)

    def get_job_status(self, request: cluster_pb2.GetStatusRequest) -> cluster_pb2.JobStatus:
        return self.job_statuses.get(request.job_id, cluster_pb2.JobStatus())

    def list_jobs(self, request: cluster_pb2.ListJobsRequest) -> cluster_pb2.ListJobsResponse:
        return cluster_pb2.ListJobsResponse(jobs=list(self.job_statuses.values()))

    def health_check(self, request: cluster_pb2.Empty) -> cluster_pb2.HealthResponse:
        if not self.healthy:
            raise ConnectionError("Worker unavailable")
        return cluster_pb2.HealthResponse(healthy=True)

    def set_job_completed(self, job_id: str, state: int, exit_code: int = 0, error: str = ""):
        self.job_statuses[job_id] = cluster_pb2.JobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            error=error,
            finished_at_ms=1000,
        )


class MockWorkerStubFactory:
    """Factory that returns mock stubs for testing."""

    def __init__(self):
        self.stubs: dict[str, MockWorkerStub] = {}

    def get_stub(self, address: str) -> MockWorkerStub:
        if address not in self.stubs:
            self.stubs[address] = MockWorkerStub()
        return self.stubs[address]

    def get_stub_for_worker(self, worker_id: str) -> MockWorkerStub:
        """Helper to get stub by worker_id (assumes host{N}:8080 format)."""
        address = f"host{worker_id[1:]}:8080" if worker_id.startswith("w") else f"{worker_id}:8080"
        return self.get_stub(address)


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job", cpu: int = 1) -> cluster_pb2.LaunchJobRequest:
        return cluster_pb2.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=cpu, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def make_resource_spec():
    """Create a minimal ResourceSpec for testing."""

    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=4, memory="8g", disk="10g")

    return _make


@pytest.fixture
def controller_with_workers(make_resource_spec):
    """Create a Controller with mock stub factory and register workers."""

    def _make(worker_ids: list[str], resources: cluster_pb2.ResourceSpec | None = None):
        stub_factory = MockWorkerStubFactory()
        config = ControllerConfig(port=0)
        controller = Controller(config, stub_factory)

        res = resources or make_resource_spec()
        workers = [WorkerConfig(wid, f"host{wid[1:]}:8080", res) for wid in worker_ids]
        load_workers_from_config(controller.state, workers)

        return controller, stub_factory

    return _make


def test_full_job_lifecycle(make_job_request, controller_with_workers):
    """Integration test: full job lifecycle from submission to completion."""
    controller, stub_factory = controller_with_workers(["w1"])

    # Submit job
    response = controller.launch_job(make_job_request("test-job"))
    job_id = response.job_id

    # Verify job is PENDING
    status = controller.get_job_status(job_id)
    assert status.job.state == cluster_pb2.JOB_STATE_PENDING

    # Run scheduling
    controller._run_scheduling()

    # Verify job is RUNNING
    status = controller.get_job_status(job_id)
    assert status.job.state == cluster_pb2.JOB_STATE_RUNNING

    # Verify dispatch RPC was called
    stub = stub_factory.get_stub("host1:8080")
    assert len(stub.run_job_calls) == 1
    assert stub.run_job_calls[0].job_id == job_id

    # Configure mock to report job succeeded
    stub.set_job_completed(job_id, cluster_pb2.JOB_STATE_SUCCEEDED)

    # Run heartbeat
    controller._run_heartbeats()

    # Verify job succeeded
    status = controller.get_job_status(job_id)
    assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert status.job.exit_code == 0


def test_job_failure_and_retry(make_job_request, controller_with_workers):
    """Job fails on first attempt, succeeds after retry."""
    controller, stub_factory = controller_with_workers(["w1"])

    # Submit job with retries enabled
    response = controller.launch_job(make_job_request("test-job"))
    job_id = response.job_id

    job = controller.state.get_job(JobId(job_id))
    job.max_retries_failure = 1

    # Run scheduling - job starts
    controller._run_scheduling()
    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_RUNNING

    # Configure mock to report failure
    stub = stub_factory.get_stub("host1:8080")
    stub.set_job_completed(job_id, cluster_pb2.JOB_STATE_FAILED, exit_code=1, error="Simulated failure")

    # Run heartbeat - picks up failure
    controller._run_heartbeats()

    # Job should be FAILED
    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_FAILED

    # Trigger retry
    handle_job_failure(controller.state, JobId(job_id), is_worker_failure=False)

    # Job should be back to PENDING
    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.failure_count == 1

    # Clear the failure status and run scheduling again
    stub.job_statuses.clear()
    controller._run_scheduling()

    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_RUNNING

    # This time configure success
    stub.set_job_completed(job_id, cluster_pb2.JOB_STATE_SUCCEEDED)
    controller._run_heartbeats()

    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert job.failure_count == 1  # Retained from retry


def test_worker_failure_triggers_retry(make_job_request, controller_with_workers):
    """Worker dies, job is retried on another worker."""
    controller, stub_factory = controller_with_workers(["w1", "w2"])

    # Submit job
    response = controller.launch_job(make_job_request("test-job"))
    job_id = response.job_id

    job = controller.state.get_job(JobId(job_id))
    job.max_retries_preemption = 10

    # Run scheduling - job goes to w1
    controller._run_scheduling()

    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_RUNNING
    assert job.worker_id == "w1"

    # Make worker 1 unhealthy (heartbeat fails)
    stub1 = stub_factory.get_stub("host1:8080")
    stub1.healthy = False

    # Run heartbeats 3 times to trigger failure threshold
    for _ in range(3):
        controller._run_heartbeats()

    # Worker should be marked unhealthy, job should be pending for retry
    worker1 = controller.state.get_worker("w1")
    assert worker1.healthy is False
    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.preemption_count == 1

    # Run scheduling again - job goes to w2
    controller._run_scheduling()

    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_RUNNING
    assert job.worker_id == "w2"

    # Worker 2 reports success
    stub2 = stub_factory.get_stub("host2:8080")
    stub2.set_job_completed(job_id, cluster_pb2.JOB_STATE_SUCCEEDED)
    controller._run_heartbeats()

    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert job.preemption_count == 1


def test_multiple_jobs_scheduled(make_job_request, controller_with_workers):
    """Submit multiple jobs, all complete successfully."""
    controller, stub_factory = controller_with_workers(
        ["w1"],
        resources=cluster_pb2.ResourceSpec(cpu=10, memory="8g"),
    )

    # Submit multiple jobs
    job_ids = []
    for i in range(5):
        response = controller.launch_job(make_job_request(f"job-{i}"))
        job_ids.append(response.job_id)

    # Run scheduling - all jobs should be assigned
    controller._run_scheduling()

    # Verify all jobs are running
    for job_id in job_ids:
        assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_RUNNING

    # Configure mock to report all jobs succeeded
    stub = stub_factory.get_stub("host1:8080")
    for job_id in job_ids:
        stub.set_job_completed(job_id, cluster_pb2.JOB_STATE_SUCCEEDED)

    controller._run_heartbeats()

    # Verify all jobs succeeded
    for job_id in job_ids:
        assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_terminated_during_execution(make_job_request, controller_with_workers):
    """Terminate a running job mid-execution."""
    controller, _ = controller_with_workers(["w1"])

    # Submit job
    response = controller.launch_job(make_job_request("test-job"))
    job_id = response.job_id

    # Run scheduling
    controller._run_scheduling()
    assert controller.get_job_status(job_id).job.state == cluster_pb2.JOB_STATE_RUNNING

    # Terminate the job
    controller.terminate_job(job_id)

    # Verify job is killed
    status = controller.get_job_status(job_id)
    assert status.job.state == cluster_pb2.JOB_STATE_KILLED


def test_concurrent_job_execution_on_multiple_workers(make_job_request, controller_with_workers):
    """Multiple workers can run jobs concurrently."""
    controller, stub_factory = controller_with_workers(
        ["w1", "w2", "w3"],
        resources=cluster_pb2.ResourceSpec(cpu=2, memory="8g"),
    )

    # Submit 6 jobs, each needing 2 CPUs
    job_ids = []
    for i in range(6):
        response = controller.launch_job(make_job_request(f"job-{i}", cpu=2))
        job_ids.append(response.job_id)

    # Run scheduling - each worker gets 1 job (2 CPU each)
    controller._run_scheduling()

    # Count running jobs - should be 3 (one per worker)
    running = [jid for jid in job_ids if controller.get_job_status(jid).job.state == cluster_pb2.JOB_STATE_RUNNING]
    assert len(running) == 3

    # Complete all running jobs
    for worker_id in ["w1", "w2", "w3"]:
        stub = stub_factory.get_stub(f"host{worker_id[1:]}:8080")
        worker = controller.state.get_worker(worker_id)
        for job_id in list(worker.running_jobs):
            stub.set_job_completed(str(job_id), cluster_pb2.JOB_STATE_SUCCEEDED)

    controller._run_heartbeats()

    # Run scheduling again to pick up remaining jobs
    controller._run_scheduling()

    # Remaining 3 jobs should now be running
    running = [jid for jid in job_ids if controller.get_job_status(jid).job.state == cluster_pb2.JOB_STATE_RUNNING]
    assert len(running) == 3


def test_scheduler_respects_resource_limits(make_job_request, controller_with_workers):
    """Scheduler doesn't over-commit worker resources."""
    controller, _ = controller_with_workers(
        ["w1"],
        resources=cluster_pb2.ResourceSpec(cpu=4, memory="8g"),
    )

    # Submit 3 jobs needing 2 CPUs each (total 6 CPUs, but only 4 available)
    job_ids = []
    for i in range(3):
        response = controller.launch_job(make_job_request(f"job-{i}", cpu=2))
        job_ids.append(response.job_id)

    # Run scheduling
    controller._run_scheduling()

    # Only 2 jobs should be running (4 CPUs / 2 CPUs per job = 2 jobs)
    running = [jid for jid in job_ids if controller.get_job_status(jid).job.state == cluster_pb2.JOB_STATE_RUNNING]
    pending = [jid for jid in job_ids if controller.get_job_status(jid).job.state == cluster_pb2.JOB_STATE_PENDING]

    assert len(running) == 2
    assert len(pending) == 1


def test_ports_forwarded_from_launch_to_run_request(controller_with_workers):
    """Verify port names are forwarded from LaunchJobRequest to RunJobRequest."""
    controller, stub_factory = controller_with_workers(["w1"])

    # Submit job with port requests
    request = cluster_pb2.LaunchJobRequest(
        name="port-test-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        ports=["http", "grpc", "actor"],
    )
    response = controller.launch_job(request)
    job_id = response.job_id

    # Run scheduling
    controller._run_scheduling()

    # Verify job is running
    status = controller.get_job_status(job_id)
    assert status.job.state == cluster_pb2.JOB_STATE_RUNNING

    # Verify the RunJobRequest sent to worker includes ports
    stub = stub_factory.get_stub("host1:8080")
    assert len(stub.run_job_calls) == 1

    run_request = stub.run_job_calls[0]
    assert list(run_request.ports) == ["http", "grpc", "actor"]
