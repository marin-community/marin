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
from fluster.cluster.controller.workers import (
    WorkerConfig,
    find_worker_for_job,
    get_committed_resources,
    load_workers_from_config,
    worker_can_fit_job,
)
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


# =============================================================================
# Resource Matching Tests
# =============================================================================


def test_worker_can_fit_job_cpu_constraint():
    """Job requiring more CPU than available should not fit."""
    state = ControllerState()

    # Worker with 4 CPUs total, running a job using 2 CPUs
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(cpu=4, memory="32g"),
    )
    state.add_worker(worker)

    # Job already running on worker (uses 2 CPUs)
    running_job = ControllerJob(
        job_id=JobId("running"),
        request=cluster_pb2.LaunchJobRequest(
            name="running",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=2, memory="1g"),
        ),
    )
    state._jobs[running_job.job_id] = running_job
    worker.running_jobs.add(running_job.job_id)

    # New job requiring 4 CPUs (only 2 available)
    new_job = ControllerJob(
        job_id=JobId("new"),
        request=cluster_pb2.LaunchJobRequest(
            name="new",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=4, memory="1g"),
        ),
    )

    assert not worker_can_fit_job(state, worker, new_job)


def test_worker_can_fit_job_memory_constraint():
    """Job requiring more memory than available should not fit."""
    state = ControllerState()

    # Worker with 16g memory, 12g already committed
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(cpu=8, memory="16g"),
    )
    state.add_worker(worker)

    # Running job uses 12g
    running_job = ControllerJob(
        job_id=JobId("running"),
        request=cluster_pb2.LaunchJobRequest(
            name="running",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="12g"),
        ),
    )
    state._jobs[running_job.job_id] = running_job
    worker.running_jobs.add(running_job.job_id)

    # New job requiring 8g (only 4g available)
    new_job = ControllerJob(
        job_id=JobId("new"),
        request=cluster_pb2.LaunchJobRequest(
            name="new",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="8g"),
        ),
    )

    assert not worker_can_fit_job(state, worker, new_job)


def test_worker_can_fit_job_device_type_mismatch():
    """GPU job should not fit on CPU-only worker."""
    state = ControllerState()

    # CPU-only worker
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(
            cpu=8,
            memory="32g",
            device=cluster_pb2.DeviceConfig(cpu=cluster_pb2.CpuDevice()),
        ),
    )
    state.add_worker(worker)

    # GPU job
    gpu_job = ControllerJob(
        job_id=JobId("gpu-job"),
        request=cluster_pb2.LaunchJobRequest(
            name="gpu-job",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(
                cpu=1,
                memory="8g",
                device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=1)),
            ),
        ),
    )

    assert not worker_can_fit_job(state, worker, gpu_job)


def test_worker_can_fit_job_gpu_variant_match():
    """Job specifying GPU variant should match worker with same variant."""
    state = ControllerState()

    # H100 GPU worker
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(
            cpu=32,
            memory="256g",
            device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="H100", count=8)),
        ),
    )
    state.add_worker(worker)

    # Job requiring H100
    h100_job = ControllerJob(
        job_id=JobId("h100-job"),
        request=cluster_pb2.LaunchJobRequest(
            name="h100-job",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(
                cpu=4,
                memory="32g",
                device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="H100", count=2)),
            ),
        ),
    )

    assert worker_can_fit_job(state, worker, h100_job)


def test_worker_can_fit_job_gpu_variant_mismatch():
    """Job specifying specific variant should not match different variant."""
    state = ControllerState()

    # A100 GPU worker
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(
            cpu=32,
            memory="256g",
            device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=8)),
        ),
    )
    state.add_worker(worker)

    # Job requiring H100
    h100_job = ControllerJob(
        job_id=JobId("h100-job"),
        request=cluster_pb2.LaunchJobRequest(
            name="h100-job",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(
                cpu=4,
                memory="32g",
                device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="H100", count=2)),
            ),
        ),
    )

    assert not worker_can_fit_job(state, worker, h100_job)


def test_worker_can_fit_job_gpu_variant_auto():
    """Job with variant='auto' should match any GPU worker."""
    state = ControllerState()

    # A100 GPU worker
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(
            cpu=32,
            memory="256g",
            device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=8)),
        ),
    )
    state.add_worker(worker)

    # Job with auto variant
    auto_job = ControllerJob(
        job_id=JobId("auto-job"),
        request=cluster_pb2.LaunchJobRequest(
            name="auto-job",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(
                cpu=4,
                memory="32g",
                device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="auto", count=1)),
            ),
        ),
    )

    assert worker_can_fit_job(state, worker, auto_job)


def test_worker_can_fit_job_gpu_count_constraint():
    """Job requiring more GPUs than available should not fit."""
    state = ControllerState()

    # Worker with 8 GPUs, 6 already in use
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(
            cpu=32,
            memory="256g",
            device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=8)),
        ),
    )
    state.add_worker(worker)

    # Running job uses 6 GPUs
    running_job = ControllerJob(
        job_id=JobId("running"),
        request=cluster_pb2.LaunchJobRequest(
            name="running",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(
                cpu=4,
                memory="32g",
                device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=6)),
            ),
        ),
    )
    state._jobs[running_job.job_id] = running_job
    worker.running_jobs.add(running_job.job_id)

    # New job requiring 4 GPUs (only 2 available)
    new_job = ControllerJob(
        job_id=JobId("new"),
        request=cluster_pb2.LaunchJobRequest(
            name="new",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(
                cpu=4,
                memory="32g",
                device=cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=4)),
            ),
        ),
    )

    assert not worker_can_fit_job(state, worker, new_job)


def test_get_committed_resources():
    """Verify committed resources are computed from running jobs."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr",
        resources=cluster_pb2.ResourceSpec(cpu=16, memory="64g"),
    )
    state.add_worker(worker)

    # Add two running jobs
    job1 = ControllerJob(
        job_id=JobId("j1"),
        request=cluster_pb2.LaunchJobRequest(
            name="j1",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=2, memory="8g"),
        ),
    )
    job2 = ControllerJob(
        job_id=JobId("j2"),
        request=cluster_pb2.LaunchJobRequest(
            name="j2",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=4, memory="16g"),
        ),
    )
    state._jobs[job1.job_id] = job1
    state._jobs[job2.job_id] = job2
    worker.running_jobs.add(job1.job_id)
    worker.running_jobs.add(job2.job_id)

    cpu, memory, _gpu = get_committed_resources(state, worker)
    assert cpu == 6  # 2 + 4
    assert memory == 24 * 1024**3  # 8g + 16g


def test_find_worker_for_job_respects_capacity():
    """Verify find_worker_for_job skips workers without capacity."""
    state = ControllerState()

    # Worker 1: only 2 CPUs total
    worker1 = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="addr1",
        resources=cluster_pb2.ResourceSpec(cpu=2, memory="16g"),
    )
    state.add_worker(worker1)

    # Worker 2: has 8 CPUs
    worker2 = ControllerWorker(
        worker_id=WorkerId("w2"),
        address="addr2",
        resources=cluster_pb2.ResourceSpec(cpu=8, memory="32g"),
    )
    state.add_worker(worker2)

    # Job requiring 4 CPUs
    job = ControllerJob(
        job_id=JobId("j1"),
        request=cluster_pb2.LaunchJobRequest(
            name="j1",
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=4, memory="1g"),
        ),
    )

    result = find_worker_for_job(state, job)
    assert result is not None
    assert result.worker_id == "w2"  # Should skip w1 (only 2 CPUs)
