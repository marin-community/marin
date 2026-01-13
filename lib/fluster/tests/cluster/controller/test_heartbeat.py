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

"""Tests for worker heartbeat monitor."""

import time

import pytest

from fluster import cluster_pb2
from fluster.cluster.controller.heartbeat import HeartbeatMonitor
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


def test_heartbeat_success_resets_failures(make_resource_spec):
    """Verify successful heartbeat resets consecutive_failures counter."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    worker.consecutive_failures = 2  # Pre-set some failures
    state.add_worker(worker)

    def mock_heartbeat(address):
        return cluster_pb2.HeartbeatResponse(jobs=[], timestamp_ms=int(time.time() * 1000))

    callbacks = []

    def on_failed(worker_id, jobs):
        callbacks.append((worker_id, jobs))

    monitor = HeartbeatMonitor(state, mock_heartbeat, on_failed, interval_seconds=0.1)

    monitor.start()
    time.sleep(0.2)
    monitor.stop()

    # Consecutive failures should be reset to 0
    assert worker.consecutive_failures == 0
    assert worker.healthy is True
    assert worker.last_heartbeat_ms > 0
    # Callback should not have been called
    assert callbacks == []


def test_heartbeat_failure_increments_count(make_resource_spec):
    """Verify failed heartbeat increments consecutive_failures counter."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)

    # Heartbeat always fails
    def mock_heartbeat_fail(address):
        return None

    callbacks = []

    def on_failed(worker_id, jobs):
        callbacks.append((worker_id, jobs))

    monitor = HeartbeatMonitor(state, mock_heartbeat_fail, on_failed, interval_seconds=0.1)

    assert worker.consecutive_failures == 0

    monitor.start()
    time.sleep(0.15)  # Wait for one check
    monitor.stop()

    # Should have incremented but not yet reached threshold
    assert worker.consecutive_failures >= 1
    assert worker.healthy is True  # Still healthy, haven't hit threshold yet
    assert callbacks == []  # No callback yet


def test_heartbeat_marks_worker_failed_after_n_failures(make_job_request, make_resource_spec):
    """Verify worker is marked unhealthy after MAX_CONSECUTIVE_FAILURES."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    job = ControllerJob(JobId("j1"), request=make_job_request())
    job.state = cluster_pb2.JOB_STATE_RUNNING
    job.worker_id = worker.worker_id
    worker.running_jobs.add(job.job_id)

    state.add_worker(worker)
    state.add_job(job)

    failed_workers = []

    def on_failed(wid, jobs):
        failed_workers.append((wid, jobs))

    # Heartbeat always fails
    monitor = HeartbeatMonitor(
        state,
        heartbeat_fn=lambda addr: None,
        on_worker_failed=on_failed,
        interval_seconds=0.05,
    )

    monitor.start()
    time.sleep(0.3)  # Wait for 3+ failures (3 * 0.05 = 0.15s + buffer)
    monitor.stop()

    # Worker should be marked unhealthy
    assert not worker.healthy
    assert worker.consecutive_failures >= HeartbeatMonitor.MAX_CONSECUTIVE_FAILURES
    # Callback should have been called once with the failed job
    assert len(failed_workers) == 1
    assert failed_workers[0] == ("w1", ["j1"])


def test_heartbeat_marks_jobs_as_worker_failed(make_job_request, make_resource_spec):
    """Verify jobs on failed worker get JOB_STATE_WORKER_FAILED status."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())

    # Create multiple jobs running on worker
    job1 = ControllerJob(JobId("j1"), request=make_job_request("job1"))
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job1.worker_id = worker.worker_id

    job2 = ControllerJob(JobId("j2"), request=make_job_request("job2"))
    job2.state = cluster_pb2.JOB_STATE_RUNNING
    job2.worker_id = worker.worker_id

    worker.running_jobs.add(job1.job_id)
    worker.running_jobs.add(job2.job_id)

    state.add_worker(worker)
    state.add_job(job1)
    state.add_job(job2)

    def on_failed(wid, jobs):
        pass

    # Heartbeat always fails
    monitor = HeartbeatMonitor(
        state,
        heartbeat_fn=lambda addr: None,
        on_worker_failed=on_failed,
        interval_seconds=0.05,
    )

    monitor.start()
    time.sleep(0.3)  # Wait for failures
    monitor.stop()

    # Both jobs should be marked as WORKER_FAILED
    assert job1.state == cluster_pb2.JOB_STATE_WORKER_FAILED
    assert job2.state == cluster_pb2.JOB_STATE_WORKER_FAILED

    # Jobs should have error messages
    assert job1.error == "Worker w1 failed"
    assert job2.error == "Worker w1 failed"

    # Jobs should have finished_at_ms set
    assert job1.finished_at_ms is not None
    assert job1.finished_at_ms > 0
    assert job2.finished_at_ms is not None
    assert job2.finished_at_ms > 0

    # Worker's running_jobs should be cleared
    assert len(worker.running_jobs) == 0


def test_heartbeat_calls_on_worker_failed_callback(make_job_request, make_resource_spec):
    """Verify on_worker_failed callback is called with correct arguments."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())

    job1 = ControllerJob(JobId("j1"), request=make_job_request())
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job1.worker_id = worker.worker_id

    job2 = ControllerJob(JobId("j2"), request=make_job_request())
    job2.state = cluster_pb2.JOB_STATE_RUNNING
    job2.worker_id = worker.worker_id

    worker.running_jobs.add(job1.job_id)
    worker.running_jobs.add(job2.job_id)

    state.add_worker(worker)
    state.add_job(job1)
    state.add_job(job2)

    callback_calls = []

    def on_failed(worker_id, job_ids):
        callback_calls.append((worker_id, set(job_ids)))

    monitor = HeartbeatMonitor(
        state,
        heartbeat_fn=lambda addr: None,
        on_worker_failed=on_failed,
        interval_seconds=0.05,
    )

    monitor.start()
    time.sleep(0.3)
    monitor.stop()

    # Callback should be called once
    assert len(callback_calls) == 1

    # Verify callback arguments
    worker_id, job_ids = callback_calls[0]
    assert worker_id == "w1"
    assert job_ids == {"j1", "j2"}


def test_heartbeat_skips_unhealthy_workers(make_resource_spec):
    """Verify monitor doesn't poll workers already marked unhealthy."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    worker.healthy = False  # Pre-mark as unhealthy
    state.add_worker(worker)

    heartbeat_calls = []

    def mock_heartbeat(address):
        heartbeat_calls.append(address)
        return cluster_pb2.HeartbeatResponse(jobs=[], timestamp_ms=int(time.time() * 1000))

    def on_failed(wid, jobs):
        pass

    monitor = HeartbeatMonitor(state, mock_heartbeat, on_failed, interval_seconds=0.1)

    monitor.start()
    time.sleep(0.2)
    monitor.stop()

    # Heartbeat function should not have been called for unhealthy worker
    assert len(heartbeat_calls) == 0


def test_heartbeat_syncs_job_states_from_response(make_job_request, make_resource_spec):
    """Verify job states are updated from heartbeat response."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())

    # Create jobs in RUNNING state
    job1 = ControllerJob(JobId("j1"), request=make_job_request())
    job1.state = cluster_pb2.JOB_STATE_RUNNING
    job1.worker_id = worker.worker_id

    job2 = ControllerJob(JobId("j2"), request=make_job_request())
    job2.state = cluster_pb2.JOB_STATE_RUNNING
    job2.worker_id = worker.worker_id

    worker.running_jobs.add(job1.job_id)
    worker.running_jobs.add(job2.job_id)

    state.add_worker(worker)
    state.add_job(job1)
    state.add_job(job2)

    # Heartbeat reports j1 succeeded, j2 failed
    def mock_heartbeat(address):
        return cluster_pb2.HeartbeatResponse(
            jobs=[
                cluster_pb2.JobStatus(
                    job_id="j1",
                    state=cluster_pb2.JOB_STATE_SUCCEEDED,
                    exit_code=0,
                    finished_at_ms=123456,
                ),
                cluster_pb2.JobStatus(
                    job_id="j2",
                    state=cluster_pb2.JOB_STATE_FAILED,
                    exit_code=1,
                    error="Job crashed",
                    finished_at_ms=123457,
                ),
            ],
            timestamp_ms=int(time.time() * 1000),
        )

    def on_failed(wid, jobs):
        pass

    monitor = HeartbeatMonitor(state, mock_heartbeat, on_failed, interval_seconds=0.1)

    monitor.start()
    time.sleep(0.2)
    monitor.stop()

    # Job states should be updated
    assert job1.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert job1.exit_code == 0
    assert job1.finished_at_ms == 123456

    assert job2.state == cluster_pb2.JOB_STATE_FAILED
    assert job2.exit_code == 1
    assert job2.error == "Job crashed"
    assert job2.finished_at_ms == 123457

    # Jobs should be removed from worker's running_jobs
    assert len(worker.running_jobs) == 0


def test_heartbeat_monitor_stops_cleanly(make_resource_spec):
    """Verify stop() returns without hanging."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    state.add_worker(worker)

    def mock_heartbeat(address):
        return cluster_pb2.HeartbeatResponse(jobs=[], timestamp_ms=int(time.time() * 1000))

    def on_failed(wid, jobs):
        pass

    monitor = HeartbeatMonitor(state, mock_heartbeat, on_failed, interval_seconds=0.1)

    monitor.start()
    time.sleep(0.2)

    # Stop should return quickly
    start_time = time.time()
    monitor.stop()
    stop_time = time.time()

    # Should complete in well under 5 seconds
    assert stop_time - start_time < 2.0


def test_heartbeat_updates_last_heartbeat_timestamp(make_resource_spec):
    """Verify last_heartbeat_ms is updated on successful heartbeat."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())
    worker.last_heartbeat_ms = 0  # Start at 0
    state.add_worker(worker)

    def mock_heartbeat(address):
        return cluster_pb2.HeartbeatResponse(jobs=[], timestamp_ms=int(time.time() * 1000))

    def on_failed(wid, jobs):
        pass

    monitor = HeartbeatMonitor(state, mock_heartbeat, on_failed, interval_seconds=0.1)

    monitor.start()
    time.sleep(0.15)
    monitor.stop()

    # Timestamp should be updated
    assert worker.last_heartbeat_ms > 0


def test_heartbeat_only_syncs_terminal_states(make_job_request, make_resource_spec):
    """Verify monitor only updates jobs that are in terminal states."""
    state = ControllerState()
    worker = ControllerWorker(WorkerId("w1"), "addr", make_resource_spec())

    job = ControllerJob(JobId("j1"), request=make_job_request())
    job.state = cluster_pb2.JOB_STATE_RUNNING
    job.worker_id = worker.worker_id
    worker.running_jobs.add(job.job_id)

    state.add_worker(worker)
    state.add_job(job)

    # Heartbeat reports job still running (not terminal)
    def mock_heartbeat(address):
        return cluster_pb2.HeartbeatResponse(
            jobs=[
                cluster_pb2.JobStatus(
                    job_id="j1",
                    state=cluster_pb2.JOB_STATE_RUNNING,  # Still running, not terminal
                ),
            ],
            timestamp_ms=int(time.time() * 1000),
        )

    def on_failed(wid, jobs):
        pass

    monitor = HeartbeatMonitor(state, mock_heartbeat, on_failed, interval_seconds=0.1)

    monitor.start()
    time.sleep(0.2)
    monitor.stop()

    # Job should still be in worker's running_jobs (not removed)
    assert job.job_id in worker.running_jobs
    # Job state should still be RUNNING
    assert job.state == cluster_pb2.JOB_STATE_RUNNING
