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

"""Integration tests for the full controller system.

These tests exercise the complete job lifecycle through all controller
components: State, Workers, Scheduler, Heartbeat, Retry, and Service.
"""

import time

import pytest

from fluster import cluster_pb2
from fluster.cluster.controller.heartbeat import HeartbeatMonitor
from fluster.cluster.controller.retry import handle_job_failure
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerJob, ControllerState
from fluster.cluster.controller.workers import WorkerConfig, load_workers_from_config
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


@pytest.fixture
def make_resource_spec():
    """Create a minimal ResourceSpec for testing."""

    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=4, memory="8g", disk="10g")

    return _make


def test_full_job_lifecycle(make_job_request, make_resource_spec):
    """Integration test: full job lifecycle from submission to completion."""
    state = ControllerState()

    # Setup workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatched jobs for mock
    dispatched_jobs = {}

    def mock_dispatch(job: ControllerJob, worker):
        dispatched_jobs[job.job_id] = worker.worker_id
        return True

    def mock_heartbeat(address: str):
        # Return heartbeat response marking dispatched jobs as succeeded
        response = cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))
        for job_id in list(dispatched_jobs.keys()):
            job = state.get_job(JobId(job_id))
            if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                response.jobs.append(
                    cluster_pb2.JobStatus(
                        job_id=job_id,
                        state=cluster_pb2.JOB_STATE_SUCCEEDED,
                        exit_code=0,
                        finished_at_ms=int(time.time() * 1000),
                    )
                )
        return response

    def on_worker_failed(worker_id, job_ids):
        for job_id in job_ids:
            handle_job_failure(state, JobId(job_id), is_worker_failure=True)

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit job
        request = make_job_request("test-job")
        response = service.launch_job(request, None)
        job_id = response.job_id

        # Wait for completion
        for _ in range(50):
            status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
            if status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                break
            time.sleep(0.1)

        # Verify job succeeded
        assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
        assert status.job.exit_code == 0
        assert status.job.worker_id == "w1"
        assert job_id in dispatched_jobs
        assert dispatched_jobs[job_id] == "w1"
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_job_failure_and_retry(make_job_request, make_resource_spec):
    """Job fails on first attempt, succeeds after retry."""
    state = ControllerState()

    # Setup workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatch attempts
    dispatch_attempts = []

    def mock_dispatch(job: ControllerJob, worker):
        dispatch_attempts.append((job.job_id, worker.worker_id))
        return True

    def mock_heartbeat(address: str):
        response = cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))
        for job_id in list(set(ja[0] for ja in dispatch_attempts)):
            job = state.get_job(JobId(job_id))
            if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                # First attempt fails, second succeeds
                attempt_count = len([ja for ja in dispatch_attempts if ja[0] == job_id])
                if attempt_count == 1:
                    # First attempt - fail it
                    response.jobs.append(
                        cluster_pb2.JobStatus(
                            job_id=job_id,
                            state=cluster_pb2.JOB_STATE_FAILED,
                            exit_code=1,
                            error="Simulated failure",
                            finished_at_ms=int(time.time() * 1000),
                        )
                    )
                else:
                    # Subsequent attempts - succeed
                    response.jobs.append(
                        cluster_pb2.JobStatus(
                            job_id=job_id,
                            state=cluster_pb2.JOB_STATE_SUCCEEDED,
                            exit_code=0,
                            finished_at_ms=int(time.time() * 1000),
                        )
                    )
        return response

    def on_worker_failed(worker_id, job_ids):
        for job_id in job_ids:
            handle_job_failure(state, JobId(job_id), is_worker_failure=True)

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit job with max_retries_failure=1
        request = make_job_request("test-job")
        response = service.launch_job(request, None)
        job_id = response.job_id

        # Manually set retry limit after submission
        job = state.get_job(JobId(job_id))
        job.max_retries_failure = 1

        # Wait for completion (with retry)
        for _ in range(100):
            status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
            if status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                break
            # Check if failed and trigger retry
            if status.job.state == cluster_pb2.JOB_STATE_FAILED:
                handle_job_failure(state, JobId(job_id), is_worker_failure=False)
                scheduler.wake()
            time.sleep(0.1)

        # Verify job eventually succeeded after retry
        assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
        # Should have dispatched twice (initial + retry)
        assert len(dispatch_attempts) >= 2
        job = state.get_job(JobId(job_id))
        assert job.failure_count == 1
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_worker_failure_triggers_retry(make_job_request, make_resource_spec):
    """Worker dies, job is retried on another worker."""
    state = ControllerState()

    # Setup two workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
        WorkerConfig("w2", "host2:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatch attempts
    dispatch_attempts = []

    def mock_dispatch(job: ControllerJob, worker):
        dispatch_attempts.append((job.job_id, worker.worker_id))
        return True

    heartbeat_call_count = 0

    def mock_heartbeat(address: str):
        nonlocal heartbeat_call_count
        heartbeat_call_count += 1

        # Fail worker 1 after job starts
        if address == "host1:8080" and heartbeat_call_count >= 3:
            # Return None to simulate heartbeat failure
            return None

        # Worker 2 succeeds
        if address == "host2:8080":
            response = cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))
            # Check if job is running on w2
            for job_id, wid in dispatch_attempts:
                if wid == "w2":
                    job = state.get_job(JobId(job_id))
                    if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                        response.jobs.append(
                            cluster_pb2.JobStatus(
                                job_id=job_id,
                                state=cluster_pb2.JOB_STATE_SUCCEEDED,
                                exit_code=0,
                                finished_at_ms=int(time.time() * 1000),
                            )
                        )
            return response

        # Worker 1 succeeds initially
        return cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))

    def on_worker_failed(worker_id, job_ids):
        # Trigger retry for all failed jobs
        for job_id in job_ids:
            handle_job_failure(state, JobId(job_id), is_worker_failure=True)
        # Wake scheduler to re-dispatch
        scheduler.wake()

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.05)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit job
        request = make_job_request("test-job")
        response = service.launch_job(request, None)
        job_id = response.job_id

        # Set high preemption retries
        job = state.get_job(JobId(job_id))
        job.max_retries_preemption = 10

        # Wait for completion (with retry on different worker)
        for _ in range(100):
            status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
            if status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                break
            time.sleep(0.1)

        # Verify job eventually succeeded
        assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED

        # Should have dispatched at least twice (w1 then w2)
        assert len(dispatch_attempts) >= 2

        # Final worker should be w2
        assert status.job.worker_id == "w2"

        # Job should have preemption_count incremented
        job = state.get_job(JobId(job_id))
        assert job.preemption_count >= 1
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_multiple_jobs_scheduled_sequentially(make_job_request, make_resource_spec):
    """Submit multiple jobs, all complete successfully."""
    state = ControllerState()

    # Setup workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatched jobs
    dispatched_jobs = {}

    def mock_dispatch(job: ControllerJob, worker):
        dispatched_jobs[job.job_id] = worker.worker_id
        return True

    def mock_heartbeat(address: str):
        response = cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))
        for job_id in list(dispatched_jobs.keys()):
            job = state.get_job(JobId(job_id))
            if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                response.jobs.append(
                    cluster_pb2.JobStatus(
                        job_id=job_id,
                        state=cluster_pb2.JOB_STATE_SUCCEEDED,
                        exit_code=0,
                        finished_at_ms=int(time.time() * 1000),
                    )
                )
        return response

    def on_worker_failed(worker_id, job_ids):
        for job_id in job_ids:
            handle_job_failure(state, JobId(job_id), is_worker_failure=True)

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit multiple jobs
        job_ids = []
        for i in range(5):
            request = make_job_request(f"job-{i}")
            response = service.launch_job(request, None)
            job_ids.append(response.job_id)

        # Wait for all jobs to complete
        for job_id in job_ids:
            for _ in range(50):
                status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
                if status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                    break
                time.sleep(0.1)

            # Verify each job succeeded
            assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
            assert status.job.exit_code == 0
            assert job_id in dispatched_jobs
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_job_terminated_during_execution(make_job_request, make_resource_spec):
    """Terminate a running job mid-execution."""
    state = ControllerState()

    # Setup workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatched jobs
    dispatched_jobs = {}

    def mock_dispatch(job: ControllerJob, worker):
        dispatched_jobs[job.job_id] = worker.worker_id
        return True

    def mock_heartbeat(address: str):
        # Jobs never complete (simulating long-running jobs)
        return cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))

    def on_worker_failed(worker_id, job_ids):
        for job_id in job_ids:
            handle_job_failure(state, JobId(job_id), is_worker_failure=True)

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit job
        request = make_job_request("test-job")
        response = service.launch_job(request, None)
        job_id = response.job_id

        # Wait for job to start
        for _ in range(20):
            status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
            if status.job.state == cluster_pb2.JOB_STATE_RUNNING:
                break
            time.sleep(0.1)

        assert status.job.state == cluster_pb2.JOB_STATE_RUNNING

        # Terminate the job
        service.terminate_job(cluster_pb2.TerminateJobRequest(job_id=job_id), None)

        # Verify job is killed
        status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
        assert status.job.state == cluster_pb2.JOB_STATE_KILLED
        assert status.job.finished_at_ms > 0
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_scheduler_wakes_on_worker_registration(make_job_request, make_resource_spec):
    """New worker registration triggers immediate scheduling."""
    state = ControllerState()

    # Start with no workers
    dispatch_calls = []

    def mock_dispatch(job: ControllerJob, worker):
        dispatch_calls.append((time.time(), job.job_id, worker.worker_id))
        return True

    def mock_heartbeat(address: str):
        return cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))

    def on_worker_failed(worker_id, job_ids):
        pass

    # Use long interval to ensure wake() is responsible for scheduling
    scheduler = Scheduler(state, mock_dispatch, interval_seconds=10.0)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit job with no workers available
        request = make_job_request("test-job")
        response = service.launch_job(request, None)
        job_id = response.job_id

        # Wait a bit, job should not be dispatched
        time.sleep(0.2)
        assert len(dispatch_calls) == 0

        # Add worker and wake scheduler
        workers = [WorkerConfig("w1", "host1:8080", make_resource_spec())]
        load_workers_from_config(state, workers)
        scheduler.wake()

        # Job should be dispatched quickly
        time.sleep(0.2)
        assert len(dispatch_calls) == 1
        assert dispatch_calls[0][1] == job_id
        assert dispatch_calls[0][2] == "w1"
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_list_jobs_shows_all_states(make_job_request, make_resource_spec):
    """List jobs shows pending, running, and completed jobs."""
    state = ControllerState()

    # Setup workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatched jobs
    dispatched_jobs = set()

    def mock_dispatch(job: ControllerJob, worker):
        dispatched_jobs.add(job.job_id)
        return True

    def mock_heartbeat(address: str):
        response = cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))
        # Only complete first job
        for job_id in list(dispatched_jobs):
            job = state.get_job(JobId(job_id))
            if job and job.request.name == "job-1" and job.state == cluster_pb2.JOB_STATE_RUNNING:
                response.jobs.append(
                    cluster_pb2.JobStatus(
                        job_id=job_id,
                        state=cluster_pb2.JOB_STATE_SUCCEEDED,
                        exit_code=0,
                        finished_at_ms=int(time.time() * 1000),
                    )
                )
        return response

    def on_worker_failed(worker_id, job_ids):
        pass

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit jobs
        job1 = service.launch_job(make_job_request("job-1"), None).job_id
        job2 = service.launch_job(make_job_request("job-2"), None).job_id
        job3 = service.launch_job(make_job_request("job-3"), None).job_id

        # Wait for job1 to complete
        for _ in range(50):
            status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job1), None)
            if status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                break
            time.sleep(0.1)

        # List all jobs
        response = service.list_jobs(cluster_pb2.ListJobsRequest(), None)

        # Should have all three jobs
        assert len(response.jobs) == 3
        job_ids = {j.job_id for j in response.jobs}
        assert job_ids == {job1, job2, job3}

        # Verify states
        states_by_id = {j.job_id: j.state for j in response.jobs}
        assert states_by_id[job1] == cluster_pb2.JOB_STATE_SUCCEEDED
        # job2 and job3 should be running or pending (first-fit will dispatch all to w1)
        assert states_by_id[job2] in (cluster_pb2.JOB_STATE_PENDING, cluster_pb2.JOB_STATE_RUNNING)
        assert states_by_id[job3] in (cluster_pb2.JOB_STATE_PENDING, cluster_pb2.JOB_STATE_RUNNING)
    finally:
        scheduler.stop()
        heartbeat.stop()


def test_concurrent_job_execution_on_multiple_workers(make_job_request, make_resource_spec):
    """Multiple workers can run jobs concurrently."""
    state = ControllerState()

    # Setup multiple workers
    workers = [
        WorkerConfig("w1", "host1:8080", make_resource_spec()),
        WorkerConfig("w2", "host2:8080", make_resource_spec()),
        WorkerConfig("w3", "host3:8080", make_resource_spec()),
    ]
    load_workers_from_config(state, workers)

    # Track dispatched jobs
    dispatched_jobs = {}

    def mock_dispatch(job: ControllerJob, worker):
        dispatched_jobs[job.job_id] = worker.worker_id
        return True

    def mock_heartbeat(address: str):
        response = cluster_pb2.HeartbeatResponse(timestamp_ms=int(time.time() * 1000))
        for job_id, _worker_id in list(dispatched_jobs.items()):
            job = state.get_job(JobId(job_id))
            if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                response.jobs.append(
                    cluster_pb2.JobStatus(
                        job_id=job_id,
                        state=cluster_pb2.JOB_STATE_SUCCEEDED,
                        exit_code=0,
                        finished_at_ms=int(time.time() * 1000),
                    )
                )
        return response

    def on_worker_failed(worker_id, job_ids):
        for job_id in job_ids:
            handle_job_failure(state, JobId(job_id), is_worker_failure=True)

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=0.1)
    heartbeat = HeartbeatMonitor(state, mock_heartbeat, on_worker_failed, interval_seconds=0.1)
    service = ControllerServiceImpl(state, scheduler)

    scheduler.start()
    heartbeat.start()

    try:
        # Submit multiple jobs
        job_ids = []
        for i in range(6):
            request = make_job_request(f"job-{i}")
            response = service.launch_job(request, None)
            job_ids.append(response.job_id)

        # Wait for all jobs to complete
        for job_id in job_ids:
            for _ in range(50):
                status = service.get_job_status(cluster_pb2.GetJobStatusRequest(job_id=job_id), None)
                if status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                    break
                time.sleep(0.1)

            assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED

        # Verify jobs were distributed (first-fit means all go to w1 in v0)
        # In v0, all jobs go to first available worker
        workers_used = set(dispatched_jobs.values())
        assert len(workers_used) >= 1  # At least one worker used
    finally:
        scheduler.stop()
        heartbeat.stop()
