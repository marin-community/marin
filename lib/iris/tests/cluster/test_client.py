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

"""Tests for cluster client hierarchical name handling."""

import pytest
from connectrpc.errors import ConnectError

from iris.client import IrisClient, LocalClientConfig
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import cluster_pb2


def dummy_entrypoint():
    """A simple entrypoint for testing."""
    pass


@pytest.fixture
def local_client():
    """Create a IrisClient for testing."""
    config = LocalClientConfig(max_workers=2)
    with IrisClient.local(config) as client:
        yield client


def test_submit_rejects_name_with_slash(local_client):
    """Verify submit raises ValueError for names containing '/'."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    with pytest.raises(ValueError) as exc_info:
        local_client.submit(entrypoint, "invalid/name", resources)

    assert "/" in str(exc_info.value)


def test_submit_rejects_duplicate_name(local_client):
    """Verify submit rejects duplicate job names."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    # First submit should succeed
    job = local_client.submit(entrypoint, "duplicate-job", resources)
    assert job.job_id == "duplicate-job"

    # Second submit with same name should fail with RPC conflict error
    with pytest.raises(ConnectError) as exc_info:
        local_client.submit(entrypoint, "duplicate-job", resources)

    assert "already exists" in str(exc_info.value)


def test_list_jobs_returns_all_jobs(local_client):
    """Verify list_jobs returns all submitted jobs."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job1 = local_client.submit(entrypoint, "list-job-1", resources)
    job2 = local_client.submit(entrypoint, "list-job-2", resources)

    jobs = local_client.list_jobs()
    job_ids = {j.job_id for j in jobs}

    assert job1.job_id in job_ids
    assert job2.job_id in job_ids


def test_list_jobs_filter_by_state(local_client):
    """Verify list_jobs can filter by state."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job = local_client.submit(entrypoint, "state-filter-job", resources)
    job.wait()  # Wait for completion

    # Filter for SUCCEEDED only
    succeeded_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_SUCCEEDED])
    assert any(j.job_id == job.job_id for j in succeeded_jobs)

    # Filter for PENDING only - should not include completed job
    pending_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_PENDING])
    assert not any(j.job_id == job.job_id for j in pending_jobs)


def test_list_jobs_filter_by_prefix(local_client):
    """Verify list_jobs can filter by job_id prefix."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    local_client.submit(entrypoint, "exp-a-job", resources)
    local_client.submit(entrypoint, "exp-b-job", resources)
    local_client.submit(entrypoint, "other-job", resources)

    # Filter by prefix
    jobs = local_client.list_jobs(prefix="exp-")
    job_ids = {j.job_id for j in jobs}

    assert "exp-a-job" in job_ids
    assert "exp-b-job" in job_ids
    assert "other-job" not in job_ids


def test_terminate_prefix_basic(local_client):
    """Verify terminate_prefix terminates matching jobs."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    # Submit jobs with different prefixes
    local_client.submit(entrypoint, "exp-a-job1", resources)
    local_client.submit(entrypoint, "exp-a-job2", resources)
    local_client.submit(entrypoint, "exp-b-job1", resources)

    # Terminate exp-a jobs
    terminated = local_client.terminate_prefix("exp-a")

    assert len(terminated) == 2
    assert "exp-a-job1" in terminated
    assert "exp-a-job2" in terminated
    assert "exp-b-job1" not in terminated


def test_terminate_prefix_excludes_finished(local_client):
    """Verify terminate_prefix skips finished jobs by default."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job = local_client.submit(entrypoint, "finished-test", resources)
    job.wait()  # Wait for completion

    # Job should be SUCCEEDED now
    status = job.status()
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # terminate_prefix should not include it
    terminated = local_client.terminate_prefix("finished-test")
    assert job.job_id not in terminated


# =============================================================================
# Task API Tests (using new Job/Task objects)
# =============================================================================


def test_job_tasks_returns_single_task_for_job(local_client):
    """Verify job.tasks() returns a task for a single-task job."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job = local_client.submit(entrypoint, "task-test-job", resources)
    job.wait()

    tasks = job.tasks()
    assert len(tasks) == 1
    assert tasks[0].job_id == job.job_id
    assert tasks[0].task_index == 0
    assert tasks[0].task_id == f"{job.job_id}/task-0"


def test_task_status_returns_task_info(local_client):
    """Verify task.status() returns task-level information."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job = local_client.submit(entrypoint, "task-status-job", resources)
    job.wait()

    tasks = job.tasks()
    assert len(tasks) == 1
    task = tasks[0]
    status = task.status()
    assert status.task_id == f"{job.job_id}/task-0"
    assert status.job_id == job.job_id
    assert status.task_index == 0
    assert status.state == cluster_pb2.TASK_STATE_SUCCEEDED


def test_task_logs_returns_logs(local_client):
    """Verify task.logs() returns logs when available (integration test)."""

    def logging_entrypoint():
        print("Hello from task")

    entrypoint = Entrypoint.from_callable(logging_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job = local_client.submit(entrypoint, "task-logs-job", resources)
    job.wait()

    # The method should be callable without error
    # In local mode, stdout capture may or may not produce logs depending on timing
    tasks = job.tasks()
    logs = tasks[0].logs()
    # Just verify we get a list back (logs may be empty in fast local execution)
    assert isinstance(logs, list)
