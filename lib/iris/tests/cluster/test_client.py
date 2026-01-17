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
    job_id = local_client.submit(entrypoint, "duplicate-job", resources)
    assert job_id == "duplicate-job"

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

    assert job1 in job_ids
    assert job2 in job_ids


def test_list_jobs_filter_by_state(local_client):
    """Verify list_jobs can filter by state."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job_id = local_client.submit(entrypoint, "state-filter-job", resources)
    local_client.wait(job_id)  # Wait for completion

    # Filter for SUCCEEDED only
    succeeded_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_SUCCEEDED])
    assert any(j.job_id == job_id for j in succeeded_jobs)

    # Filter for PENDING only - should not include completed job
    pending_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_PENDING])
    assert not any(j.job_id == job_id for j in pending_jobs)


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

    job_id = local_client.submit(entrypoint, "finished-test", resources)
    local_client.wait(job_id)  # Wait for completion

    # Job should be SUCCEEDED now
    status = local_client.status(job_id)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # terminate_prefix should not include it
    terminated = local_client.terminate_prefix("finished-test")
    assert job_id not in terminated
