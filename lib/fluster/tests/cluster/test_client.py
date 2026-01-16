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

from fluster.client import FlusterClient, LocalClientConfig
from fluster.cluster.types import Entrypoint
from fluster.rpc import cluster_pb2


def dummy_entrypoint():
    """A simple entrypoint for testing."""
    pass


@pytest.fixture
def resources():
    """Create minimal ResourceSpec for testing."""
    return cluster_pb2.ResourceSpec(cpu=1, memory="1g")


@pytest.fixture
def local_client():
    """Create a FlusterClient for testing."""
    config = LocalClientConfig(max_workers=2)
    with FlusterClient.local(config) as client:
        yield client


def test_submit_rejects_name_with_slash(local_client, resources):
    """Verify submit raises ValueError for names containing '/'."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)

    with pytest.raises(ValueError) as exc_info:
        local_client.submit(entrypoint, "invalid/name", resources)

    assert "/" in str(exc_info.value)


def test_submit_root_job_uses_name_directly(local_client, resources):
    """Verify root job (no parent) uses name directly as job_id."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)

    job_id = local_client.submit(entrypoint, "my-root-job", resources)

    assert job_id == "my-root-job"


def test_submit_rejects_duplicate_name(local_client, resources):
    """Verify submit rejects duplicate job names."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)

    # First submit should succeed
    job_id = local_client.submit(entrypoint, "duplicate-job", resources)
    assert job_id == "duplicate-job"

    # Second submit with same name should fail with RPC conflict error
    with pytest.raises(ConnectError) as exc_info:
        local_client.submit(entrypoint, "duplicate-job", resources)

    assert "already exists" in str(exc_info.value)


def test_list_jobs_returns_all_jobs(local_client, resources):
    """Verify list_jobs returns all submitted jobs."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)

    job1 = local_client.submit(entrypoint, "list-job-1", resources)
    job2 = local_client.submit(entrypoint, "list-job-2", resources)

    jobs = local_client.list_jobs()
    job_ids = {j.job_id for j in jobs}

    assert job1 in job_ids
    assert job2 in job_ids


def test_list_jobs_filter_by_state(local_client, resources):
    """Verify list_jobs can filter by state."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)

    job_id = local_client.submit(entrypoint, "state-filter-job", resources)
    local_client.wait(job_id)  # Wait for completion

    # Filter for SUCCEEDED only
    succeeded_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_SUCCEEDED])
    assert any(j.job_id == job_id for j in succeeded_jobs)

    # Filter for PENDING only - should not include completed job
    pending_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_PENDING])
    assert not any(j.job_id == job_id for j in pending_jobs)


def test_list_jobs_filter_by_prefix(local_client, resources):
    """Verify list_jobs can filter by job_id prefix."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)

    local_client.submit(entrypoint, "exp-a-job", resources)
    local_client.submit(entrypoint, "exp-b-job", resources)
    local_client.submit(entrypoint, "other-job", resources)

    # Filter by prefix
    jobs = local_client.list_jobs(prefix="exp-")
    job_ids = {j.job_id for j in jobs}

    assert "exp-a-job" in job_ids
    assert "exp-b-job" in job_ids
    assert "other-job" not in job_ids
