# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster client hierarchical name handling."""

import pytest
from connectrpc.errors import ConnectError

from iris.client import IrisClient, LocalClientConfig
from iris.cluster.types import Entrypoint, JobName, ResourceSpec
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
    assert job.job_id == JobName.root("duplicate-job")

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

    assert job1.job_id.to_wire() in job_ids
    assert job2.job_id.to_wire() in job_ids


def test_list_jobs_filter_by_state(local_client):
    """Verify list_jobs can filter by state."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    job = local_client.submit(entrypoint, "state-filter-job", resources)
    job.wait()  # Wait for completion

    # Filter for SUCCEEDED only
    succeeded_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_SUCCEEDED])
    assert any(j.job_id == job.job_id.to_wire() for j in succeeded_jobs)

    # Filter for PENDING only - should not include completed job
    pending_jobs = local_client.list_jobs(states=[cluster_pb2.JOB_STATE_PENDING])
    assert not any(j.job_id == job.job_id.to_wire() for j in pending_jobs)


def test_list_jobs_filter_by_prefix(local_client):
    """Verify list_jobs can filter by job_id prefix."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    local_client.submit(entrypoint, "exp-a-job", resources)
    local_client.submit(entrypoint, "exp-b-job", resources)
    local_client.submit(entrypoint, "other-job", resources)

    # Filter by prefix
    jobs = local_client.list_jobs(prefix=JobName.root("exp-"))
    job_ids = {j.job_id for j in jobs}

    assert JobName.root("exp-a-job").to_wire() in job_ids
    assert JobName.root("exp-b-job").to_wire() in job_ids
    assert JobName.root("other-job").to_wire() not in job_ids


def test_terminate_prefix_basic(local_client):
    """Verify terminate_prefix terminates matching jobs."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    # Submit jobs with different prefixes
    local_client.submit(entrypoint, "exp-a-job1", resources)
    local_client.submit(entrypoint, "exp-a-job2", resources)
    local_client.submit(entrypoint, "exp-b-job1", resources)

    # Terminate exp-a jobs
    terminated = local_client.terminate_prefix(JobName.root("exp-a"))

    assert len(terminated) == 2
    assert JobName.root("exp-a-job1") in terminated
    assert JobName.root("exp-a-job2") in terminated
    assert JobName.root("exp-b-job1") not in terminated


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
    terminated = local_client.terminate_prefix(JobName.root("finished-test"))
    assert job.job_id not in terminated
