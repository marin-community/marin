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

"""Tests for terminate_prefix."""

import pytest

from fluster.client import FlusterClient, LocalClientConfig
from fluster.cluster.types import Entrypoint
from fluster.rpc import cluster_pb2


def dummy_job():
    pass


@pytest.fixture
def local_client():
    config = LocalClientConfig(max_workers=4)
    with FlusterClient.local(config) as client:
        yield client


@pytest.fixture
def resources():
    return cluster_pb2.ResourceSpec(cpu=1, memory="1g")


def test_terminate_prefix_basic(local_client, resources):
    """Verify terminate_prefix terminates matching jobs."""
    entrypoint = Entrypoint.from_callable(dummy_job)

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


def test_terminate_prefix_excludes_finished(local_client, resources):
    """Verify terminate_prefix skips finished jobs by default."""
    entrypoint = Entrypoint.from_callable(dummy_job)

    job_id = local_client.submit(entrypoint, "finished-test", resources)
    local_client.wait(job_id)  # Wait for completion

    # Job should be SUCCEEDED now
    status = local_client.status(job_id)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # terminate_prefix should not include it
    terminated = local_client.terminate_prefix("finished-test")
    assert job_id not in terminated
