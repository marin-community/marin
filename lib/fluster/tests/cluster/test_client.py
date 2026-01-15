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

from fluster.cluster.client import LocalClient, LocalClientConfig
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
    """Create a LocalClient for testing."""
    config = LocalClientConfig(max_workers=2)
    client = LocalClient(config)
    client.__enter__()
    yield client
    client.__exit__(None, None, None)


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

    # Second submit with same name should fail
    with pytest.raises(ValueError) as exc_info:
        local_client.submit(entrypoint, "duplicate-job", resources)

    assert "already exists" in str(exc_info.value)
