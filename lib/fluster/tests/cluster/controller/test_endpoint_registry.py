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

"""Tests for endpoint registry in controller state.

Endpoints now use prefixed names for namespace isolation:
- Registration stores "{namespace}/{actor_name}"
- Lookup uses exact name match on the full prefixed name
- List uses prefix matching on the full prefixed name

This provides automatic namespace isolation without a separate namespace field.
"""

import pytest

from fluster.rpc import cluster_pb2
from fluster.cluster.controller.state import ControllerEndpoint, ControllerJob, ControllerState
from fluster.cluster.types import JobId


@pytest.fixture
def state() -> ControllerState:
    return ControllerState()


def test_add_and_lookup_endpoint(state: ControllerState):
    """Test basic endpoint registration and lookup."""
    # Create a running job first
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Register endpoint with prefixed name
    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",  # Prefixed with namespace
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Lookup by full prefixed name
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 1
    assert results[0].address == "10.0.0.1:8080"
    assert results[0].endpoint_id == "ep-1"


def test_endpoint_not_returned_for_non_running_job(state: ControllerState):
    """Test that endpoints for non-RUNNING jobs are filtered out."""
    # Create a completed job
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Should not return endpoint because job is not running
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0


def test_remove_endpoints_on_job_termination(state: ControllerState):
    """Test that endpoints are removed when a job terminates."""
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Verify endpoint is visible
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 1

    # Simulate job termination
    removed = state.remove_endpoints_for_job(JobId("ns-1"))
    assert len(removed) == 1
    assert removed[0].endpoint_id == "ep-1"

    # Endpoint should be gone
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0


def test_namespace_isolation_via_prefix(state: ControllerState):
    """Test that namespace isolation works via name prefixing."""
    job1 = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    job2 = ControllerJob(
        job_id=JobId("ns-2"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job1)
    state.add_job(job2)

    # Same actor name, different namespace prefixes
    ep1 = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="ns-2/actor",
        address="10.0.0.2:8080",
        job_id=JobId("ns-2"),
    )
    state.add_endpoint(ep1)
    state.add_endpoint(ep2)

    # Each namespace prefix only sees its own endpoint
    results_ns1 = state.lookup_endpoints("ns-1/actor")
    assert len(results_ns1) == 1
    assert results_ns1[0].address == "10.0.0.1:8080"

    results_ns2 = state.lookup_endpoints("ns-2/actor")
    assert len(results_ns2) == 1
    assert results_ns2[0].address == "10.0.0.2:8080"


def test_list_endpoints_by_prefix(state: ControllerState):
    """Test prefix-based endpoint listing."""
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Register multiple endpoints with shared prefix
    ep1 = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/inference/model-a",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="ns-1/inference/model-b",
        address="10.0.0.2:8080",
        job_id=JobId("ns-1"),
    )
    ep3 = ControllerEndpoint(
        endpoint_id="ep-3",
        name="ns-1/training/main",
        address="10.0.0.3:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep1)
    state.add_endpoint(ep2)
    state.add_endpoint(ep3)

    # List by prefix (includes namespace)
    results = state.list_endpoints_by_prefix("ns-1/inference/")
    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"ns-1/inference/model-a", "ns-1/inference/model-b"}

    results_training = state.list_endpoints_by_prefix("ns-1/training/")
    assert len(results_training) == 1
    assert results_training[0].name == "ns-1/training/main"

    # Listing all in namespace
    results_all = state.list_endpoints_by_prefix("ns-1/")
    assert len(results_all) == 3


def test_multiple_endpoints_for_same_name(state: ControllerState):
    """Test that multiple endpoints can be registered for the same name."""
    job1 = ControllerJob(
        job_id=JobId("ns-1/worker-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    job2 = ControllerJob(
        job_id=JobId("ns-1/worker-2"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job1)
    state.add_job(job2)

    # Register multiple endpoints with same name (for load balancing)
    ep1 = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/inference",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1/worker-1"),
    )
    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="ns-1/inference",
        address="10.0.0.2:8080",
        job_id=JobId("ns-1/worker-2"),
    )
    state.add_endpoint(ep1)
    state.add_endpoint(ep2)

    results = state.lookup_endpoints("ns-1/inference")
    assert len(results) == 2
    addresses = {r.address for r in results}
    assert addresses == {"10.0.0.1:8080", "10.0.0.2:8080"}


def test_remove_endpoint_by_id(state: ControllerState):
    """Test explicit endpoint removal by ID."""
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Remove by ID
    removed = state.remove_endpoint("ep-1")
    assert removed is not None
    assert removed.endpoint_id == "ep-1"

    # Should no longer be found
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0

    # Removing again should be idempotent
    removed_again = state.remove_endpoint("ep-1")
    assert removed_again is None


def test_pending_job_endpoints_not_returned(state: ControllerState):
    """Test that endpoints for PENDING jobs are not returned."""
    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_PENDING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Should not return because job is pending
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 0

    # Transition to running
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # Now should be visible
    results = state.lookup_endpoints("ns-1/my-actor")
    assert len(results) == 1
