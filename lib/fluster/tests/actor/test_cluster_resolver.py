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

"""Tests for ClusterResolver integration with controller."""

import socket
import threading
import time

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from fluster import cluster_pb2
from fluster.actor.resolver import ClusterResolver
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerEndpoint, ControllerJob, ControllerState
from fluster.cluster.types import JobId, Namespace


def create_controller_app(state: ControllerState) -> Starlette:
    """Create a minimal controller app with ListEndpoints handler."""

    def mock_dispatch(job, worker):
        """Mock dispatch function for scheduler (unused in tests)."""
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=1.0)
    service = ControllerServiceImpl(state, scheduler)

    async def list_endpoints_handler(request):
        body = await request.body()
        req = cluster_pb2.ListEndpointsRequest()
        req.ParseFromString(body)
        resp = service.list_endpoints(req, None)
        return Response(resp.SerializeToString(), media_type="application/proto")

    return Starlette(
        routes=[
            Route("/fluster.cluster.ControllerService/ListEndpoints", list_endpoints_handler, methods=["POST"]),
        ]
    )


@pytest.fixture
def controller_with_endpoint():
    """Start a controller with a registered endpoint."""
    state = ControllerState()

    # Add a running job
    job = ControllerJob(
        job_id=JobId("job-1"),
        request=cluster_pb2.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Add an endpoint
    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="inference",
        address="10.0.0.1:8080",
        job_id=JobId("job-1"),
        namespace="<local>",
    )
    state.add_endpoint(ep)

    # Find free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    app = create_controller_app(state)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)

    yield f"http://127.0.0.1:{port}", state


def test_cluster_resolver_finds_endpoint(controller_with_endpoint):
    """Test that ClusterResolver successfully resolves a registered endpoint."""
    address, _state = controller_with_endpoint

    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert "10.0.0.1:8080" in result.first().url
    assert result.first().actor_id == "ep-1"


def test_cluster_resolver_missing_endpoint(controller_with_endpoint):
    """Test that ClusterResolver returns empty result for non-existent actor."""
    address, _state = controller_with_endpoint

    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("nonexistent")

    assert result.is_empty


def test_cluster_resolver_multiple_endpoints(controller_with_endpoint):
    """Test that ClusterResolver returns all matching endpoints."""
    address, state = controller_with_endpoint

    # Add another job and endpoint with the same name
    job2 = ControllerJob(
        job_id=JobId("job-2"),
        request=cluster_pb2.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job2)

    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="inference",
        address="10.0.0.2:8080",
        job_id=JobId("job-2"),
        namespace="<local>",
    )
    state.add_endpoint(ep2)

    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 2
    addresses = {ep.url for ep in result.endpoints}
    assert "http://10.0.0.1:8080" in addresses
    assert "http://10.0.0.2:8080" in addresses


def test_cluster_resolver_namespace_isolation(controller_with_endpoint):
    """Test that ClusterResolver respects namespace boundaries."""
    address, state = controller_with_endpoint

    # Add endpoint in different namespace
    job2 = ControllerJob(
        job_id=JobId("job-2"),
        request=cluster_pb2.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job2)

    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="inference",
        address="10.0.0.2:8080",
        job_id=JobId("job-2"),
        namespace="other-namespace",
    )
    state.add_endpoint(ep2)

    # Resolve in <local> namespace should only find ep-1
    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert result.first().url == "http://10.0.0.1:8080"

    # Resolve in other-namespace should only find ep-2
    result_other = resolver.resolve("inference", namespace=Namespace("other-namespace"))
    assert len(result_other.endpoints) == 1
    assert result_other.first().url == "http://10.0.0.2:8080"


def test_cluster_resolver_filters_exact_name_match(controller_with_endpoint):
    """Test that ClusterResolver filters to exact name matches despite prefix API."""
    address, state = controller_with_endpoint

    # Add endpoint with similar but different name
    job2 = ControllerJob(
        job_id=JobId("job-2"),
        request=cluster_pb2.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job2)

    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="inference-v2",
        address="10.0.0.2:8080",
        job_id=JobId("job-2"),
        namespace="<local>",
    )
    state.add_endpoint(ep2)

    # Resolve "inference" should not return "inference-v2"
    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert result.first().url == "http://10.0.0.1:8080"


def test_cluster_resolver_only_running_jobs(controller_with_endpoint):
    """Test that ClusterResolver only returns endpoints for RUNNING jobs."""
    address, state = controller_with_endpoint

    # Add a completed job with endpoint
    job2 = ControllerJob(
        job_id=JobId("job-2"),
        request=cluster_pb2.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
    )
    state.add_job(job2)

    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="inference",
        address="10.0.0.2:8080",
        job_id=JobId("job-2"),
        namespace="<local>",
    )
    state.add_endpoint(ep2)

    # Should only find the running job's endpoint
    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert result.first().url == "http://10.0.0.1:8080"


def test_cluster_resolver_metadata(controller_with_endpoint):
    """Test that ClusterResolver preserves endpoint metadata."""
    address, state = controller_with_endpoint

    # Add endpoint with metadata
    job2 = ControllerJob(
        job_id=JobId("job-2"),
        request=cluster_pb2.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job2)

    ep2 = ControllerEndpoint(
        endpoint_id="ep-2",
        name="tagged-actor",
        address="10.0.0.2:8080",
        job_id=JobId("job-2"),
        namespace="<local>",
        metadata={"model": "gpt-4", "version": "1.0"},
    )
    state.add_endpoint(ep2)

    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("tagged-actor")

    assert len(result.endpoints) == 1
    assert result.first().metadata["model"] == "gpt-4"
    assert result.first().metadata["version"] == "1.0"
