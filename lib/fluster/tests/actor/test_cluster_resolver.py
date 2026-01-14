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

from connectrpc.request import RequestContext

from fluster import cluster_pb2
from fluster.actor.resolver import ClusterResolver
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerEndpoint, ControllerJob, ControllerState
from fluster.cluster.types import JobId, Namespace
from fluster.cluster_connect import ControllerServiceASGIApplication


class AsyncControllerServiceWrapper:
    """Async wrapper around synchronous ControllerServiceImpl for testing."""

    def __init__(self, sync_service: ControllerServiceImpl):
        self._service = sync_service

    async def launch_job(
        self, request: cluster_pb2.LaunchJobRequest, ctx: RequestContext
    ) -> cluster_pb2.LaunchJobResponse:
        return self._service.launch_job(request, ctx)

    async def get_job_status(
        self, request: cluster_pb2.GetJobStatusRequest, ctx: RequestContext
    ) -> cluster_pb2.GetJobStatusResponse:
        return self._service.get_job_status(request, ctx)

    async def terminate_job(self, request: cluster_pb2.TerminateJobRequest, ctx: RequestContext) -> cluster_pb2.Empty:
        return self._service.terminate_job(request, ctx)

    async def list_jobs(self, request: cluster_pb2.ListJobsRequest, ctx: RequestContext) -> cluster_pb2.ListJobsResponse:
        return self._service.list_jobs(request, ctx)

    async def register_worker(
        self, request: cluster_pb2.RegisterWorkerRequest, ctx: RequestContext
    ) -> cluster_pb2.RegisterWorkerResponse:
        return self._service.register_worker(request, ctx)

    async def heartbeat(
        self, request: cluster_pb2.HeartbeatRequest, ctx: RequestContext
    ) -> cluster_pb2.HeartbeatResponse:
        return self._service.heartbeat(request, ctx)

    async def list_workers(
        self, request: cluster_pb2.ListWorkersRequest, ctx: RequestContext
    ) -> cluster_pb2.ListWorkersResponse:
        return self._service.list_workers(request, ctx)

    async def register_endpoint(
        self, request: cluster_pb2.RegisterEndpointRequest, ctx: RequestContext
    ) -> cluster_pb2.RegisterEndpointResponse:
        return self._service.register_endpoint(request, ctx)

    async def unregister_endpoint(
        self, request: cluster_pb2.UnregisterEndpointRequest, ctx: RequestContext
    ) -> cluster_pb2.Empty:
        return self._service.unregister_endpoint(request, ctx)

    async def lookup_endpoint(
        self, request: cluster_pb2.LookupEndpointRequest, ctx: RequestContext
    ) -> cluster_pb2.LookupEndpointResponse:
        return self._service.lookup_endpoint(request, ctx)

    async def list_endpoints(
        self, request: cluster_pb2.ListEndpointsRequest, ctx: RequestContext
    ) -> cluster_pb2.ListEndpointsResponse:
        return self._service.list_endpoints(request, ctx)


def create_controller_app(state: ControllerState) -> ControllerServiceASGIApplication:
    """Create a minimal controller app with ListEndpoints handler."""

    def mock_dispatch(job, worker):
        """Mock dispatch function for scheduler (unused in tests)."""
        return True

    scheduler = Scheduler(state, mock_dispatch, interval_seconds=1.0)
    service = ControllerServiceImpl(state, scheduler)
    async_service = AsyncControllerServiceWrapper(service)

    return ControllerServiceASGIApplication(service=async_service)


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
