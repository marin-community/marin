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

"""Tests for LocalClusterClient."""

import pytest

from fluster.cluster.client import LocalClusterClient, get_job_info
from fluster.cluster.types import Entrypoint, is_job_finished
from fluster.rpc import cluster_pb2


@pytest.fixture
def local_ops():
    """Create and start a LocalClusterClient instance."""
    ops = LocalClusterClient(max_workers=2)
    ops.start()
    yield ops
    ops.shutdown()


def test_basic_job_submission(local_ops):
    """Test basic job submission and execution."""

    def test_fn(x, y):
        return x + y

    entrypoint = Entrypoint.from_callable(test_fn, 10, 20)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job("test-job", entrypoint, resources)
    status = local_ops.wait_for_job("test-job", timeout=5.0)

    assert is_job_finished(status.state)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_info_contextvar(local_ops):
    """Test JobInfo is available in job via contextvar."""

    def test_fn():
        info = get_job_info()
        assert info is not None
        assert info.job_id == "test-job-info"
        assert info.worker_id is not None
        assert "actor" in info.ports
        return info.job_id

    entrypoint = Entrypoint.from_callable(test_fn)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job("test-job-info", entrypoint, resources, ports=["actor"])
    status = local_ops.wait_for_job("test-job-info", timeout=5.0)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_info_with_multiple_ports(local_ops):
    """Test JobInfo contains all allocated ports."""

    def test_fn():
        info = get_job_info()
        assert info is not None
        assert "actor" in info.ports
        assert "metrics" in info.ports
        assert "custom" in info.ports
        # Ports should be unique
        port_values = list(info.ports.values())
        assert len(port_values) == len(set(port_values))
        return info.ports

    entrypoint = Entrypoint.from_callable(test_fn)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job(
        "test-job-ports",
        entrypoint,
        resources,
        ports=["actor", "metrics", "custom"],
    )
    status = local_ops.wait_for_job("test-job-ports", timeout=5.0)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_failure(local_ops):
    """Test that job failures are captured correctly."""

    def failing_fn():
        raise ValueError("Intentional failure")

    entrypoint = Entrypoint.from_callable(failing_fn)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job("failing-job", entrypoint, resources)
    status = local_ops.wait_for_job("failing-job", timeout=5.0)

    assert status.state == cluster_pb2.JOB_STATE_FAILED
    assert "Intentional failure" in status.error


def test_hierarchical_job_id(local_ops):
    """Test that hierarchical job IDs work correctly."""

    def test_fn():
        info = get_job_info()
        assert info is not None
        assert info.job_id == "root/child/grandchild"
        return info.job_id

    entrypoint = Entrypoint.from_callable(test_fn)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job("root/child/grandchild", entrypoint, resources)
    status = local_ops.wait_for_job("root/child/grandchild", timeout=5.0)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_duplicate_job_id_rejected(local_ops):
    """Test that duplicate job IDs are rejected."""

    def test_fn():
        return 42

    entrypoint = Entrypoint.from_callable(test_fn)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job("duplicate-job", entrypoint, resources)

    # Try to submit again with same ID
    with pytest.raises(ValueError, match="already exists"):
        local_ops.submit_job("duplicate-job", entrypoint, resources)

    # Clean up
    local_ops.wait_for_job("duplicate-job", timeout=5.0)


def test_job_status_for_nonexistent_job(local_ops):
    """Test that querying status for nonexistent job returns UNSCHEDULABLE."""
    status = local_ops.get_job_status("nonexistent-job")
    assert status.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE


def test_endpoint_registration(local_ops):
    """Test endpoint registration and listing."""
    # Register some endpoints
    ep1_id = local_ops.register_endpoint("ns1/actor1", "host1:5000", "job1", {"key": "val1"})
    ep2_id = local_ops.register_endpoint("ns1/actor2", "host2:5001", "job1", {"key": "val2"})
    ep3_id = local_ops.register_endpoint("ns2/actor1", "host3:5002", "job2", {"key": "val3"})

    # List with prefix
    ns1_eps = local_ops.list_endpoints("ns1/")
    assert len(ns1_eps) == 2
    assert all(ep.name.startswith("ns1/") for ep in ns1_eps)

    # Verify metadata
    ep1 = next(ep for ep in ns1_eps if ep.endpoint_id == ep1_id)
    assert ep1.address == "host1:5000"
    assert ep1.metadata["key"] == "val1"

    # Unregister
    local_ops.unregister_endpoint(ep1_id)
    ns1_eps = local_ops.list_endpoints("ns1/")
    assert len(ns1_eps) == 1


def test_endpoint_prefix_matching(local_ops):
    """Test that endpoint prefix matching works correctly."""
    # Register endpoints with various prefixes
    local_ops.register_endpoint("ns1/actor1", "host1:5000", "job1")
    local_ops.register_endpoint("ns1/actor2", "host2:5001", "job1")
    local_ops.register_endpoint("ns1/service/actor3", "host3:5002", "job1")
    local_ops.register_endpoint("ns2/actor1", "host4:5003", "job2")

    # List with different prefixes
    all_ns1 = local_ops.list_endpoints("ns1/")
    assert len(all_ns1) == 3

    ns1_service = local_ops.list_endpoints("ns1/service/")
    assert len(ns1_service) == 1
    assert ns1_service[0].name == "ns1/service/actor3"

    ns2 = local_ops.list_endpoints("ns2/")
    assert len(ns2) == 1


def test_job_termination(local_ops):
    """Test that job termination marks job as killed."""
    import time

    def long_running_fn():
        time.sleep(10)
        return 42

    entrypoint = Entrypoint.from_callable(long_running_fn)
    resources = cluster_pb2.ResourceSpec()

    local_ops.submit_job("long-job", entrypoint, resources)

    # Wait a bit for job to start
    time.sleep(0.1)

    # Terminate it
    local_ops.terminate_job("long-job")

    # Check status
    status = local_ops.get_job_status("long-job")
    assert status.state == cluster_pb2.JOB_STATE_KILLED
    assert "Terminated by user" in status.error
