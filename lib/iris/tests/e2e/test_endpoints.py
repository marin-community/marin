# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint registration, prefix matching, port allocation, and JobInfo context tests.

Migrated from tests/cluster/test_e2e.py::TestEndpoints, TestPorts, and TestJobInfo.

Tests that validate runtime context (get_job_info(), ControllerServiceClientSync)
work by submitting jobs whose inner functions use those APIs. The inner functions
run inside the worker -- only the test harness uses TestCluster.
"""

import time
import uuid

import pytest
from iris.cluster.client import get_job_info
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

from .conftest import assert_visible, dashboard_click, dashboard_goto, wait_for_dashboard_ready

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------


def _register_endpoint_job(prefix):
    """Runs inside the worker. Registers an endpoint and verifies it via RPC."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    if not info.controller_address:
        raise ValueError("controller_address not set in JobInfo")

    client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
    try:
        endpoint_name = f"{prefix}/actor1"
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=endpoint_name,
            address="localhost:5000",
            job_id=info.job_id.to_wire(),
            metadata={"type": "actor"},
        )
        response = client.register_endpoint(request)
        assert response.endpoint_id

        list_request = cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{prefix}/")
        list_response = client.list_endpoints(list_request)
        assert len(list_response.endpoints) == 1
        assert list_response.endpoints[0].name == endpoint_name
        assert list_response.endpoints[0].metadata["type"] == "actor"

        time.sleep(0.5)
    finally:
        client.close()


def test_endpoint_registration_from_job(cluster):
    """Job can register endpoints that are visible via the controller RPC."""
    prefix = f"test-{uuid.uuid4().hex[:8]}"
    job = cluster.submit(_register_endpoint_job, "endpoint-reg", prefix)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed: {status}"


def _register_multiple_endpoints(ns1_prefix, ns2_prefix):
    """Runs inside the worker. Registers endpoints across two namespaces and verifies prefix matching."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    if not info.controller_address:
        raise ValueError("controller_address not set in JobInfo")

    client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
    try:
        for name, addr in [
            (f"{ns1_prefix}/actor1", "host1:5000"),
            (f"{ns1_prefix}/actor2", "host2:5001"),
            (f"{ns1_prefix}/service/actor3", "host3:5002"),
            (f"{ns2_prefix}/actor1", "host4:5003"),
        ]:
            request = cluster_pb2.Controller.RegisterEndpointRequest(
                name=name,
                address=addr,
                job_id=info.job_id.to_wire(),
            )
            client.register_endpoint(request)

        ns1_all = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{ns1_prefix}/"))
        assert len(ns1_all.endpoints) == 3

        ns1_service = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{ns1_prefix}/service/"))
        assert len(ns1_service.endpoints) == 1
        assert ns1_service.endpoints[0].name == f"{ns1_prefix}/service/actor3"

        ns2_all = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{ns2_prefix}/"))
        assert len(ns2_all.endpoints) == 1

        time.sleep(0.5)
    finally:
        client.close()


def test_endpoint_prefix_matching(cluster):
    """Endpoint prefix matching correctly filters by namespace."""
    run_id = uuid.uuid4().hex[:8]
    ns1 = f"ns1-{run_id}"
    ns2 = f"ns2-{run_id}"
    job = cluster.submit(_register_multiple_endpoints, "prefix-match", ns1, ns2)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed: {status}"


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


def _port_job():
    """Runs inside the worker. Validates that requested ports are allocated."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    if "http" not in info.ports or "grpc" not in info.ports:
        raise ValueError(f"Ports not set: {info.ports}")
    assert info.ports["http"] > 0
    assert info.ports["grpc"] > 0


def test_port_allocation(cluster):
    """Requested ports are allocated and accessible via JobInfo."""
    job = cluster.submit(_port_job, "port-alloc", ports=["http", "grpc"])
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed: {status}"


# ---------------------------------------------------------------------------
# JobInfo context
# ---------------------------------------------------------------------------


def _job_info_context_fn(expected_job_id):
    """Runs inside the worker. Validates JobInfo provides correct job_id, worker_id, and ports."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    if info.job_id.to_wire() != expected_job_id:
        raise ValueError(f"JobInfo has wrong job_id: {info.job_id}")
    if info.worker_id is None:
        raise ValueError("JobInfo missing worker_id")
    if "actor" not in info.ports:
        raise ValueError("JobInfo missing expected port 'actor'")
    return "success"


def test_job_info_provides_context(cluster):
    """JobInfo provides job_id, worker_id, and ports during execution."""
    job_name = "job-info-ctx"
    expected_job_id = JobName.root(job_name).to_wire()
    job = cluster.submit(_job_info_context_fn, job_name, expected_job_id, ports=["actor"])
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def _job_info_ports_fn():
    """Runs inside the worker. Validates all requested ports are present and unique."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    required_ports = {"actor", "metrics", "custom"}
    if not required_ports.issubset(info.ports.keys()):
        raise ValueError(f"Missing ports. Expected {required_ports}, got {info.ports.keys()}")
    port_values = list(info.ports.values())
    if len(port_values) != len(set(port_values)):
        raise ValueError(f"Ports are not unique: {port_values}")
    return "success"


def test_job_info_port_allocation(cluster):
    """JobInfo provides all requested ports and they are unique."""
    job = cluster.submit(_job_info_ports_fn, "job-info-ports", ports=["actor", "metrics", "custom"])
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def _job_info_task_context_fn(expected_job_name):
    """Runs inside the worker. Validates task-specific context in JobInfo."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")
    expected_task_id = JobName.root(expected_job_name).task(0).to_wire()
    if info.task_id.to_wire() != expected_task_id:
        raise ValueError(f"Expected task_id {expected_task_id}, got {info.task_id}")
    if info.task_index != 0:
        raise ValueError(f"Expected task_index 0, got {info.task_index}")
    if info.num_tasks != 1:
        raise ValueError(f"Expected num_tasks 1, got {info.num_tasks}")
    return "success"


def test_job_info_task_context(cluster):
    """JobInfo provides task-specific context: task_id, task_index, num_tasks."""
    job_name = "task-ctx"
    job = cluster.submit(_job_info_task_context_fn, job_name, job_name)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Dashboard assertions (require Playwright via the `page` fixture)
# ---------------------------------------------------------------------------


def _register_and_hold(prefix):
    """Registers an endpoint and holds the job running so the dashboard can observe it."""
    info = get_job_info()
    if info is None:
        raise ValueError("JobInfo not available")

    client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
    try:
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=f"{prefix}/dashboard-check",
            address="localhost:9999",
            job_id=info.job_id.to_wire(),
            metadata={"purpose": "dashboard-test"},
        )
        client.register_endpoint(request)
        time.sleep(30)
    finally:
        client.close()


def test_endpoints_visible_in_dashboard(cluster, page, screenshot):
    """Endpoints tab shows a registered endpoint while the job is running."""
    prefix = f"dash-ep-{uuid.uuid4().hex[:8]}"
    job = cluster.submit(_register_and_hold, "dash-endpoint", prefix)
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)
    time.sleep(2)

    dashboard_goto(page, f"{cluster.url}/")
    wait_for_dashboard_ready(page)
    dashboard_click(page, 'button.tab-btn:has-text("Endpoints")')

    assert_visible(page, f"text={prefix}/dashboard-check")
    screenshot("endpoints-tab")

    cluster.kill(job)
