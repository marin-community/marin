# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for multi-region constraint-based routing.

Verifies that:
- Jobs with region constraints are scheduled on workers in the correct region.
- Child jobs inherit region constraints from parents when no explicit constraints are given.
- Child jobs can override parent region constraints with their own.

These tests boot a local cluster with two CPU scale groups, each tagged with a
different region attribute. The constraint-based scheduler matches job constraints
against worker attributes to route jobs to the correct region.
"""


import pytest
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    region_constraint,
)
from iris.client.client import IrisClient
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

from .conftest import IRIS_ROOT, DEFAULT_CONFIG, TestCluster

pytestmark = pytest.mark.e2e

REGION_A = "us-central1"
REGION_B = "europe-west4"


def _make_multi_region_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with two CPU scale groups in different regions."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()

    for name, region in [("cpu-region-a", REGION_A), ("cpu-region-b", REGION_B)]:
        sg = config.scale_groups[name]
        sg.name = name
        sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.num_vms = 1
        sg.min_slices = 1
        sg.max_slices = 2
        sg.resources.cpu_millicores = 8000
        sg.resources.memory_bytes = 16 * 1024**3
        sg.resources.disk_bytes = 50 * 1024**3
        sg.slice_template.local.SetInParent()
        sg.worker.attributes["region"] = region

    return make_local_config(config)


@pytest.fixture
def multi_region_cluster():
    """Boot a local cluster with two scale groups in different regions.

    Waits for at least 2 workers (one from each region) before yielding.
    """
    config = _make_multi_region_config()
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = TestCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(2, timeout=30)
        yield tc
        controller_client.close()


def _get_worker_region(cluster: TestCluster, worker_id: str) -> str | None:
    """Look up the region attribute for a worker by its ID."""
    request = cluster_pb2.Controller.ListWorkersRequest()
    response = cluster.controller_client.list_workers(request)
    for w in response.workers:
        if w.worker_id == worker_id:
            region_attr = w.metadata.attributes.get("region")
            if region_attr and region_attr.HasField("string_value"):
                return region_attr.string_value
    return None


def _noop():
    return "ok"


def test_region_constrained_job_routes_correctly(multi_region_cluster):
    """A job with a region constraint lands on a worker in that region."""
    cluster = multi_region_cluster

    job = cluster.submit(
        _noop,
        "region-a-job",
        constraints=[region_constraint([REGION_A])],
    )
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    task = cluster.task_status(job, task_index=0)
    assert task.worker_id, "Task should have been assigned to a worker"

    worker_region = _get_worker_region(cluster, task.worker_id)
    assert worker_region == REGION_A, f"Expected worker in {REGION_A}, got {worker_region} (worker_id={task.worker_id})"


def _submit_child_no_constraints():
    """Parent task that submits a child job without explicit constraints.

    Runs inside an Iris job. Uses iris_ctx() to get the client and submit a child.
    The child should inherit the parent's region constraint.
    """
    from iris.client.client import iris_ctx

    ctx = iris_ctx()
    child = ctx.client.submit(
        entrypoint=Entrypoint.from_callable(lambda: "child-ok"),
        name="inherited-child",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
    )
    child.wait(timeout=60, raise_on_failure=True)


def test_child_inherits_parent_region_constraint(multi_region_cluster):
    """A child job without explicit constraints inherits the parent's region."""
    cluster = multi_region_cluster

    parent_job = cluster.submit(
        _submit_child_no_constraints,
        "parent-with-region",
        constraints=[region_constraint([REGION_A])],
    )
    parent_status = cluster.wait(parent_job, timeout=60)
    assert parent_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Verify the parent ran in region A
    parent_task = cluster.task_status(parent_job, task_index=0)
    parent_region = _get_worker_region(cluster, parent_task.worker_id)
    assert parent_region == REGION_A, f"Parent expected in {REGION_A}, got {parent_region}"

    # Find the child job by listing jobs and looking for "inherited-child"
    request = cluster_pb2.Controller.ListJobsRequest()
    response = cluster.controller_client.list_jobs(request)
    child_jobs = [j for j in response.jobs if "inherited-child" in j.name]
    assert child_jobs, "Child job 'inherited-child' should exist"

    child_status = child_jobs[0]
    assert child_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Verify the child also ran in region A (inherited constraint)
    child_task_request = cluster_pb2.Controller.GetTaskStatusRequest(
        task_id=child_status.name + "/0",
    )
    child_task = cluster.controller_client.get_task_status(child_task_request).task
    assert child_task.worker_id, "Child task should have been assigned to a worker"

    child_region = _get_worker_region(cluster, child_task.worker_id)
    assert child_region == REGION_A, f"Child expected to inherit parent region {REGION_A}, got {child_region}"


def _submit_child_with_region_override():
    """Parent task that submits a child job with an explicit region override.

    The child specifies region-b, overriding the parent's region-a constraint.
    """
    from iris.client.client import iris_ctx
    from iris.cluster.types import region_constraint

    ctx = iris_ctx()
    child = ctx.client.submit(
        entrypoint=Entrypoint.from_callable(lambda: "child-override-ok"),
        name="overridden-child",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
        constraints=[region_constraint(["europe-west4"])],
    )
    child.wait(timeout=60, raise_on_failure=True)


def test_child_overrides_parent_region_constraint(multi_region_cluster):
    """A child job with its own region constraint overrides the parent's region."""
    cluster = multi_region_cluster

    parent_job = cluster.submit(
        _submit_child_with_region_override,
        "parent-region-a-child-overrides",
        constraints=[region_constraint([REGION_A])],
    )
    parent_status = cluster.wait(parent_job, timeout=60)
    assert parent_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Verify the parent ran in region A
    parent_task = cluster.task_status(parent_job, task_index=0)
    parent_region = _get_worker_region(cluster, parent_task.worker_id)
    assert parent_region == REGION_A, f"Parent expected in {REGION_A}, got {parent_region}"

    # Find the child job
    request = cluster_pb2.Controller.ListJobsRequest()
    response = cluster.controller_client.list_jobs(request)
    child_jobs = [j for j in response.jobs if "overridden-child" in j.name]
    assert child_jobs, "Child job 'overridden-child' should exist"

    child_status = child_jobs[0]
    assert child_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Verify the child ran in region B (overridden constraint)
    child_task_request = cluster_pb2.Controller.GetTaskStatusRequest(
        task_id=child_status.name + "/0",
    )
    child_task = cluster.controller_client.get_task_status(child_task_request).task
    assert child_task.worker_id, "Child task should have been assigned to a worker"

    child_region = _get_worker_region(cluster, child_task.worker_id)
    assert child_region == REGION_B, f"Child expected in {REGION_B} (overriding parent's {REGION_A}), got {child_region}"
