# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests verifying the demo notebook submission patterns work end-to-end."""

from pathlib import Path

import pytest
from iris.client import IrisClient
from iris.cluster.config import (
    IrisClusterConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    make_local_config,
)
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import AcceleratorType, CapacityType, Entrypoint, ResourceSpec

pytestmark = pytest.mark.requires_cluster


def _make_demo_config() -> IrisClusterConfig:
    config = IrisClusterConfig(
        scale_groups={
            "cpu": ScaleGroupConfig(
                name="cpu",
                buffer_slices=0,
                max_slices=1,
                num_vms=1,
                resources=ScaleGroupResources(
                    cpu_millicores=1000,
                    memory_bytes=1024**3,
                    disk_bytes=0,
                    device_type=AcceleratorType.CPU,
                    device_count=0,
                    capacity_type=CapacityType.ON_DEMAND,
                ),
            )
        }
    )
    return make_local_config(config)


@pytest.fixture(scope="module")
def demo_client() -> IrisClient:
    controller = LocalCluster(_make_demo_config())
    address = controller.start()
    try:
        client = IrisClient.remote(
            address,
            workspace=Path(__file__).resolve().parents[3],
        )
        yield client
    finally:
        controller.close()


def test_demo_notebook_hello_world_submit(demo_client: IrisClient) -> None:
    # Notebook cell snippet (verbatim structure).
    def hello_world():
        print("Hello from the cluster!")
        return 42

    job = demo_client.submit(
        entrypoint=Entrypoint.from_callable(hello_world),
        name="notebook-hello",
        resources=ResourceSpec(cpu=1, memory="512m"),
    )
    status = job.wait(timeout=30.0, raise_on_failure=False)
    assert status is not None


def test_demo_notebook_name_normalizes_to_absolute_job_id(demo_client: IrisClient) -> None:
    def hello_world():
        print("Hello from the cluster!")
        return 42

    job = demo_client.submit(
        entrypoint=Entrypoint.from_callable(hello_world),
        name="notebook-hello",
        resources=ResourceSpec(cpu=1, memory="512m"),
    )
    # Regression coverage: ensure names without leading "/" are normalized
    # to absolute job IDs before reaching the controller.
    status = job.wait(timeout=30.0, raise_on_failure=False)
    assert status is not None
    assert job.job_id.to_wire().endswith("/notebook-hello")


def test_demo_notebook_job_tasks_returns_tasks(demo_client: IrisClient) -> None:
    def hello_world():
        print("Hello from the cluster!")
        return 42

    job = demo_client.submit(
        entrypoint=Entrypoint.from_callable(hello_world),
        name="notebook-hello",
        resources=ResourceSpec(cpu=1, memory="512m"),
    )
    status = job.wait(timeout=30.0, raise_on_failure=False)
    assert status is not None

    # Regression coverage: job.tasks() should pass a JobName to the RPC
    # client (not a string), avoiding "'str' object has no attribute to_wire".
    tasks = job.tasks()
    assert len(tasks) == 1
    assert tasks[0].job_id == job.job_id
