# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.config import IrisConfig
from iris.cluster.controller.local import LocalController
from iris.rpc import config_pb2


def _make_demo_config() -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    cpu_sg = config.scale_groups["cpu"]
    cpu_sg.name = "cpu"
    cpu_sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    cpu_sg.min_slices = 0
    cpu_sg.max_slices = 1
    cpu_sg.num_vms = 1
    cpu_sg.resources.cpu_millicores = 1000
    cpu_sg.resources.memory_bytes = 1024**3
    cpu_sg.resources.disk_bytes = 0
    cpu_sg.resources.gpu_count = 0
    cpu_sg.resources.tpu_count = 0
    return IrisConfig(config).as_local().proto


@pytest.fixture
def demo_client() -> IrisClient:
    controller = LocalController(_make_demo_config())
    address = controller.start()
    try:
        client = IrisClient.remote(
            address,
            workspace=Path(__file__).resolve().parents[3],
        )
        yield client
    finally:
        controller.stop()


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
