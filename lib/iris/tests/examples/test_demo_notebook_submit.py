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

from pathlib import Path

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.rpc import config_pb2


def _make_demo_config() -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    cpu_sg = config.scale_groups["cpu"]
    cpu_sg.name = "cpu"
    cpu_sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    cpu_sg.min_slices = 0
    cpu_sg.max_slices = 1
    return config


def test_demo_notebook_hello_world_submit() -> None:
    config = make_local_config(_make_demo_config())
    manager = ClusterManager(config)
    manager.start()
    try:
        controller_url = manager.controller.discover()
        assert controller_url is not None

        client = IrisClient.remote(
            controller_url,
            workspace=Path(__file__).resolve().parents[3],
        )

        # Notebook cell snippet (verbatim structure).
        def hello_world():
            print("Hello from the cluster!")
            return 42

        job = client.submit(
            entrypoint=Entrypoint.from_callable(hello_world),
            name="notebook-hello",
            resources=ResourceSpec(cpu=1, memory="512m"),
        )
        status = job.wait(timeout=30.0, raise_on_failure=False)
        assert status is not None
    finally:
        manager.stop()


def test_demo_notebook_name_normalizes_to_absolute_job_id() -> None:
    config = make_local_config(_make_demo_config())
    manager = ClusterManager(config)
    manager.start()
    try:
        controller_url = manager.controller.discover()
        assert controller_url is not None

        client = IrisClient.remote(
            controller_url,
            workspace=Path(__file__).resolve().parents[3],
        )

        def hello_world():
            print("Hello from the cluster!")
            return 42

        job = client.submit(
            entrypoint=Entrypoint.from_callable(hello_world),
            name="notebook-hello",
            resources=ResourceSpec(cpu=1, memory="512m"),
        )
        # Regression coverage: ensure names without leading "/" are normalized
        # to absolute job IDs before reaching the controller.
        status = job.wait(timeout=30.0, raise_on_failure=False)
        assert status is not None
        assert job.job_id.to_wire() == "/notebook-hello"
    finally:
        manager.stop()


def test_demo_notebook_job_tasks_returns_tasks() -> None:
    config = make_local_config(_make_demo_config())
    manager = ClusterManager(config)
    manager.start()
    try:
        controller_url = manager.controller.discover()
        assert controller_url is not None

        client = IrisClient.remote(
            controller_url,
            workspace=Path(__file__).resolve().parents[3],
        )

        def hello_world():
            print("Hello from the cluster!")
            return 42

        job = client.submit(
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
    finally:
        manager.stop()
