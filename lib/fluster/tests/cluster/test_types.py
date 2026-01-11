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

"""Unit tests for cluster types."""

import os
import pytest

from fluster.cluster.types import (
    CpuConfig,
    Endpoint,
    EndpointId,
    Entrypoint,
    GpuConfig,
    JobId,
    JobInfo,
    JobRequest,
    Namespace,
    ResourceConfig,
    TaskStatus,
    TpuConfig,
    VMId,
    VMInfo,
    create_environment,
    get_tpu_topology,
    is_job_finished,
)
from fluster import cluster_pb2


class TestJobState:
    def test_finished(self):
        assert not is_job_finished(cluster_pb2.JOB_STATE_PENDING)
        assert not is_job_finished(cluster_pb2.JOB_STATE_RUNNING)
        assert is_job_finished(cluster_pb2.JOB_STATE_SUCCEEDED)
        assert is_job_finished(cluster_pb2.JOB_STATE_FAILED)
        assert is_job_finished(cluster_pb2.JOB_STATE_KILLED)


class TestTpuTopology:
    def test_get_tpu_topology_v4(self):
        topo = get_tpu_topology("v4-8")
        assert topo.name == "v4-8"
        assert topo.chip_count == 4
        assert topo.host_count == 1
        assert topo.vm_count == 1
        assert topo.chips_per_vm == 4

    def test_get_tpu_topology_v5litepod(self):
        topo = get_tpu_topology("v5litepod-16")
        assert topo.name == "v5litepod-16"
        assert topo.chip_count == 16
        assert topo.host_count == 2
        assert topo.vm_count == 4
        assert topo.chips_per_vm == 4

    def test_get_tpu_topology_v5p(self):
        topo = get_tpu_topology("v5p-8192")
        assert topo.name == "v5p-8192"
        assert topo.chip_count == 4096
        assert topo.host_count == 1024
        assert topo.vm_count == 1024
        assert topo.chips_per_vm == 4

    def test_get_tpu_topology_v6e(self):
        topo = get_tpu_topology("v6e-256")
        assert topo.name == "v6e-256"
        assert topo.chip_count == 256
        assert topo.host_count == 64
        assert topo.vm_count == 64
        assert topo.chips_per_vm == 4

    def test_get_tpu_topology_unknown(self):
        with pytest.raises(ValueError, match="Unknown TPU type"):
            get_tpu_topology("nonexistent-tpu")


class TestDeviceConfigs:
    def test_cpu_config(self):
        cpu = CpuConfig()
        assert cpu.kind == "cpu"
        assert cpu.variant == "cpu"
        assert cpu.chip_count() == 0

    def test_gpu_config(self):
        gpu = GpuConfig(variant="A100", count=4)
        assert gpu.kind == "gpu"
        assert gpu.variant == "A100"
        assert gpu.count == 4
        assert gpu.chip_count() == 4

    def test_tpu_config(self):
        tpu = TpuConfig(variant="v5litepod-16")
        assert tpu.kind == "tpu"
        assert tpu.variant == "v5litepod-16"
        assert tpu.chip_count() == 16
        assert tpu.vm_count() == 4

    def test_tpu_config_with_topology(self):
        tpu = TpuConfig(variant="v5p-8", topology="2x2x1")
        assert tpu.topology == "2x2x1"
        assert tpu.chip_count() == 4


class TestResourceConfig:
    def test_default_config(self):
        config = ResourceConfig()
        assert config.cpu == 1
        assert config.ram == "128m"
        assert config.disk == "1g"
        assert isinstance(config.device, CpuConfig)
        assert config.replicas == 1
        assert config.preemptible is True
        assert config.regions is None
        assert config.chip_count() == 0

    def test_with_cpu(self):
        config = ResourceConfig.with_cpu(cpu=4, ram="8g")
        assert config.cpu == 4
        assert config.ram == "8g"
        assert isinstance(config.device, CpuConfig)
        assert config.chip_count() == 0

    def test_with_gpu(self):
        config = ResourceConfig.with_gpu(gpu_type="H100", count=8, cpu=16, ram="64g")
        assert config.cpu == 16
        assert config.ram == "64g"
        assert isinstance(config.device, GpuConfig)
        assert config.device.variant == "H100"
        assert config.device.count == 8
        assert config.chip_count() == 8

    def test_with_tpu(self):
        config = ResourceConfig.with_tpu("v5litepod-16", slice_count=2)
        assert isinstance(config.device, TpuConfig)
        assert config.device.variant == "v5litepod-16"
        assert config.replicas == 2
        assert config.chip_count() == 32  # 16 chips per slice * 2 slices

    def test_chip_count_multislice(self):
        config = ResourceConfig.with_tpu("v4-64", slice_count=4)
        assert config.chip_count() == 128  # 32 chips * 4 slices


class TestCreateEnvironment:
    def test_default_workspace(self):
        config = create_environment()
        assert config.workspace == os.getcwd()
        assert config.WhichOneof("source") == "workspace"

    def test_explicit_workspace(self):
        config = create_environment(workspace="/custom/path")
        assert config.workspace == "/custom/path"
        assert config.WhichOneof("source") == "workspace"

    def test_docker_image(self):
        config = create_environment(docker_image="python:3.11")
        assert config.docker_image == "python:3.11"
        assert config.WhichOneof("source") == "docker_image"

    def test_default_env_vars(self):
        config = create_environment()
        assert config.env_vars["HF_DATASETS_TRUST_REMOTE_CODE"] == "1"
        assert config.env_vars["TOKENIZERS_PARALLELISM"] == "false"

    def test_custom_env_vars(self):
        config = create_environment(env_vars={"CUSTOM_VAR": "value"})
        assert config.env_vars["CUSTOM_VAR"] == "value"
        assert config.env_vars["HF_DATASETS_TRUST_REMOTE_CODE"] == "1"

    def test_override_default_env_vars(self):
        config = create_environment(env_vars={"HF_DATASETS_TRUST_REMOTE_CODE": "0"})
        assert config.env_vars["HF_DATASETS_TRUST_REMOTE_CODE"] == "0"

    def test_env_vars_from_environment(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        config = create_environment()
        assert config.env_vars["HF_TOKEN"] == "test-token"
        assert config.env_vars["WANDB_API_KEY"] == "test-key"

    def test_none_env_vars_filtered(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        config = create_environment()
        assert "HF_TOKEN" not in config.env_vars
        assert "WANDB_API_KEY" not in config.env_vars

    def test_pip_packages_and_extras(self):
        config = create_environment(pip_packages=["numpy"], extras=["tpu"])
        assert list(config.pip_packages) == ["numpy"]
        assert list(config.extras) == ["tpu"]

    def test_both_workspace_and_docker_fails(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_environment(workspace="/path", docker_image="image:tag")


class TestEntrypoint:
    def test_basic_callable(self):
        def my_func():
            return 42

        entry = Entrypoint(callable=my_func)
        assert entry.callable == my_func
        assert entry.args == ()
        assert entry.kwargs == {}

    def test_with_args(self):
        def my_func(a, b):
            return a + b

        entry = Entrypoint(callable=my_func, args=(1, 2))
        assert entry.args == (1, 2)

    def test_with_kwargs(self):
        def my_func(a=1, b=2):
            return a + b

        entry = Entrypoint(callable=my_func, kwargs={"a": 10, "b": 20})
        assert entry.kwargs == {"a": 10, "b": 20}


class TestJobRequest:
    def test_basic_request(self):
        def my_func():
            pass

        req = JobRequest(name="test-job", entrypoint=Entrypoint(callable=my_func))
        assert req.name == "test-job"
        assert isinstance(req.resources, ResourceConfig)
        assert req.environment is None
        assert req.max_retries_failure == 0
        assert req.max_retries_preemption == 100

    def test_name_with_space_fails(self):
        def my_func():
            pass

        with pytest.raises(ValueError, match="must not contain spaces"):
            JobRequest(name="test job", entrypoint=Entrypoint(callable=my_func))

    def test_with_resources(self):
        def my_func():
            pass

        resources = ResourceConfig.with_tpu("v5litepod-8")
        req = JobRequest(name="test-job", entrypoint=Entrypoint(callable=my_func), resources=resources)
        assert req.resources.device.variant == "v5litepod-8"

    def test_with_environment(self):
        def my_func():
            pass

        env = create_environment(docker_image="python:3.11")
        req = JobRequest(name="test-job", entrypoint=Entrypoint(callable=my_func), environment=env)
        assert req.environment.docker_image == "python:3.11"


class TestTaskStatusAndJobInfo:
    def test_task_status(self):
        task = TaskStatus(state=cluster_pb2.JOB_STATE_RUNNING)
        assert task.state == cluster_pb2.JOB_STATE_RUNNING
        assert task.error_message is None

    def test_task_status_with_error(self):
        task = TaskStatus(state=cluster_pb2.JOB_STATE_FAILED, error_message="Connection refused")
        assert task.state == cluster_pb2.JOB_STATE_FAILED
        assert task.error_message == "Connection refused"

    def test_job_info(self):
        job = JobInfo(
            job_id=JobId("job-123"),
            state=cluster_pb2.JOB_STATE_RUNNING,
            tasks=[TaskStatus(state=cluster_pb2.JOB_STATE_RUNNING)],
            name="test-job",
        )
        assert job.job_id == JobId("job-123")
        assert job.state == cluster_pb2.JOB_STATE_RUNNING
        assert len(job.tasks) == 1
        assert job.name == "test-job"
        assert job.error_message is None


class TestEndpoint:
    def test_endpoint_creation(self):
        endpoint = Endpoint(
            endpoint_id=EndpointId("ep-123"),
            name="my-service",
            address="localhost:8080",
            job_id=JobId("job-456"),
            namespace=Namespace("default"),
        )
        assert endpoint.endpoint_id == EndpointId("ep-123")
        assert endpoint.name == "my-service"
        assert endpoint.address == "localhost:8080"
        assert endpoint.job_id == JobId("job-456")
        assert endpoint.namespace == Namespace("default")
        assert endpoint.metadata == {}

    def test_endpoint_with_metadata(self):
        endpoint = Endpoint(
            endpoint_id=EndpointId("ep-123"),
            name="my-service",
            address="localhost:8080",
            job_id=JobId("job-456"),
            namespace=Namespace("default"),
            metadata={"version": "1.0", "region": "us-west"},
        )
        assert endpoint.metadata == {"version": "1.0", "region": "us-west"}


class TestVMInfo:
    def test_vm_info(self):
        resources = ResourceConfig.with_tpu("v5litepod-16")
        vm = VMInfo(
            vm_id=VMId("vm-abc"),
            address="10.0.0.1",
            status="ready",
            resources=resources,
        )
        assert vm.vm_id == VMId("vm-abc")
        assert vm.address == "10.0.0.1"
        assert vm.status == "ready"
        assert vm.resources.device.variant == "v5litepod-16"
