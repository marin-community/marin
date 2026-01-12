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

from fluster import cluster_pb2
from fluster.cluster.types import (
    JobRequest,
    ResourceConfig,
    create_environment,
    get_tpu_topology,
    is_job_finished,
)


class TestJobState:
    def test_finished(self):
        assert not is_job_finished(cluster_pb2.JOB_STATE_PENDING)
        assert not is_job_finished(cluster_pb2.JOB_STATE_RUNNING)
        assert is_job_finished(cluster_pb2.JOB_STATE_SUCCEEDED)
        assert is_job_finished(cluster_pb2.JOB_STATE_FAILED)
        assert is_job_finished(cluster_pb2.JOB_STATE_KILLED)


class TestTpuTopology:
    """Tests for TPU topology calculations."""

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


class TestResourceConfig:
    """Tests for ResourceConfig chip count calculations."""

    def test_chip_count_cpu(self):
        config = ResourceConfig.with_cpu(cpu=4, ram="8g")
        assert config.chip_count() == 0

    def test_chip_count_gpu(self):
        config = ResourceConfig.with_gpu(gpu_type="H100", count=8, cpu=16, ram="64g")
        assert config.chip_count() == 8

    def test_chip_count_tpu(self):
        config = ResourceConfig.with_tpu("v5litepod-16", slice_count=2)
        assert config.chip_count() == 32  # 16 chips per slice * 2 slices

    def test_chip_count_multislice_tpu(self):
        config = ResourceConfig.with_tpu("v4-64", slice_count=4)
        assert config.chip_count() == 128  # 32 chips * 4 slices


class TestCreateEnvironment:
    """Tests for create_environment behavior."""

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

    def test_both_workspace_and_docker_fails(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_environment(workspace="/path", docker_image="image:tag")


class TestJobRequest:
    """Tests for JobRequest validation."""

    def test_name_with_space_fails(self):
        def my_func():
            pass

        from fluster.cluster.types import Entrypoint

        with pytest.raises(ValueError, match="must not contain spaces"):
            JobRequest(name="test job", entrypoint=Entrypoint(callable=my_func))
