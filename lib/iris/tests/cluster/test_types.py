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

"""Tests for cluster types and utilities."""

import os
from unittest.mock import patch

from iris.cluster.types import EnvironmentSpec, ResourceSpec
from iris.rpc import cluster_pb2


def test_resource_spec_with_device():
    """Verify ResourceSpec copies device configuration to proto."""
    gpu_device = cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(count=2, variant="a100"))
    spec = ResourceSpec(cpu=4, memory="16g", device=gpu_device)
    proto = spec.to_proto()
    assert proto.device.HasField("gpu")
    assert proto.device.gpu.count == 2
    assert proto.device.gpu.variant == "a100"


def test_resource_spec_with_all_fields():
    """Verify all ResourceSpec fields are copied to proto."""
    tpu_device = cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16"))
    spec = ResourceSpec(
        cpu=8,
        memory="32g",
        disk="500g",
        device=tpu_device,
        replicas=4,
        preemptible=True,
        regions=["us-central1", "us-east1"],
    )
    proto = spec.to_proto()
    assert proto.cpu == 8
    assert proto.memory_bytes == 32 * 1024**3
    assert proto.disk_bytes == 500 * 1024**3
    assert proto.device.HasField("tpu")
    assert proto.device.tpu.variant == "v5litepod-16"
    assert proto.replicas == 4
    assert proto.preemptible is True
    assert list(proto.regions) == ["us-central1", "us-east1"]


def test_environment_spec_env_vars_override_defaults():
    """User env_vars override default values."""
    spec = EnvironmentSpec(env_vars={"TOKENIZERS_PARALLELISM": "true"})
    proto = spec.to_proto()
    assert proto.env_vars["TOKENIZERS_PARALLELISM"] == "true"


def test_environment_spec_inherits_env_tokens():
    """EnvironmentSpec inherits HF_TOKEN and WANDB_API_KEY from environment."""
    with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token", "WANDB_API_KEY": "test-wandb-key"}):
        spec = EnvironmentSpec()
        proto = spec.to_proto()
        assert proto.env_vars["HF_TOKEN"] == "test-hf-token"
        assert proto.env_vars["WANDB_API_KEY"] == "test-wandb-key"


def test_environment_spec_omits_none_env_vars():
    """EnvironmentSpec omits env vars with None values."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove HF_TOKEN and WANDB_API_KEY from environment
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("WANDB_API_KEY", None)
        spec = EnvironmentSpec()
        proto = spec.to_proto()
        assert "HF_TOKEN" not in proto.env_vars
        assert "WANDB_API_KEY" not in proto.env_vars
