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

"""Tests for cluster type definitions."""

import pytest

from fray.cluster import (
    CpuConfig,
    EnvironmentConfig,
    GpuConfig,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    create_environment,
)


def test_cpu_config():
    """Test CpuConfig creation."""
    config = CpuConfig()
    assert isinstance(config, CpuConfig)


def test_gpu_config():
    """Test GpuConfig creation."""
    config = GpuConfig(type="A100", count=4)
    assert config.type == "A100"
    assert config.count == 4


def test_gpu_config_default_count():
    """Test GpuConfig with default count."""
    config = GpuConfig(type="H100")
    assert config.type == "H100"
    assert config.count == 1


def test_tpu_config():
    """Test TpuConfig creation."""
    config = TpuConfig(type="v5e-16", count=8)
    assert config.type == "v5e-16"
    assert config.count == 8
    assert config.topology is None


def test_tpu_config_with_topology():
    """Test TpuConfig with topology."""
    config = TpuConfig(type="v5p-8", count=16, topology="2x2x4")
    assert config.type == "v5p-8"
    assert config.count == 16
    assert config.topology == "2x2x4"


def test_resource_config_defaults():
    """Test ResourceConfig with defaults."""
    config = ResourceConfig()
    assert config.cpu == 1
    assert config.ram == "4g"
    assert config.disk == "10g"
    assert isinstance(config.device, CpuConfig)
    assert config.count == 1
    assert config.regions is None


def test_resource_config_with_gpu():
    """Test ResourceConfig with GPU."""
    config = ResourceConfig(
        cpu=8,
        ram="32g",
        device=GpuConfig(type="A100", count=4),
        count=2,
    )
    assert config.cpu == 8
    assert config.ram == "32g"
    assert isinstance(config.device, GpuConfig)
    assert config.device.type == "A100"
    assert config.device.count == 4
    assert config.count == 2


def test_resource_config_with_tpu():
    """Test ResourceConfig with TPU."""
    config = ResourceConfig(
        cpu=96,
        ram="512g",
        device=TpuConfig(type="v5e-16", count=8),
        regions=["us-central1"],
    )
    assert config.cpu == 96
    assert isinstance(config.device, TpuConfig)
    assert config.device.type == "v5e-16"
    assert config.regions == ["us-central1"]


def test_environment_config_workspace():
    """Test EnvironmentConfig with workspace."""
    config = EnvironmentConfig(
        workspace="/path/to/workspace",
        pip_packages=["numpy", "pandas"],
        env_vars={"MY_VAR": "value"},
        extra_dependency_groups=["tpu", "eval"],
    )
    assert config.workspace == "/path/to/workspace"
    assert config.docker_image is None
    assert config.pip_packages == ["numpy", "pandas"]
    assert config.env_vars == {"MY_VAR": "value"}
    assert config.extra_dependency_groups == ["tpu", "eval"]


def test_environment_config_docker():
    """Test EnvironmentConfig with docker image."""
    config = EnvironmentConfig(
        docker_image="my-image:latest",
        env_vars={"MY_VAR": "value"},
    )
    assert config.docker_image == "my-image:latest"
    assert config.workspace is None


def test_environment_config_requires_one():
    """Test EnvironmentConfig requires workspace or docker_image."""
    with pytest.raises(ValueError, match="Must specify either workspace or docker_image"):
        EnvironmentConfig()


def test_environment_config_rejects_both():
    """Test EnvironmentConfig rejects both workspace and docker_image."""
    with pytest.raises(ValueError, match="Cannot specify both workspace and docker_image"):
        EnvironmentConfig(workspace="/path", docker_image="image:latest")


def test_create_environment_defaults():
    """Test create_environment with defaults."""
    config = create_environment()
    assert config.workspace is not None  # Should use current directory
    assert config.docker_image is None
    assert "HF_DATASETS_TRUST_REMOTE_CODE" in config.env_vars
    assert config.env_vars["HF_DATASETS_TRUST_REMOTE_CODE"] == "1"
    assert "TOKENIZERS_PARALLELISM" in config.env_vars
    assert config.env_vars["TOKENIZERS_PARALLELISM"] == "false"


def test_create_environment_workspace():
    """Test create_environment with workspace."""
    config = create_environment(workspace="/my/workspace")
    assert config.workspace == "/my/workspace"
    assert config.docker_image is None


def test_create_environment_docker():
    """Test create_environment with docker image."""
    config = create_environment(docker_image="my-image:latest")
    assert config.docker_image == "my-image:latest"
    assert config.workspace is None


def test_create_environment_merge_env_vars():
    """Test create_environment merges env vars with defaults."""
    config = create_environment(env_vars={"CUSTOM_VAR": "value", "HF_DATASETS_TRUST_REMOTE_CODE": "0"})
    # Custom var should be present
    assert config.env_vars["CUSTOM_VAR"] == "value"
    # User value should override default
    assert config.env_vars["HF_DATASETS_TRUST_REMOTE_CODE"] == "0"
    # Default should still be present
    assert config.env_vars["TOKENIZERS_PARALLELISM"] == "false"


def test_create_environment_extra_groups():
    """Test create_environment with extra dependency groups."""
    config = create_environment(extra_dependency_groups=["tpu", "eval"])
    assert config.extra_dependency_groups == ["tpu", "eval"]


def test_job_request_minimal():
    """Test minimal JobRequest."""
    request = JobRequest(name="test-job", entrypoint="my_module")
    assert request.name == "test-job"
    assert request.entrypoint == "my_module"
    assert request.entrypoint_args == []
    assert isinstance(request.resources, ResourceConfig)
    assert request.environment is None


def test_job_request_with_args():
    """Test JobRequest with arguments."""
    request = JobRequest(
        name="test-job",
        entrypoint="my_module.main",
        entrypoint_args=["--arg1", "value1", "--arg2", "value2"],
    )
    assert request.entrypoint == "my_module.main"
    assert request.entrypoint_args == ["--arg1", "value1", "--arg2", "value2"]


def test_job_request_with_resources():
    """Test JobRequest with custom resources."""
    request = JobRequest(
        name="tpu-job",
        entrypoint="train",
        resources=ResourceConfig(
            cpu=96,
            ram="512g",
            device=TpuConfig(type="v5e-16", count=8),
        ),
    )
    assert isinstance(request.resources.device, TpuConfig)
    assert request.resources.device.type == "v5e-16"


def test_job_request_with_environment():
    """Test JobRequest with environment."""
    env = create_environment(extra_dependency_groups=["tpu"])
    request = JobRequest(
        name="test-job",
        entrypoint="my_module",
        environment=env,
    )
    assert request.environment is not None
    assert request.environment.extra_dependency_groups == ["tpu"]
