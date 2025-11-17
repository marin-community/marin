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

"""Tests for Cluster ABC implementations (LocalCluster and RayCluster).

Parameterized tests run for both implementations via fixtures.
Implementation-specific tests are kept as separate manual tests.
"""

import pytest
from fray.cluster import (
    GpuConfig,
    JobId,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    create_environment,
)


def test_cluster_launch(cluster):
    request = JobRequest(
        name="test-simple-job",
        entrypoint="json.tool",  # Built-in module
        entrypoint_args=["--help"],
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    assert job_id is not None

    info = cluster.poll(job_id)
    assert info.job_id == job_id
    assert info.name == "test-simple-job"
    assert info.status in ["pending", "running", "succeeded", "failed"]


def test_cluster_list_jobs(cluster):
    request = JobRequest(
        name="list-test-job",
        entrypoint="json.tool",
        environment=create_environment(),
    )

    job_id = cluster.launch(request)

    # List jobs
    jobs = cluster.list_jobs()
    assert isinstance(jobs, list)
    assert len(jobs) >= 1
    assert any(job.job_id == job_id for job in jobs)


def test_cluster_poll_unknown_job(cluster):
    with pytest.raises(KeyError):
        cluster.poll(JobId("unknown-job-id-12345"))


def test_cluster_terminate(cluster, cluster_type):
    request = JobRequest(
        name="terminate-test-job",
        entrypoint="time",  # time module
        entrypoint_args=["sleep", "10"],
        environment=create_environment(),
    )

    job_id = cluster.launch(request)

    # Terminate waits for termination to complete
    cluster.terminate(job_id)

    info = cluster.poll(job_id)
    # Ray jobs may still be pending if they never started due to slow runtime env setup
    if cluster_type == "ray":
        assert info.status in ["pending", "stopped", "failed", "succeeded"]
    else:
        assert info.status in ["stopped", "failed", "succeeded"]


def test_cluster_job_success(cluster, cluster_type):
    request = JobRequest(
        name="success-test-job",
        entrypoint="json.tool",
        entrypoint_args=["--help"],
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    info = cluster.wait(job_id)

    assert info.status == "succeeded"
    assert info.error_message is None
    assert info.end_time is not None


def test_cluster_monitor_logs(cluster):
    request = JobRequest(
        name="log-test-job",
        entrypoint="json.tool",
        entrypoint_args=["--help"],
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    logs = list(cluster.monitor(job_id))
    assert len(logs) > 0


def test_ray_cluster_get_runtime_env(ray_cluster):
    request = JobRequest(
        name="runtime-env-test",
        entrypoint="json.tool",
        environment=create_environment(
            extra_dependency_groups=["cpu"],
            env_vars={"CUSTOM_VAR": "value"},
        ),
    )

    runtime_env = ray_cluster._get_runtime_env(request)

    # Should have env_vars and pip
    assert "env_vars" in runtime_env
    assert "PYTHONPATH" in runtime_env["env_vars"]
    assert "CUSTOM_VAR" in runtime_env["env_vars"]
    assert runtime_env["env_vars"]["CUSTOM_VAR"] == "value"
    assert "pip" in runtime_env


def test_ray_cluster_get_ray_resources_cpu(ray_cluster):
    request = JobRequest(
        name="cpu-resource-test",
        entrypoint="my_module",
        resources=ResourceConfig(),  # Default is CPU
    )

    resources = ray_cluster.get_ray_resources(request)
    assert resources == {}


def test_ray_cluster_get_ray_resources_gpu(ray_cluster):
    """GPU resources should map to Ray GPU resource."""

    request = JobRequest(
        name="gpu-resource-test",
        entrypoint="my_module",
        resources=ResourceConfig(device=GpuConfig(type="A100", count=4)),
    )

    resources = ray_cluster.get_ray_resources(request)
    assert resources == {"GPU": 4.0}


def test_ray_cluster_get_ray_resources_tpu(ray_cluster):
    """TPU resources should map to Ray TPU resources with head."""

    request = JobRequest(
        name="tpu-resource-test",
        entrypoint="my_module",
        resources=ResourceConfig(device=TpuConfig(type="v5e-16", count=8)),
    )

    resources = ray_cluster.get_ray_resources(request)
    assert resources == {"TPU": 8.0, "v5e-16-head": 1.0}


def test_ray_cluster_environment_variable_injection(ray_cluster):
    request = JobRequest(
        name="env-var-test",
        entrypoint="json.tool",
        environment=create_environment(
            env_vars={"TEST_VAR": "test_value"},
        ),
    )

    runtime_env = ray_cluster._get_runtime_env(request)

    # Check that our custom var is present
    assert "TEST_VAR" in runtime_env["env_vars"]
    assert runtime_env["env_vars"]["TEST_VAR"] == "test_value"

    # Check that default vars are present
    assert "HF_DATASETS_TRUST_REMOTE_CODE" in runtime_env["env_vars"]
    assert "TOKENIZERS_PARALLELISM" in runtime_env["env_vars"]


def test_ray_cluster_extra_dependency_groups(ray_cluster):
    request = JobRequest(
        name="deps-test",
        entrypoint="json.tool",
        environment=create_environment(
            extra_dependency_groups=["cpu", "eval"],
        ),
    )

    runtime_env = ray_cluster._get_runtime_env(request)
    assert "pip" in runtime_env
    assert "env_vars" in runtime_env


def test_ray_cluster_job_with_custom_environment(ray_cluster):
    request = JobRequest(
        name="custom-env-job",
        entrypoint="json.tool",
        entrypoint_args=["--help"],
        environment=create_environment(
            extra_dependency_groups=["cpu"],
            env_vars={"MY_CUSTOM_VAR": "my_value"},
        ),
    )

    job_id = ray_cluster.launch(request)

    # Verify job was created
    info = ray_cluster.poll(job_id)
    assert info.job_id == job_id
    assert info.name == "custom-env-job"
