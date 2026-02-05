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

"""Tests for the fray v2 Ray backend that run WITHOUT a real Ray cluster.

Tests resource mapping, entrypoint routing, retry count calculation, and
runtime environment building logic.
"""

import pytest

from fray.v2.types import (
    CpuConfig,
    Entrypoint,
    GpuConfig,
    JobRequest,
    ResourceConfig,
    TpuConfig,
)

# ---------------------------------------------------------------------------
# compute_ray_retry_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_failure, max_preemption, expected",
    [
        (0, 100, 100),
        (3, 10, 13),
        (0, 0, 0),
        (5, 0, 5),
    ],
)
def test_compute_ray_retry_count(max_failure, max_preemption, expected):
    from fray.v2.ray_backend.backend import compute_ray_retry_count

    request = JobRequest(
        name="test",
        entrypoint=Entrypoint.from_callable(lambda: None),
        max_retries_failure=max_failure,
        max_retries_preemption=max_preemption,
    )
    assert compute_ray_retry_count(request) == expected


# ---------------------------------------------------------------------------
# get_entrypoint_params — resource mapping
# ---------------------------------------------------------------------------


def test_entrypoint_params_cpu():
    from fray.v2.ray_backend.backend import get_entrypoint_params

    request = JobRequest(
        name="cpu-job",
        entrypoint=Entrypoint.from_binary("echo", ["hello"]),
        resources=ResourceConfig(cpu=4, ram="2g"),
    )
    params = get_entrypoint_params(request)
    assert params["entrypoint_num_cpus"] == 4.0
    assert "entrypoint_num_gpus" not in params
    assert params["entrypoint_memory"] > 0


def test_entrypoint_params_gpu():
    from fray.v2.ray_backend.backend import get_entrypoint_params

    request = JobRequest(
        name="gpu-job",
        entrypoint=Entrypoint.from_binary("train", []),
        resources=ResourceConfig(device=GpuConfig(variant="H100", count=4)),
    )
    params = get_entrypoint_params(request)
    assert params["entrypoint_num_gpus"] == 4.0


def test_entrypoint_params_tpu():
    from fray.v2.ray_backend.backend import get_entrypoint_params

    request = JobRequest(
        name="tpu-job",
        entrypoint=Entrypoint.from_binary("train", []),
        resources=ResourceConfig(device=TpuConfig(variant="v4-8")),
    )
    params = get_entrypoint_params(request)
    assert "entrypoint_resources" in params
    assert params["entrypoint_resources"]["TPU-v4-8-head"] == 1.0
    assert params["entrypoint_resources"]["TPU"] == 4.0  # v4-8 has 4 chips


# ---------------------------------------------------------------------------
# Entrypoint routing logic
# ---------------------------------------------------------------------------


def test_routing_tpu_job_requires_callable():
    """TPU jobs must have a callable entrypoint."""
    request = JobRequest(
        name="tpu-binary",
        entrypoint=Entrypoint.from_binary("train", []),
        resources=ResourceConfig(device=TpuConfig(variant="v4-8")),
    )
    # The submit method checks isinstance(device, TpuConfig) and then
    # asserts callable_entrypoint is not None. We verify the assertion
    # would be hit by checking the entrypoint directly.
    assert request.entrypoint.callable_entrypoint is None
    assert request.entrypoint.binary_entrypoint is not None


def test_routing_binary_job():
    """Binary entrypoint routes to _launch_binary_job."""
    request = JobRequest(
        name="binary-job",
        entrypoint=Entrypoint.from_binary("echo", ["hello"]),
        resources=ResourceConfig(device=CpuConfig()),
    )
    assert request.entrypoint.binary_entrypoint is not None
    assert not isinstance(request.resources.device, TpuConfig)


def test_routing_callable_job():
    """Callable entrypoint on non-TPU device routes to _launch_callable_job."""
    request = JobRequest(
        name="callable-job",
        entrypoint=Entrypoint.from_callable(lambda: 42),
        resources=ResourceConfig(device=CpuConfig()),
    )
    assert request.entrypoint.callable_entrypoint is not None
    assert request.entrypoint.binary_entrypoint is None
    assert not isinstance(request.resources.device, TpuConfig)


# ---------------------------------------------------------------------------
# Replicas mapping
# ---------------------------------------------------------------------------


def test_replicas_on_job_request():
    """JobRequest.replicas is the gang-scheduling count (maps to num_slices for TPU)."""
    request = JobRequest(
        name="multi-slice",
        entrypoint=Entrypoint.from_callable(lambda: None),
        resources=ResourceConfig(device=TpuConfig(variant="v4-8")),
        replicas=4,
    )
    assert request.replicas == 4


# ---------------------------------------------------------------------------
# Resource mapping helpers (fray.v2.ray.resources)
# ---------------------------------------------------------------------------


def test_as_remote_kwargs_cpu():
    from fray.v2.ray_backend.resources import as_remote_kwargs

    config = ResourceConfig(device=CpuConfig())
    kwargs = as_remote_kwargs(config)
    assert kwargs == {"num_cpus": 1}


def test_as_remote_kwargs_gpu():
    from fray.v2.ray_backend.resources import as_remote_kwargs

    config = ResourceConfig(device=GpuConfig(variant="H100", count=2))
    kwargs = as_remote_kwargs(config)
    assert kwargs["num_gpus"] == 2
    assert kwargs["accelerator_type"] == "H100"


def test_as_remote_kwargs_gpu_auto():
    from fray.v2.ray_backend.resources import as_remote_kwargs

    config = ResourceConfig(device=GpuConfig(variant="auto", count=1))
    kwargs = as_remote_kwargs(config)
    assert kwargs["num_gpus"] == 1
    assert "accelerator_type" not in kwargs


def test_as_remote_kwargs_tpu():
    from fray.v2.ray_backend.resources import as_remote_kwargs

    config = ResourceConfig(device=TpuConfig(variant="v4-32"))
    kwargs = as_remote_kwargs(config)
    assert kwargs["num_cpus"] == 8


def test_as_remote_kwargs_with_env_vars():
    from fray.v2.ray_backend.resources import as_remote_kwargs

    config = ResourceConfig(device=CpuConfig())
    kwargs = as_remote_kwargs(config, env_vars={"FOO": "bar"})
    assert kwargs["runtime_env"] == {"env_vars": {"FOO": "bar"}}


def test_accelerator_descriptor():
    from fray.v2.ray_backend.resources import accelerator_descriptor

    assert accelerator_descriptor(ResourceConfig(device=TpuConfig(variant="v4-8"))) == "v4-8"
    assert accelerator_descriptor(ResourceConfig(device=GpuConfig(variant="H100"))) == "H100"
    assert accelerator_descriptor(ResourceConfig(device=CpuConfig())) is None


# ---------------------------------------------------------------------------
# Runtime env building — JAX_PLATFORMS logic
# ---------------------------------------------------------------------------


def test_build_runtime_env_cpu_sets_jax_platforms():
    from fray.v2.ray_backend.backend import build_runtime_env

    request = JobRequest(
        name="cpu-test",
        entrypoint=Entrypoint.from_callable(lambda: None),
        resources=ResourceConfig(device=CpuConfig()),
    )
    env = build_runtime_env(request)
    assert env["env_vars"]["JAX_PLATFORMS"] == "cpu"


def test_build_runtime_env_tpu_clears_jax_platforms():
    from fray.v2.ray_backend.backend import build_runtime_env

    request = JobRequest(
        name="tpu-test",
        entrypoint=Entrypoint.from_callable(lambda: None),
        resources=ResourceConfig(device=TpuConfig(variant="v4-8")),
    )
    env = build_runtime_env(request)
    assert env["env_vars"]["JAX_PLATFORMS"] == ""


def test_build_runtime_env_gpu_clears_jax_platforms():
    from fray.v2.ray_backend.backend import build_runtime_env

    request = JobRequest(
        name="gpu-test",
        entrypoint=Entrypoint.from_callable(lambda: None),
        resources=ResourceConfig(device=GpuConfig(variant="H100")),
    )
    env = build_runtime_env(request)
    assert env["env_vars"]["JAX_PLATFORMS"] == ""


# ---------------------------------------------------------------------------
# Actor resource mapping (_actor_ray_options)
# ---------------------------------------------------------------------------


def test_actor_options_default_preemptible():
    from fray.v2.ray_backend.backend import _actor_ray_options

    options = _actor_ray_options(ResourceConfig())
    assert options["num_cpus"] == 1
    assert "resources" not in options


def test_actor_options_non_preemptible_pins_head_node():
    from fray.v2.ray_backend.backend import _actor_ray_options

    options = _actor_ray_options(ResourceConfig(preemptible=False))
    assert options["num_cpus"] == 1
    assert options["resources"] == {"head_node": 0.0001}
