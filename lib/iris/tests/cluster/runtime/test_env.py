# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_device_env_vars in iris.cluster.runtime.env."""

from iris.cluster.runtime.env import build_device_env_vars
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2


def _tpu_container_config(variant: str) -> ContainerConfig:
    """Create a ContainerConfig with TPU resources."""
    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=32000,
        memory_bytes=128 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            tpu=cluster_pb2.TpuDevice(variant=variant, count=4),
        ),
    )
    return ContainerConfig(
        image="test-image",
        entrypoint=cluster_pb2.RuntimeEntrypoint(),
        env={},
        resources=resources,
    )


def _cpu_container_config() -> ContainerConfig:
    """Create a ContainerConfig with CPU-only resources."""
    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=4000,
        memory_bytes=8 * 1024**3,
    )
    return ContainerConfig(
        image="test-image",
        entrypoint=cluster_pb2.RuntimeEntrypoint(),
        env={},
        resources=resources,
    )


def test_tpu_env_vars_include_accelerator_type():
    """TPU containers must set TPU_ACCELERATOR_TYPE for libtpu topology init."""
    config = _tpu_container_config("v4-8")
    env = build_device_env_vars(config)

    assert env["JAX_PLATFORMS"] == "tpu,cpu"
    assert env["PJRT_DEVICE"] == "TPU"
    assert env["JAX_FORCE_TPU_INIT"] == "1"
    assert env["TPU_ACCELERATOR_TYPE"] == "v4-8"


def test_tpu_env_vars_v5litepod():
    config = _tpu_container_config("v5litepod-16")
    env = build_device_env_vars(config)

    assert env["TPU_ACCELERATOR_TYPE"] == "v5litepod-16"


def test_cpu_config_has_no_tpu_env_vars():
    config = _cpu_container_config()
    env = build_device_env_vars(config)

    assert "TPU_ACCELERATOR_TYPE" not in env
    assert "JAX_PLATFORMS" not in env


def test_no_resources_returns_empty():
    config = ContainerConfig(
        image="test-image",
        entrypoint=cluster_pb2.RuntimeEntrypoint(),
        env={},
        resources=None,
    )
    env = build_device_env_vars(config)
    assert env == {}
