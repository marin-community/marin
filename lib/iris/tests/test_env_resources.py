# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.env_resources."""

import os

import pytest
from google.protobuf import json_format

from iris.env_resources import TaskResources, _read_iris_resource_proto, _read_proc_meminfo_total
from iris.rpc import job_pb2


@pytest.fixture(autouse=True)
def _clear_proto_cache():
    """Clear the cached proto between tests so env var changes take effect."""
    _read_iris_resource_proto.cache_clear()
    yield
    _read_iris_resource_proto.cache_clear()


def _make_resource_env(
    cpu_millicores: int = 0,
    memory_bytes: int = 0,
    gpu_count: int = 0,
    tpu_count: int = 0,
) -> str:
    """Build an IRIS_TASK_RESOURCES JSON string from a proto."""
    proto = job_pb2.ResourceSpecProto(
        cpu_millicores=cpu_millicores,
        memory_bytes=memory_bytes,
    )
    if gpu_count:
        proto.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=gpu_count))
    if tpu_count:
        proto.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4", count=tpu_count))
    return json_format.MessageToJson(proto, preserving_proto_field_name=True)


def test_from_environment_with_full_env(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_RESOURCES", _make_resource_env(cpu_millicores=4000, memory_bytes=8 * 1024**3))
    res = TaskResources.from_environment()
    assert res.memory_bytes == 8 * 1024**3
    assert res.cpu_cores == 4.0
    assert res.gpu_count == 0
    assert res.tpu_count == 0


def test_from_environment_with_gpu(monkeypatch):
    monkeypatch.setenv(
        "IRIS_TASK_RESOURCES", _make_resource_env(cpu_millicores=8000, memory_bytes=16 * 1024**3, gpu_count=4)
    )
    res = TaskResources.from_environment()
    assert res.gpu_count == 4
    assert res.tpu_count == 0


def test_from_environment_with_tpu(monkeypatch):
    monkeypatch.setenv(
        "IRIS_TASK_RESOURCES", _make_resource_env(cpu_millicores=8000, memory_bytes=16 * 1024**3, tpu_count=8)
    )
    res = TaskResources.from_environment()
    assert res.tpu_count == 8
    assert res.gpu_count == 0


def test_from_environment_falls_back_without_env(monkeypatch):
    """Without IRIS_TASK_RESOURCES, should fall back to OS-level detection."""
    monkeypatch.delenv("IRIS_TASK_RESOURCES", raising=False)
    res = TaskResources.from_environment()
    # Should get some positive values from the host
    assert res.cpu_cores > 0
    assert res.gpu_count == 0
    assert res.tpu_count == 0


def test_from_environment_partial_env_falls_back(monkeypatch):
    """When only GPU is specified, CPU/memory should fall back to OS-level."""
    monkeypatch.setenv("IRIS_TASK_RESOURCES", _make_resource_env(gpu_count=2))
    res = TaskResources.from_environment()
    # cpu_millicores and memory_bytes are 0 in the proto, so should fall back
    assert res.cpu_cores > 0
    assert res.gpu_count == 2


def test_malformed_env_falls_back(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_RESOURCES", "not-valid-json{{{")
    res = TaskResources.from_environment()
    # Should fall back to OS-level, not crash
    assert res.cpu_cores > 0
    assert res.gpu_count == 0


def test_read_proc_meminfo_total():
    """Smoke test: on Linux CI this should return a positive value."""
    total = _read_proc_meminfo_total()
    if os.path.exists("/proc/meminfo"):
        assert total is not None
        assert total > 0
    # On non-Linux, it's fine to return None
