# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.env_resources."""

import io
import json
import os

import pytest
from iris.env_resources import TaskResources, _read_iris_resources


@pytest.fixture(autouse=True)
def _clear_resource_cache():
    _read_iris_resources.cache_clear()
    yield
    _read_iris_resources.cache_clear()


def _make_resource_env(
    cpu_millicores: int = 0,
    memory_bytes: int = 0,
    gpu_count: int = 0,
    tpu_count: int = 0,
) -> str:
    """Build the stable ResourceSpecProto JSON wire used by task environments."""
    value: dict[str, object] = {
        "cpu_millicores": cpu_millicores,
        "memory_bytes": str(memory_bytes),
        "disk_bytes": "0",
    }
    if gpu_count:
        value["device"] = {"gpu": {"variant": "H100", "count": gpu_count}}
    if tpu_count:
        value["device"] = {"tpu": {"variant": "v4", "topology": "", "count": tpu_count}}
    return json.dumps(value)


@pytest.fixture
def host_resources(monkeypatch):
    real_open = open

    def fake_open(path, *args, **kwargs):
        path_string = str(path)
        if path_string.startswith("/sys/fs/cgroup/"):
            raise FileNotFoundError(path_string)
        if path_string == "/proc/meminfo":
            return io.StringIO("MemTotal:       16777216 kB\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)


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


def test_from_environment_falls_back_without_env(monkeypatch, host_resources):
    """Without IRIS_TASK_RESOURCES, should fall back to OS-level detection."""
    monkeypatch.delenv("IRIS_TASK_RESOURCES", raising=False)
    res = TaskResources.from_environment()
    assert res.cpu_cores == 12
    assert res.memory_bytes == 16 * 1024**3
    assert res.gpu_count == 0
    assert res.tpu_count == 0


def test_from_environment_partial_env_falls_back(monkeypatch, host_resources):
    """When only GPU is specified, CPU/memory should fall back to OS-level."""
    monkeypatch.setenv("IRIS_TASK_RESOURCES", _make_resource_env(gpu_count=2))
    res = TaskResources.from_environment()
    assert res.cpu_cores == 12
    assert res.memory_bytes == 16 * 1024**3
    assert res.gpu_count == 2


def test_malformed_env_falls_back(monkeypatch, host_resources):
    monkeypatch.setenv("IRIS_TASK_RESOURCES", "not-valid-json{{{")
    res = TaskResources.from_environment()
    assert res.cpu_cores == 12
    assert res.memory_bytes == 16 * 1024**3
    assert res.gpu_count == 0
