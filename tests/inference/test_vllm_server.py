# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TPU detection in vllm_server."""

import pytest

from marin.inference import vllm_server

_TPU_ENV_VARS = (
    "TPU_NAME",
    "TPU_ACCELERATOR_TYPE",
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_MESH_CONTROLLER_ADDRESS",
    "TPU_VISIBLE_DEVICES",
)


@pytest.fixture
def clean_tpu_env(monkeypatch):
    """Clear all TPU-related env vars and stub glob to return no device nodes."""
    for key in _TPU_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(vllm_server.glob, "glob", lambda _pattern: [])
    return monkeypatch


def test_detect_tpu_via_tpu_name(clean_tpu_env):
    clean_tpu_env.setenv("TPU_NAME", "my-tpu")
    assert vllm_server._detect_tpu_environment() is True


def test_detect_tpu_via_dev_accel(clean_tpu_env):
    clean_tpu_env.setattr(
        vllm_server.glob,
        "glob",
        lambda pattern: ["/dev/accel0"] if pattern == "/dev/accel*" else [],
    )
    assert vllm_server._detect_tpu_environment() is True


def test_detect_tpu_via_dev_vfio(clean_tpu_env):
    clean_tpu_env.setattr(
        vllm_server.glob,
        "glob",
        lambda pattern: ["/dev/vfio/0"] if pattern == "/dev/vfio/*" else [],
    )
    assert vllm_server._detect_tpu_environment() is True


def test_vfio_control_node_alone_is_not_tpu(clean_tpu_env):
    clean_tpu_env.setattr(
        vllm_server.glob,
        "glob",
        lambda pattern: ["/dev/vfio/vfio"] if pattern == "/dev/vfio/*" else [],
    )
    assert vllm_server._detect_tpu_environment() is False


@pytest.mark.parametrize(
    "env_key",
    [
        "TPU_ACCELERATOR_TYPE",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_MESH_CONTROLLER_ADDRESS",
        "TPU_VISIBLE_DEVICES",
    ],
)
def test_detect_tpu_via_fallback_env_var(clean_tpu_env, env_key):
    clean_tpu_env.setenv(env_key, "1")
    assert vllm_server._detect_tpu_environment() is True


def test_no_tpu_signals(clean_tpu_env):
    assert vllm_server._detect_tpu_environment() is False
