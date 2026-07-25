# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig

from experiments.grug import dispatch
from experiments.grug.moe.ngc_xla_plugin import copy_artifact


def _entrypoint(config: str) -> None:
    del config


def _capture_request(monkeypatch, resources: ResourceConfig):
    requests = []
    handle = SimpleNamespace(wait=lambda **kwargs: None)
    monkeypatch.setattr(
        dispatch,
        "current_client",
        lambda: SimpleNamespace(submit=lambda request: requests.append(request) or handle),
    )
    dispatch.dispatch_grug_training_run(
        run_id="ngc-test",
        config="config",
        local_entrypoint=_entrypoint,
        resources=resources,
        max_retries_failure=0,
    )
    assert len(requests) == 1
    return requests[0]


def test_dispatch_preserves_default_environment(monkeypatch) -> None:
    request = _capture_request(monkeypatch, ResourceConfig.with_gpu("GB200", count=4))

    assert request.environment.setup_scripts is None


def test_dispatch_forwards_scale_knobs_with_nvidia_jax_image(monkeypatch) -> None:
    monkeypatch.setenv("SCALE_A2A_GATHER_DISPATCH", "1")

    request = _capture_request(
        monkeypatch,
        ResourceConfig.with_gpu("GB200", count=4, image="nvcr.io/nvidia/jax:26.06-py3"),
    )

    assert request.environment.env_vars["SCALE_A2A_GATHER_DISPATCH"] == "1"


def test_dispatch_with_nvidia_jax_image_preserves_container_jax(monkeypatch) -> None:
    request = _capture_request(
        monkeypatch,
        ResourceConfig.with_gpu("GB200", count=4, image="nvcr.io/nvidia/jax:26.06-py3"),
    )

    assert request.max_retries_failure == 0
    assert request.environment.setup_scripts is not None
    script = request.environment.setup_scripts[0]
    excluded = set(re.findall(r"--no-install-package ([a-z0-9-]+)", script))
    assert {
        "jax",
        "jaxlib",
        "jax-cuda13-pjrt",
        "jax-cuda13-plugin",
        "cuda-python",
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
        "nvidia-nccl-cu13",
        "torch",
        "torchvision",
    } <= excluded
    assert "nvidia-cutlass-dsl" not in excluded
    assert "nvidia-cutlass-dsl-libs-base" not in excluded
    assert "nvidia-cutlass-dsl-libs-cu13" not in excluded
    assert "venv --system-site-packages" in script
    assert "--active --inexact" in script
    assert "--package marin-root" in script
    assert "nvidia_cutlass_dsl_libs_cu13-*.dist-info" in script
    assert 'test -d "$IRIS_VENV/lib/python3.12/site-packages/"nvidia_cutlass_dsl_libs_base-*.dist-info' in script
    assert 'assert jax.__file__.startswith("/opt/jax/")' in script
    assert 'assert jaxlib.__file__.startswith("/opt/jaxlibs/")' in script
    root_sync = script.split("--package marin-root", 1)[1].split('"$uv" sync', 1)[0]
    gpu_sync = script.split("--package marin-levanter", 1)[1]
    assert "--no-install-package torch" not in root_sync
    assert "--no-install-package torch" in gpu_sync
    assert "torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl" in script
    assert "assert torch.__file__.startswith(venv)" in script
    assert 'assert "+cpu" in torch.__version__' in script


def test_dispatch_with_ngc_xla_plugin_installs_verified_overlay(monkeypatch) -> None:
    artifact_uri = "s3://marin-us-east-02a/tmp/ttl=30d/cubin7421/fix/xla_cuda_plugin.so"
    artifact_sha256 = "a" * 64
    monkeypatch.setenv("MARIN_NGC_XLA_CUDA_PLUGIN_URI", artifact_uri)
    monkeypatch.setenv("MARIN_NGC_XLA_CUDA_PLUGIN_SHA256", artifact_sha256)

    request = _capture_request(
        monkeypatch,
        ResourceConfig.with_gpu("GB200", count=4, image="nvcr.io/nvidia/jax:26.06-py3"),
    )

    assert request.environment.setup_scripts is not None
    assert len(request.environment.setup_scripts) == 2
    assert artifact_uri in request.environment.setup_scripts[1]
    assert artifact_sha256 in request.environment.setup_scripts[1]
    plugin_path = "/opt/jaxlibs/jax_cuda13_pjrt/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
    assert plugin_path in request.environment.setup_scripts[1]
    assert "/usr/lib/aarch64-linux-gnu/nvshmem/13" in request.environment.env_vars["LD_LIBRARY_PATH"]
    assert "NGC CUDA PJRT plugin not present" in request.environment.setup_scripts[1]


def test_copy_ngc_xla_plugin_writes_only_hash_verified_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "overlay" / "xla_cuda_plugin.so"
    payload = b"patched-cuda-plugin"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    source.write_bytes(payload)

    copied_sha256 = copy_artifact(str(source), destination, expected_sha256)

    assert copied_sha256 == expected_sha256
    assert destination.read_bytes() == payload


def test_copy_ngc_xla_plugin_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.so"
    destination = tmp_path / "overlay" / "xla_cuda_plugin.so"
    source.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        copy_artifact(str(source), destination, "0" * 64)

    assert not destination.exists()
