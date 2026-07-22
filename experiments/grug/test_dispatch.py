# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

from fray.cluster import ResourceConfig

from experiments.grug import dispatch


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
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-nccl-cu13",
        "torch",
        "torchvision",
    } <= excluded
    assert "nvidia-cutlass-dsl" not in excluded
    assert "nvidia-cutlass-dsl-libs-cu13" not in excluded
    assert "venv --system-site-packages" in script
    assert "--active --inexact" in script
    assert "--package marin-root" in script
    assert "nvidia_cutlass_dsl_libs_cu13-*.dist-info" in script
    assert "nvidia_cutlass_dsl_libs_base-*.dist-info" in script
    assert 'assert jax.__file__.startswith("/opt/jax/")' in script
    assert 'assert jaxlib.__file__.startswith("/opt/jaxlibs/")' in script
    root_sync = script.split("--package marin-root", 1)[1].split('"$uv" sync', 1)[0]
    gpu_sync = script.split("--package marin-levanter", 1)[1]
    assert "--no-install-package torch" not in root_sync
    assert "--no-install-package torch" in gpu_sync
    assert "assert torch.__file__.startswith(venv)" in script
    assert 'assert "+cpu" in torch.__version__' in script
