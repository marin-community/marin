# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig

from experiments.grug import dispatch
from experiments.grug.te_setup import transformer_engine_build_env, transformer_engine_setup_scripts


def _noop(_: object) -> None:
    pass


def _capture_submitted_request(monkeypatch) -> list:
    submitted = []
    job = SimpleNamespace(wait=lambda **_: None)
    client = SimpleNamespace(submit=lambda request: submitted.append(request) or job)
    monkeypatch.setattr(dispatch, "current_client", lambda: client)
    return submitted


def test_dispatch_forwards_allocator_environment(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "libjemalloc.so.2")
    monkeypatch.setenv("MALLOC_CONF", "background_thread:true,narenas:2")
    submitted = _capture_submitted_request(monkeypatch)

    dispatch.dispatch_grug_training_run(
        run_id="allocator-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
    )

    assert submitted[0].environment.env_vars["LD_PRELOAD"] == "libjemalloc.so.2"
    assert submitted[0].environment.env_vars["MALLOC_CONF"] == "background_thread:true,narenas:2"


def test_dispatch_installs_transformer_engine_when_asked(monkeypatch):
    submitted = _capture_submitted_request(monkeypatch)

    dispatch.dispatch_grug_training_run(
        run_id="te-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
        extra_env_vars=transformer_engine_build_env(),
        setup_scripts=transformer_engine_setup_scripts(),
    )

    environment = submitted[0].environment
    assert environment.env_vars["NVTE_CUDA_ARCHS"] == "100"
    assert environment.env_vars["CUDNN_FRONTEND_CUDART_LIB_NAME"] == "libcudart.so.13"
    # The TE install replaces the task's default setup, so both must be present.
    assert any("uv sync" in script for script in environment.setup_scripts)
    assert any("transformer_engine_jax" in script for script in environment.setup_scripts)


def test_dispatch_leaves_setup_alone_without_a_pjrt_wheel(monkeypatch):
    monkeypatch.delenv(dispatch.PJRT_WHEEL_ENV, raising=False)
    submitted = _capture_submitted_request(monkeypatch)

    dispatch.dispatch_grug_training_run(
        run_id="stock-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
    )

    assert not submitted[0].environment.setup_scripts


def test_dispatch_installs_the_pjrt_wheel_override_when_the_env_names_one(monkeypatch):
    monkeypatch.setenv(dispatch.PJRT_WHEEL_ENV, "s3://bucket/jax_cuda13_pjrt-custom.whl")
    submitted = _capture_submitted_request(monkeypatch)

    dispatch.dispatch_grug_training_run(
        run_id="pjrt-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
    )

    scripts = submitted[0].environment.setup_scripts
    # The override replaces the task's default setup, so the standard sync has to be rebuilt alongside it.
    assert any("uv sync" in script for script in scripts)
    assert any("jax_cuda13_pjrt-custom.whl" in script for script in scripts)


def test_dispatch_rejects_pip_packages_that_setup_scripts_would_drop(monkeypatch):
    monkeypatch.delenv(dispatch.PJRT_WHEEL_ENV, raising=False)
    _capture_submitted_request(monkeypatch)

    with pytest.raises(ValueError, match="ignores pip_packages"):
        dispatch.dispatch_grug_training_run(
            run_id="te-test",
            config=object(),
            local_entrypoint=_noop,
            resources=ResourceConfig.with_cpu(),
            pip_packages=("transformer_engine_jax==2.17.1",),
            setup_scripts=transformer_engine_setup_scripts(),
        )
