# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig

from experiments.grug import dispatch
from experiments.june_tpu_67b_a2b import dispatch as historical_dispatch


def _noop(_: object) -> None:
    pass


def test_dispatch_forwards_allocator_environment(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "libjemalloc.so.2")
    monkeypatch.setenv("MALLOC_CONF", "background_thread:true,narenas:2")
    submitted = []
    job = SimpleNamespace(wait=lambda **_: None)
    client = SimpleNamespace(submit=lambda request: submitted.append(request) or job)
    monkeypatch.setattr(dispatch, "current_client", lambda: client)

    dispatch.dispatch_grug_training_run(
        run_id="allocator-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
    )

    assert submitted[0].environment.env_vars["LD_PRELOAD"] == "libjemalloc.so.2"
    assert submitted[0].environment.env_vars["MALLOC_CONF"] == "background_thread:true,narenas:2"


@pytest.mark.parametrize("dispatch_module", [dispatch, historical_dispatch])
def test_dispatch_forwards_levanter_runtime_controls(monkeypatch, dispatch_module):
    monkeypatch.setenv("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    submitted = []
    job = SimpleNamespace(wait=lambda **_: None)
    monkeypatch.setattr(
        dispatch_module,
        "current_client",
        lambda: SimpleNamespace(submit=lambda request: submitted.append(request) or job),
    )
    monkeypatch.setattr(dispatch_module, "resolve_training_env", lambda base_env, resources: base_env)

    dispatch_module.dispatch_grug_training_run(
        run_id="runtime-control-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
    )

    env = submitted[0].environment.env_vars
    assert env["LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS"] == "0"
    assert "JAX_PLATFORMS" not in env
