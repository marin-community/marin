# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.dispatch import _forwarded_env_vars


def test_forwarded_env_vars_includes_runtime_settings_and_excludes_dispatcher_platform(monkeypatch) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    monkeypatch.setenv("SCALE_MUON_SYRK", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("UNRELATED_SETTING", "excluded")

    forwarded = _forwarded_env_vars()

    assert forwarded["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
    assert forwarded["SCALE_MUON_SYRK"] == "1"
    assert "JAX_PLATFORMS" not in forwarded
    assert "UNRELATED_SETTING" not in forwarded
