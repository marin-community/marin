# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.dispatch import _forwarded_env_vars


def test_forwarded_env_vars_includes_xla_allocator_and_excludes_dispatcher_platform(monkeypatch) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    forwarded = _forwarded_env_vars()

    assert forwarded["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
    assert "JAX_PLATFORMS" not in forwarded
