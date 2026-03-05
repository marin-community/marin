# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging

import jax
import jax.numpy as jnp

import haliax.nn.gmm as gmm_module
from haliax.nn import gmm_sharded


def _inputs():
    lhs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def test_gmm_auto_prefers_megablox_on_cpu(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

    def fake_megablox(lhs, rhs, group_sizes):
        calls.append("megablox")
        return jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    def fake_ragged(lhs, rhs, group_sizes):
        calls.append("ragged")
        return jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    monkeypatch.setattr(gmm_module, "_gmm_megablox_impl", fake_megablox)
    monkeypatch.setattr(gmm_module, "_gmm_ragged_impl", fake_ragged)

    out = gmm_sharded(lhs, rhs, group_sizes, implementation="auto")

    assert out.shape == (3, 5)
    assert calls == ["megablox"]


def test_gmm_auto_falls_back_on_gpu_triton_dynamic_grid(monkeypatch, caplog):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    def fake_megablox(lhs, rhs, group_sizes):
        calls.append("megablox")
        raise NotImplementedError("dynamic grid bounds not supported in the Triton backend")

    def fake_ragged(lhs, rhs, group_sizes):
        calls.append("ragged")
        return jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    monkeypatch.setattr(gmm_module, "_gmm_megablox_impl", fake_megablox)
    monkeypatch.setattr(gmm_module, "_gmm_ragged_impl", fake_ragged)
    gmm_module._warn_gpu_fallback_once.cache_clear()

    with caplog.at_level(logging.WARNING):
        out = gmm_sharded(lhs, rhs, group_sizes, implementation="auto")

    assert out.shape == (3, 5)
    assert calls == ["megablox", "ragged"]
    assert "falling back from megablox to ragged_dot_general on GPU" in caplog.text


def test_gmm_explicit_implementations_numerically_close():
    lhs, rhs, group_sizes = _inputs()

    megablox_out = gmm_sharded(lhs, rhs, group_sizes, implementation="megablox")
    ragged_out = gmm_sharded(lhs, rhs, group_sizes, implementation="ragged_dot_general")

    assert jnp.allclose(megablox_out, ragged_out, rtol=1e-5, atol=1e-5)
