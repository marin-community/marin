# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax
import jax.numpy as jnp
import pytest

from haliax.nn import ragged_dot

ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")


def _inputs():
    lhs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_ragged_dot_auto_prefers_xla_on_non_tpu(monkeypatch, backend):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []

    monkeypatch.setattr(jax, "default_backend", lambda: backend)

    def fake_megablox(lhs, rhs, group_sizes):
        calls.append("megablox")
        return jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    def fake_xla(lhs, rhs, group_sizes):
        calls.append("xla")
        return jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_megablox_impl", fake_megablox)
    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_xla_impl", fake_xla)

    out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert out.shape == (3, 5)
    assert calls == ["xla"]


def test_ragged_dot_auto_falls_back_to_xla_on_tpu(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(ragged_dot_module, "_HAS_WARNED_AUTO_FALLBACK", False)

    def fake_megablox(lhs, rhs, group_sizes):
        calls.append("megablox")
        raise RuntimeError("megablox unavailable")

    def fake_xla(lhs, rhs, group_sizes):
        calls.append("xla")
        return jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_megablox_impl", fake_megablox)
    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_xla_impl", fake_xla)

    with pytest.warns(RuntimeWarning, match="auto fallback"):
        out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert out.shape == (3, 5)
    assert calls == ["megablox", "xla"]


def test_ragged_dot_auto_does_not_swallow_unexpected_errors(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    def fake_megablox(lhs, rhs, group_sizes):
        calls.append("megablox")
        raise ValueError("unexpected failure")

    def fake_xla(lhs, rhs, group_sizes):
        calls.append("xla")
        return jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=lhs.dtype)

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_megablox_impl", fake_megablox)
    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_xla_impl", fake_xla)

    with pytest.raises(ValueError, match="unexpected failure"):
        ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert calls == ["megablox"]


def test_ragged_dot_platform_default_is_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    default_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")
    xla_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert jnp.allclose(default_out, xla_out, rtol=1e-5, atol=1e-5)
