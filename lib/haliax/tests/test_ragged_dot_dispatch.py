# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax
import jax.numpy as jnp

from haliax.nn import ragged_dot

ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")


def _inputs():
    lhs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def test_ragged_dot_auto_prefers_xla_on_cpu(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

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


def test_ragged_dot_auto_prefers_xla_on_gpu(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls: list[str] = []

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

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


def test_ragged_dot_platform_default_is_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    default_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")
    xla_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert jnp.allclose(default_out, xla_out, rtol=1e-5, atol=1e-5)
