# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax
import jax.numpy as jnp

from haliax.nn import ragged_dot

ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")


def _inputs():
    contraction_dim = 32
    output_dim = 17
    lhs = jnp.arange(3 * contraction_dim, dtype=jnp.float32).reshape(3, contraction_dim) / 100
    rhs = jnp.arange(2 * contraction_dim * output_dim, dtype=jnp.float32).reshape(2, contraction_dim, output_dim) / 100
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def test_ragged_dot_platform_default_is_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    default_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")
    xla_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert jnp.allclose(default_out, xla_out, rtol=1e-5, atol=1e-5)


def test_ragged_dot_platform_default_gradients_are_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    def loss(lhs, rhs, implementation):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation=implementation) ** 2)

    default_value, default_gradients = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs, "auto")
    xla_value, xla_gradients = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs, "xla")

    assert jnp.allclose(default_value, xla_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(default_gradients[0], xla_gradients[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(default_gradients[1], xla_gradients[1], rtol=1e-5, atol=1e-5)


def test_ragged_dot_gpu_auto_uses_triton_when_available(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    expected = jnp.full((lhs.shape[0], rhs.shape[2]), 17.0, dtype=lhs.dtype)
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)

    def triton_result(lhs, rhs, group_sizes):
        return jnp.full((lhs.shape[0], rhs.shape[2]), expected[0, 0], dtype=lhs.dtype)

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_triton_impl", triton_result)

    auto_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert jnp.array_equal(auto_out, expected)


def test_triton_default_block_sizes_use_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: True)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 256, 32)


def test_triton_default_block_sizes_keep_non_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: False)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 128, 32)


def test_triton_custom_vjp_routes_backward_through_triton_layouts(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls = []

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
    ):
        calls.append(ragged_dot_dimension_numbers)
        return jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)

    def triton_loss(lhs, rhs):
        return jnp.sum(ragged_dot_module._ragged_dot_triton_impl(lhs, rhs, group_sizes))

    def xla_loss(lhs, rhs):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation="xla"))

    triton_value, triton_grads = jax.value_and_grad(triton_loss, argnums=(0, 1))(lhs, rhs)
    xla_value, xla_grads = jax.value_and_grad(xla_loss, argnums=(0, 1))(lhs, rhs)

    assert jnp.allclose(triton_value, xla_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[0], xla_grads[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[1], xla_grads[1], rtol=1e-5, atol=1e-5)
    assert calls == [
        ragged_dot_module._DEFAULT_DIM_NUMS,
        ragged_dot_module._DLHS_DIM_NUMS,
        ragged_dot_module._DRHS_DIM_NUMS,
    ]
