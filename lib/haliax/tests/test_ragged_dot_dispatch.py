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


def test_ragged_dot_platform_default_is_close_to_xla_call():
    lhs, rhs, group_sizes = _inputs()

    default_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")
    xla_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert jnp.allclose(default_out, xla_out, rtol=1e-5, atol=1e-5)


def test_triton_kernel_traces_with_jax_0_9_pallas_memory_api_on_cpu_interpreter():
    if not ragged_dot_module._has_pallas_triton:
        pytest.skip("Pallas Triton backend is not available")

    lhs = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
    rhs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    lo = jnp.array(0, dtype=jnp.int32)
    hi = jnp.array(lhs.shape[0], dtype=jnp.int32)

    pallas_call = ragged_dot_module.pl.pallas_call(
        lambda a, b, lo, hi, out: ragged_dot_module._triton_ragged_dot_kernel(
            a, b, lo, hi, out, block_m=lhs.shape[0], block_k=lhs.shape[1]
        ),
        out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[1]), lhs.dtype),
        in_specs=[
            ragged_dot_module.pl.no_block_spec,
            ragged_dot_module.pl.no_block_spec,
            ragged_dot_module.pl.no_block_spec,
            ragged_dot_module.pl.no_block_spec,
        ],
        out_specs=ragged_dot_module.pl.no_block_spec,
        grid=(1,),
        interpret=True,
    )

    assert jnp.allclose(pallas_call(lhs, rhs, lo, hi), lhs @ rhs, rtol=1e-5, atol=1e-5)


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
