# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax._src.ragged_dot_mgpu import mgpu_dwgrad, mgpu_ragged_dot
from haliax.nn import ragged_dot
from haliax.quantization import Fp8RaggedDotOp, _jax_supports_mixed_fp8_wgmma

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


def test_triton_default_block_sizes_use_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: True)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 256, 32)


def test_triton_default_block_sizes_keep_non_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: False)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 128, 32)


def test_triton_block_k_override(monkeypatch):
    monkeypatch.setenv("HALIAX_RAGGED_DOT_TRITON_BLOCK_K", "64")

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 128, 64)
    assert ragged_dot_module._triton_block_k(16) == 16


def test_triton_block_k_rejects_unsupported_value(monkeypatch):
    monkeypatch.setenv("HALIAX_RAGGED_DOT_TRITON_BLOCK_K", "48")

    with pytest.raises(ValueError, match="must be 32, 64, or 128"):
        ragged_dot_module._triton_block_k(5120)


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


def test_triton_fp32_weight_gradient_preserves_forward_and_input_gradient_dtype(monkeypatch):
    lhs = jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.bfloat16).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        output_dtype=None,
    ):
        output = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        if output_dtype is not None:
            output = output.astype(output_dtype)
        return output

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)

    def loss(lhs, rhs):
        output = ragged_dot_module._ragged_dot_triton_fp32_weight_gradient_impl(lhs, rhs, group_sizes)
        return jnp.sum(output.astype(jnp.float32))

    value, (lhs_gradient, rhs_gradient) = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs)

    assert value.dtype == jnp.float32
    assert lhs_gradient.dtype == jnp.bfloat16
    assert rhs_gradient.dtype == jnp.float32


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0, 2.0])
def test_triton_accumulating_weight_gradient_scales_only_prior_accumulator(monkeypatch, scale):
    lhs = jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4)
    rhs = jnp.arange(2 * 4 * 5, dtype=jnp.bfloat16).reshape(2, 4, 5)
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    accumulator = jnp.arange(rhs.size, dtype=jnp.float32).reshape(rhs.shape) / 10
    accumulation_scale = jnp.asarray(scale, dtype=jnp.float32)

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        output_dtype=None,
    ):
        output = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        if output_dtype is not None:
            output = output.astype(output_dtype)
        return output

    def fake_accumulating_pallas_call(lhs, rhs, group_sizes, accumulator, accumulation_scale):
        gradient = fake_triton_pallas_call(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_module._DRHS_DIM_NUMS,
            output_dtype=jnp.float32,
        )
        return gradient + accumulation_scale * accumulator

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)
    monkeypatch.setattr(
        ragged_dot_module,
        "_triton_ragged_contracting_dim_accumulating_pallas_call",
        fake_accumulating_pallas_call,
    )

    def loss(lhs, rhs, accumulator):
        output, token = ragged_dot_module._ragged_dot_triton_accumulating_weight_gradient_impl(
            lhs,
            rhs,
            group_sizes,
            accumulator,
        )
        return jnp.sum(output.astype(jnp.float32)) + accumulation_scale * token

    value, (lhs_gradient, rhs_gradient, accumulator_gradient) = jax.value_and_grad(loss, argnums=(0, 1, 2))(
        lhs,
        rhs,
        accumulator,
    )
    expected_output = fake_triton_pallas_call(lhs, rhs, group_sizes)
    expected_rhs_gradient = fake_triton_pallas_call(
        lhs,
        jnp.ones_like(expected_output),
        group_sizes,
        ragged_dot_module._DRHS_DIM_NUMS,
        output_dtype=jnp.float32,
    )

    assert value == jnp.sum(expected_output.astype(jnp.float32))
    assert lhs_gradient.dtype == jnp.bfloat16
    assert rhs_gradient.dtype == jnp.float32
    assert jnp.array_equal(rhs_gradient, expected_rhs_gradient + accumulation_scale * accumulator)
    assert jnp.array_equal(accumulator_gradient, jnp.zeros_like(accumulator))


# ---------------------------------------------------------------------------
# FP8 op dispatch tests (op=/implementation= routing, init contract)
# ---------------------------------------------------------------------------


def _fp8_inputs(T=64, K=128, E=4, N=128, seed=0):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    group_sizes = jnp.asarray(rng.multinomial(T, np.ones(E) / E), jnp.int32)  # non-uniform, sums to T
    return lhs, rhs, group_sizes


def test_op_with_explicit_implementation_raises():
    lhs, rhs, gs = _fp8_inputs()
    # Uniform rev_dtype so the op constructs on jax 0.10.x too; the dtype
    # recipe is irrelevant to the dispatch contract under test.
    op = Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e4m3fn)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ragged_dot(lhs, rhs, gs, implementation="xla", op=op)


def test_fp32_weight_gradient_requires_explicit_triton():
    lhs, rhs, group_sizes = _fp8_inputs()

    with pytest.raises(ValueError, match="requires implementation='triton'"):
        ragged_dot(lhs, rhs, group_sizes, fp32_weight_gradient=True)


@pytest.mark.skipif(_jax_supports_mixed_fp8_wgmma(), reason="guard only fires on jax < 0.11.0")
def test_init_mixed_rev_dtype_fails_fast_on_old_jax():
    # On jax without mixed-dtype wgmma (jax-ml/jax#38859) the default mixed
    # recipe must fail at init with an actionable error, not deep inside
    # Mosaic lowering on the first backward pass.
    with pytest.raises(ValueError, match="38859"):
        Fp8RaggedDotOp.init()
    # The uniform approximation stays constructible.
    Fp8RaggedDotOp.init(rev_dtype=jnp.float8_e4m3fn)


@pytest.mark.skipif(not _jax_supports_mixed_fp8_wgmma(), reason="mixed wgmma needs jax >= 0.11.0")
def test_init_defaults_to_mixed_backward():
    op = Fp8RaggedDotOp.init()
    assert jnp.dtype(op.fwd_dtype) == jnp.dtype(jnp.float8_e4m3fn)
    assert jnp.dtype(op.rev_dtype) == jnp.dtype(jnp.float8_e5m2)


# Dummy tile config: the dtype gates under test fire before any tile-shape or
# kernel work, so the values never matter.
_KERNEL_KW = dict(block_m=64, block_n=64, block_k=64, max_concurrent_steps=2, grid_block_n=1)


def test_mgpu_ragged_dot_rejects_non_fp8_mixed_dtypes():
    lhs = jnp.zeros((64, 64), jnp.bfloat16)
    rhs = jnp.zeros((2, 64, 64), jnp.float8_e4m3fn)
    gs = jnp.asarray([32, 32], jnp.int32)
    with pytest.raises(NotImplementedError, match="same dtype or both be FP8"):
        mgpu_ragged_dot(lhs, rhs, group_sizes=gs, **_KERNEL_KW)


def test_mgpu_ragged_dot_accepts_mixed_fp8_pair_past_dtype_gate():
    # k != k2 so the call dies on the later shape check -- reaching it proves
    # the e5m2 x e4m3 pair passed the dtype gate that used to reject any
    # mismatch, without needing a Hopper GPU to run the kernel.
    lhs = jnp.zeros((64, 64), jnp.float8_e5m2)
    rhs = jnp.zeros((2, 128, 64), jnp.float8_e4m3fn)
    gs = jnp.asarray([32, 32], jnp.int32)
    with pytest.raises(ValueError, match="must match"):
        mgpu_ragged_dot(lhs, rhs, group_sizes=gs, **_KERNEL_KW)


def test_mgpu_dwgrad_rejects_non_fp8_operands():
    lhs_t = jnp.zeros((64, 128), jnp.bfloat16)
    grad_t = jnp.zeros((64, 128), jnp.float8_e5m2)
    gs = jnp.asarray([64, 64], jnp.int32)
    with pytest.raises(NotImplementedError, match="expects FP8 operands"):
        mgpu_dwgrad(lhs_t, grad_t, group_sizes=gs, **_KERNEL_KW)
