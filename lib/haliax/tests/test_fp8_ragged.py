# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for per-tensor delayed-scaling FP8 ragged (grouped) matmul.

These exercise the XLA backend (the CPU fallback). The Triton f8 path is validated
separately on GPU. Numerics and gradients are checked against the bf16 ``ragged_dot``;
fp8's E4M3/E5M2 quantization should track bf16 to a few percent on well-scaled inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax._src.fp8_ragged import fp8_scaled_ragged_dot
from haliax._src.fp8_ragged_guards import (
    Fp8Contraction,
    assert_fp8_contraction,
    fp8_ragged_lowered_text,
    lowering_contains_fp8,
)
from haliax.nn.ragged_dot import ragged_dot

_AMAX_LEN = 1024


def _state():
    """Fresh per-tensor delayed-scaling state: unit scales, empty amax windows."""
    return dict(
        lhs_scale=jnp.ones(1, jnp.float32),
        rhs_scale=jnp.ones(1, jnp.float32),
        grad_scale=jnp.ones(1, jnp.float32),
        lhs_amax_history=jnp.zeros(_AMAX_LEN, jnp.float32),
        rhs_amax_history=jnp.zeros(_AMAX_LEN, jnp.float32),
        grad_amax_history=jnp.zeros(_AMAX_LEN, jnp.float32),
    )


def _inputs(M=128, D=64, N=96, G=4, seed=0):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((M, D)), jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((G, D, N)), jnp.bfloat16)
    # Group sizes sum to M so every token is assigned (mirrors a saturated MoE dispatch).
    counts = rng.multinomial(M, np.ones(G) / G)
    group_sizes = jnp.asarray(counts, jnp.int32)
    return lhs, rhs, group_sizes


def _fp8(lhs, rhs, group_sizes):
    return fp8_scaled_ragged_dot(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=jnp.bfloat16,
        quantize_compute_type=jnp.bfloat16,
        **_state(),
    )


def _rel_frob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)


def test_forward_tracks_bf16():
    lhs, rhs, group_sizes = _inputs()
    fp8_out = _fp8(lhs, rhs, group_sizes)
    bf16_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")

    assert fp8_out.shape == bf16_out.shape
    assert fp8_out.dtype == bf16_out.dtype
    # E4M3 quantization of both operands over a D=64 contraction: a few percent.
    assert _rel_frob(fp8_out, bf16_out) < 0.1


def test_backward_layouts_track_bf16():
    """grad_lhs (dlhs layout) and grad_rhs (drhs layout) on f8 operands track the bf16 grads."""
    lhs, rhs, group_sizes = _inputs()

    def fp8_loss(lhs, rhs):
        return jnp.sum(_fp8(lhs, rhs, group_sizes).astype(jnp.float32))

    def bf16_loss(lhs, rhs):
        return jnp.sum(
            ragged_dot(lhs, rhs, group_sizes, implementation="xla").astype(jnp.float32)
        )

    g_lhs_fp8, g_rhs_fp8 = jax.grad(fp8_loss, argnums=(0, 1))(lhs, rhs)
    g_lhs_bf16, g_rhs_bf16 = jax.grad(bf16_loss, argnums=(0, 1))(lhs, rhs)

    assert g_lhs_fp8.shape == lhs.shape
    assert g_rhs_fp8.shape == rhs.shape
    # E5M2 output-grad quantization is coarser than E4M3; allow a wider band than the forward.
    assert _rel_frob(g_lhs_fp8, g_lhs_bf16) < 0.2
    assert _rel_frob(g_rhs_fp8, g_rhs_bf16) < 0.2


def test_delayed_scaling_state_captures_amax():
    """The scale/amax-history args receive their delayed-scaling updates as (overwrite) gradients."""
    lhs, rhs, group_sizes = _inputs()

    def loss(state):
        y = fp8_scaled_ragged_dot(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=jnp.bfloat16,
            quantize_compute_type=jnp.bfloat16,
            **state,
        )
        return jnp.sum(y.astype(jnp.float32))

    state_grad = jax.grad(loss)(_state())

    # Newest history slot records this step's amax for each tensor.
    assert np.allclose(
        state_grad["lhs_amax_history"][0], float(jnp.max(jnp.abs(lhs))), rtol=1e-3
    )
    assert np.allclose(
        state_grad["rhs_amax_history"][0], float(jnp.max(jnp.abs(rhs))), rtol=1e-3
    )
    # Output cotangent of sum(y) is 1 everywhere a token is assigned, so the grad amax is 1.
    assert np.allclose(state_grad["grad_amax_history"][0], 1.0, rtol=1e-3)


def test_unpadded_tokens_not_a_multiple_of_512():
    """Token counts that are not a multiple of the pad granularity round-trip correctly."""
    lhs, rhs, group_sizes = _inputs(M=130)
    fp8_out = _fp8(lhs, rhs, group_sizes)
    bf16_out = ragged_dot(lhs, rhs, group_sizes, implementation="xla")
    assert fp8_out.shape == (130, rhs.shape[2])
    assert _rel_frob(fp8_out, bf16_out) < 0.1


@pytest.mark.parametrize("M", [128, 130])
def test_fp8_runs_under_jit(M):
    lhs, rhs, group_sizes = _inputs(M=M)
    out = jax.jit(_fp8)(lhs, rhs, group_sizes)
    assert out.shape == (M, rhs.shape[2])
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))


def test_assert_fp8_contraction_accepts_genuine_mixed_recipe():
    """The hybrid recipe: forward all-E4M3, both backward dots genuine mixed E5M2×E4M3."""
    a = jnp.ones((4, 4), jnp.float8_e4m3fn)
    g = jnp.ones((4, 4), jnp.float8_e5m2)
    assert_fp8_contraction(
        a, a, contraction=Fp8Contraction.FORWARD, grad_dtype=jnp.float8_e5m2
    )
    assert_fp8_contraction(
        g, a, contraction=Fp8Contraction.DLHS, grad_dtype=jnp.float8_e5m2
    )
    assert_fp8_contraction(
        a, g, contraction=Fp8Contraction.DRHS, grad_dtype=jnp.float8_e5m2
    )


def test_assert_fp8_contraction_rejects_bf16_fallback():
    """A backward operand that dequantized to bf16 (QDQ regression) must fail loudly."""
    a = jnp.ones((4, 4), jnp.float8_e4m3fn)
    g_bf16 = jnp.ones((4, 4), jnp.bfloat16)
    with pytest.raises(AssertionError, match="genuine f8 operands"):
        assert_fp8_contraction(
            g_bf16, a, contraction=Fp8Contraction.DLHS, grad_dtype=jnp.float8_e5m2
        )


def test_assert_fp8_contraction_rejects_all_e4m3_collapse():
    """An E5M2-grad recipe whose gradient collapsed to E4M3 is no longer the required mix."""
    a = jnp.ones((4, 4), jnp.float8_e4m3fn)
    with pytest.raises(AssertionError, match="all-E4M3 collapse"):
        assert_fp8_contraction(
            a, a, contraction=Fp8Contraction.DLHS, grad_dtype=jnp.float8_e5m2
        )


def test_forward_lowers_with_operands_still_fp8():
    """The compiled forward keeps f8 operands at the GEMM — not pre-dequantized to bf16."""
    lhs, rhs, group_sizes = _inputs()
    text = fp8_ragged_lowered_text(_fp8, lhs, rhs, group_sizes)
    assert lowering_contains_fp8(text)
