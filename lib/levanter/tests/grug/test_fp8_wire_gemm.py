# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical gates for the fp8-wire expert GEMM before it is plumbed into the MoE path.

The design carries unscaled e4m3 activations over the dispatch collective and applies the
per-token scale to the GEMM *output* instead of its input, because XLA only folds an operand
scale into ``__cublas$lt$matmul$f8`` when that scale is a scalar. These tests check what has to hold
for that to be sound, independent of the plumbing: the row-linearity identity that lets the scale
move past the dot; error no worse than the input-side arrangement the earlier fp8-wire attempt
used; a per-expert scalar weight scale (which the rewriter requires) costing nothing against finer
granularity; and per-token scaling that never shares a scale across token rows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.ep_ragged_all_to_all import _wire_quantize

TOKENS = 512
HIDDEN = 256
FFN = 128


def _quantize_weight_per_expert(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-expert scalar weight scale: allowed by the rewriter and token-independent."""
    amax = jnp.max(jnp.abs(w.astype(jnp.float32)))
    scale = jnp.maximum(amax, 1e-12) / 448.0
    return (w.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn), scale


def _reference(x, w):
    return x.astype(jnp.float32) @ w.astype(jnp.float32)


def _output_side(x, w):
    """The proposed arrangement: unscaled fp8 into the dot, per-token scale on the output."""
    q, row_scale = _wire_quantize(x, jnp.float8_e4m3fn)
    wq, w_scale = _quantize_weight_per_expert(w)
    raw = q.astype(jnp.float32) @ wq.astype(jnp.float32)
    return raw * (row_scale * w_scale)[:, None]


def _input_side(x, w):
    """The earlier arrangement: dequantize the activations, then a bf16 dot."""
    q, row_scale = _wire_quantize(x, jnp.float8_e4m3fn)
    wq, w_scale = _quantize_weight_per_expert(w)
    return (q.astype(jnp.float32) * row_scale[:, None]) @ (wq.astype(jnp.float32) * w_scale)


@pytest.fixture
def inputs():
    key = jax.random.PRNGKey(0)
    x_key, w_key = jax.random.split(key)
    # Wide per-token dynamic range: per-token scaling has to earn its keep here.
    x = jax.random.normal(x_key, (TOKENS, HIDDEN), jnp.float32)
    x = x * jnp.exp(jax.random.uniform(x_key, (TOKENS, 1), minval=-4.0, maxval=4.0))
    w = jax.random.normal(w_key, (HIDDEN, FFN), jnp.float32) * 0.05
    return x, w


def test_row_linearity_moves_the_scale_past_the_dot(inputs):
    """Scaling the output rows equals scaling the input rows, exactly, in exact arithmetic."""
    x, w = inputs
    q, row_scale = _wire_quantize(x, jnp.float8_e4m3fn)
    wq, w_scale = _quantize_weight_per_expert(w)
    out_side = (q.astype(jnp.float32) @ wq.astype(jnp.float32)) * (row_scale * w_scale)[:, None]
    in_side = (q.astype(jnp.float32) * row_scale[:, None]) @ (wq.astype(jnp.float32) * w_scale)
    # Exact in exact arithmetic; in fp32 the two orderings differ only by accumulation rounding.
    deviation = float(jnp.max(jnp.abs(out_side - in_side)) / jnp.mean(jnp.abs(out_side)))
    assert deviation < 1e-4, f"scale-past-the-dot deviation {deviation:.2e} exceeds fp32 rounding"


def test_output_side_scaling_is_no_less_accurate_than_input_side(inputs):
    """Moving the scale past the dot must not cost accuracy against the fp32 reference."""
    x, w = inputs
    reference = _reference(x, w)
    scale = jnp.mean(jnp.abs(reference))
    out_err = float(jnp.mean(jnp.abs(_output_side(x, w) - reference)) / scale)
    in_err = float(jnp.mean(jnp.abs(_input_side(x, w) - reference)) / scale)
    assert out_err <= in_err * 1.01, f"output-side {out_err:.5f} worse than input-side {in_err:.5f}"


def test_per_expert_scalar_weight_scale_costs_nothing_against_finer_granularity(inputs):
    """The rewriter needs a scalar weight scale; check that granularity is not what limits error.

    A per-column scale is strictly more expressive than one scalar per expert. If the two land at
    the same error, the limit is e4m3's 3-bit mantissa rather than the scale granularity, and the
    scalar scale the fp8 GEMM path requires costs nothing.
    """
    x, w = inputs
    reference = _reference(x, w)
    denominator = jnp.mean(jnp.abs(reference))
    q, row_scale = _wire_quantize(x, jnp.float8_e4m3fn)

    wq, w_scale = _quantize_weight_per_expert(w)
    scalar_err = float(
        jnp.mean(jnp.abs((q.astype(jnp.float32) @ wq.astype(jnp.float32)) * (row_scale * w_scale)[:, None] - reference))
        / denominator
    )

    column_scale = jnp.maximum(jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0), 1e-12) / 448.0
    wq_column = (w.astype(jnp.float32) / column_scale).astype(jnp.float8_e4m3fn)
    column_err = float(
        jnp.mean(
            jnp.abs(
                (q.astype(jnp.float32) @ wq_column.astype(jnp.float32)) * row_scale[:, None] * column_scale[None, :]
                - reference
            )
        )
        / denominator
    )
    assert scalar_err <= column_err * 1.05, f"scalar weight scale {scalar_err:.5f} vs per-column {column_err:.5f}"


def test_quantization_scale_is_per_token_and_never_shared(inputs):
    """Each row's scale depends only on that row: the causal-safety invariant."""
    x, _ = inputs
    _, scale = _wire_quantize(x, jnp.float8_e4m3fn)
    perturbed = x.at[0].multiply(1000.0)
    _, perturbed_scale = _wire_quantize(perturbed, jnp.float8_e4m3fn)
    assert perturbed_scale[0] != scale[0]
    np.testing.assert_array_equal(np.asarray(perturbed_scale[1:]), np.asarray(scale[1:]))


def test_padding_rows_quantize_to_exact_zero():
    """Dropped/padded capacity slots must contribute exactly nothing to the GEMM."""
    x = jnp.zeros((8, HIDDEN), jnp.bfloat16).at[0].set(1.0)
    q, scale = _wire_quantize(x, jnp.float8_e4m3fn)
    dequantized = q.astype(jnp.float32) * scale[:, None]
    np.testing.assert_array_equal(np.asarray(dequantized[1:]), np.zeros((7, HIDDEN), np.float32))


def test_wire_cotangent_uses_e5m2_range():
    """The backward wire must survive cotangent magnitudes that overflow e4m3 (max 448)."""
    ct = jnp.full((4, HIDDEN), 5000.0, jnp.float32)
    q_e5m2, scale_e5m2 = _wire_quantize(ct, jnp.float8_e5m2)
    recovered = q_e5m2.astype(jnp.float32) * scale_e5m2[:, None]
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(ct), rtol=0.2)
    assert jnp.isfinite(recovered).all()


def test_straight_through_gradient_is_finite_and_shaped():
    """The QDQ pair must pass a finite cotangent of the input's shape straight through."""

    def quantize_dequantize(v):
        q, scale = _wire_quantize(v, jnp.float8_e4m3fn)
        return (q.astype(jnp.float32) * scale[..., None]).astype(v.dtype)

    x = jax.random.normal(jax.random.PRNGKey(1), (8, HIDDEN), jnp.float32)
    grad = jax.grad(lambda v: jnp.sum(quantize_dequantize(v) ** 2))(x)
    assert grad.shape == x.shape
    assert jnp.isfinite(grad).all()
    assert float(jnp.max(jnp.abs(grad))) > 0.0
