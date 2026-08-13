# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the SM100 RMS-GatedNorm reverse kernels against their pure-JAX references."""

import jax
import jax.numpy as jnp
import pytest
from levanter.grug._moe.rms_gated_norm import (
    exact_rms_backward_consumer,
    exact_rms_backward_producer_reference,
    exact_silu_backward_reference,
)

# The bridges import cutlass/quack at module scope; skip the file where they aren't installed.
quack_rms_cute = pytest.importorskip("levanter.grug._moe.quack_rms_cute")

_ROWS = 512
_HIDDEN_DIM = 1024
_RANK = 128
_NORM_EPS = 1e-5

# The kernels accumulate in float32 but round their outputs to BF16, and CUTLASS reduces the
# contraction in a different order than the cuBLAS call XLA lowers the reference to. Compare
# relative to the magnitude of the tensor; one BF16 ulp is ~2^-8 of the peak.
_BF16_TOLERANCE = 1e-2
_FLOAT32_TOLERANCE = 1e-4


def _require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("The QuACK RMS-GatedNorm reverse kernels require a GPU backend.")


def _assert_close(actual, expected, *, tolerance, label):
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(expected)), 1e-6)
    error = float(jnp.max(jnp.abs(actual - expected)) / scale)
    assert error <= tolerance, f"{label}: max relative error {error:.8g} exceeds {tolerance:.8g}"


def _producer_inputs():
    keys = jax.random.split(jax.random.key(23), 4)
    gate_preactivation_cotangent = jax.random.normal(keys[0], (_ROWS, _RANK), dtype=jnp.bfloat16)
    w_down = (0.1 * jax.random.normal(keys[1], (_HIDDEN_DIM, _RANK), dtype=jnp.float32)).astype(jnp.bfloat16)
    direct_cotangent = jax.random.normal(keys[2], (_ROWS, _HIDDEN_DIM), dtype=jnp.bfloat16)
    x = jax.random.normal(keys[3], (_ROWS, _HIDDEN_DIM), dtype=jnp.bfloat16)
    norm_weight = jnp.ones((_HIDDEN_DIM,), dtype=jnp.bfloat16)
    inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1) + _NORM_EPS)
    return gate_preactivation_cotangent, w_down, direct_cotangent, x, norm_weight, inverse_rms


def test_backward_producer_matches_reference():
    _require_gpu()

    args = _producer_inputs()
    _, w_down, _, x, norm_weight, inverse_rms = args
    unweighted_cotangent, row_dot_partial = quack_rms_cute.quack_coda_rms_backward_producer(*args)
    expected_unweighted, _ = exact_rms_backward_producer_reference(*args)

    _assert_close(unweighted_cotangent, expected_unweighted, tolerance=_BF16_TOLERANCE, label="unweighted cotangent")

    # Check the row reduction against the kernel's *own* unweighted cotangent so a GEMM rounding
    # difference doesn't leak into the comparison. Feeding it as the direct term with a zero rank
    # cotangent makes the reference's unweighted value identical by construction, leaving only the
    # float32 reduction order (per-tile partials plus a host sum) to differ.
    _, expected_row_dot_partial = exact_rms_backward_producer_reference(
        jnp.zeros_like(args[0]), w_down, unweighted_cotangent, x, norm_weight, inverse_rms
    )
    _assert_close(
        jnp.sum(row_dot_partial, axis=-1),
        jnp.sum(expected_row_dot_partial, axis=-1),
        tolerance=_FLOAT32_TOLERANCE,
        label="row dot",
    )


def test_backward_producer_row_partials_have_one_column_per_tile():
    _require_gpu()

    _, row_dot_partial = quack_rms_cute.quack_coda_rms_backward_producer(*_producer_inputs())

    tile_n = quack_rms_cute._DEFAULT_BACKWARD_TILE_MN[1]
    assert row_dot_partial.shape == (_ROWS, (_HIDDEN_DIM + tile_n - 1) // tile_n)
    assert row_dot_partial.dtype == jnp.float32


def test_backward_consumer_matches_reference():
    _require_gpu()

    args = _producer_inputs()
    _, _, _, x, norm_weight, inverse_rms = args
    unweighted_cotangent, row_dot_partial = quack_rms_cute.quack_coda_rms_backward_producer(*args)
    row_dot = jnp.sum(row_dot_partial, axis=-1)
    expected = exact_rms_backward_consumer(unweighted_cotangent, row_dot, x, norm_weight, inverse_rms)
    actual = quack_rms_cute.quack_coda_rms_backward_consumer(
        unweighted_cotangent, row_dot, x, norm_weight, inverse_rms
    )

    _assert_close(actual, expected, tolerance=_BF16_TOLERANCE, label="input cotangent")


def test_backward_producer_rejects_non_float32_inverse_rms():
    _require_gpu()

    *head, inverse_rms = _producer_inputs()
    with pytest.raises(ValueError, match="inverse_rms must be float32"):
        quack_rms_cute.quack_coda_rms_backward_producer(*head, inverse_rms.astype(jnp.bfloat16))


def test_silu_backward_gemm_matches_reference():
    _require_gpu()

    keys = jax.random.split(jax.random.key(29), 3)
    output_cotangent = jax.random.normal(keys[0], (_ROWS, _HIDDEN_DIM), dtype=jnp.bfloat16)
    w_up = (0.1 * jax.random.normal(keys[1], (_RANK, _HIDDEN_DIM), dtype=jnp.float32)).astype(jnp.bfloat16)
    preactivation = jax.random.normal(keys[2], (_ROWS, _RANK), dtype=jnp.bfloat16)

    preactivation_cotangent, postactivation = quack_rms_cute.quack_silu_backward_gemm(
        output_cotangent, w_up, preactivation
    )
    expected_cotangent, expected_postactivation = exact_silu_backward_reference(output_cotangent, w_up, preactivation)

    _assert_close(
        preactivation_cotangent, expected_cotangent, tolerance=_BF16_TOLERANCE, label="preactivation cotangent"
    )
    _assert_close(postactivation, expected_postactivation, tolerance=_BF16_TOLERANCE, label="postactivation")
