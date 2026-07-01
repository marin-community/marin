# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax._src.fp8_cast_transpose import cast_transpose_amax_2d, cast_transpose_amax_3d, quantize_amax_2d
from haliax._src.fp8 import get_fp8_max

gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="fused quantize+amax only lowers on GPU")


@gpu_only
def test_fused_quantize_matches_naive_and_amax():
    """quantize_amax_2d output must be bit-identical to naive clip-and-cast, amax matches."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((256, 128)) * 3.0, jnp.bfloat16)
    inv = jnp.asarray([0.5], jnp.float32)
    q, amax = quantize_amax_2d(x, inv, jnp.float8_e4m3fn)
    naive = jnp.clip(x.astype(jnp.float32) * inv[0], -448.0, 448.0).astype(jnp.float8_e4m3fn)
    np.testing.assert_array_equal(np.asarray(q), np.asarray(naive))
    np.testing.assert_allclose(float(amax[0]), float(jnp.max(jnp.abs(x.astype(jnp.float32)))), rtol=1e-5)


@gpu_only
def test_cast_transpose_natural_equals_c4_and_transposed_is_T():
    """cast_transpose_amax_2d: natural == quantize_amax_2d bit-for-bit; transposed == natural.T; amax matches."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((256, 128)) * 3.0, jnp.bfloat16)
    inv = jnp.asarray([0.5], jnp.float32)
    q_nat, q_t, amax = cast_transpose_amax_2d(x, inv, jnp.float8_e4m3fn)
    q_ref, amax_ref = quantize_amax_2d(x, inv, jnp.float8_e4m3fn)  # natural-only reference kernel
    np.testing.assert_array_equal(np.asarray(q_nat), np.asarray(q_ref))  # natural bit-for-bit
    np.testing.assert_array_equal(np.asarray(q_t), np.asarray(q_ref).T)  # transposed == its transpose
    np.testing.assert_allclose(float(amax[0]), float(amax_ref[0]), rtol=1e-5)


@gpu_only
def test_cast_transpose_3d_natural_and_transposed():
    """cast_transpose_amax_3d: natural == naive quantize bit-for-bit; q_t[ei] == q_nat[ei].T; amax matches."""
    rng = np.random.default_rng(2)
    e, k, n = 4, 128, 64
    x = jnp.asarray(rng.standard_normal((e, k, n)) * 3.0, jnp.bfloat16)
    inv = jnp.asarray([0.5], jnp.float32)
    q_nat, q_t, amax = cast_transpose_amax_3d(x, inv, jnp.float8_e4m3fn)

    assert q_nat.shape == (e, k, n), f"natural shape mismatch: {q_nat.shape}"
    assert q_t.shape == (e, n, k), f"transposed shape mismatch: {q_t.shape}"

    # Natural output must match naive element-wise quantization bit-for-bit.
    dtype_max = float(get_fp8_max(jnp.float8_e4m3fn, jnp.float32))
    naive = jnp.clip(x.astype(jnp.float32) * inv[0], -dtype_max, dtype_max).astype(jnp.float8_e4m3fn)
    np.testing.assert_array_equal(np.asarray(q_nat), np.asarray(naive))

    # Transposed layout: q_t[ei] must equal q_nat[ei].T for every expert.
    q_nat_np = np.asarray(q_nat)
    q_t_np = np.asarray(q_t)
    for ei in range(e):
        np.testing.assert_array_equal(q_t_np[ei], q_nat_np[ei].T)

    # Amax: max absolute value of the float32-cast input across all experts.
    expected_amax = float(jnp.max(jnp.abs(x.astype(jnp.float32))))
    np.testing.assert_allclose(float(amax[0]), expected_amax, rtol=1e-5)
