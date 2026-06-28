# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness tests for the FP8 cast-transpose (TE-style dual-output quant).

``cast_transpose(x, scale)`` must return exactly ``(quantize(x), quantize(x).T)`` byte-for-byte:
the transposed output is a relabel of the same quantized values, never a second cast. These run on
CPU against the vanilla reference; the same assertions gate the Mosaic-GPU fast path on H100 (M2).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from haliax._src.fp8 import quantize
from haliax._src.fp8_cast_transpose import cast_transpose, cast_transpose_reference

# (M, K) grid: square, the real wgrad activation/grad shapes, and odd/non-square sizes to catch
# transpose-indexing bugs the square cases would hide.
_SHAPES = [(128, 128), (256, 64), (64, 256), (8192, 2048), (8192, 11264), (130, 97)]
_OUT_DTYPES = [jnp.float8_e4m3fn, jnp.float8_e5m2]


def _x(shape, seed):
    rng = np.random.default_rng(seed)
    # Spread across the f8 range (some clipping at low scale) so quantization isn't trivial.
    return jnp.asarray(rng.standard_normal(shape) * 4.0, jnp.bfloat16)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("out_dtype", _OUT_DTYPES)
def test_cast_transpose_matches_quantize_then_transpose(shape, out_dtype):
    x = _x(shape, seed=hash(shape) & 0xFFFF)
    scale = jnp.full(1, 0.5, jnp.float32)

    q, qt = cast_transpose(x, scale, out_dtype=out_dtype)
    q_ref = quantize(x, out_dtype, scale, jnp.bfloat16)

    # Rowwise output is the plain quantization; transposed output is exactly its transpose.
    assert q.dtype == out_dtype and qt.dtype == out_dtype
    assert q.shape == shape and qt.shape == (shape[1], shape[0])
    # f8 has no NaN-laden comparison hazards here; compare the raw bytes via uint8 view.
    np.testing.assert_array_equal(jnp.asarray(q).view(jnp.uint8), jnp.asarray(q_ref).view(jnp.uint8))
    np.testing.assert_array_equal(jnp.asarray(qt).view(jnp.uint8), jnp.asarray(q_ref.T).view(jnp.uint8))


@pytest.mark.parametrize("shape", _SHAPES)
def test_public_matches_reference(shape):
    """The public wrapper is bit-identical to the oracle — this is the gate M2's kernel must pass."""
    x = _x(shape, seed=1)
    scale = jnp.full(1, 1.0, jnp.float32)
    q, qt = cast_transpose(x, scale)
    q_ref, qt_ref = cast_transpose_reference(x, scale)
    np.testing.assert_array_equal(jnp.asarray(q).view(jnp.uint8), jnp.asarray(q_ref).view(jnp.uint8))
    np.testing.assert_array_equal(jnp.asarray(qt).view(jnp.uint8), jnp.asarray(qt_ref).view(jnp.uint8))


def test_rejects_non_2d():
    x = jnp.zeros((2, 3, 4), jnp.bfloat16)
    with pytest.raises(ValueError, match="2D"):
        cast_transpose(x, jnp.ones(1, jnp.float32))
