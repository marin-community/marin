# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax._src.fp8_cast_transpose import quantize_amax_2d

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
