# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="CuTe fp8 GEMM only lowers on GPU")


def _rel_fro(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def _nonuniform_fp8(T, K, E, N, seed=0):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    parts = rng.multinomial(T, np.ones(E) / E)
    return lhs, rhs, jnp.asarray(parts, jnp.int32)


@gpu_only
def test_cute_ragged_dot_matches_reference():
    from haliax._src.ragged_dot_cute import cute_ragged_dot  # noqa: PLC0415

    lhs, rhs, gs = _nonuniform_fp8(512, 256, 8, 256)
    E4M3 = jnp.float8_e4m3fn
    a = (lhs / jnp.max(jnp.abs(lhs)) * 448.0).astype(E4M3)  # [T,K] E4M3
    b = (jnp.swapaxes(rhs, 1, 2) / jnp.max(jnp.abs(rhs)) * 448.0).astype(E4M3)  # [E,N,K] E4M3
    out_scale = jnp.ones((1,), jnp.float32)  # dequant handled in the test's ref
    out = cute_ragged_dot(a, b, gs, out_dtype=jnp.bfloat16, out_scale=out_scale)
    # Reference: dequantize operands (no scaling here) and contract in f32, same layout.
    ref = jax.lax.ragged_dot(a.astype(jnp.float32), jnp.swapaxes(b, 1, 2).astype(jnp.float32), gs)
    assert _rel_fro(out, ref) < 5e-2
