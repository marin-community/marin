# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import AttentionMask, attention, reference_attention

pytest.importorskip("jax_aiter.mha")
if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability != "gfx950":
    pytest.skip("JAX-AITER ships gfx950 kernels only", allow_module_level=True)

# (batch, seq_len, query heads, kv heads, head_dim) and mask. The small window is narrow enough that an off-by-one
# window fails the tolerance; the snowball cases are its long and short layers.
CASES = {
    "small_causal": ((2, 512, 8, 2, 128), AttentionMask.causal()),
    "small_window": ((2, 512, 8, 2, 128), AttentionMask.causal(sliding_window=16)),
    "snowball_causal": ((2, 4096, 20, 5, 128), AttentionMask.causal()),
    "snowball_window": ((2, 4096, 20, 5, 128), AttentionMask.causal(sliding_window=2048)),
}
# bf16 P and dS feed the kernels' matmuls, as in every flash kernel; the Pallas kernel's worst case is 3.5e-3.
TOLERANCE = 1e-2


def _qkv(shape):
    batch, seq_len, heads, kv_heads, head_dim = shape
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (batch, seq_len, heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (batch, seq_len, kv_heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (batch, seq_len, kv_heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    return q, k, v


def _reference(q, k, v, mask):
    with jax.default_matmul_precision("highest"):
        f32 = [x.astype(jnp.float32) for x in (q, k, v)]
        return reference_attention(*f32, mask, logits_dtype=jnp.float32)


def _aiter(q, k, v, mask):
    return attention(q, k, v, mask, implementation="gpu_aiter_flash")


@pytest.mark.parametrize("case", list(CASES))
def test_aiter_flash_forward_matches_reference(case):
    shape, mask = CASES[case]
    q, k, v = _qkv(shape)
    actual = _aiter(q, k, v, mask)
    expected = _reference(q, k, v, mask)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected), atol=TOLERANCE, rtol=TOLERANCE)


@pytest.mark.parametrize("case", list(CASES))
def test_aiter_flash_gradients_match_reference(case):
    shape, mask = CASES[case]
    q, k, v = _qkv(shape)
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v, mask).astype(jnp.float32) * cotangent)

    expected = jax.grad(lambda *a: loss(_reference, *a), argnums=(0, 1, 2))(q, k, v)
    actual = jax.grad(lambda *a: loss(_aiter, *a), argnums=(0, 1, 2))(q, k, v)
    for name, want, got in zip("qkv", expected, actual, strict=True):
        want = np.asarray(want, np.float32)
        got = np.asarray(got, np.float32)
        # Gradients of K/V sum over a GQA group and many queries, so compare relative to their scale.
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(got / scale, want / scale, atol=TOLERANCE, rtol=TOLERANCE, err_msg=f"d{name}")
