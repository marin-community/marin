# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import levanter.grug.attention._fa4_cute as fa4_cute
import levanter.grug.attention._fa4_cute_backend as fa4_cute_backend
from levanter.grug.attention import (
    AttentionMask,
    gpu_fa4_cute_attention,
    reference_attention,
)
from levanter.grug.attention._fa4_cute import _simple_causal_lower_bounds


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


def test_packed_segment_backward_block_sparse_indices_are_q_direction():
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, -1]], dtype=jnp.int32)
    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=8,
        sliding_window=None,
    )

    mask_block_cnt, mask_block_idx = fa4_cute_backend._packed_segment_backward_block_sparse_indices(
        lower_bounds,
        valid,
        tile_m=2,
        tile_n=4,
    )

    np.testing.assert_array_equal(mask_block_cnt, jnp.array([[[2, 2]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        mask_block_idx,
        jnp.array([[[[0, 1, 0, 0], [2, 3, 0, 0]]]], dtype=jnp.int32),
    )


def test_packed_segment_backward_block_sparse_indices_split_full_blocks():
    segment_ids = jnp.zeros((1, 8), dtype=jnp.int32)
    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=8,
        sliding_window=None,
    )

    sparse_metadata = fa4_cute_backend._packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        tile_m=2,
        tile_n=2,
    )

    np.testing.assert_array_equal(sparse_metadata.partial_block_cnt, jnp.array([[[1, 1, 1, 1]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        sparse_metadata.partial_block_idx,
        jnp.array([[[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]]], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(sparse_metadata.full_block_cnt, jnp.array([[[3, 2, 1, 0]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        sparse_metadata.full_block_idx,
        jnp.array([[[[1, 2, 3, 0], [2, 3, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]]], dtype=jnp.int32),
    )


def test_fa4_frontend_rejects_mismatched_q_kv_segment_ids():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe validation requires a GPU backend.")
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    q_segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    kv_segment_ids = jnp.array([[1, 1, 3, 3]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    with pytest.raises(Exception, match="requires matching q/kv segment_ids"):
        jax.block_until_ready(gpu_fa4_cute_attention(q, k, v, mask))


def test_simple_causal_lower_bounds_match_sliding_window_semantics():
    lower_bounds, valid = _simple_causal_lower_bounds(batch_size=2, seq_len=6, sliding_window=3)

    np.testing.assert_array_equal(
        lower_bounds,
        np.array(
            [
                [0, 0, 0, 1, 2, 3],
                [0, 0, 0, 1, 2, 3],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(valid, np.ones((2, 6), dtype=np.bool_))


def test_simple_causal_lower_bounds_match_full_causal_semantics():
    lower_bounds, valid = _simple_causal_lower_bounds(batch_size=2, seq_len=4, sliding_window=None)

    np.testing.assert_array_equal(lower_bounds, np.zeros((2, 4), dtype=np.int32))
    np.testing.assert_array_equal(valid, np.ones((2, 4), dtype=np.bool_))


def _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent, *, valid_tokens=None):
    actual = jax.jit(gpu_fa4_cute_attention)(q, k, v, mask)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    if valid_tokens is not None:
        actual = jnp.where(valid_tokens[..., None, None], actual, expected)

    np.testing.assert_allclose(actual, expected, atol=7e-2, rtol=7e-2)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def fa4_loss(q_arg, k_arg, v_arg):
        out = gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual_grads = jax.jit(jax.grad(fa4_loss, argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)


@pytest.mark.parametrize(("q_heads", "kv_heads", "head_dim"), [(4, 1, 64), (2, 2, 64), (4, 1, 128)])
def test_real_gpu_fa4_cute_attention_matches_reference_for_valid_dynamic_packed_segments(q_heads, kv_heads, head_dim):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(4)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 64, q_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, kv_heads, head_dim), dtype=jnp.bfloat16)
    segment_ids = jnp.array(
        [[37] * 17 + [42] * 23 + [43] * 21 + [-1] * 3],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal(sliding_window=5).with_segment_ids(segment_ids)
    valid = segment_ids >= 0
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent, valid_tokens=valid)


def test_real_gpu_fa4_cute_attention_matches_reference_for_simple_sliding_mask():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(5)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (2, 64, 4, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (2, 64, 2, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (2, 64, 2, 64), dtype=jnp.bfloat16)
    mask = AttentionMask.causal(sliding_window=7)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)

    _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent)
