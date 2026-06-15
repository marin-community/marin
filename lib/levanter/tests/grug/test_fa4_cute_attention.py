# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import levanter.grug.attention._fa4_cute as fa4_cute
from levanter.grug.attention import (
    AttentionMask,
    Fa4CuteMetadata,
    gpu_fa4_cute_attention,
    reference_attention,
    with_fa4_cute_metadata,
)
from levanter.grug.attention._fa4_cute import (
    _causal_lower_bounds,
    _packed_segment_causal_lower_bounds,
    _packed_self_attention_segment_ids,
)


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


def test_fa4_frontend_uses_query_segment_ids_without_dynamic_equality_check():
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=1)
    del v
    q_segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    kv_segment_ids = jnp.array([[1, 1, 3, 3]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    actual = _packed_self_attention_segment_ids(q, k, mask, backend_name="gpu_fa4_cute_attention")

    np.testing.assert_array_equal(actual, q_segment_ids)


def test_fa4_cute_metadata_is_precomputed_per_sliding_window():
    segment_ids = jnp.array([[7, 7, 8, 8, 8, -1]], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids)

    mask = with_fa4_cute_metadata(mask, batch_size=1, seq_len=6)

    assert mask.fa4_cute_metadata is not None
    expected_lower_bounds, _ = _packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=6,
        sliding_window=3,
    )
    np.testing.assert_array_equal(mask.fa4_cute_metadata.lower_bounds, expected_lower_bounds)

    changed_window = mask.with_sliding_window(4)
    assert changed_window.fa4_cute_metadata is None


def test_fa4_cute_lower_bounds_encode_invalid_queries():
    seq_len = 6
    segment_ids = jnp.array([[7, 7, 8, 8, -1, -1]], dtype=jnp.int32)

    lower_bounds, valid = _packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=seq_len,
        sliding_window=3,
    )

    np.testing.assert_array_equal(valid, segment_ids >= 0)
    np.testing.assert_array_equal(lower_bounds[segment_ids < 0], jnp.full((2,), seq_len, dtype=jnp.int32))


def test_fa4_cute_metadata_supports_unsegmented_causal_masks():
    mask = AttentionMask.causal(sliding_window=3)

    mask = with_fa4_cute_metadata(mask, batch_size=2, seq_len=6)

    assert mask.fa4_cute_metadata is not None
    expected_lower_bounds, _ = _causal_lower_bounds(batch_size=2, seq_len=6, sliding_window=3)
    np.testing.assert_array_equal(mask.fa4_cute_metadata.lower_bounds, expected_lower_bounds)
    np.testing.assert_array_equal(
        mask.fa4_cute_metadata.lower_bounds,
        jnp.array([[0, 0, 0, 1, 2, 3], [0, 0, 0, 1, 2, 3]], dtype=jnp.int32),
    )

    changed_window = mask.with_sliding_window(4)
    assert changed_window.fa4_cute_metadata is None


def test_fa4_cute_full_sequence_window_matches_unwindowed_metadata():
    segment_ids = jnp.array([[0, 0, 1, 1, 1, -1], [2, 2, 2, 3, 3, 3]], dtype=jnp.int32)
    packed_mask = AttentionMask.causal().with_segment_ids(segment_ids)
    windowed_packed_mask = packed_mask.with_sliding_window(6)

    packed_metadata = with_fa4_cute_metadata(packed_mask, batch_size=2, seq_len=6).fa4_cute_metadata
    windowed_packed_metadata = with_fa4_cute_metadata(
        windowed_packed_mask,
        batch_size=2,
        seq_len=6,
    ).fa4_cute_metadata

    assert packed_metadata is not None
    assert windowed_packed_metadata is not None
    np.testing.assert_array_equal(windowed_packed_metadata.lower_bounds, packed_metadata.lower_bounds)

    causal_metadata = with_fa4_cute_metadata(AttentionMask.causal(), batch_size=2, seq_len=6).fa4_cute_metadata
    windowed_causal_metadata = with_fa4_cute_metadata(
        AttentionMask.causal(sliding_window=6),
        batch_size=2,
        seq_len=6,
    ).fa4_cute_metadata

    assert causal_metadata is not None
    assert windowed_causal_metadata is not None
    np.testing.assert_array_equal(windowed_causal_metadata.lower_bounds, causal_metadata.lower_bounds)


def test_fa4_cute_metadata_reuses_compatible_metadata():
    segment_ids = jnp.array([[7, 7, 8, 8, 8, -1]], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids)
    mask = with_fa4_cute_metadata(mask, batch_size=1, seq_len=6)

    reused = with_fa4_cute_metadata(mask, batch_size=1, seq_len=6)

    assert reused is mask


def test_fa4_cute_metadata_refreshes_stale_window_metadata():
    stale_lower_bounds = jnp.zeros((1, 6), dtype=jnp.int32)
    mask = AttentionMask(
        is_causal=True,
        fa4_cute_metadata=Fa4CuteMetadata(
            lower_bounds=stale_lower_bounds,
            sliding_window=4,
        ),
        sliding_window=3,
    )

    refreshed = with_fa4_cute_metadata(mask, batch_size=1, seq_len=6)

    assert refreshed.fa4_cute_metadata is not None
    np.testing.assert_array_equal(
        refreshed.fa4_cute_metadata.lower_bounds,
        jnp.array([[0, 0, 0, 1, 2, 3]], dtype=jnp.int32),
    )
    assert refreshed.fa4_cute_metadata.sliding_window == 3


def test_gpu_fa4_cute_attention_rejects_stale_metadata_window(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    q, k, v = _make_qkv(batch=1, q_len=6, k_len=6, q_heads=2, kv_heads=1)
    mask = AttentionMask(
        is_causal=True,
        fa4_cute_metadata=Fa4CuteMetadata(
            lower_bounds=jnp.zeros((1, 6), dtype=jnp.int32),
            sliding_window=4,
        ),
        sliding_window=3,
    )

    with pytest.raises(ValueError, match="metadata sliding_window"):
        gpu_fa4_cute_attention(q, k, v, mask)


def test_gpu_fa4_cute_attention_does_not_retrace_for_dynamic_segment_ids(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "_segmented_kernel_config", lambda head_dim: object())

    trace_count = 0

    def fake_attention_forward(q, k, v, lower_bounds, *, sm_scale, kernel_config):
        nonlocal trace_count
        del k, v, sm_scale, kernel_config
        trace_count += 1
        metadata_dependency = lower_bounds.astype(q.dtype)[..., None, None] * jnp.asarray(0, dtype=q.dtype)
        return q + metadata_dependency

    monkeypatch.setattr(fa4_cute, "fa4_cute_attention_forward", fake_attention_forward)
    q, k, v = _make_qkv(batch=1, q_len=6, k_len=6, q_heads=2, kv_heads=1)

    @jax.jit
    def run_attention(segment_ids):
        mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids)
        mask = with_fa4_cute_metadata(mask, batch_size=1, seq_len=6)
        return gpu_fa4_cute_attention(q, k, v, mask)

    first_segments = jnp.array([[0, 0, 1, 1, 1, -1]], dtype=jnp.int32)
    second_segments = jnp.array([[5, 5, 6, 6, 7, -1]], dtype=jnp.int32)

    np.testing.assert_array_equal(run_attention(first_segments), q)
    np.testing.assert_array_equal(run_attention(second_segments), q)
    np.testing.assert_array_equal(run_attention(first_segments), q)
    assert trace_count == 1


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(4, 1), (2, 2)])
def test_real_gpu_fa4_cute_attention_matches_reference_for_valid_dynamic_packed_segments(q_heads, kv_heads):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(4)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 64, q_heads, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, kv_heads, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, kv_heads, 64), dtype=jnp.bfloat16)
    segment_ids = jnp.array(
        [[37] * 17 + [42] * 23 + [43] * 21 + [-1] * 3],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal(sliding_window=5).with_segment_ids(segment_ids)

    actual = jax.jit(gpu_fa4_cute_attention)(q, k, v, mask)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    valid = segment_ids >= 0

    np.testing.assert_allclose(
        jnp.where(valid[..., None, None], actual, expected),
        expected,
        atol=7e-2,
        rtol=7e-2,
    )

    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

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
