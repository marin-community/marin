# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import (
    AttentionMask,
    _flattened_packed_segment_seqlens_offsets,
    _jax_dot_product_attention,
    _packed_segment_ids_positions,
    _packed_segment_seqlens_offsets,
    attention,
    gpu_cudnn_attention,
    gpu_te_attention,
    gpu_xla_attention,
    reference_attention,
)


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


@pytest.mark.parametrize(
    "mask",
    [
        None,
        AttentionMask.causal(),
        AttentionMask.causal(sliding_window=3),
        AttentionMask().with_sliding_window(3),
    ],
)
def test_jax_dot_product_attention_matches_reference(mask):
    q, k, v = _make_qkv()

    actual = _jax_dot_product_attention(q, k, v, mask, implementation="xla")
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_jax_dot_product_attention_matches_reference_for_dense_bool_mask():
    q, k, v = _make_qkv(q_len=4, k_len=5)
    dense_mask = jnp.array(
        [
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
            ],
            [
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
        ],
        dtype=jnp.bool_,
    )

    actual = _jax_dot_product_attention(q, k, v, dense_mask, implementation="xla")
    expected = reference_attention(q, k, v, dense_mask, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_jax_dot_product_attention_grad_matches_reference_with_gqa():
    q, k, v = _make_qkv(batch=2, q_len=5, k_len=5, q_heads=4, kv_heads=1)
    mask = AttentionMask.causal(sliding_window=3)
    cotangent = jax.random.normal(jax.random.PRNGKey(1), q.shape, dtype=jnp.float32)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out * cotangent)

    def jax_loss(q_arg, k_arg, v_arg):
        out = _jax_dot_product_attention(q_arg, k_arg, v_arg, mask, implementation="xla")
        return jnp.sum(out * cotangent)

    actual = jax.grad(jax_loss, argnums=(0, 1, 2))(q, k, v)
    expected = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-4, rtol=2e-4)


def test_jax_dot_product_attention_rejects_segment_ids_without_materializing(monkeypatch):
    q, k, v = _make_qkv()
    mask = AttentionMask.causal().with_segment_ids(
        jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32)
    )

    def fail_materialize_mask(self, q_len, k_len):
        raise AssertionError("JAX SDPA should not materialize segment_ids masks")

    monkeypatch.setattr(AttentionMask, "materialize_mask", fail_materialize_mask)
    with pytest.raises(NotImplementedError, match="do not support segment_ids"):
        _jax_dot_product_attention(q, k, v, mask, implementation="xla")


def test_attention_explicit_gpu_cudnn_requires_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only failure behavior is covered off-GPU.")
    q, k, v = _make_qkv()
    with pytest.raises(RuntimeError, match="requires the JAX GPU backend"):
        attention(q, k, v, AttentionMask.causal(), implementation="gpu_cudnn")


def test_attention_explicit_gpu_xla_requires_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only failure behavior is covered off-GPU.")
    q, k, v = _make_qkv()
    with pytest.raises(RuntimeError, match="requires the JAX GPU backend"):
        attention(q, k, v, AttentionMask.causal(), implementation="gpu_xla")


def test_attention_explicit_gpu_te_requires_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("GPU behavior requires Transformer Engine on a GPU backend.")
    q, k, v = _make_qkv()
    mask = AttentionMask.causal(max_segments_per_seq=4).with_segment_ids(
        jnp.array([[0, 0, 1, 1, 2, -1], [10, 10, 11, 11, 12, -1]], dtype=jnp.int32)
    )
    with pytest.raises(RuntimeError, match="requires the JAX GPU backend"):
        attention(q, k, v, mask, implementation="gpu_te")


def test_packed_segment_seqlens_offsets_from_dynamic_loader_ids():
    segment_ids = jnp.array(
        [
            [37, 37, 42, 42, 42, -1],
            [1000, 1000, 2000, 2000, 3000, -1],
        ],
        dtype=jnp.int32,
    )

    @jax.jit
    def lengths_offsets(dynamic_segment_ids):
        return _packed_segment_seqlens_offsets(
            dynamic_segment_ids,
            batch_size=2,
            seq_len=6,
            max_segments_per_seq=4,
        )

    seqlens, offsets = lengths_offsets(segment_ids)

    np.testing.assert_array_equal(seqlens, jnp.array([[2, 3, -1, -1], [2, 2, 1, -1]], dtype=jnp.int32))
    np.testing.assert_array_equal(offsets, jnp.array([[0, 2, -1, -1, -1], [0, 2, 4, -1, -1]], dtype=jnp.int32))


def test_flattened_packed_segment_offsets_skip_inter_row_padding():
    segment_ids = jnp.array(
        [
            [37, 37, 42, 42, 42, -1],
            [1000, 1000, 2000, 2000, 3000, -1],
        ],
        dtype=jnp.int32,
    )

    @jax.jit
    def lengths_offsets(dynamic_segment_ids):
        return _flattened_packed_segment_seqlens_offsets(
            dynamic_segment_ids,
            batch_size=2,
            seq_len=6,
            max_segments_per_seq=4,
        )

    seqlens, offsets = lengths_offsets(segment_ids)

    np.testing.assert_array_equal(
        seqlens,
        jnp.array([[2, 3, 2, 2, 1, -1, -1, -1]], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        offsets,
        jnp.array([[0, 2, 6, 8, 10, 11, -1, -1, -1]], dtype=jnp.int32),
    )


def test_flattened_packed_segment_offsets_use_cumulative_fast_path_for_single_row():
    segment_ids = jnp.array([[37, 37, 42, 42, 42, -1]], dtype=jnp.int32)

    @jax.jit
    def lengths_offsets(dynamic_segment_ids):
        return _flattened_packed_segment_seqlens_offsets(
            dynamic_segment_ids,
            batch_size=1,
            seq_len=6,
            max_segments_per_seq=4,
        )

    seqlens, offsets = lengths_offsets(segment_ids)

    np.testing.assert_array_equal(seqlens, jnp.array([[2, 3, -1, -1]], dtype=jnp.int32))
    np.testing.assert_array_equal(offsets, jnp.array([[0, 2, 5, -1, -1]], dtype=jnp.int32))


def test_packed_segment_ids_positions_from_dynamic_loader_ids():
    segment_ids = jnp.array(
        [
            [37, 37, 42, 42, 42, -1],
            [1000, 1000, 2000, 2000, 3000, -1],
        ],
        dtype=jnp.int32,
    )

    @jax.jit
    def ids_positions(dynamic_segment_ids):
        return _packed_segment_ids_positions(dynamic_segment_ids, batch_size=2, seq_len=6)

    local_ids, positions = ids_positions(segment_ids)

    np.testing.assert_array_equal(local_ids, jnp.array([[1, 1, 2, 2, 2, 0], [1, 1, 2, 2, 3, 0]], dtype=jnp.int32))
    np.testing.assert_array_equal(positions, jnp.array([[0, 1, 0, 1, 2, 0], [0, 1, 0, 1, 0, 0]], dtype=jnp.int32))


def test_packed_segment_seqlens_offsets_rejects_too_many_segments():
    segment_ids = jnp.array([[37, 42, 105, -1]], dtype=jnp.int32)

    @jax.jit
    def lengths_offsets(dynamic_segment_ids):
        return _packed_segment_seqlens_offsets(
            dynamic_segment_ids,
            batch_size=1,
            seq_len=4,
            max_segments_per_seq=2,
        )

    with pytest.raises(Exception, match="packed segment count exceeds max_segments_per_seq"):
        jax.block_until_ready(lengths_offsets(segment_ids))


def test_gpu_xla_attention_matches_reference_without_segments():
    if jax.default_backend() != "gpu":
        pytest.skip("GPU XLA SDPA correctness requires a GPU backend.")
    q, k, v = _make_qkv(batch=2, q_len=7, k_len=7, q_heads=4, kv_heads=2)
    mask = AttentionMask.causal(sliding_window=4)

    actual = jax.jit(gpu_xla_attention)(q, k, v, mask)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


def test_gpu_cudnn_attention_matches_reference_without_segments():
    if jax.default_backend() != "gpu":
        pytest.skip("cuDNN SDPA correctness requires a GPU backend.")
    q, k, v = _make_qkv(batch=2, q_len=7, k_len=7, q_heads=4, kv_heads=2)
    mask = AttentionMask.causal(sliding_window=4)

    actual = jax.jit(gpu_cudnn_attention)(q, k, v, mask)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


def test_gpu_cudnn_attention_returns_value_dtype():
    if jax.default_backend() != "gpu":
        pytest.skip("cuDNN SDPA correctness requires a GPU backend.")
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)

    actual = gpu_cudnn_attention(q, k, v, AttentionMask.causal())

    assert actual.dtype == v.dtype


def test_gpu_te_attention_matches_reference_for_valid_dynamic_packed_segments():
    if jax.default_backend() != "gpu":
        pytest.skip("Transformer Engine correctness requires a GPU backend.")
    pytest.importorskip("transformer_engine")
    q, k, v = _make_qkv(batch=2, q_len=6, k_len=6, q_heads=4, kv_heads=2)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    segment_ids = jnp.array([[37, 37, 42, 42, 42, -1], [1000, 1000, 2000, 2000, 3000, -1]], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=4, max_segments_per_seq=4).with_segment_ids(segment_ids)

    actual = jax.jit(gpu_te_attention)(q, k, v, mask)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    valid = segment_ids >= 0

    np.testing.assert_allclose(
        jnp.where(valid[..., None, None], actual, expected),
        expected,
        atol=7e-2,
        rtol=7e-2,
    )

    cotangent = jax.random.normal(jax.random.PRNGKey(3), q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def te_loss(q_arg, k_arg, v_arg):
        out = gpu_te_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual_grads = jax.jit(jax.grad(te_loss, argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)
