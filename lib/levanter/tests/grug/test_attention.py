# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import levanter.grug.attention as attention_lib
from levanter.grug.attention import (
    AttentionMask,
    _flattened_packed_segment_seqlens_offsets,
    _flash4_cute_kernel_config,
    _jax_dot_product_attention,
    _packed_segment_causal_lower_bounds,
    _packed_segment_ids_positions,
    _packed_segment_seqlens_offsets,
    _packed_segment_start_positions,
    attention,
    gpu_cudnn_attention,
    gpu_fa4_cute_attention,
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


def test_attention_explicit_gpu_fa4_cute_requires_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("GPU behavior requires the FA4/CuTe JAX FFI target.")
    q, k, v = _make_qkv()
    mask = AttentionMask.causal(max_segments_per_seq=4).with_segment_ids(
        jnp.array([[0, 0, 1, 1, 2, -1], [10, 10, 11, 11, 12, -1]], dtype=jnp.int32)
    )
    with pytest.raises(RuntimeError, match="requires the JAX GPU backend"):
        attention(q, k, v, mask, implementation="gpu_fa4_cute")


def test_gpu_fa4_cute_attention_uses_segment_lower_bounds_without_materializing(monkeypatch):
    q, k, v = _make_qkv(batch=1, q_len=6, k_len=6, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    segment_ids = jnp.array([[5, 5, 5, 7, 7, -1]], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=3, max_segments_per_seq=3).with_segment_ids(segment_ids)
    calls = {}

    def fail_materialize_mask(self, q_len, k_len):
        raise AssertionError("gpu_fa4_cute_attention should not materialize segment_ids masks")

    def fake_backend(q_arg, k_arg, v_arg, lower_bounds, valid, *, sm_scale, kernel_config):
        calls["q_shape"] = q_arg.shape
        calls["k_shape"] = k_arg.shape
        calls["v_shape"] = v_arg.shape
        calls["lower_bounds"] = lower_bounds
        calls["valid"] = valid
        calls["sm_scale"] = sm_scale
        calls["kernel_config"] = kernel_config
        return q_arg.astype(v_arg.dtype)

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(AttentionMask, "materialize_mask", fail_materialize_mask)
    monkeypatch.setattr(attention_lib, "_gpu_compute_arch", lambda: 90)
    monkeypatch.setattr(attention_lib, "fa4_cute_attention_forward", fake_backend)

    actual = attention_lib.gpu_fa4_cute_attention(q, k, v, mask)

    assert actual.shape == q.shape
    assert actual.dtype == v.dtype
    assert calls["q_shape"] == q.shape
    assert calls["k_shape"] == k.shape
    assert calls["v_shape"] == v.shape
    assert calls["kernel_config"].forward_tile == (128, 128)
    np.testing.assert_array_equal(calls["lower_bounds"], jnp.array([[0, 0, 0, 3, 3, 6]], dtype=jnp.int32))
    np.testing.assert_array_equal(calls["valid"], segment_ids >= 0)


def test_flash4_cute_kernel_config_uses_sm12x_nerfed_tiles_for_gb10():
    config = _flash4_cute_kernel_config(head_dim=128, head_dim_v=128, arch=121)

    assert config.forward_tile == (128, 64)
    assert config.backward_tile == (64, 64)
    assert config.num_threads == 128
    assert config.forward_num_stages == 1
    assert config.backward_num_stages_q == 1
    assert config.backward_num_stages_do == 1
    assert config.use_sm80_mma
    assert not config.allow_split_kv
    assert not config.allow_paged_kv
    assert not config.allow_block_sparse
    assert not config.allow_mask_mod_backward


def test_flash4_cute_kernel_config_keeps_sm120_wider_tiles_for_small_heads():
    config = _flash4_cute_kernel_config(head_dim=64, head_dim_v=64, arch=120)

    assert config.forward_tile == (128, 128)
    assert config.backward_tile == (64, 64)
    assert config.backward_num_stages_q == 2
    assert config.backward_num_stages_do == 2


def test_flash4_cute_kernel_config_uses_ampere_compatible_tiles_on_gh200():
    config = _flash4_cute_kernel_config(head_dim=128, head_dim_v=128, arch=90)

    assert config.forward_tile == (128, 64)
    assert config.num_threads == 128
    assert config.use_sm80_mma


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


def test_packed_segment_start_positions_from_dynamic_loader_ids():
    segment_ids = jnp.array(
        [
            [37, 37, 42, 42, 42, -1],
            [1000, 1000, 2000, 2000, 3000, -1],
        ],
        dtype=jnp.int32,
    )

    @jax.jit
    def start_positions(dynamic_segment_ids):
        return _packed_segment_start_positions(dynamic_segment_ids, batch_size=2, seq_len=6)

    starts = start_positions(segment_ids)

    np.testing.assert_array_equal(starts, jnp.array([[0, 0, 2, 2, 2, 6], [0, 0, 2, 2, 4, 6]], dtype=jnp.int32))


def test_packed_segment_causal_lower_bounds_from_non_block_aligned_dynamic_ids():
    segment_ids = jnp.array(
        [
            [5, 5, 5, 7, 7, 7, 7, 9, 9, -1],
            [100, 101, 101, 101, 102, 102, 103, 103, 103, -1],
        ],
        dtype=jnp.int32,
    )

    @jax.jit
    def bounds(dynamic_segment_ids):
        return _packed_segment_causal_lower_bounds(
            dynamic_segment_ids,
            batch_size=2,
            seq_len=10,
            sliding_window=3,
        )

    lower_bounds, valid = bounds(segment_ids)

    np.testing.assert_array_equal(
        lower_bounds,
        jnp.array(
            [
                [0, 0, 0, 3, 3, 3, 4, 7, 7, 10],
                [0, 1, 1, 1, 4, 4, 6, 6, 6, 10],
            ],
            dtype=jnp.int32,
        ),
    )
    np.testing.assert_array_equal(valid, segment_ids >= 0)


def test_packed_segment_causal_lower_bounds_without_sliding_window():
    segment_ids = jnp.array([[37, 37, 42, 42, 42, -1]], dtype=jnp.int32)

    @jax.jit
    def bounds(dynamic_segment_ids):
        return _packed_segment_causal_lower_bounds(
            dynamic_segment_ids,
            batch_size=1,
            seq_len=6,
            sliding_window=None,
        )

    lower_bounds, valid = bounds(segment_ids)

    np.testing.assert_array_equal(lower_bounds, jnp.array([[0, 0, 2, 2, 2, 6]], dtype=jnp.int32))
    np.testing.assert_array_equal(valid, jnp.array([[True, True, True, True, True, False]]))


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


def test_gpu_fa4_cute_attention_matches_reference_for_valid_dynamic_packed_segments():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(4)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 64, 4, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, 1, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, 1, 64), dtype=jnp.bfloat16)
    segment_ids = jnp.array(
        [[37] * 17 + [42] * 23 + [43] * 21 + [-1] * 3],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal(sliding_window=5, max_segments_per_seq=5).with_segment_ids(segment_ids)

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
