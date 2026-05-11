# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import (
    AttentionMask,
    _jax_dot_product_attention,
    attention,
    gpu_cudnn_attention,
    gpu_xla_attention,
    reference_attention,
)
from levanter.grug.flex_attention import grug_flex_mask_mod, tokamax_flex_attention


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
    with pytest.raises(NotImplementedError, match="use gpu_flex_pallas"):
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


def test_attention_explicit_gpu_flex_pallas_requires_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("GPU behavior requires the Tokamax Pallas path on a GPU backend.")
    q, k, v = _make_qkv()
    with pytest.raises(NotImplementedError, match="requires the JAX GPU backend"):
        attention(q, k, v, AttentionMask.causal(), implementation="gpu_flex_pallas")


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


@pytest.mark.parametrize(
    "mask",
    [
        AttentionMask.causal(),
        AttentionMask.causal(sliding_window=3),
        AttentionMask().with_sliding_window(3),
        AttentionMask().with_segment_ids(
            jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32),
            jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        ),
        AttentionMask.causal(sliding_window=3).with_segment_ids(
            jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32),
            jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32),
        ),
    ],
)
def test_grug_flex_mask_mod_matches_attention_mask_materialization(mask):
    scores_shape = (2, 4, 6, 6)
    mask_mod = grug_flex_mask_mod(mask)
    assert mask_mod is not None

    actual = jnp.broadcast_to(mask_mod(scores_shape), scores_shape)
    expected = mask.materialize_mask(q_len=6, k_len=6)
    if expected.ndim == 2:
        expected = expected[None, None, :, :]
    else:
        expected = expected[:, None, :, :]
    expected = jnp.broadcast_to(expected, scores_shape)

    np.testing.assert_array_equal(actual, expected)


def test_grug_flex_mask_mod_none_stays_none():
    assert grug_flex_mask_mod(None) is None


def test_grug_flex_mask_mod_empty_attention_mask_stays_none():
    assert grug_flex_mask_mod(AttentionMask()) is None


def test_grug_flex_mask_mod_does_not_call_attention_mask_materialize(monkeypatch):
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(
        jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32),
        jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32),
    )
    expected = mask.materialize_mask(q_len=6, k_len=6)[:, None, :, :]

    def fail_materialize_mask(self, q_len, k_len):
        raise AssertionError("flex mask_mod should be index-derived, not materialized through AttentionMask")

    monkeypatch.setattr(AttentionMask, "materialize_mask", fail_materialize_mask)
    mask_mod = grug_flex_mask_mod(mask)
    assert mask_mod is not None

    actual = mask_mod((2, 4, 6, 6))
    np.testing.assert_array_equal(jnp.broadcast_to(actual, (2, 4, 6, 6)), jnp.broadcast_to(expected, (2, 4, 6, 6)))


@pytest.mark.parametrize(
    "mask",
    [
        AttentionMask.causal(),
        AttentionMask.causal(sliding_window=3),
        AttentionMask.causal(sliding_window=3).with_segment_ids(
            jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32),
            jnp.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]], dtype=jnp.int32),
        ),
    ],
)
def test_tokamax_flex_xla_attention_matches_reference(mask):
    pytest.importorskip("tokamax")
    q, k, v = _make_qkv()

    actual = tokamax_flex_attention(q, k, v, mask, implementation="xla")
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_tokamax_flex_attention_with_segments_does_not_call_materialize(monkeypatch):
    pytest.importorskip("tokamax")
    q, k, v = _make_qkv(batch=2, q_len=5, k_len=5, q_heads=4, kv_heads=2)
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(
        jnp.array([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]], dtype=jnp.int32)
    )
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    def fail_materialize_mask(self, q_len, k_len):
        raise AssertionError("Tokamax flex attention should use mask_mod, not AttentionMask.materialize_mask")

    monkeypatch.setattr(AttentionMask, "materialize_mask", fail_materialize_mask)
    actual = tokamax_flex_attention(q, k, v, mask, implementation="xla")

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_tokamax_flex_xla_attention_grad_matches_reference_with_segments():
    pytest.importorskip("tokamax")
    q, k, v = _make_qkv(batch=2, q_len=5, k_len=5, q_heads=4, kv_heads=2)
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(
        jnp.array([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]], dtype=jnp.int32)
    )
    cotangent = jax.random.normal(jax.random.PRNGKey(2), q.shape, dtype=jnp.float32)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out * cotangent)

    def flex_loss(q_arg, k_arg, v_arg):
        out = tokamax_flex_attention(q_arg, k_arg, v_arg, mask, implementation="xla")
        return jnp.sum(out * cotangent)

    actual = jax.grad(flex_loss, argnums=(0, 1, 2))(q, k, v)
    expected = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-4, rtol=2e-4)
