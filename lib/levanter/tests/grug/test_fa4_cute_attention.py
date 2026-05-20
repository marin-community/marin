# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import (
    AttentionMask,
    attention,
    gpu_fa4_cute_attention,
    reference_attention,
)
from levanter.grug.attention import _fa4_cute as fa4_cute_lib


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


def _install_fake_fa4_backend(monkeypatch, fake_backend):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute_lib, "_gpu_compute_arch", lambda: 90)
    monkeypatch.setattr(fa4_cute_lib, "fa4_cute_attention_forward", fake_backend)


def test_fa4_dispatch_requires_gpu_backend_when_selected():
    if jax.default_backend() == "gpu":
        pytest.skip("GPU behavior requires the FA4/CuTe JAX FFI target.")
    q, k, v = _make_qkv()
    mask = AttentionMask.causal().with_segment_ids(
        jnp.array([[0, 0, 1, 1, 2, -1], [10, 10, 11, 11, 12, -1]], dtype=jnp.int32)
    )
    with pytest.raises(RuntimeError, match="requires the JAX GPU backend"):
        attention(q, k, v, mask, implementation="gpu_fa4_cute")


def test_fa4_frontend_uses_dynamic_segment_metadata_without_materializing(monkeypatch):
    q, k, v = _make_qkv(batch=2, q_len=10, k_len=10, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    segment_ids = jnp.array(
        [
            [5, 5, 5, 7, 7, 7, 7, 9, 9, -1],
            [100, 101, 101, 101, 102, 102, 103, 103, 103, -1],
        ],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids)
    calls = {}

    def fail_materialize_mask(_self, _q_len, _k_len):
        raise AssertionError("gpu_fa4_cute_attention should not materialize segment_ids masks")

    def fake_backend(q_arg, k_arg, v_arg, lower_bounds, valid, **_kwargs):
        calls["q_shape"] = q_arg.shape
        calls["k_shape"] = k_arg.shape
        calls["v_shape"] = v_arg.shape
        calls["lower_bounds"] = lower_bounds
        calls["valid"] = valid
        return q_arg.astype(v_arg.dtype)

    monkeypatch.setattr(AttentionMask, "materialize_mask", fail_materialize_mask)
    _install_fake_fa4_backend(monkeypatch, fake_backend)

    actual = gpu_fa4_cute_attention(q, k, v, mask)

    assert actual.shape == q.shape
    assert actual.dtype == v.dtype
    assert calls["q_shape"] == q.shape
    assert calls["k_shape"] == k.shape
    assert calls["v_shape"] == v.shape
    np.testing.assert_array_equal(
        calls["lower_bounds"],
        jnp.array(
            [
                [0, 0, 0, 3, 3, 3, 4, 7, 7, 10],
                [0, 1, 1, 1, 4, 4, 6, 6, 6, 10],
            ],
            dtype=jnp.int32,
        ),
    )
    np.testing.assert_array_equal(calls["valid"], segment_ids >= 0)


def test_fa4_frontend_rejects_mismatched_q_kv_segment_ids(monkeypatch):
    if jax.default_backend() == "tpu":
        pytest.skip("TPU runtime does not surface eqx.error_if as a Python exception.")
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    q_segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    kv_segment_ids = jnp.array([[1, 1, 3, 3]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(Exception, match="requires matching q/kv segment_ids"):
        jax.block_until_ready(gpu_fa4_cute_attention(q, k, v, mask))


def test_fa4_frontend_supports_mha(monkeypatch):
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=2)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(segment_ids)
    calls = {}

    def fake_backend(q_arg, k_arg, v_arg, lower_bounds, valid, **_kwargs):
        calls["q_shape"] = q_arg.shape
        calls["k_shape"] = k_arg.shape
        calls["v_shape"] = v_arg.shape
        calls["lower_bounds"] = lower_bounds
        calls["valid"] = valid
        return q_arg.astype(v_arg.dtype)

    _install_fake_fa4_backend(monkeypatch, fake_backend)

    actual = gpu_fa4_cute_attention(q, k, v, mask)

    assert actual.shape == q.shape
    assert calls["q_shape"] == q.shape
    assert calls["k_shape"] == k.shape
    assert calls["v_shape"] == v.shape
    np.testing.assert_array_equal(calls["lower_bounds"], jnp.array([[0, 0, 2, 2]], dtype=jnp.int32))
    np.testing.assert_array_equal(calls["valid"], jnp.ones_like(segment_ids, dtype=jnp.bool_))


def test_real_gpu_fa4_cute_attention_matches_reference_for_valid_dynamic_packed_segments():
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
