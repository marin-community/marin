# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import AttentionMask, attention
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import SnowballConfig

S, HQ, HKV, D = 9, 4, 2, 12


def _bf16(key: int, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(key), shape).astype(jnp.bfloat16)


def _qkv(batch: int = 1) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _bf16(0, (batch, S, HQ, D)), _bf16(1, (batch, S, HKV, D)), _bf16(2, (batch, S, HKV, D))


def _torch_eager_reference(q, k, v, sliding_window):
    """An independent transcription of torch-order attention: the head-repeated einsum with FP32 softmax that
    matched torch eager on Snowball's first attention layer to 2 of 814,080 elements."""
    repeats = q.shape[2] // k.shape[2]
    k, v = jnp.repeat(k, repeats, axis=2), jnp.repeat(v, repeats, axis=2)
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q.astype(jnp.float32) / math.sqrt(q.shape[-1]),
        k.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    query_pos = jnp.arange(q.shape[1])[:, None]
    key_pos = jnp.arange(k.shape[1])[None, :]
    allowed = key_pos <= query_pos
    if sliding_window is not None:
        allowed = allowed & (key_pos >= query_pos - (sliding_window - 1))
    scores = jnp.where(allowed[None, None], scores, jnp.float32(-1e9))
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


@pytest.mark.parametrize("window", [4, None])
def test_torch_order_matches_torch_eager_rounding(window):
    q, k, v = _qkv(batch=2)
    mask = AttentionMask(is_causal=True, sliding_window=window)
    ours = attention(q, k, v, mask, implementation="torch_order")
    assert ours.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(_torch_eager_reference(q, k, v, window)))
    # A dense mask selects the reference backend by default on CPU, GPU, and TPU.
    dense_mask = jnp.broadcast_to(mask.materialize_mask(S, S), (q.shape[0], S, S))
    default = attention(q, k, v, dense_mask)
    reference = attention(q, k, v, dense_mask, implementation="reference")
    np.testing.assert_array_equal(np.asarray(default), np.asarray(reference))
    assert not np.array_equal(np.asarray(ours), np.asarray(default))


@pytest.mark.parametrize("window", [3, None])
def test_torch_order_keeps_packed_segments_apart(window):
    """A packed two-segment row: each segment's output equals that segment run alone, and changing one segment's
    keys and values leaves the other's output unchanged."""
    q, k, v = _qkv()
    split = 4
    segments = jnp.asarray([[0] * split + [1] * (S - split)])
    packed = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segments, segments))
    alone = AttentionMask(is_causal=True, sliding_window=window)
    out = attention(q, k, v, packed, implementation="torch_order")
    for lo, hi in ((0, split), (split, S)):
        single = attention(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], alone, implementation="torch_order")
        np.testing.assert_array_equal(np.asarray(out[:, lo:hi]), np.asarray(single))
    moved = attention(q, k.at[:, :split].set(3.0), v.at[:, :split].set(3.0), packed, implementation="torch_order")
    np.testing.assert_array_equal(np.asarray(out[:, split:]), np.asarray(moved[:, split:]))


@pytest.mark.parametrize("window", [3, None])
def test_torch_order_padded_single_document_matches_the_transcription(window):
    """The SFT row shape: one document with trailing pads (segment id -1). Real positions equal the dump's
    transcription on the document alone."""
    q, k, v = _qkv()
    real = 6
    segments = jnp.asarray([[0] * real + [-1] * (S - real)])
    mask = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segments, segments))
    out = attention(q, k, v, mask, implementation="torch_order")
    expected = _torch_eager_reference(q[:, :real], k[:, :real], v[:, :real], window)
    np.testing.assert_array_equal(np.asarray(out[:, :real]), np.asarray(expected))


def _tiny(**overrides) -> SnowballConfig:
    base = dict(
        vocab_size=48,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=5,
        num_heads=HQ,
        num_kv_heads=HKV,
        head_dim=D,
        max_seq_len=32,
        sliding_window=4,
        qk_mult=1.75,
        layer_norm_eps=1e-5,
        initializer_std=0.2,
    )
    return SnowballConfig(**(base | overrides))


def test_hf_reload_keeps_the_attention_implementation():
    requested = _tiny(attention_implementation="torch_order")
    reloaded = SnowballConfig.from_hf_config(requested.to_hf_config(requested.vocab_size))
    assert reloaded.attention_implementation is None
    restored = preserve_hf_runtime_settings(reloaded, requested)
    assert restored.attention_implementation == "torch_order"
