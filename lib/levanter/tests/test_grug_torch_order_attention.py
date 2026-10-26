# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.attention import AttentionMask, attention
from levanter.grug.attention import _core
from levanter.grug.sharding import compact_grug_mesh
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models import snowball
from levanter.models.snowball import SnowballAttention, SnowballConfig

S, HQ, HKV, D = 9, 4, 2, 12


def _bf16(key: int, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(key), shape).astype(jnp.bfloat16)


def _qkv(batch: int = 1) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _bf16(0, (batch, S, HQ, D)), _bf16(1, (batch, S, HKV, D)), _bf16(2, (batch, S, HKV, D))


def _l0_dump_transcription(q, k, v, sliding_window):
    """``torch_order_core`` from the #310 L0 attention dump (levanter_l0_attn.py at 440e9376), verbatim; it matched
    torch eager on the real model to 2 of 814,080 elements."""
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
def test_torch_order_equals_the_l0_dump_transcription(window):
    q, k, v = _qkv(batch=2)
    mask = AttentionMask(is_causal=True, sliding_window=window)
    ours = attention(q, k, v, mask, implementation="torch_order")
    assert ours.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(_l0_dump_transcription(q, k, v, window)))


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


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="gpu_fa4_cute needs a GPU")
def test_torch_order_masks_packed_segments_as_fa4_does():
    """On a GPU: the same packed two-segment row under torch_order and gpu_fa4_cute; neither lets one segment see the
    other, and the two backends agree within one BF16 step (they round P differently)."""
    q, k, v = (x.astype(jnp.bfloat16) for x in _qkv())
    q, k, v = (jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 128 - D))) for x in (q, k, v))
    split = 4
    segments = jnp.asarray([[0] * split + [1] * (S - split)])
    packed = AttentionMask(is_causal=True, sliding_window=None, segment_ids=(segments, segments))
    outputs = {}
    for name in ("torch_order", "gpu_fa4_cute"):
        out = attention(q, k, v, packed, implementation=name)
        moved = attention(q, k.at[:, :split].set(3.0), v.at[:, :split].set(3.0), packed, implementation=name)
        np.testing.assert_array_equal(np.asarray(out[:, split:]), np.asarray(moved[:, split:]))
        outputs[name] = np.asarray(out.astype(jnp.float32))
    np.testing.assert_allclose(outputs["torch_order"], outputs["gpu_fa4_cute"], rtol=2**-7, atol=2**-7)


@pytest.mark.parametrize("window", [3, None])
def test_torch_order_padded_single_document_matches_the_transcription(window):
    """The SFT row shape: one document with trailing pads (segment id -1). Real positions equal the dump's
    transcription on the document alone."""
    q, k, v = _qkv()
    real = 6
    segments = jnp.asarray([[0] * real + [-1] * (S - real)])
    mask = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segments, segments))
    out = attention(q, k, v, mask, implementation="torch_order")
    expected = _l0_dump_transcription(q[:, :real], k[:, :real], v[:, :real], window)
    np.testing.assert_array_equal(np.asarray(out[:, :real]), np.asarray(expected))


@pytest.mark.parametrize("window", [2048, None])
def test_torch_order_short_and_global_layer_masks(window):
    """The production short-layer window (2048) on a row longer than it, and the global (full causal) mask: equal to
    the transcription, and the window changes the last positions."""
    length = 2056
    q = _bf16(5, (1, length, 2, 8))
    k, v = _bf16(6, (1, length, 1, 8)), _bf16(7, (1, length, 1, 8))
    ours = attention(q, k, v, AttentionMask(is_causal=True, sliding_window=window), implementation="torch_order")
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(_l0_dump_transcription(q, k, v, window)))
    if window is not None:
        full = attention(q, k, v, AttentionMask(is_causal=True), implementation="torch_order")
        np.testing.assert_array_equal(np.asarray(ours[:, :window]), np.asarray(full[:, :window]))
        assert not np.array_equal(np.asarray(ours[:, window:]), np.asarray(full[:, window:]))


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="gpu_fa4_cute needs a GPU")
@pytest.mark.parametrize("window", [3, None])
def test_torch_order_padded_row_masks_as_fa4_does(window):
    """On a GPU: a padded single-document row; on real positions torch_order and gpu_fa4_cute agree within one BF16
    step, and neither depends on the pads."""
    q, k, v = (jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 128 - D))) for x in _qkv())
    real = 6
    segments = jnp.asarray([[0] * real + [-1] * (S - real)])
    mask = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segments, segments))
    outputs = {}
    for name in ("torch_order", "gpu_fa4_cute"):
        out = attention(q, k, v, mask, implementation=name)[:, :real]
        moved = attention(q, k.at[:, real:].set(3.0), v.at[:, real:].set(3.0), mask, implementation=name)[:, :real]
        np.testing.assert_array_equal(np.asarray(out), np.asarray(moved))
        outputs[name] = np.asarray(out.astype(jnp.float32))
    np.testing.assert_allclose(outputs["torch_order"], outputs["gpu_fa4_cute"], rtol=2**-7, atol=2**-7)


def _old_reference_math(q, k, v, mask, *, logits_dtype):
    """``_reference_attention_math`` before the mask code moved into ``_explicit_mask`` (23152e4d), for the
    unchanged-path check."""
    k = _core.align_kv_heads(k, num_q_heads=q.shape[2])
    v = _core.align_kv_heads(v, num_q_heads=q.shape[2])
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * (1.0 / math.sqrt(q.shape[-1])), k)
    explicit = mask.materialize_mask(scores.shape[-2], scores.shape[-1]) if isinstance(mask, AttentionMask) else mask
    if explicit is not None:
        if explicit.ndim == 2:
            explicit = explicit[None, :, :]
        explicit = explicit[:, None, :, :]
        if explicit.dtype == jnp.bool_:
            scores = jnp.where(explicit, scores, jnp.array(-1e9, dtype=scores.dtype))
        else:
            scores = scores + explicit
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v).astype(v.dtype)


def test_reference_path_is_unchanged():
    q, k, v = _qkv(batch=2)
    segments = jnp.asarray([[0] * 4 + [1] * 5, [0] * 9])
    masks = [
        None,
        AttentionMask(is_causal=True, sliding_window=4),
        AttentionMask(is_causal=True, segment_ids=(segments, segments)),
        jnp.tril(jnp.ones((S, S), dtype=jnp.bool_)),
        jnp.zeros((2, S, S), dtype=jnp.bfloat16),
    ]
    for mask in masks:
        new = _core._reference_attention_math(q, k, v, mask, logits_dtype=jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(new), np.asarray(_old_reference_math(q, k, v, mask, logits_dtype=jnp.float32))
        )


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


@pytest.mark.parametrize(
    ("configured", "backend", "passed"),
    [
        ("torch_order", "gpu", "torch_order"),
        (None, "gpu", "gpu_fa4_cute"),
        (None, "cpu", None),
    ],
)
def test_the_configured_implementation_reaches_the_attention_call(monkeypatch, configured, backend, passed):
    seen = []

    def record(q, k, v, mask, *, implementation=None):
        seen.append(implementation)
        return _core.reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    monkeypatch.setattr(snowball, "attention", record)
    monkeypatch.setattr(snowball.jax, "default_backend", lambda: backend)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        attn = SnowballAttention.init(_tiny(attention_implementation=configured), key=jax.random.PRNGKey(3))
        attn(_bf16(4, (1, S, 32)).astype(attn.w_q.dtype), AttentionMask(is_causal=True, sliding_window=4))
    assert seen == [passed]


def test_hf_reload_keeps_the_attention_implementation():
    requested = _tiny(attention_implementation="torch_order")
    reloaded = SnowballConfig.from_hf_config(requested.to_hf_config(requested.vocab_size))
    assert reloaded.attention_implementation is None
    restored = preserve_hf_runtime_settings(replace(reloaded, router_dot_fp32=False), requested, None)
    assert restored.attention_implementation == "torch_order"
