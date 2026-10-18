# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inkling relative position on plain GQA layers (``inkling_relpos`` without MLA), local sliding-window and global:
the banded bias equals a naive per-distance bias, the Pallas forward the GPU runs matches a naive GQA attention, and
the whole model stays causal."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.attention._fa4_cute import fa4_cute_segment_bounds
from levanter.kernels.pallas.parallax.pallas_gpu import softmax_forward

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import InklingRelPos, LocalMixer

_B, _S, _D, _HQ, _HKV, _HEAD = 2, 384, 32, 4, 2, 16
_WINDOW, _EXTENT, _REL_DIM = 256, 128, 4


def _naive_attention(q, k, v, relative_states, proj, window):
    """Softmax attention with GQA, a causal (optionally sliding) mask and ``bias[b,h,i,j] = R[b,h,i] . proj[:, i-j]``
    for ``0 <= i - j < extent``, written out per distance."""
    q, k, v, relative_states, proj = (np.asarray(x, np.float64) for x in (q, k, v, relative_states, proj))
    s, hq, hkv = q.shape[1], q.shape[2], k.shape[2]
    extent = proj.shape[1]
    i, j = np.arange(s)[:, None], np.arange(s)[None, :]
    distance = i - j
    allowed = distance >= 0
    if window is not None:
        allowed &= distance < window
    per_distance = np.einsum("bhir,rl->bhil", relative_states, proj)  # [B, H, S, extent]
    bias = np.where(
        (distance >= 0) & (distance < extent),
        np.take_along_axis(
            per_distance, np.broadcast_to(np.clip(distance, 0, extent - 1), (*per_distance.shape[:2], s, s)), -1
        ),
        0.0,
    )
    k = np.repeat(k, hq // hkv, axis=2)
    v = np.repeat(v, hq // hkv, axis=2)
    scores = np.einsum("bihd,bjhd->bhij", q, k) / np.sqrt(q.shape[-1]) + bias
    scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return np.einsum("bhij,bjhd->bihd", weights, v)


def _inputs():
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    x = jax.random.normal(keys[0], (_B, _S, _D))
    q = jax.random.normal(keys[1], (_B, _S, _HQ, _HEAD))
    k = jax.random.normal(keys[2], (_B, _S, _HKV, _HEAD))
    v = jax.random.normal(keys[3], (_B, _S, _HKV, _HEAD))
    cfg = t._config(hidden_dim=_D, num_heads=_HQ, rel_dim=_REL_DIM, rel_extent=_EXTENT, initializer_std=0.3)
    with jax.set_mesh(t._mesh()):
        relpos = InklingRelPos.init(cfg, key=keys[4])
        x = reshard(x, P(("replica_dcn", "data", "expert"), None, None))
    relative_states = jnp.einsum("bsd,dk->bsk", x, relpos.r_proj).reshape(_B, _S, _HQ, _REL_DIM).transpose(0, 2, 1, 3)
    return x, q, k, v, relpos, relative_states


@pytest.mark.parametrize("window", [None, _WINDOW])
def test_banded_bias_matches_naive_gqa_attention(window):
    x, q, k, v, relpos, relative_states = _inputs()
    with jax.set_mesh(t._mesh()):
        band = np.asarray(relpos(x))
    band = jnp.asarray(band)  # outside the mesh: the attention runs unsharded
    out = attention(
        q, k, v, AttentionMask(is_causal=True, sliding_window=window), implementation="reference", rel_bias=band
    )
    expected = _naive_attention(q, k, v, relative_states, relpos.proj, window)
    np.testing.assert_allclose(np.asarray(out), expected, atol=2e-5, rtol=2e-5)


def _pallas_forward(q, k, v, band, window):
    mask = AttentionMask(is_causal=True, sliding_window=window)
    lower, valid = fa4_cute_segment_bounds(mask, batch_size=_B, seq_len=_S, sliding_window=window)
    out, _ = softmax_forward(
        q, k, v, lower, valid, band, scale=1.0 / np.sqrt(_HEAD), block_q=128, block_k=64, num_warps=4, interpret=True
    )
    return out


@pytest.mark.parametrize("window", [None, _WINDOW])
def test_pallas_forward_matches_naive_and_ignores_the_future(window):
    """The ``gpu_pallas_fa4`` forward kernel (interpreted) on GQA + sliding window + band; scaling every value
    and key after position ``cut`` by 1e6 leaves the outputs at and before ``cut`` unchanged."""
    x, q, k, v, relpos, relative_states = _inputs()
    with jax.set_mesh(t._mesh()):
        band = np.asarray(relpos(x))
    band = jnp.asarray(band)
    out = _pallas_forward(q, k, v, band, window)
    expected = _naive_attention(q, k, v, relative_states, relpos.proj, window)
    np.testing.assert_allclose(np.asarray(out), expected, atol=2e-5, rtol=2e-5)

    cut = 200
    future = (jnp.arange(_S) > cut)[None, :, None, None]
    stressed = _pallas_forward(q, jnp.where(future, k * 1e6, k), jnp.where(future, v * 1e6, v), band, window)
    np.testing.assert_array_equal(np.asarray(stressed[:, : cut + 1]), np.asarray(out[:, : cut + 1]))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(attn_res=True, attn_res_num_blocks=4, scan_layers=True),
    ],
)
def test_model_with_gqa_inkling_on_every_layer_is_causal(overrides):
    """Local sliding-window and global GQA layers, all with the Inkling bias (no RoPE): changing the tokens after
    ``cut`` leaves every logit at and before ``cut`` unchanged."""
    seq = 256
    base = dict(
        max_seq_len=seq,
        sliding_window=128,
        num_layers=8,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        local_mixer=LocalMixer.SLIDING_WINDOW,
        attn_res=False,
        inkling_relpos=True,
        rel_extent=64,
        rel_dim=4,
        moe_implementation="fixed_pooled_wave_all_to_all",
    )
    mesh, model = t._model(**{**base, **overrides})
    assert all(layer.attn.rel_pos is not None for layer in model.layers())
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, seq), 0, t._VOCAB)
    cut = 150
    changed = tokens.at[:, cut + 1 :].set((tokens[:, cut + 1 :] + 7) % t._VOCAB)
    with jax.set_mesh(mesh):
        logits_fn = eqx.filter_jit(lambda m, x: m.logits(x))
        logits, logits_changed = logits_fn(model, tokens), logits_fn(model, changed)
    np.testing.assert_allclose(np.asarray(logits_changed[:, : cut + 1]), np.asarray(logits[:, : cut + 1]), atol=1e-6)
    assert not np.allclose(np.asarray(logits_changed[:, cut + 1 :]), np.asarray(logits[:, cut + 1 :]))
