# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``head_probe``: per-token, per-query-head attention statistics from the dense baseline's softmax layers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import (
    HEAD_PROBE_FIELDS,
    HEAD_PROBE_STAT,
    HEAD_QCOS_STAT,
    LocalMixer,
    _head_attention_stats,
    head_probe,
)

_DENSE = dict(
    ngram_stat_rows=0,
    dense_mlp=True,
    attn_res=False,
    latent_dim=None,
    local_mixer=LocalMixer.SLIDING_WINDOW,
    num_heads=4,
    num_kv_heads=1,
    local_kv_heads=1,
    global_kv_heads=1,
    max_seq_len=16,
    sliding_window=8,
)


def _reference(q, k, allowed):
    """Brute-force softmax over ``allowed`` [B, Q, K], expanded to all query heads."""
    k = jnp.repeat(k, q.shape[2] // k.shape[2], axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(q.shape[-1])
    logits = jnp.where(allowed[:, None], logits, -jnp.inf)
    return np.asarray(jax.nn.softmax(logits, axis=-1))


def test_attention_stats_match_brute_force_with_documents():
    q = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 8))
    k = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 1, 8))
    segments = jnp.array([[0] * 6 + [1] * 10, [0] * 16])
    mask = AttentionMask.causal().with_segment_ids(segments, segments)
    with jax.set_mesh(t._mesh()):
        stats = np.asarray(_head_attention_stats(q, k, mask, None, segments))
    p = _reference(q, k, mask.materialize_mask(16, 16))
    starts = np.array([[0] * 6 + [6] * 10, [0] * 16])
    for b in range(2):
        for s in range(16):
            np.testing.assert_allclose(stats[b, s, :, 0], p[b, :, s, starts[b, s]], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(stats[b, s, :, 1], p[b, :, s, s], rtol=1e-5, atol=1e-6)
            ent = -np.sum(np.where(p[b, :, s] > 0, p[b, :, s] * np.log(np.maximum(p[b, :, s], 1e-30)), 0), -1)
            np.testing.assert_allclose(stats[b, s, :, 2], ent, rtol=1e-4, atol=1e-5)
    # The first token of a document can only attend to itself.
    np.testing.assert_allclose(stats[0, 6, :, :2], 1.0, rtol=1e-6)


def test_dense_model_records_every_layer_and_head():
    mesh, model = t._model(**_DENSE)
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, 16), 0, t._VOCAB)
    with jax.set_mesh(mesh), head_probe():
        _, metrics = eqx.filter_jit(lambda m: m(tokens))(model)
    stats = np.asarray(metrics[HEAD_PROBE_STAT])
    assert stats.shape == (model.config.num_layers, 2, 16, 4, len(HEAD_PROBE_FIELDS))
    assert np.all((stats[..., :2] >= 0) & (stats[..., :2] <= 1 + 1e-5))
    assert np.all(stats[..., 4:] >= 0) and float(stats[..., 5].max()) > 0
    with jax.set_mesh(mesh):
        _, plain = eqx.filter_jit(lambda m: m(tokens))(model)
    assert HEAD_PROBE_STAT not in plain
    qcos = np.asarray(metrics[HEAD_QCOS_STAT])  # [L, 2, H, H]
    assert qcos.shape == (model.config.num_layers, 2, 4, 4)
    np.testing.assert_allclose(np.diagonal(qcos[:, 0], axis1=-2, axis2=-1), 1.0, rtol=1e-4)
    np.testing.assert_allclose(qcos[:, 0], np.swapaxes(qcos[:, 0], -1, -2), atol=1e-5)
    assert np.all(qcos[:, 1] >= np.abs(qcos[:, 0]) - 1e-5)


def test_zeroing_a_heads_output_rows_silences_only_its_contribution():
    """The ablation the head-probe dump runs: zero one (layer, head)'s ``w_o`` rows."""
    mesh, model = t._model(**_DENSE)
    head_dim = model.config.inferred_head_dim
    w_o = model.stacked_blocks.stacked.attn.w_o
    rows = (jnp.arange(w_o.shape[1]) // head_dim) == 1
    zero = (jnp.arange(w_o.shape[0]) == 0)[:, None, None] & rows[None, :, None]
    ablated = eqx.tree_at(lambda m: m.stacked_blocks.stacked.attn.w_o, model, jnp.where(zero, 0.0, w_o))
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, t._VOCAB)
    with jax.set_mesh(mesh), head_probe():
        _, metrics = eqx.filter_jit(lambda m: m(tokens))(ablated)
    contrib = np.asarray(metrics[HEAD_PROBE_STAT])[..., HEAD_PROBE_FIELDS.index("contrib_norm")]
    assert float(np.abs(contrib[0, :, :, 1]).max()) == 0.0
    assert float(contrib[0, :, :, 0].min()) > 0 and float(contrib[1, :, :, 1].min()) > 0


def test_moe_baseline_records_head_stats_too():
    moe = {k: v for k, v in _DENSE.items() if k not in ("dense_mlp", "latent_dim")}
    mesh, model = t._model(**moe)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, t._VOCAB)
    with jax.set_mesh(mesh), head_probe():
        _, metrics = eqx.filter_jit(lambda m: m(tokens))(model)
    assert np.asarray(metrics[HEAD_PROBE_STAT]).shape == (model.config.num_layers, 2, 16, 4, len(HEAD_PROBE_FIELDS))
