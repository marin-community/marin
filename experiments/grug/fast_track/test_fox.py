# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FoX forget gate (``mla_forget_gate``) on the MLA layers: the q/k channel augmentation adds exactly the
per-document decay bias, and the gated model stays causal with independent packed documents."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask, reference_attention

import experiments.grug.fast_track.test_kda_local as kda_test
from experiments.grug.fast_track.model import (
    AttnResLayerBackward,
    _fox_augment,
    _fox_head_dim,
    _segment_cumsum,
)

_MLA = dict(
    mla=True,
    mla_kv_latent_dim=16,
    qk_norm=False,
    mla_k_norm=True,
    mla_q_norm=True,
    attn_res_layer_backward=AttnResLayerBackward.SAVE,
)


def test_fox_augment_adds_per_document_decay_bias():
    b, s, n, h = 2, 40, 3, 16
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    q, k, v = (jax.random.normal(key, (b, s, n, h), jnp.float32) for key in keys[:3])
    log_forget = jax.nn.log_sigmoid(jax.random.normal(keys[3], (b, s, n)))
    segment_ids = jnp.asarray(np.repeat([0, 1, 2], [13, 20, 7])[None].repeat(b, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids))
    c = _segment_cumsum(log_forget, segment_ids)

    padded = _fox_head_dim(h)
    with jax.set_mesh(kda_test._mesh()):
        qa, ka, va = _fox_augment(q * math.sqrt(padded / h), k, v, -c)
        got = reference_attention(qa, ka, va, mask, logits_dtype=jnp.float32)[..., :h]

    logits = jnp.einsum("bqnd,bknd->bnqk", q, k) / math.sqrt(h)
    logits = logits + jnp.transpose(c, (0, 2, 1))[..., :, None] - jnp.transpose(c, (0, 2, 1))[..., None, :]
    allowed = mask.materialize_mask(s, s)[:, None]
    weights = jax.nn.softmax(jnp.where(allowed, logits, -1e9), axis=-1)
    expected = jnp.einsum("bnqk,bknd->bqnd", weights, v)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-4, atol=1e-5)


def test_fox_bias_split_is_precise_in_bf16():
    """A 3-part bf16 split keeps a 4096-token-document cumsum (f = 0.5 everywhere) within ~1e-3 logits."""
    c = jnp.linspace(0.0, 4096 * math.log(2.0), 4096, dtype=jnp.float32).reshape(1, -1, 1)
    ones = jnp.ones((1, c.shape[1], 1, 128), jnp.bfloat16)
    with jax.set_mesh(kda_test._mesh()):
        qa, ka, _ = _fox_augment(ones, ones, ones, c)
    padded = _fox_head_dim(128)
    bias = jnp.sum(ka[..., 128:].astype(jnp.float32) * qa[..., 128:].astype(jnp.float32), axis=-1)
    np.testing.assert_allclose(np.asarray(bias[..., 0] / math.sqrt(padded)), np.asarray(c[..., 0]), atol=2e-3)


def _gated_model():
    mesh, model = kda_test._model(**_MLA, mla_forget_gate=True)
    # Random gate directions so each token's forget value differs (the zero-init gate is data independent).
    model = jax.tree_util.tree_map_with_path(
        lambda path, x: (
            0.5 * jax.random.normal(jax.random.PRNGKey(7), x.shape, x.dtype)
            if jax.tree_util.keystr(path).endswith("forget_gate_w")
            else x
        ),
        model,
    )
    return mesh, model


def test_forget_gate_model_is_causal():
    mesh, model = _gated_model()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, kda_test._SEQ), 0, kda_test._VOCAB)
    perturbed = tokens.at[:, -1].set((tokens[:, -1] + 1) % kda_test._VOCAB)
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        hidden, hidden_perturbed = forward(model, tokens), forward(model, perturbed)
    np.testing.assert_allclose(np.asarray(hidden[:, :-1]), np.asarray(hidden_perturbed[:, :-1]), rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(hidden[:, -1]), np.asarray(hidden_perturbed[:, -1]))


def test_forget_gate_resets_at_document_boundaries():
    mesh, model = _gated_model()
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, kda_test._SEQ), 0, kda_test._VOCAB)
    segment_ids = jnp.asarray(np.repeat([0, 1], [11, 13])[None].repeat(2, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids))
    perturbed = tokens.at[:, 3].set((tokens[:, 3] + 1) % kda_test._VOCAB)
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, t: m(t, mask=mask)[0])
        hidden, hidden_perturbed = forward(model, tokens), forward(model, perturbed)
    np.testing.assert_array_equal(np.asarray(hidden[:, 11:]), np.asarray(hidden_perturbed[:, 11:]))
    assert not np.allclose(np.asarray(hidden[:, 3:11]), np.asarray(hidden_perturbed[:, 3:11]))


def test_saturated_forget_gate_matches_plain_attention():
    """With f ~= 1 (bias +30) the gate adds nothing and the padded kernel inputs reproduce the base loss."""
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, kda_test._SEQ), 0, kda_test._VOCAB)
    weights = jnp.ones(tokens.shape, jnp.float32)
    losses = []
    for extra in ({}, dict(mla_forget_gate=True, mla_forget_gate_bias_init=30.0)):
        mesh, model = kda_test._model(**_MLA, learnable_qk_mult=True, qk_mult_per_head=True, **extra)
        with jax.set_mesh(mesh):
            losses.append(float(eqx.filter_jit(lambda m: m.next_token_loss(tokens, weights))(model)))
    np.testing.assert_allclose(losses[1], losses[0], rtol=1e-6)
