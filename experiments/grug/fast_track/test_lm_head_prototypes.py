# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``lm_head_prototypes``: K lm_head vectors per token, ``p(v) ∝ sum_k exp(logit_{v,k})``."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t


@pytest.mark.parametrize("cap", [None, 5.0])
def test_two_prototype_loss_is_cross_entropy_over_summed_sub_token_probabilities(cap):
    mesh, model = t._model(ngram_stat_rows=0, lm_head_prototypes=2, logit_soft_cap=cap)
    v = model.config.vocab_size
    assert model.output_proj.shape[-1] == 2 * v
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    weight = jnp.ones(tokens.shape).at[:, -1].set(0.0)
    with jax.set_mesh(mesh):
        loss = float(eqx.filter_jit(lambda m: m.next_token_loss(tokens, weight))(model))
        hidden, _ = eqx.filter_jit(lambda m: m(tokens))(model)
        raw = jnp.einsum("bsd,dv->bsv", hidden.astype(jnp.float32), model.output_proj.astype(jnp.float32))
    raw = np.asarray(raw if cap is None else jnp.tanh(raw / cap) * cap).reshape(2, t._SEQ, 2, v)
    per_token = np.logaddexp(raw[:, :, 0], raw[:, :, 1])  # [B, S, V]
    log_p = per_token - np.log(np.exp(per_token).sum(-1, keepdims=True))
    labels = np.pad(np.asarray(tokens)[:, 1:], ((0, 0), (0, 1)))
    nll = -np.take_along_axis(log_p, labels[..., None], -1)[..., 0]
    expected = (nll * np.asarray(weight)).sum() / np.asarray(weight).sum()
    np.testing.assert_allclose(loss, expected, rtol=2e-3)


def test_logits_marginalize_the_prototypes():
    mesh, model = t._model(ngram_stat_rows=0, lm_head_prototypes=2)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        logits = eqx.filter_jit(lambda m: m.logits(tokens))(model)
    assert logits.shape[-1] == model.config.vocab_size
