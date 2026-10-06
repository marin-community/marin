# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``latent_up_count``: several sigmoid-gated MoE output projections from the expert latent."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import mixture_weights
from experiments.grug.fast_track.optimizer import _is_gate_or_router_weight


@pytest.mark.parametrize("count", [2, 4])
def test_gated_projections_train_every_projection_and_the_gate(count):
    mesh, model = t._model(ngram_stat_rows=0, latent_up_count=count)
    mlp = model.stacked_blocks.stacked.mlp
    assert len(mlp.w_latent_up_extra) == count - 1
    assert all(w.shape == mlp.w_latent_up.shape for w in mlp.w_latent_up_extra)
    assert not np.allclose(np.asarray(mlp.w_latent_up_extra[0]), np.asarray(mlp.w_latent_up))
    assert float(jnp.abs(mlp.latent_up_gate).max()) == 0.0
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert np.isfinite(float(loss))
    g = grads.stacked_blocks.stacked.mlp
    for leaf in (g.w_latent_up, *g.w_latent_up_extra, g.latent_up_gate):
        assert float(jnp.abs(leaf).max()) > 0
    assert _is_gate_or_router_weight("stacked_blocks.stacked.mlp.latent_up_gate")


def test_top2_of_4_zeroes_the_unchosen_projections_per_token():
    mesh, model = t._model(ngram_stat_rows=0, latent_up_count=4, latent_up_topk=2)
    mlp = jax.tree.map(lambda a: a[0] if eqx.is_array(a) else a, model.stacked_blocks.stacked.mlp)
    assert float(jnp.abs(mlp.latent_up_gate).max()) > 0
    x = jax.random.normal(jax.random.PRNGKey(3), (32, model.config.hidden_dim))
    weights = np.asarray(mixture_weights(x, mlp.latent_up_gate, 4, 2))
    assert ((weights > 0).sum(-1) == 2).all()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))(model)
    assert np.isfinite(float(loss))


def test_renormed_top2_weights_sum_to_one():
    mesh, model = t._model(ngram_stat_rows=0, latent_up_count=4, latent_up_topk=2, latent_up_renorm=True)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))(model)
    assert np.isfinite(float(loss))
