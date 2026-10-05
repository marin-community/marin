# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``router_on_embed``: routing depends only on the token, so a token picks the same experts in any context."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import ROUTING_SELECTED_KEY


def _selected(tokens, **overrides):
    mesh, model = t._model(ngram_stat_rows=0, **overrides)
    with jax.set_mesh(mesh):
        _, metrics = eqx.filter_jit(lambda m: m(tokens, return_routing=True))(model)
    return np.asarray(metrics[ROUTING_SELECTED_KEY])  # [L, T, K]


def _sorted(x):
    return np.sort(x, axis=-1)


def test_a_token_routes_the_same_in_every_context_and_the_router_trains():
    positions = (3, 7, 11, 14)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (1, t._SEQ), 0, t._VOCAB)
    for p in positions:
        tokens = tokens.at[0, p].set(7)  # the same token in four different contexts
    sel = _selected(tokens, router_on_embed=True, num_experts=16)
    for p in positions[1:]:
        np.testing.assert_array_equal(_sorted(sel[:, positions[0]]), _sorted(sel[:, p]))

    mesh, model = t._model(ngram_stat_rows=0, router_on_embed=True)
    assert model.stacked_blocks.stacked.mlp.router.shape[-2] == model.config.hidden_dim
    with jax.set_mesh(mesh):
        grads = eqx.filter_jit(eqx.filter_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape))))(model)
    assert float(jnp.abs(grads.stacked_blocks.stacked.mlp.router).max()) > 0


def test_router_on_embed_excludes_router_on_latent():
    with pytest.raises(ValueError, match="router_on_embed"):
        t._config(router_on_embed=True, router_on_latent=True)
