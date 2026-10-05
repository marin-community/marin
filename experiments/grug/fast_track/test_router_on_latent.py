# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``router_on_latent``: the router reads the normed MoE latent, so its weight is [latent_dim, E] and it trains."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t


def test_router_reads_the_latent_and_trains():
    mesh, model = t._model(ngram_stat_rows=0, router_on_latent=True)
    cfg = model.config
    assert model.stacked_blocks.stacked.mlp.router.shape[-2:] == (cfg.latent_dim, cfg.num_experts)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert np.isfinite(float(loss))
    assert float(jnp.abs(grads.stacked_blocks.stacked.mlp.router).max()) > 0


def test_router_on_latent_needs_a_latent():
    with pytest.raises(ValueError, match="router_on_latent"):
        t._config(router_on_latent=True, latent_dim=None)
