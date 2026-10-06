# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``latent_up_pair``: two sigmoid-gated MoE output projections from the expert latent."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.optimizer import _is_gate_or_router_weight


def test_pair_starts_as_the_half_sum_and_trains_both_projections_and_the_gate():
    mesh, model = t._model(ngram_stat_rows=0, latent_up_pair=True)
    mlp = model.stacked_blocks.stacked.mlp
    assert mlp.w_latent_up_b.shape == mlp.w_latent_up.shape
    assert not np.allclose(np.asarray(mlp.w_latent_up_b), np.asarray(mlp.w_latent_up))
    assert float(jnp.abs(mlp.latent_up_gate).max()) == 0.0
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
        )(model)
    assert np.isfinite(float(loss))
    g = grads.stacked_blocks.stacked.mlp
    for leaf in (g.w_latent_up, g.w_latent_up_b, g.latent_up_gate):
        assert float(jnp.abs(leaf).max()) > 0
    assert _is_gate_or_router_weight("stacked_blocks.stacked.mlp.latent_up_gate")
