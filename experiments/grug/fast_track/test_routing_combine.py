# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine-weight variants: the raw sqrt-softplus combine matches the renormalized one at zero logits, and a
learnable routing sum starts at ``routing_renorm_sum``, trains and is logged."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import RouterCombine


def _loss_and_grads(**overrides):
    mesh, model = t._model(ngram_stat_rows=0, **overrides)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        (loss, metrics), grads = eqx.filter_jit(
            eqx.filter_value_and_grad(
                lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape), return_router_metrics=True), has_aux=True
            )
        )(model)
    return model, float(loss), metrics, grads


def test_raw_sqrt_softplus_matches_the_renormalized_combine_when_router_logits_are_zero():
    zero_router = lambda m: eqx.tree_at(  # noqa: E731
        lambda x: (x.kda_blocks.stacked.mlp.router, x.stacked_blocks.stacked.mlp.router),
        m,
        replace_fn=jnp.zeros_like,
    )
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape)))
    losses = []
    for combine in (RouterCombine.SQRT_SOFTPLUS_RENORM, RouterCombine.SQRT_SOFTPLUS_RAW):
        mesh, model = t._model(ngram_stat_rows=0, router_combine=combine)
        with jax.set_mesh(mesh):
            losses.append(float(loss(zero_router(model))))
    np.testing.assert_allclose(losses[1], losses[0], rtol=1e-5)


def test_learnable_routing_sum_starts_at_the_fixed_sum_trains_and_is_logged():
    model, _, metrics, grads = _loss_and_grads(routing_sum_learnable=True, routing_renorm_sum=2.5)
    np.testing.assert_allclose(np.asarray(model.stacked_blocks.stacked.mlp.routing_sum), 2.5)
    assert float(jnp.abs(grads.stacked_blocks.stacked.mlp.routing_sum).max()) > 0
    assert float(metrics["train/attn_res/knob_routing_sum_L0"]) == 2.5
