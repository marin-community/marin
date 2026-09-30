# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HERO-MoE router history (``router_history``): an additive, learned bias from earlier layers' routing."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import _router_history_bias


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


def _loss(model, tokens):
    return model.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32))


def test_zero_history_weights_reproduce_the_model_without_history():
    mesh, plain = t._model()
    _, hero = t._model(router_history=True)
    zeroed = eqx.tree_at(lambda m: m.router_hist_w, hero, jnp.zeros_like(hero.router_hist_w))
    tokens = _tokens()
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        np.testing.assert_allclose(
            np.asarray(forward(zeroed, tokens)), np.asarray(forward(plain, tokens)), rtol=1e-5, atol=1e-5
        )
        assert not np.allclose(np.asarray(forward(hero, tokens)), np.asarray(forward(plain, tokens)), atol=1e-6)


def test_history_weights_get_a_gradient_from_every_later_layer():
    mesh, hero = t._model(router_history=True)
    tokens = _tokens()
    with jax.set_mesh(mesh):
        grads = eqx.filter_jit(eqx.filter_grad(_loss))(hero, tokens)
    per_layer = np.abs(np.asarray(grads.router_hist_w)).sum(axis=(1, 2))
    # Layer 0 has no earlier routing, so its weights stay unused; every later layer reads the history.
    assert per_layer[0] == 0
    assert np.all(per_layer[1:] > 0)


def test_history_bias_shape_and_first_layer():
    probs = [jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(i), (2, 3, 4)), axis=-1) for i in range(2)]
    w = jnp.ones((5 * 4, 4))
    assert _router_history_bias(w, [], num_layers=6) is None
    assert _router_history_bias(w, probs, num_layers=6).shape == (2, 3, 4)


def test_router_history_needs_attn_res():
    with pytest.raises(ValueError):
        t._config(router_history=True, attn_res=False)
