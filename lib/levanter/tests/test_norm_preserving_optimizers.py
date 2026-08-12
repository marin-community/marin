# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The defining invariant of AdamH/MuonH: matrix parameters keep their initial norm."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import haliax as hax

from levanter.optim.adamh import AdamHConfig
from levanter.optim.muonh import MuonHConfig
from levanter.optim.util import norm_preserving_update


@pytest.mark.parametrize("learning_rate", [1e-3, 0.5, 10.0])
def test_norm_preserving_update_keeps_matrix_norm(learning_rate):
    key = jax.random.PRNGKey(0)
    param_key, update_key = jax.random.split(key)
    param = jax.random.normal(param_key, (16, 8))
    update = jax.random.normal(update_key, (16, 8))

    delta = norm_preserving_update(param, update, learning_rate)
    new_param = param + delta

    assert jnp.linalg.norm(new_param) == pytest.approx(float(jnp.linalg.norm(param)), rel=1e-5)
    # a nonzero step was actually taken, otherwise norm preservation is trivially satisfied
    assert float(jnp.linalg.norm(delta)) > 1e-4
    # only the direction of the update matters: rescaling it leaves the step unchanged
    assert jnp.allclose(norm_preserving_update(param, 5.0 * update, learning_rate), delta, atol=1e-6)


def test_norm_preserving_update_treats_stacked_layers_independently():
    key = jax.random.PRNGKey(1)
    param_key, update_key, redirect_key = jax.random.split(key, 3)
    param = jax.random.normal(param_key, (4, 16, 8)) * jnp.array([1.0, 3.0, 0.5, 7.0])[:, None, None]
    update = jax.random.normal(update_key, (4, 16, 8))

    delta = norm_preserving_update(param, update, 0.1)
    new_param = param + delta

    per_layer_norm = lambda x: jnp.sqrt(jnp.sum(jnp.square(x), axis=(1, 2)))  # noqa: E731
    # each stacked layer keeps its own norm, not just the norm of the whole stack
    assert jnp.allclose(per_layer_norm(new_param), per_layer_norm(param), rtol=1e-5)

    # redirecting one layer's update must not leak into the other layers' deltas
    redirected_update = update.at[0].set(jax.random.normal(redirect_key, (16, 8)))
    redirected_delta = norm_preserving_update(param, redirected_update, 0.1)
    assert not jnp.allclose(redirected_delta[0], delta[0])
    assert jnp.allclose(redirected_delta[1:], delta[1:])


class _TwoLayer(eqx.Module):
    first: hax.nn.Linear
    second: hax.nn.Linear


def _two_layer_model(key) -> _TwoLayer:
    In = hax.Axis("In", 16)
    Mid = hax.Axis("Mid", 8)
    Out = hax.Axis("Out", 4)
    k1, k2 = jax.random.split(key)
    return _TwoLayer(
        first=hax.nn.Linear.init(In=In, Out=Mid, key=k1, use_bias=True),
        second=hax.nn.Linear.init(In=Mid, Out=Out, key=k2, use_bias=True),
    )


@pytest.mark.parametrize("config", [AdamHConfig(learning_rate=0.1), MuonHConfig(learning_rate=0.1)])
def test_optimizer_step_preserves_linear_weight_norms(config):
    model = _two_layer_model(jax.random.PRNGKey(2))
    optimizer = config.build(num_train_steps=10)
    state = optimizer.init(model)

    grad_key = jax.random.PRNGKey(3)
    leaves, treedef = jax.tree.flatten(model)
    grads = jax.tree.unflatten(
        treedef,
        [jax.random.normal(k, leaf.shape) for k, leaf in zip(jax.random.split(grad_key, len(leaves)), leaves)],
    )

    updates, _ = optimizer.update(grads, state, model)
    new_model = optax.apply_updates(model, updates)

    for old, new in ((model.first, new_model.first), (model.second, new_model.second)):
        old_norm = float(jnp.linalg.norm(old.weight.array))
        assert jnp.linalg.norm(new.weight.array) == pytest.approx(old_norm, rel=1e-5)
        assert float(jnp.linalg.norm(new.weight.array - old.weight.array)) > 1e-6
        # biases are routed to the plain Adam branch, which does not preserve their norm
        assert not jnp.allclose(new.bias.array, old.bias.array)
