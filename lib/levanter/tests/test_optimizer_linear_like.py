# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import haliax as hax

from levanter.models.linear import LinearLikeModule
from levanter.optim.adamh import AdamHConfig
from levanter.optim.muon import MuonConfig
from levanter.optim.muonh import MuonHConfig
from levanter.optim.namo import _create_namo_mask
from levanter.optim.scion import ScionConfig
from levanter.optim.util import (
    is_linear_like_module,
    label_linear_like_module,
    norm_preserving_update,
)


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


def test_is_linear_like_module_detects_haliax_and_eqx_linears():
    In = hax.Axis("in", 4)
    Out = hax.Axis("out", 3)
    haliax_linear = hax.nn.Linear.init(In=In, Out=Out, key=jax.random.PRNGKey(0))
    eqx_linear = eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(1))

    assert is_linear_like_module(haliax_linear)
    assert is_linear_like_module(eqx_linear)
    assert not is_linear_like_module(jax.numpy.ones((2, 2)))


def test_is_linear_like_module_detects_marker_modules():
    class _MarkedLinear(LinearLikeModule):
        weight: jax.Array
        bias: jax.Array | None

    marked = _MarkedLinear(weight=jnp.ones((4, 3)), bias=jnp.zeros((3,)))
    assert is_linear_like_module(marked)


def test_label_linear_like_module_labels_weight_and_bias():
    In = hax.Axis("in", 4)
    Out = hax.Axis("out", 3)
    haliax_linear = hax.nn.Linear.init(In=In, Out=Out, key=jax.random.PRNGKey(2))
    eqx_linear = eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(3))

    masked_haliax = label_linear_like_module(haliax_linear, weight_label="namo", bias_label="adamw")
    masked_eqx = label_linear_like_module(eqx_linear, weight_label="namo", bias_label="adamw")

    assert masked_haliax.weight == "namo"
    assert masked_haliax.bias == "adamw"
    assert masked_eqx.weight == "namo"
    assert masked_eqx.bias == "adamw"


def test_muon_mask_routes_eqx_linear_to_adamw_fallback():
    class _Module(eqx.Module):
        linear: eqx.nn.Linear

    params = _Module(linear=eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(4)))
    mask = MuonConfig(use_kimi_scaling=False).create_mask(params, use_kimi_scaling=False)

    assert mask.linear.weight == "adamw"
    assert mask.linear.bias == "adamw"


def test_muonh_mask_routes_eqx_linear_to_adam_fallback():
    class _Module(eqx.Module):
        linear: eqx.nn.Linear

    params = _Module(linear=eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(5)))
    mask = MuonHConfig().create_mask(params)

    assert mask.linear.weight == "adam"
    assert mask.linear.bias == "adam"


def test_scion_mask_routes_eqx_linear_to_signum_fallback():
    class _Module(eqx.Module):
        linear: eqx.nn.Linear

    params = _Module(linear=eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(6)))
    mask = ScionConfig().create_mask(params)

    assert mask.linear.weight == "signum"
    assert mask.linear.bias == "signum"


def test_namo_mask_routes_eqx_linear_to_adamw_fallback():
    class _Module(eqx.Module):
        linear: eqx.nn.Linear

    params = _Module(linear=eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(7)))
    mask = _create_namo_mask(params)

    assert mask.linear.weight == "adamw"
    assert mask.linear.bias == "adamw"
