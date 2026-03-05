# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import haliax as hax

from levanter.models.linear import LinearLikeModule
from levanter.optim.muon import MuonConfig
from levanter.optim.muonh import MuonHConfig
from levanter.optim.namo import NamoConfig, _create_namo_mask
from levanter.optim.scion import ScionConfig
from levanter.optim.util import (
    is_linear_like_module,
    label_linear_like_module,
    linear_like_weight_array,
    map_flattened_linear_layers,
    replace_linear_like_weight_array,
)


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


def test_map_flattened_linear_layers_updates_eqx_linear_weight():
    class _Module(eqx.Module):
        linear: eqx.nn.Linear

    module = _Module(linear=eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(4)))

    def _set_unit_weight(linear):
        return replace_linear_like_weight_array(linear, jnp.ones_like(linear_like_weight_array(linear)))

    updated = map_flattened_linear_layers(_set_unit_weight, module)
    assert jnp.all(updated.linear.weight == 1.0)


def test_map_flattened_linear_layers_updates_marker_linear_weight():
    class _MarkedLinear(LinearLikeModule):
        weight: jax.Array
        bias: jax.Array | None

    class _Module(eqx.Module):
        linear: _MarkedLinear

    module = _Module(linear=_MarkedLinear(weight=jnp.zeros((4, 3)), bias=jnp.zeros((3,))))

    def _set_unit_weight(linear):
        return replace_linear_like_weight_array(linear, jnp.ones_like(linear_like_weight_array(linear)))

    updated = map_flattened_linear_layers(_set_unit_weight, module)
    assert jnp.all(updated.linear.weight == 1.0)


def _vmapped_eqx_linear(batch: int = 2) -> eqx.nn.Linear:
    keys = jax.random.split(jax.random.PRNGKey(123), batch)
    return eqx.filter_vmap(lambda key: eqx.nn.Linear(4, 3, key=key))(keys)


def _assert_update_smoke_for_optimizer(config) -> None:
    params = {"linear": _vmapped_eqx_linear()}
    grads = jax.tree.map(jnp.ones_like, params)
    optimizer = config.build(num_train_steps=4)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)
    assert updates["linear"].weight.shape == params["linear"].weight.shape
    assert updates["linear"].bias is not None
    assert updates["linear"].bias.shape == params["linear"].bias.shape


def test_optimizer_masks_route_eqx_linear_to_linear_transforms():
    params = {"linear": eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(10))}

    muon_mask = MuonConfig().create_mask(params)
    muonh_mask = MuonHConfig().create_mask(params)
    scion_mask = ScionConfig().create_mask(params)
    namo_mask = _create_namo_mask(params)

    assert muon_mask["linear"].weight == "muon"
    assert muon_mask["linear"].bias == "adamw"
    assert muonh_mask["linear"].weight == "muonh"
    assert muonh_mask["linear"].bias == "adam"
    assert scion_mask["linear"].weight == "scion"
    assert scion_mask["linear"].bias == "signum"
    assert namo_mask["linear"].weight == "namo"
    assert namo_mask["linear"].bias == "adamw"


@pytest.mark.parametrize(
    "config",
    [
        MuonConfig(max_grad_norm=0.0),
        MuonHConfig(max_grad_norm=0.0),
        ScionConfig(max_grad_norm=0.0),
        NamoConfig(max_grad_norm=0.0),
    ],
)
def test_optimizers_support_vmapped_eqx_linear(config):
    _assert_update_smoke_for_optimizer(config)
