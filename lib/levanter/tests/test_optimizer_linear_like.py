# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax

from levanter.models.linear import LinearLikeModule
from levanter.optim.muon import MuonConfig
from levanter.optim.muonh import MuonHConfig
from levanter.optim.namo import _create_namo_mask
from levanter.optim.scion import ScionConfig
from levanter.optim.util import (
    is_linear_like_module,
    label_linear_like_module,
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
