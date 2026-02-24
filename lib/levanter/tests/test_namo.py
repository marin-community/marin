# Copyright 2026 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import haliax as hax

from levanter.optim import NamoConfig, NamoDConfig, OptimizerConfig
from levanter.optim.namo import _clamp_to_mean
from levanter.optim.util import flatten_linear_layers


@pytest.fixture
def toy_params_and_grads():
    in_axis = hax.Axis("In", 4)
    out_axis = hax.Axis("Out", 3)
    linear = hax.nn.Linear.init(in_axis, out_axis, key=jax.random.PRNGKey(0), out_first=True)

    params = {
        "linear": linear,
        "extra": jnp.ones((2,), dtype=jnp.float32),
    }

    grad_weight = dataclasses.replace(linear.weight, array=jnp.ones_like(linear.weight.array))
    grad_bias = None
    if linear.bias is not None:
        grad_bias = dataclasses.replace(linear.bias, array=jnp.ones_like(linear.bias.array))

    grads = {
        "linear": dataclasses.replace(linear, weight=grad_weight, bias=grad_bias),
        "extra": jnp.ones((2,), dtype=jnp.float32),
    }

    return params, grads


def test_namo_registration():
    assert OptimizerConfig.get_choice_class("namo") is NamoConfig
    assert OptimizerConfig.get_choice_class("namoD") is NamoDConfig


def test_namo_build_and_update_smoke(toy_params_and_grads):
    params, grads = toy_params_and_grads
    config = NamoConfig(
        learning_rate=1e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    )

    optimizer = config.build(num_train_steps=10)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)

    flat_updates = flatten_linear_layers(updates)
    linear_update = flat_updates["linear"].weight.array

    assert linear_update.shape == (3, 4)
    assert jnp.all(jnp.isfinite(linear_update))
    assert jnp.any(linear_update != 0)

    assert updates["extra"].shape == (2,)
    assert jnp.all(jnp.isfinite(updates["extra"]))


def test_namod_build_and_update_smoke(toy_params_and_grads):
    params, grads = toy_params_and_grads
    config = NamoDConfig(
        learning_rate=1e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
        col_state_clamp_c=0.5,
    )

    optimizer = config.build(num_train_steps=10)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)

    flat_updates = flatten_linear_layers(updates)
    linear_update = flat_updates["linear"].weight.array

    assert linear_update.shape == (3, 4)
    assert jnp.all(jnp.isfinite(linear_update))
    assert jnp.any(linear_update != 0)

    assert updates["extra"].shape == (2,)
    assert jnp.all(jnp.isfinite(updates["extra"]))


def test_namod_clamp_bounds_to_mean_window():
    col_scale = jnp.array([0.1, 1.0, 10.0], dtype=jnp.float32)
    clamped = _clamp_to_mean(col_scale, clamp_c=0.5)

    mean = jnp.nanmean(col_scale)
    floor = mean * 0.5
    ceil = mean / 0.5

    assert jnp.all(clamped >= floor)
    assert jnp.all(clamped <= ceil)
