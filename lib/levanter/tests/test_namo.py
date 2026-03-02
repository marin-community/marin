# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import haliax as hax

from levanter.optim import NamoConfig, NamoDConfig, OptimizerConfig
from levanter.optim.namo import _clamp_to_mean, _create_namo_mask, scale_with_namo, scale_with_namod
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


@pytest.fixture
def stacked_params_and_grads():
    in_axis = hax.Axis("In", 4)
    out_axis = hax.Axis("Out", 3)
    layers = hax.Axis("Layers", 2)

    class StackedLinearModule(eqx.Module):
        linear: hax.nn.Linear

        @staticmethod
        def init(*, key):
            return StackedLinearModule(hax.nn.Linear.init(in_axis, out_axis, key=key, out_first=True))

    stacked_module = hax.nn.Stacked.init(layers, StackedLinearModule)(key=jax.random.split(jax.random.PRNGKey(1), 2))

    params = {
        "stacked": stacked_module,
        "extra": jnp.ones((2,), dtype=jnp.float32),
    }

    def grad_leaf(x):
        if isinstance(x, hax.NamedArray):
            return dataclasses.replace(x, array=jnp.ones_like(x.array))
        return x

    grads = jax.tree.map(grad_leaf, params, is_leaf=lambda x: isinstance(x, hax.NamedArray))

    return params, grads


def test_namo_registration():
    assert OptimizerConfig.get_choice_class("namo") is NamoConfig
    assert OptimizerConfig.get_choice_class("namoD") is NamoDConfig


def test_namo_defaults_to_reference_simple_coefficients():
    assert NamoConfig().coefficient_type == "simple"


def test_namod_defaults_to_reference_simple_coefficients():
    assert NamoDConfig().coefficient_type == "simple"


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


def test_namo_build_and_update_smoke_nesterov_disabled(toy_params_and_grads):
    params, grads = toy_params_and_grads
    config = NamoConfig(
        learning_rate=1e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
        nesterov=False,
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


def test_namod_build_and_update_smoke_nesterov_disabled(toy_params_and_grads):
    params, grads = toy_params_and_grads
    config = NamoDConfig(
        learning_rate=1e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
        col_state_clamp_c=0.5,
        nesterov=False,
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


def test_namo_build_and_update_with_stacked_linear(stacked_params_and_grads):
    params, grads = stacked_params_and_grads
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
    stacked_linear_update = flat_updates["stacked"].stacked.linear.weight.array

    assert stacked_linear_update.shape == (2, 3, 4)
    assert jnp.all(jnp.isfinite(stacked_linear_update))
    assert jnp.any(stacked_linear_update != 0)

    assert updates["extra"].shape == (2,)
    assert jnp.all(jnp.isfinite(updates["extra"]))


def test_namod_build_and_update_with_stacked_linear(stacked_params_and_grads):
    params, grads = stacked_params_and_grads
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
    stacked_linear_update = flat_updates["stacked"].stacked.linear.weight.array

    assert stacked_linear_update.shape == (2, 3, 4)
    assert jnp.all(jnp.isfinite(stacked_linear_update))
    assert jnp.any(stacked_linear_update != 0)

    assert updates["extra"].shape == (2,)
    assert jnp.all(jnp.isfinite(updates["extra"]))


def test_scale_with_namo_requires_params(toy_params_and_grads):
    params, grads = toy_params_and_grads
    transform = scale_with_namo()
    state = transform.init(params)

    with pytest.raises(ValueError, match="requires params"):
        transform.update(grads, state, params=None)


def test_scale_with_namod_requires_params(toy_params_and_grads):
    params, grads = toy_params_and_grads
    transform = scale_with_namod()
    state = transform.init(params)

    with pytest.raises(ValueError, match="requires params"):
        transform.update(grads, state, params=None)


def test_create_namo_mask_routes_expected_groups():
    in_axis = hax.Axis("In", 4)
    out_axis = hax.Axis("Out", 3)
    linear = hax.nn.Linear.init(in_axis, out_axis, key=jax.random.PRNGKey(3), out_first=True)

    params = {
        "linear": linear,
        "Embedding": linear,
        "lm_head": linear,
        "extra": jnp.ones((2,), dtype=jnp.float32),
    }

    mask = _create_namo_mask(params)

    assert mask["extra"] == "adamw"
    assert mask["Embedding"] == "adamw"
    assert mask["lm_head"] == "adamw"
    assert mask["linear"].weight == "namo"
    assert mask["linear"].bias == ("adamw" if linear.bias is not None else None)


def test_namod_clamp_bounds_to_mean_window():
    col_scale = jnp.array([0.1, 1.0, 10.0], dtype=jnp.float32)
    clamped = _clamp_to_mean(col_scale, clamp_c=0.5)

    mean = jnp.nanmean(col_scale)
    floor = mean * 0.5
    ceil = mean / 0.5

    assert jnp.all(clamped >= floor)
    assert jnp.all(clamped <= ceil)


def test_namod_clamp_out_of_range_returns_original():
    col_scale = jnp.array([[1.0, 100.0], [1.0, 2.0]], dtype=jnp.float32)

    for clamp_c in [0.0, -0.5, 1.5]:
        clamped = _clamp_to_mean(col_scale, clamp_c=clamp_c)
        assert jnp.allclose(clamped, col_scale)


def test_namod_clamp_applies_per_matrix_for_stacked_inputs():
    col_scale = jnp.array([[1.0, 100.0], [1.0, 2.0]], dtype=jnp.float32)
    clamped = _clamp_to_mean(col_scale, clamp_c=0.5)

    expected = jnp.array([[25.25, 100.0], [1.0, 2.0]], dtype=jnp.float32)
    assert jnp.allclose(clamped, expected)


def test_namod_clamp_nan_and_inf_behavior():
    col_scale = jnp.array([[jnp.nan, 2.0, jnp.inf], [1.0, jnp.nan, 3.0]], dtype=jnp.float32)
    clamped = _clamp_to_mean(col_scale, clamp_c=0.5)

    assert clamped.shape == col_scale.shape
    assert jnp.isnan(clamped[0, 0])
    assert jnp.isinf(clamped[0, 2])
    assert jnp.isnan(clamped[1, 1])
    assert jnp.isfinite(clamped[0, 1])
    assert jnp.isfinite(clamped[1, 0])
    assert jnp.isfinite(clamped[1, 2])


def test_namo_dual_lr_linear_path_depends_on_learning_rate_only(toy_params_and_grads):
    params, grads = toy_params_and_grads

    config_a = NamoConfig(
        learning_rate=1e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    )
    config_b = NamoConfig(
        learning_rate=5e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    )

    opt_a = config_a.build(num_train_steps=10)
    opt_b = config_b.build(num_train_steps=10)
    state_a = opt_a.init(params)
    state_b = opt_b.init(params)
    updates_a, _ = opt_a.update(grads, state_a, params)
    updates_b, _ = opt_b.update(grads, state_b, params)

    flat_updates_a = flatten_linear_layers(updates_a)
    flat_updates_b = flatten_linear_layers(updates_b)
    linear_a = flat_updates_a["linear"].weight.array
    linear_b = flat_updates_b["linear"].weight.array

    assert not jnp.allclose(linear_a, linear_b)
    assert jnp.allclose(updates_a["extra"], updates_b["extra"])


def test_namo_dual_lr_adamw_path_depends_on_adam_lr_only(toy_params_and_grads):
    params, grads = toy_params_and_grads

    config_a = NamoConfig(
        learning_rate=1e-2,
        adam_lr=1e-4,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    )
    config_b = NamoConfig(
        learning_rate=1e-2,
        adam_lr=1e-2,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    )

    opt_a = config_a.build(num_train_steps=10)
    opt_b = config_b.build(num_train_steps=10)
    state_a = opt_a.init(params)
    state_b = opt_b.init(params)
    updates_a, _ = opt_a.update(grads, state_a, params)
    updates_b, _ = opt_b.update(grads, state_b, params)

    flat_updates_a = flatten_linear_layers(updates_a)
    flat_updates_b = flatten_linear_layers(updates_b)
    linear_a = flat_updates_a["linear"].weight.array
    linear_b = flat_updates_b["linear"].weight.array

    assert jnp.allclose(linear_a, linear_b)
    assert not jnp.allclose(updates_a["extra"], updates_b["extra"])


def test_namo_default_matches_simple_and_differs_from_quintic():
    in_axis = hax.Axis("In", 4)
    out_axis = hax.Axis("Out", 3)
    linear = hax.nn.Linear.init(in_axis, out_axis, key=jax.random.PRNGKey(0), out_first=True)

    params = {
        "linear": linear,
        "extra": jnp.ones((2,), dtype=jnp.float32),
    }

    w_key, b_key = jax.random.split(jax.random.PRNGKey(42))
    grad_weight = dataclasses.replace(linear.weight, array=jax.random.normal(w_key, linear.weight.array.shape))
    grad_bias = None
    if linear.bias is not None:
        grad_bias = dataclasses.replace(linear.bias, array=jax.random.normal(b_key, linear.bias.array.shape))
    grads = {
        "linear": dataclasses.replace(linear, weight=grad_weight, bias=grad_bias),
        "extra": jnp.ones((2,), dtype=jnp.float32),
    }

    def linear_update_for(config: NamoConfig):
        optimizer = config.build(num_train_steps=10)
        state = optimizer.init(params)
        updates, _ = optimizer.update(grads, state, params)
        flat_updates = flatten_linear_layers(updates)
        return flat_updates["linear"].weight.array

    base_kwargs = dict(
        learning_rate=1e-2,
        adam_lr=1e-3,
        weight_decay=0.1,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    )
    default_update = linear_update_for(NamoConfig(**base_kwargs))
    simple_update = linear_update_for(NamoConfig(**base_kwargs, coefficient_type="simple"))
    quintic_update = linear_update_for(NamoConfig(**base_kwargs, coefficient_type="quintic"))

    assert jnp.allclose(default_update, simple_update)
    assert not jnp.allclose(default_update, quintic_update)
