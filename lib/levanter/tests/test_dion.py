# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import haliax as hax

from levanter.optim import DionConfig, OptimizerConfig, ScaleByDionState
from levanter.optim.dion import _dion_update_matrix, scale_with_dion
from levanter.optim.util import flatten_linear_layers


def _linear_params_and_grads(*, in_size: int = 4, out_size: int = 3):
    In = hax.Axis("In", in_size)
    Out = hax.Axis("Out", out_size)
    linear = hax.nn.Linear.init(In, Out, key=jax.random.PRNGKey(0), out_first=True)
    grad_weight = dataclasses.replace(linear.weight, array=jnp.ones_like(linear.weight.array))
    grad_bias = None if linear.bias is None else dataclasses.replace(linear.bias, array=jnp.ones_like(linear.bias.array))
    grads = dataclasses.replace(linear, weight=grad_weight, bias=grad_bias)
    return linear, grads


def test_dion_registration():
    assert OptimizerConfig.get_choice_class("dion") is DionConfig
    assert isinstance(DionConfig().build(num_train_steps=10), optax.GradientTransformation)


def test_dion_mask_routes_only_haliax_linear_weight_to_dion():
    class _Module(eqx.Module):
        embedding: hax.nn.Embedding
        linear: hax.nn.Linear
        eqx_linear: eqx.nn.Linear
        lm_head: hax.nn.Linear

    Vocab = hax.Axis("Vocab", 8)
    Embed = hax.Axis("Embed", 4)
    Out = hax.Axis("Out", 3)
    module = _Module(
        embedding=hax.nn.Embedding.init(Vocab, Embed, key=jax.random.PRNGKey(1)),
        linear=hax.nn.Linear.init(Embed, Out, key=jax.random.PRNGKey(2), out_first=True),
        eqx_linear=eqx.nn.Linear(4, 3, key=jax.random.PRNGKey(3)),
        lm_head=hax.nn.Linear.init(Embed, Vocab, key=jax.random.PRNGKey(4), out_first=True),
    )

    mask = DionConfig().create_mask(module)

    assert mask.embedding.weight == "adamw"
    assert mask.linear.weight == "dion"
    assert mask.linear.bias == "adamw"
    assert mask.eqx_linear.weight == "adamw"
    assert mask.eqx_linear.bias == "adamw"
    assert mask.lm_head == "adamw"


def test_scale_with_dion_works_with_out_first_false():
    In = hax.Axis("In", 8)
    Out = hax.Axis("Out", 16)
    linear = hax.nn.Linear.init(In, Out, key=jax.random.PRNGKey(42), out_first=False)
    grad_weight = dataclasses.replace(linear.weight, array=jnp.ones_like(linear.weight.array))
    grads = dataclasses.replace(linear, weight=grad_weight, bias=None)

    transform = scale_with_dion(rank_fraction=0.5)
    state = transform.init(linear)

    # V should be [In=8, r=4] regardless of out_first
    assert state.right_vectors.weight.array.shape[-2:] == (8, 4)

    updates, new_state = transform.update(grads, state, linear)
    flat_updates = flatten_linear_layers(updates)

    # Output shape matches input: [In, Out] = [8, 16] for out_first=False
    assert flat_updates.weight.array.shape == (8, 16)
    assert jnp.all(jnp.isfinite(flat_updates.weight.array))
    assert jnp.any(flat_updates.weight.array != 0)


def test_scale_with_dion_initializes_rank_reduced_right_vectors():
    params, grads = _linear_params_and_grads(in_size=64, out_size=64)
    transform = scale_with_dion(rank_fraction=0.5)
    state = transform.init(params)

    assert isinstance(state, ScaleByDionState)
    assert state.right_vectors.weight.array.shape == (64, 32)

    updates, new_state = transform.update(grads, state, params)
    flat_updates = flatten_linear_layers(updates)

    assert flat_updates.weight.array.shape == (64, 64)
    assert new_state.right_vectors.weight.array.shape == (64, 32)
    assert jnp.all(jnp.isfinite(flat_updates.weight.array))
    assert jnp.any(flat_updates.weight.array != 0)


def test_rank_reduced_dion_decreases_toy_matrix_loss():
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 16)
    params = hax.nn.Linear.init(In, Out, key=jax.random.PRNGKey(11), out_first=True)
    params = dataclasses.replace(params, bias=None)
    transform = scale_with_dion(rank_fraction=0.5)
    state = transform.init(params)

    initial_loss = jnp.mean(params.weight.array**2)
    for _ in range(20):
        grads = dataclasses.replace(
            params,
            weight=dataclasses.replace(params.weight, array=params.weight.array),
            bias=None,
        )
        updates, state = transform.update(grads, state, params)
        params = dataclasses.replace(
            params,
            weight=dataclasses.replace(params.weight, array=params.weight.array - 0.1 * updates.weight.array),
        )

    assert jnp.mean(params.weight.array**2) < initial_loss


def test_dion_error_feedback_reduces_naive_momentum_growth():
    gradient = jax.random.normal(jax.random.PRNGKey(12), (8, 4))
    momentum = jnp.zeros_like(gradient)
    right_vectors = jax.random.normal(jax.random.PRNGKey(13), (4, 2))
    right_vectors = right_vectors / jnp.linalg.norm(right_vectors, axis=0, keepdims=True)

    num_steps = 20
    for _ in range(num_steps):
        _, momentum, right_vectors = _dion_update_matrix(
            gradient,
            momentum,
            right_vectors,
            fan_out=8,
            fan_in=4,
            out_first=True,
            mu=0.5,
            power_iters=1,
            epsilon=1e-8,
        )

    naive_momentum_norm = jnp.linalg.norm(num_steps * gradient)
    assert jnp.linalg.norm(momentum) < naive_momentum_norm


def test_dion_update_has_unit_singular_values_at_full_rank_before_shape_scaling():
    gradient = jax.random.normal(jax.random.PRNGKey(5), (8, 4))
    momentum = jnp.zeros_like(gradient)
    right_vectors = jax.random.normal(jax.random.PRNGKey(6), (4, 1))
    right_vectors = right_vectors / jnp.linalg.norm(right_vectors, axis=0, keepdims=True)

    update, _, _ = _dion_update_matrix(
        gradient,
        momentum,
        right_vectors,
        fan_out=8,
        fan_in=4,
        out_first=True,
        mu=0.95,
        power_iters=1,
        epsilon=1e-8,
    )
    unscaled_update = update / jnp.sqrt(8 / 4)
    singular_values = jnp.linalg.svd(unscaled_update, compute_uv=False)

    np.testing.assert_allclose(singular_values[0], jnp.array(1.0), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(singular_values[1:], jnp.zeros((3,)), rtol=2e-5, atol=2e-5)


def test_dion_update_matches_manual_reference():
    gradient = jax.random.normal(jax.random.PRNGKey(7), (5, 3))
    momentum = jax.random.normal(jax.random.PRNGKey(8), (5, 3))
    right_vectors = jax.random.normal(jax.random.PRNGKey(9), (3, 3))
    right_vectors = right_vectors / jnp.linalg.norm(right_vectors, axis=0, keepdims=True)

    update, new_momentum, new_right_vectors = _dion_update_matrix(
        gradient,
        momentum,
        right_vectors,
        fan_out=5,
        fan_in=3,
        out_first=True,
        mu=0.9,
        power_iters=1,
        epsilon=1e-8,
    )

    accumulated = momentum + gradient
    projected = accumulated @ right_vectors
    left_basis, _ = jnp.linalg.qr(projected)
    right_factor = accumulated.T @ left_basis
    expected_right_vectors = right_factor / (jnp.linalg.norm(right_factor, axis=0, keepdims=True) + 1e-8)
    expected_momentum = accumulated - 0.1 * (left_basis @ right_factor.T)
    expected_update = jnp.sqrt(jnp.asarray(5 / 3, dtype=gradient.dtype)) * (left_basis @ expected_right_vectors.T)

    np.testing.assert_allclose(update, expected_update, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new_momentum, expected_momentum, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new_right_vectors, expected_right_vectors, rtol=1e-5, atol=1e-5)


def test_dion_build_and_update_smoke():
    params, grads = _linear_params_and_grads(in_size=4, out_size=3)
    config = DionConfig(
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

    assert flat_updates.weight.array.shape == (3, 4)
    assert jnp.all(jnp.isfinite(flat_updates.weight.array))
    assert jnp.any(flat_updates.weight.array != 0)


def test_dion_lr_schedule_updates_injected_hyperparams():
    params, grads = _linear_params_and_grads(in_size=4, out_size=3)
    optimizer = DionConfig(
        learning_rate=1.0,
        adam_lr=0.5,
        warmup=0.1,
        lr_schedule="cosine",
        min_lr_ratio=0.1,
    ).build(num_train_steps=10)
    state = optimizer.init(params)

    _, state = optimizer.update(grads, state, params)
    _, state = optimizer.update(grads, state, params)

    assert state.hyperparams["learning_rate"] == 1.0
    assert state.hyperparams["adam_lr"] == 0.5


def test_dion_weight_decay_mask_excludes_embedding_and_bias():
    class _Module(eqx.Module):
        embedding: hax.nn.Embedding
        linear: hax.nn.Linear

    Vocab = hax.Axis("Vocab", 8)
    Embed = hax.Axis("Embed", 4)
    Out = hax.Axis("Out", 3)
    params = _Module(
        embedding=hax.nn.Embedding.init(Vocab, Embed, key=jax.random.PRNGKey(14)),
        linear=hax.nn.Linear.init(Embed, Out, key=jax.random.PRNGKey(15), out_first=True),
    )

    def zero_grad_leaf(x):
        if isinstance(x, hax.NamedArray):
            return dataclasses.replace(x, array=jnp.zeros_like(x.array))
        return x

    grads = jax.tree.map(zero_grad_leaf, params, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    optimizer = DionConfig(
        learning_rate=1.0,
        adam_lr=1.0,
        weight_decay=1.0,
        warmup=0.0,
        min_lr_ratio=1.0,
        lr_schedule="constant",
    ).build(num_train_steps=10)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)

    assert jnp.all(updates.embedding.weight.array == 0)
    assert jnp.all(updates.linear.bias.array == 0)
    assert jnp.any(updates.linear.weight.array != 0)


def test_scale_with_dion_handles_stacked_linear():
    In = hax.Axis("In", 4)
    Out = hax.Axis("Out", 3)
    Layers = hax.Axis("Layers", 2)

    class _StackedLinear(eqx.Module):
        linear: hax.nn.Linear

        @staticmethod
        def init(*, key):
            return _StackedLinear(hax.nn.Linear.init(In, Out, key=key, out_first=True))

    params = hax.nn.Stacked.init(Layers, _StackedLinear)(key=jax.random.split(jax.random.PRNGKey(10), 2))

    def grad_leaf(x):
        if isinstance(x, hax.NamedArray):
            return dataclasses.replace(x, array=jnp.ones_like(x.array))
        return x

    grads = jax.tree.map(grad_leaf, params, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    transform = scale_with_dion(rank_fraction=0.5)
    state = transform.init(params)
    updates, new_state = transform.update(grads, state, params)

    flat_updates = flatten_linear_layers(updates)

    assert flat_updates.stacked.linear.weight.array.shape == (2, 3, 4)
    assert new_state.right_vectors.stacked.linear.weight.array.shape == (2, 4, 1)
    assert jnp.all(jnp.isfinite(flat_updates.stacked.linear.weight.array))
