# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe.adamh import (
    normalize_module_gradients_to_unit_rms,
    scale_by_adamh,
    scale_by_adamh_with_module_gradient_normalization,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeAdamHGradientNormConfig


def _assert_tree_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol)


def _tree_rms(tree):
    leaves = jax.tree.leaves(tree)
    square_sum = sum((jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves), jnp.array(0.0))
    count = sum(leaf.size for leaf in leaves)
    return jnp.sqrt(square_sum / count)


def test_module_gradient_normalization_scales_each_module_to_unit_rms():
    grads = {
        "block": {
            "attn": {
                "w_q": jnp.full((2, 2), 2.0, dtype=jnp.float32),
                "w_k": jnp.full((2,), 4.0, dtype=jnp.float32),
            },
            "mlp": {
                "w_gate_up": jnp.full((2, 3), -3.0, dtype=jnp.float32),
            },
        }
    }

    normalized = normalize_module_gradients_to_unit_rms(grads, eps=0.0)

    np.testing.assert_allclose(_tree_rms(normalized["block"]["attn"]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(_tree_rms(normalized["block"]["mlp"]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(normalized["block"]["attn"]["w_q"], jnp.full((2, 2), 2.0 / jnp.sqrt(8.0)), rtol=1e-6)
    np.testing.assert_allclose(normalized["block"]["attn"]["w_k"], jnp.full((2,), 4.0 / jnp.sqrt(8.0)), rtol=1e-6)
    np.testing.assert_allclose(normalized["block"]["mlp"]["w_gate_up"], -jnp.ones((2, 3)), rtol=1e-6)


def test_adamh_gradient_normalized_variant_matches_adamh_for_unit_rms_gradients():
    params = {
        "layer": {
            "w_q": jnp.arange(1, 5, dtype=jnp.float32).reshape(2, 2),
            "w_k": jnp.arange(5, 9, dtype=jnp.float32).reshape(2, 2),
        }
    }
    grads = {
        "layer": {
            "w_q": jnp.array([[1.0, -1.0], [1.0, -1.0]], dtype=jnp.float32),
            "w_k": jnp.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=jnp.float32),
        }
    }
    adamh = scale_by_adamh(b1=0.9, b2=0.95, eps=1e-8, learning_rate=0.05)
    grad_norm_adamh = scale_by_adamh_with_module_gradient_normalization(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        learning_rate=0.05,
        gradient_norm_eps=0.0,
    )

    adamh_updates, _ = adamh.update(grads, adamh.init(params), params)
    grad_norm_updates, _ = grad_norm_adamh.update(grads, grad_norm_adamh.init(params), params)

    _assert_tree_allclose(grad_norm_updates, adamh_updates)


def test_adamh_gradient_normalized_variant_stores_normalized_module_gradients():
    params = {"layer": {"weight": jnp.arange(1, 5, dtype=jnp.float32).reshape(2, 2)}}
    grads = {"layer": {"weight": jnp.full((2, 2), 3.0, dtype=jnp.float32)}}
    adamh = scale_by_adamh(b1=0.9, b2=0.95, eps=1e-8, learning_rate=0.05)
    grad_norm_adamh = scale_by_adamh_with_module_gradient_normalization(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        learning_rate=0.05,
        gradient_norm_eps=0.0,
    )

    _, adamh_state = adamh.update(grads, adamh.init(params), params)
    _, grad_norm_state = grad_norm_adamh.update(grads, grad_norm_adamh.init(params), params)

    np.testing.assert_allclose(adamh_state.mu["layer"]["weight"], jnp.full((2, 2), 0.3), rtol=1e-6)
    np.testing.assert_allclose(grad_norm_state.mu["layer"]["weight"], jnp.full((2, 2), 0.1), rtol=1e-6)


def test_gradient_normalized_adamh_config_keeps_baseline_parameter_groups():
    params = {
        "token_embed": jnp.ones((8, 4), dtype=jnp.float32),
        "blocks": {
            "0": {
                "attn": {"w_q": jnp.ones((4, 4), dtype=jnp.float32)},
                "mlp": {"w_gate_up": jnp.ones((2, 4, 8), dtype=jnp.float32)},
                "rms_attn": {"weight": jnp.ones((4,), dtype=jnp.float32)},
            }
        },
    }

    baseline_mask = GrugMoeAdamHConfig().create_mask(params)
    grad_norm_mask = GrugMoeAdamHGradientNormConfig().create_mask(params)

    assert grad_norm_mask == baseline_mask
