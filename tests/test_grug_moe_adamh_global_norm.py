# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe.adamh import normalize_gradients_to_unit_rms
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeAdamHGlobalGradientNormConfig


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


def test_global_gradient_normalization_scales_full_tree_to_unit_rms():
    grads = {
        "block": {
            "attn": {"w_q": jnp.full((2, 2), 2.0, dtype=jnp.float32)},
            "mlp": {"w_gate_up": jnp.full((2,), 6.0, dtype=jnp.float32)},
        }
    }

    normalized = normalize_gradients_to_unit_rms(grads, eps=0.0)

    expected_inv_rms = jax.lax.rsqrt((4 * 2.0**2 + 2 * 6.0**2) / 6)
    np.testing.assert_allclose(_tree_rms(normalized), 1.0, rtol=1e-6)
    np.testing.assert_allclose(normalized["block"]["attn"]["w_q"], jnp.full((2, 2), 2.0 * expected_inv_rms))
    np.testing.assert_allclose(normalized["block"]["mlp"]["w_gate_up"], jnp.full((2,), 6.0 * expected_inv_rms))


def test_global_gradient_normalized_config_matches_manual_global_normalization():
    params = {
        "blocks": {
            "0": {
                "attn": {"w_q": jnp.arange(1, 5, dtype=jnp.float32).reshape(2, 2)},
                "mlp": {"w_gate_up": jnp.arange(1, 9, dtype=jnp.float32).reshape(2, 2, 2)},
                "rms_attn": {"weight": jnp.ones((2,), dtype=jnp.float32)},
            }
        }
    }
    grads = {
        "blocks": {
            "0": {
                "attn": {"w_q": jnp.full((2, 2), 2.0, dtype=jnp.float32)},
                "mlp": {"w_gate_up": jnp.full((2, 2, 2), 6.0, dtype=jnp.float32)},
                "rms_attn": {"weight": jnp.full((2,), 3.0, dtype=jnp.float32)},
            }
        }
    }
    baseline = GrugMoeAdamHConfig(
        learning_rate=0.05,
        adam_lr=0.05,
        expert_lr=0.05,
        max_grad_norm=None,
    ).build(4)
    global_norm = GrugMoeAdamHGlobalGradientNormConfig(
        learning_rate=0.05,
        adam_lr=0.05,
        expert_lr=0.05,
        max_grad_norm=None,
        gradient_norm_eps=0.0,
    ).build(4)

    manually_normalized = normalize_gradients_to_unit_rms(grads, eps=0.0)
    expected_updates, _ = baseline.update(manually_normalized, baseline.init(params), params)
    actual_updates, _ = global_norm.update(grads, global_norm.init(params), params)

    _assert_tree_allclose(actual_updates, expected_updates)


def test_global_gradient_normalized_adamh_config_keeps_baseline_parameter_groups():
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
    grad_norm_mask = GrugMoeAdamHGlobalGradientNormConfig().create_mask(params)

    assert grad_norm_mask == baseline_mask
