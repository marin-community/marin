# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from levanter.models.moe_load_balance import equilibrium_bias_delta_from_topk


def test_equilibrium_bias_delta_shapes_and_ranges():
    logits = jnp.array(
        [
            [5.0, 3.0, 0.0],
            [4.8, 3.2, 0.1],
            [5.2, 3.1, -0.1],
        ],
        dtype=jnp.float32,
    )
    topk_idx = jnp.array([[0, 1], [0, 1], [0, 1]], dtype=jnp.int32)
    topk_weights = jnp.array([[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]], dtype=jnp.float32)

    delta, weighted_load, target_load, quantile_prob = equilibrium_bias_delta_from_topk(logits, topk_idx, topk_weights)

    assert delta.shape == (3,)
    assert weighted_load.shape == (3,)
    assert quantile_prob.shape == (3,)
    assert target_load.shape == ()
    assert float(target_load) == 1.0
    assert jnp.all(quantile_prob >= 0.0)
    assert jnp.all(quantile_prob <= 1.0)


def test_equilibrium_bias_delta_penalizes_overloaded_expert():
    logits = jnp.array(
        [
            [5.0, 3.0, 0.0],
            [5.1, 3.1, 0.0],
            [5.2, 3.0, 0.0],
            [5.3, 3.1, 0.0],
        ],
        dtype=jnp.float32,
    )
    topk_idx = jnp.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=jnp.int32)
    topk_weights = jnp.array([[0.95, 0.05], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1]], dtype=jnp.float32)

    delta, weighted_load, _, _ = equilibrium_bias_delta_from_topk(logits, topk_idx, topk_weights)

    # Expert 0 is overloaded and should receive the strongest downward update.
    assert weighted_load[0] > weighted_load[1]
    assert delta[0] > 0.0
    assert delta[0] >= delta[1]
