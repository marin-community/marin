# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from levanter.models.moe_load_balance import quantile_balancing_bias_target


def _max_abs_load_violation(scores: jnp.ndarray, bias: jnp.ndarray, top_k: int) -> jnp.ndarray:
    tokens, experts = scores.shape
    idx = jax.lax.top_k(scores + bias[None, :], top_k)[1]
    counts = jnp.bincount(idx.reshape(-1), length=experts).astype(jnp.float32)
    normalized_violation = counts / (tokens * top_k) * experts - 1.0
    return jnp.max(jnp.abs(normalized_violation))


def test_quantile_balancing_bias_target_shapes_and_ranges():
    logits = jnp.array(
        [
            [5.0, 3.0, 0.0],
            [4.8, 3.2, 0.1],
            [5.2, 3.1, -0.1],
        ],
        dtype=jnp.float32,
    )
    bias = jnp.zeros((3,), dtype=jnp.float32)

    target_bias, expert_counts, target_count, quantile_level = quantile_balancing_bias_target(
        logits, bias, top_k=2, num_iterations=5
    )

    assert target_bias.shape == (3,)
    assert expert_counts.shape == (3,)
    assert target_count.shape == ()
    assert quantile_level.shape == ()
    assert float(target_count) == 2.0
    assert 0.0 <= float(quantile_level) <= 1.0


def test_quantile_balancing_bias_target_reduces_load_violation():
    tokens = 4096
    experts = 16
    top_k = 2
    key = jax.random.key(0)
    noise = jax.random.normal(key, (tokens, experts)) * 0.7
    expert_bias = jnp.array(
        [2.3, 1.6, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -1.0],
        dtype=jnp.float32,
    )
    scores = noise + expert_bias[None, :]
    initial_bias = jnp.zeros((experts,), dtype=jnp.float32)

    target_bias, _, _, _ = quantile_balancing_bias_target(scores, initial_bias, top_k=top_k, num_iterations=5)

    before = _max_abs_load_violation(scores, initial_bias, top_k)
    after = _max_abs_load_violation(scores, target_bias, top_k)

    assert float(after) < float(before)
