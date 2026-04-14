# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduction helpers for token-level objective terms."""

import jax
import jax.numpy as jnp

from .spec import ReductionConfig, ReductionKind


def compute_ppo_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
) -> jax.Array:
    """Compute PPO-style loss with per-sequence normalization."""
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / jnp.sum(loss_masks, axis=1))


def compute_dapo_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
) -> jax.Array:
    """Compute DAPO-style loss with global token normalization."""
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / jnp.sum(loss_masks))


def compute_grpo_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
    max_output_tokens: int,
) -> jax.Array:
    """Compute GRPO-style loss with fixed response-budget normalization."""
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / max_output_tokens)


def reduce_loss_objective(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
    reduction: ReductionConfig,
    *,
    max_output_tokens: int,
) -> jax.Array:
    """Reduce a token-level objective into a scalar loss."""
    if reduction.kind == ReductionKind.PPO:
        return compute_ppo_loss(loss_objective, loss_masks)
    if reduction.kind == ReductionKind.DAPO:
        return compute_dapo_loss(loss_objective, loss_masks)
    if reduction.kind == ReductionKind.GRPO:
        return compute_grpo_loss(loss_objective, loss_masks, max_output_tokens=max_output_tokens)

    raise ValueError(f"Unsupported reduction kind: {reduction.kind}")
