# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GRPO group advantages and whole-step objective weights.

Matches MarinSkyRL outcome advantages with token-mean policy loss and
sequence-mean KL. Compute recipe inputs over the complete optimizer-step
batch before packing or slicing execution microbatches.
Reference: marin-community/MarinSkyRL at
82396a73a31e776229f3240c3af2b50ad53c4336 (objective/{losses,reduction,objective}.py
and utils/{advantage_estimators,policy_math}.py under skyrl-train/skyrl_train).
"""

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray

GRPO_EPSILON = 1e-6


def grpo_advantages(
    token_rewards: NamedArray,
    response_mask: NamedArray,
    group_ids: NamedArray,
    *,
    Batch: Axis,
    Position: Axis,
    num_groups: int,
    normalize_by_std: bool,
) -> NamedArray:
    """Compute detached token advantages from complete admitted groups.

    Group IDs must be dense integers in [0, num_groups). Rewards are summed
    without masking, matching the oracle's outcome-reward contract. Singleton
    groups retain their score (divided by 1 + epsilon when normalizing).
    """
    scores = token_rewards.sum(Position).rearrange((Batch,)).array.astype(jnp.float32)
    ids = group_ids.rearrange((Batch,)).array
    counts = jax.ops.segment_sum(jnp.ones_like(scores), ids, num_groups)
    means = jax.ops.segment_sum(scores, ids, num_groups) / jnp.maximum(counts, 1)
    centered = scores - means[ids]
    variances = jax.ops.segment_sum(centered**2, ids, num_groups) / jnp.maximum(counts - 1, 1)
    group_mean = jnp.where(counts[ids] > 1, means[ids], 0)
    std = jnp.where(counts[ids] > 1, jnp.sqrt(variances[ids]), 1)
    advantages = scores - group_mean
    if normalize_by_std:
        advantages = advantages / (std + GRPO_EPSILON)
    result = hax.named(jax.lax.stop_gradient(advantages), Batch) * response_mask
    return result.rearrange((Batch, Position))


def grpo_objective_weights(
    response_mask: NamedArray,
    *,
    Batch: Axis,
    Position: Axis,
) -> tuple[NamedArray, NamedArray]:
    """Return whole-step token-mean policy and sequence-mean KL weights.

    The binary response mask must cover all rows in one optimizer step across
    data-parallel ranks. Empty responses contribute neither loss nor counts.
    Preserve these weights when packing or slicing execution microbatches.
    """
    mask = response_mask.rearrange((Batch, Position)).array.astype(jnp.float32)
    lengths = mask.sum(axis=1)
    policy = mask / jnp.maximum(lengths.sum(), 1)
    nonempty_sequences = (lengths > 0).sum()
    kl = mask / (jnp.maximum(lengths, 1)[:, None] * jnp.maximum(nonempty_sequences, 1))
    return hax.named(policy, (Batch, Position)), hax.named(kl, (Batch, Position))
