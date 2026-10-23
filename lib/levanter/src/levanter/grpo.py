# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GRPO group advantages and reference-partition objective weights.

Matches MarinSkyRL sample-standard-deviation advantages, masked token mean
policy loss, and sequence mean KL over the reference learner partitions.
Compute recipe inputs before slicing execution microbatches.
Reference: marin-community/MarinSkyRL at
8e33e01707b7225ecde1d6b8ad172a3dd4dc8661 (utils/{advantage_estimators,policy_losses,policy_math}.py).
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
    objective_partition_ids: NamedArray,
    *,
    Batch: Axis,
    Position: Axis,
    num_partitions: int,
) -> tuple[NamedArray, NamedArray]:
    """Return policy and KL weights preserving reference partition reductions.

    Partition IDs must densely enumerate all reference microbatches across DP
    ranks in [0, num_partitions). Each reference partition has equal objective
    weight. An empty response contributes zero but still counts as a sequence
    in the KL mean. The response mask must be binary.
    """
    mask = response_mask.rearrange((Batch, Position)).array.astype(jnp.float32)
    ids = objective_partition_ids.rearrange((Batch,)).array
    lengths = mask.sum(axis=1)
    tokens = jax.ops.segment_sum(lengths, ids, num_partitions)
    sequences = jax.ops.segment_sum(jnp.ones_like(lengths), ids, num_partitions)
    policy = mask / (jnp.maximum(tokens[ids], 1)[:, None] * num_partitions)
    kl = mask / (jnp.maximum(lengths, 1)[:, None] * jnp.maximum(sequences[ids], 1)[:, None] * num_partitions)
    return hax.named(policy, (Batch, Position)), hax.named(kl, (Batch, Position))
