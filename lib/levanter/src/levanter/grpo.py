# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronous GRPO math with explicit objective partitions.

Matches MarinSkyRL's regular PPO surrogate, sample-standard-deviation GRPO,
masked token mean policy loss, and sequence mean k3 KL. Objective partitions
are the reference learner's microbatches (including data-parallel ranks).
Compute weights and advantages over the complete accepted batch before slicing
execution microbatches; changing execution partitions then preserves the loss.
Reference equations: marin-community/MarinSkyRL at
8e33e01707b7225ecde1d6b8ad172a3dd4dc8661 (utils/{advantage_estimators,policy_losses,policy_math}.py).
"""

from dataclasses import dataclass
from enum import StrEnum

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray

from levanter.metrics import Metric, ReductionType


LOG_PROB_DELTA_CLIP = 20.0
GRPO_EPSILON = 1e-6


class KlGradient(StrEnum):
    DETACHED = "detached"
    DIFFERENTIABLE = "differentiable"


@dataclass(frozen=True)
class GrpoConfig:
    eps_clip_low: float
    eps_clip_high: float
    kl_loss_coef: float
    kl_gradient: KlGradient


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


def _clamp(value: NamedArray, low: float, high: float) -> NamedArray:
    # Torch clamp passes the full gradient at either boundary; JAX clip halves it.
    return hax.where(value < low, low, hax.where(value > high, high, value))


def grpo_loss(
    log_probs: NamedArray,
    old_log_probs: NamedArray,
    reference_log_probs: NamedArray | None,
    advantages: NamedArray,
    policy_weights: NamedArray,
    kl_weights: NamedArray,
    *,
    config: GrpoConfig,
    accumulation_steps: int,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Return loss and additive metrics for Levanter gradient accumulation.

    Pass the number of execution microbatches as accumulation_steps to cancel
    Levanter's gradient averaging. All other inputs are sliced from the full
    batch; old/reference logprobs, advantages and weights are detached. The
    captured MarinSkyRL oracle uses detached KL, which affects reported loss
    but contributes no gradient. Choose that behavior explicitly in config.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    old_log_probs = jax.tree_util.tree_map(jax.lax.stop_gradient, old_log_probs)
    advantages = jax.tree_util.tree_map(jax.lax.stop_gradient, advantages)
    policy_weights = jax.tree_util.tree_map(jax.lax.stop_gradient, policy_weights)
    kl_weights = jax.tree_util.tree_map(jax.lax.stop_gradient, kl_weights)
    ratio = hax.exp(
        _clamp((log_probs - old_log_probs).astype(jnp.float32), -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
    ).astype(log_probs.dtype)
    surrogate = ratio * advantages
    clipped = _clamp(ratio, 1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    policy_loss = -(hax.minimum(surrogate, clipped) * policy_weights).sum().scalar()
    kl_loss = jnp.zeros((), dtype=log_probs.dtype)
    if config.kl_loss_coef != 0:
        if reference_log_probs is None:
            raise ValueError("reference_log_probs are required for nonzero KL coefficient")
        reference_log_probs = jax.tree_util.tree_map(jax.lax.stop_gradient, reference_log_probs)
        delta = _clamp(reference_log_probs - log_probs, -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
        kl = _clamp(hax.exp(delta) - delta - 1, -10.0, 10.0)
        if config.kl_gradient == KlGradient.DETACHED:
            kl = jax.tree_util.tree_map(jax.lax.stop_gradient, kl)
        kl_loss = (kl * kl_weights).sum().scalar()
    loss = policy_loss + config.kl_loss_coef * kl_loss
    metrics = {
        "loss": Metric.from_value(loss, ReductionType.SUM),
        "policy_loss": Metric.from_value(policy_loss, ReductionType.SUM),
        "policy_kl": Metric.from_value(kl_loss, ReductionType.SUM),
        "ppo_clip_ratio": Metric.from_value(
            ((clipped < surrogate) * policy_weights).sum().scalar(), ReductionType.SUM
        ),
    }
    selected = clipped < surrogate
    low_pressure = ratio < 1 - config.eps_clip_low
    high_pressure = ratio > 1 + config.eps_clip_high
    for name, condition in (
        ("ppo_clip_ratio_low", selected & low_pressure),
        ("ppo_clip_ratio_high", selected & high_pressure),
        ("ppo_clip_pressure_low", low_pressure),
        ("ppo_clip_pressure_high", high_pressure),
        ("ppo_ratio_exact_unit_fraction", ratio == 1),
    ):
        metrics[name] = Metric.from_value((condition * policy_weights).sum().scalar(), ReductionType.SUM)
    metrics["log_ratio_abs_max"] = Metric.from_value(
        hax.where(policy_weights > 0, hax.abs(log_probs - old_log_probs), 0).max().scalar(),
        ReductionType.MAX,
    )
    return loss * accumulation_steps, metrics
