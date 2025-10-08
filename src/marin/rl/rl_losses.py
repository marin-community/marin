# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL loss functions."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from levanter.models.lm_model import LmHeadModel
from optax import softmax_cross_entropy_with_integer_labels

from marin.rl.types import Rollout, TrainingBatch

logger = logging.getLogger(__name__)

# TODO(power) - these should be refactored to accept the precomputed logits instead
# of computing outputs themselves.


def ppo_loss(
    model: LmHeadModel,
    reference_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon: float,
) -> jax.Array:
    """Compute PPO-style loss with RLOO advantages."""
    reference_output = reference_model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )

    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )

    logits = model_output.array.astype(jnp.float32)

    target_ids_array = batch.target_ids.array

    reference_logits = reference_output.array
    reference_logits_array = reference_logits.astype(jnp.float32)
    reference_logprobs_array = -softmax_cross_entropy_with_integer_labels(reference_logits_array, target_ids_array)

    token_ce_loss = softmax_cross_entropy_with_integer_labels(logits, target_ids_array)
    current_logprobs = -token_ce_loss

    # Get the old policy's log probs (from the worker policy that collected the data)
    old_logprobs = batch.policy_logprobs.array

    # Compute importance sampling ratio exp(log π_current - log π_old)
    log_ratio = current_logprobs - old_logprobs
    ratio = jnp.exp(log_ratio)

    # RLOO advantages (returned from the worker, and smeared across tokens)
    advantages = batch.loss_weights.array

    # Get the mask for valid tokens (e.g., excluding padding)
    mask = batch.loss_masks.array

    # PPO objective with clipping
    # We want to maximize advantage-weighted log probs, so we minimize the negative

    # Unclipped surrogate objective: ratio * advantage
    surrogate_1 = ratio * advantages

    # Clipped surrogate objective
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate_2 = clipped_ratio * advantages

    # PPO takes the minimum of the two (pessimistic bound)
    # We're minimizing negative rewards, so we take minimum of surrogate objectives
    # then negate to convert maximization to minimization
    ppo_loss_per_token = -jnp.minimum(surrogate_1, surrogate_2)

    # Apply mask and average
    ppo_loss = jnp.sum(ppo_loss_per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # KL penalty from reference policy (optional regularization)
    # KL(π_current || π_ref) ≈ π_current * (log π_current - log π_ref)
    kl_div = jnp.exp(current_logprobs) * (current_logprobs - reference_logprobs_array)
    kl_loss = jnp.sum(kl_div * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # Total loss
    total_loss = ppo_loss + kl_coef * kl_loss
    return total_loss


def rloo_loss_with_importance_sampling(
    model: LmHeadModel,
    reference_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon: float,
) -> jax.Array:
    """Compute RLOO (Reward Leave-One-Out) loss with importance sampling for off-policy data.

    Args:
        model: The language model
        batch: dict containing rollout data with RLOO advantages
        key: JAX random key for dropout
        kl_coef: Coefficient for KL regularization
        clip_epsilon: Clipping epsilon for importance sampling ratio

    Returns:
        Tuple of (loss, aux_metrics)
    """
    target_ids_array = batch.target_ids.array
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array

    # Get logits from current policy
    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )

    reference_output = reference_model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )

    reference_logits = reference_output.array
    reference_logits_array = reference_logits.astype(jnp.float32)
    reference_logprobs_array = -softmax_cross_entropy_with_integer_labels(reference_logits_array, target_ids_array)

    logits = model_output
    logits_array = logits.array
    logits_array = logits_array.astype(jnp.float32)
    token_loss = softmax_cross_entropy_with_integer_labels(logits_array, target_ids_array)

    current_logprobs = -token_loss

    # importance sampling since we're using off-policy data
    # ratio = π_current(a|s) / π_old(a|s) = exp(log π_current - log π_old)
    log_ratio = jnp.subtract(current_logprobs, policy_logprobs_array)
    ratio = jnp.exp(log_ratio)

    # N.B. This should be enabled, but we seem to be training far enough
    # off of policy that we're not learning anything when we clip.
    ratio_before_clip = ratio.copy()
    ratio = jnp.clip(ratio, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)

    # RLOO loss with importance sampling
    # batch["loss_weights"] contains RLOO advantages: r_i - mean(r_j for j≠i)
    weighted_loss = -ratio * loss_weights_array * loss_masks_array
    reinforce_loss = jnp.sum(weighted_loss) / jnp.maximum(jnp.sum(loss_masks_array), 1.0)

    # KL regularization
    kl_penalty = current_logprobs - reference_logprobs_array
    kl_loss = kl_coef * jnp.sum(kl_penalty * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0)

    loss = reinforce_loss + kl_loss

    ratio_clipped_percentage = jnp.sum(
        jnp.where((ratio > 1.0 + clip_epsilon) | (ratio < 1.0 - clip_epsilon), 1.0, 0.0)
    ) / jnp.maximum(jnp.sum(loss_masks_array), 1.0)

    jax.debug.print(
        "RLOO Loss with Importance Sampling Debug:\n"
        "  advantages (mean/std): {adv_mean:.4f} / {adv_std:.4f}\n"
        "  advantages (min/max): {adv_min:.4f} / {adv_max:.4f}\n"
        "  logprobs (mean/std): {lp_mean:.4f} / {lp_std:.4f}\n"
        "  reinforce_loss: {rl:.4f}\n"
        "  kl_loss: {kl:.4f}\n"
        "  total_loss: {total:.4f}\n"
        "  num_valid_tokens: {n_tokens}\n"
        "  ratio_clipped percentage: {ratio_clipped_percentage:.4f}\n"
        "  importance sampling ratio (mean/std): {ratio_mean:.4f} / {ratio_std:.4f}\n"
        "  ratio_before_clip (mean/std): {ratio_before_clip_mean:.4f} / {ratio_before_clip_std:.4f}\n",
        adv_mean=jnp.sum(loss_weights_array * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0),
        adv_std=jnp.std(loss_weights_array * loss_masks_array),
        adv_min=jnp.min(jnp.where(loss_masks_array > 0, loss_weights_array, jnp.inf)),
        adv_max=jnp.max(jnp.where(loss_masks_array > 0, loss_weights_array, -jnp.inf)),
        lp_mean=jnp.sum(current_logprobs * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0),
        lp_std=jnp.std(current_logprobs * loss_masks_array),
        rl=reinforce_loss,
        kl=kl_loss,
        total=loss,
        n_tokens=jnp.sum(loss_masks_array),
        ratio_clipped_percentage=ratio_clipped_percentage,
        ratio_mean=jnp.sum(ratio * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0),
        ratio_std=jnp.std(ratio * loss_masks_array),
        ratio_before_clip_mean=jnp.sum(ratio_before_clip * loss_masks_array)
        / jnp.maximum(jnp.sum(loss_masks_array), 1.0),
        ratio_before_clip_std=jnp.std(ratio_before_clip * loss_masks_array),
    )

    return loss


def rloo_loss_synchronous(
    model: LmHeadModel,
    reference_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon: float,
) -> jax.Array:
    """Compute synchronous RLOO (Reward Leave-One-Out) loss for on-policy data.

    This implementation assumes the rollouts are generated by the current policy
    (synchronous/on-policy RL), so no importance sampling is needed.

    Args:
        model: The current policy model being trained
        reference_model: Reference model for KL regularization (typically the initial model)
        batch: Training batch with rollout data and RLOO advantages in loss_weights
        key: JAX random key for dropout
        kl_coef: Coefficient for KL regularization (use 0 for pure REINFORCE)
        clip_epsilon: Not used in synchronous version (kept for API compatibility)

    Returns:
        Scalar loss value
    """
    target_ids_array = batch.target_ids.array
    loss_weights_array = batch.loss_weights.array  # Contains RLOO advantages
    loss_masks_array = batch.loss_masks.array

    # Get logits from current policy
    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )

    logits_array = model_output.array.astype(jnp.float32)

    # Compute log probabilities for the actions taken
    token_loss = softmax_cross_entropy_with_integer_labels(logits_array, target_ids_array)
    current_logprobs = -token_loss

    # RLOO/REINFORCE loss (on-policy, no importance sampling needed)
    # loss_weights_array contains RLOO advantages: A_i = r_i - mean(r_j for j≠i)
    # We want to maximize E[log π(a|s) * A], which means minimize -E[log π(a|s) * A]
    weighted_logprobs = current_logprobs * loss_weights_array * loss_masks_array
    reinforce_loss = -jnp.sum(weighted_logprobs) / jnp.maximum(jnp.sum(loss_masks_array), 1.0)

    # KL regularization: KL(π_current || π_ref) for stability
    # Only compute if kl_coef > 0 to save computation
    if kl_coef > 0:
        reference_output = reference_model(
            input_ids=batch.input_ids,
            attn_mask=batch.attention_mask,
            pos_ids=batch.position_ids,
            key=key,
        )
        reference_logits_array = reference_output.array.astype(jnp.float32)
        reference_logprobs_array = -softmax_cross_entropy_with_integer_labels(reference_logits_array, target_ids_array)

        # KL divergence: E[log π_current(a) - log π_ref(a)]
        kl_penalty = current_logprobs - reference_logprobs_array
        kl_loss = kl_coef * jnp.sum(kl_penalty * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0)
    else:
        kl_loss = jnp.array(0.0)

    loss = reinforce_loss + kl_loss

    # Debug logging
    jax.debug.print(
        "Synchronous RLOO Loss Debug:\n"
        "  advantages (mean/std): {adv_mean:.4f} / {adv_std:.4f}\n"
        "  advantages (min/max): {adv_min:.4f} / {adv_max:.4f}\n"
        "  logprobs (mean/std): {lp_mean:.4f} / {lp_std:.4f}\n"
        "  reinforce_loss: {rl:.4f}\n"
        "  kl_loss: {kl:.4f}\n"
        "  total_loss: {total:.4f}\n"
        "  num_valid_tokens: {n_tokens}\n",
        adv_mean=jnp.sum(loss_weights_array * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0),
        adv_std=jnp.std(loss_weights_array * loss_masks_array),
        adv_min=jnp.min(jnp.where(loss_masks_array > 0, loss_weights_array, jnp.inf)),
        adv_max=jnp.max(jnp.where(loss_masks_array > 0, loss_weights_array, -jnp.inf)),
        lp_mean=jnp.sum(current_logprobs * loss_masks_array) / jnp.maximum(jnp.sum(loss_masks_array), 1.0),
        lp_std=jnp.std(current_logprobs * loss_masks_array),
        rl=reinforce_loss,
        kl=kl_loss,
        total=loss,
        n_tokens=jnp.sum(loss_masks_array),
    )

    return loss


def compute_rloo_advantages(rollouts: list[Rollout]) -> np.ndarray:
    """Compute RLOO (Reward Leave-One-Out) advantages for a group of rollouts.

    This is a standalone version of RolloutGroup.compute_rloo_advantages() that
    can be used independently.

    Args:
        rollouts: List of rollouts to compute advantages for

    Returns:
        Array of advantages, one per rollout
    """
    rewards = np.array([r.episode_reward for r in rollouts])
    n = len(rewards)
    if n <= 1:
        return np.zeros_like(rewards)

    total = rewards.sum()
    leave_one_out_baselines = (total - rewards) / (n - 1)
    advantages = rewards - leave_one_out_baselines

    # Add small noise to avoid failure cases when all rewards are identical
    # This noise should be small enough not to bias training but large enough to provide gradient signal
    generator = np.random.default_rng()
    advantages += generator.normal(loc=0.0, scale=1e-6, size=advantages.shape)

    # Log statistics for debugging
    logger.info(
        f"RLOO Advantages: rewards (min/mean/max): {rewards.min():.4f}/{rewards.mean():.4f}/{rewards.max():.4f}, "
        f"advantages (min/mean/max/std): \
        {advantages.min():.4f}/{advantages.mean():.4f}/{advantages.max():.4f}/{advantages.std():.4f}"
    )

    return advantages
