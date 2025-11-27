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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel

from marin.rl.types import Rollout, TrainingBatch

# TODO(power) - these should be refactored to accept the precomputed logits instead
# of computing outputs themselves.


class RLLossModule(Protocol):
    """Defines the interface used for computing RL loss & advantages."""

    def build(self, reference_model: eqx.Module) -> eqx.Module:
        """Initialize any learned components (e.g., value heads)."""
        ...

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute advantages for a group of rollouts."""
        ...

    def create_loss_fn(self, reference_model: eqx.Module, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        ...


def rloo_loss_with_importance_sampling(
    model: LmHeadModel,
    reference_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
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
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array

    batch_size, seq_len = batch.input_ids.array.shape

    # Get logits from current policy
    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )

    reference_output = reference_model(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )

    # logits[i] predicts token at position i+1
    # We want logprob[j] = P(token[j] | tokens[0:j])
    # This comes from logits[j-1] indexed by token[j]
    logits_array = model_output.array.astype(jnp.float32)[:, :-1, :]  # Drop last position [batch, seq_len-1, vocab]
    reference_logits_array = reference_output.array.astype(jnp.float32)[:, :-1, :]  # Drop last position

    target_ids_array = batch.input_ids.array[:, 1:]  # Drop first position [batch, seq_len-1]

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_array, axis=-1)
    reference_log_probs = jax.nn.log_softmax(reference_logits_array, axis=-1)

    batch_idx = jnp.arange(batch_size)[:, None]
    pos_idx = jnp.arange(seq_len - 1)

    # Extract logprobs for the actual tokens
    current_logprobs_shifted = log_probs[batch_idx, pos_idx, target_ids_array]  # [batch, seq_len-1]
    reference_logprobs_shifted = reference_log_probs[batch_idx, pos_idx, target_ids_array]  # [batch, seq_len-1]

    # Prepend zeros for position 0 (no context to predict from)
    current_logprobs = jnp.concatenate([jnp.zeros((batch_size, 1)), current_logprobs_shifted], axis=1)
    reference_logprobs_array = jnp.concatenate([jnp.zeros((batch_size, 1)), reference_logprobs_shifted], axis=1)

    # jax.debug.print("predicted_logprobs_array {current_logprobs}", current_logprobs=current_logprobs)
    # jax.debug.print(
    #     "reference_logprobs_array {reference_logprobs_array}", reference_logprobs_array=reference_logprobs_array
    # )
    # jax.debug.print("policy_logprobs_array {policy_logprobs_array}", policy_logprobs_array=policy_logprobs_array)

    # importance sampling since we're using off-policy data
    # ratio = π_current(a|s) / π_old(a|s) = log(π_current) - log(π_old)
    # mask the input tokens to ignore them in the loss
    current_logprobs = current_logprobs * loss_masks_array
    reference_logprobs_array = reference_logprobs_array * loss_masks_array

    log_ratio = jnp.subtract(current_logprobs, policy_logprobs_array)
    ratio = jnp.exp(log_ratio)

    # N.B. This should be enabled, but we seem to be training far enough
    # off of policy that we're not learning anything when we clip.
    clipped_ratio = jnp.clip(ratio, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)

    # RLOO loss with importance sampling
    # batch["loss_weights"] contains RLOO advantages: r_i - mean(r_j for j≠i)
    weighted_loss = -clipped_ratio * loss_weights_array * loss_masks_array
    reinforce_loss = jnp.sum(weighted_loss) / jnp.sum(loss_masks_array)

    # KL regularization
    log_ratio = (current_logprobs - reference_logprobs_array) * loss_masks_array
    # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L151
    kl_penalty = log_ratio**2
    kl_loss = kl_coef * jnp.sum(kl_penalty * loss_masks_array) / jnp.sum(loss_masks_array)

    loss = reinforce_loss + kl_loss
    return loss, {
        "ratio_mean": jnp.mean(ratio),
        "clipped_ratio_mean": jnp.mean(clipped_ratio),
        "reinforce_loss": reinforce_loss,
        "kl_loss": kl_loss,
        "kl_penalty": jnp.mean(kl_penalty),
    }


def compute_rloo_advantages(rollouts: list[Rollout]) -> np.ndarray:
    """Compute RLOO (Reward Leave-One-Out) advantages for a group of rollouts."""
    rewards = np.array([r.episode_reward for r in rollouts])
    n = len(rewards)
    if n <= 1:
        return np.zeros_like(rewards)

    total = rewards.sum()
    leave_one_out_baselines = (total - rewards) / (n - 1)
    advantages = rewards - leave_one_out_baselines
    return advantages


@dataclass
class RLOOLoss:
    """RLOO loss with importance sampling."""

    kl_coef: float = 0.1
    clip_epsilon: float = 0.2

    def build(self, reference_model: eqx.Module) -> eqx.Module:
        """Initialize any learned components (e.g., value heads)."""
        return self  # No learned parameters

    def compute_advantages(self, rollout_group: list[Rollout]) -> list[float]:
        """Compute advantages for a group of rollouts."""
        return compute_rloo_advantages(rollout_group)

    def create_loss_fn(self, reference_model: eqx.Module, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""

        def loss_fn(model, batch, key):
            return rloo_loss_with_importance_sampling(
                model, reference_model, batch, key=key, kl_coef=self.kl_coef, clip_epsilon=self.clip_epsilon
            )

        return loss_fn


def compute_drgrpo_advantages(rollouts: list[Rollout]) -> np.ndarray:
    """Compute Dr. GRPO advantages (centered rewards) for a group of rollouts."""
    rewards = np.array([r.episode_reward for r in rollouts])
    if len(rewards) <= 1:
        raise ValueError("Not enough rollouts to compute advantages")
    
    # Dr. GRPO uses simple centering: r_i - mean(r)
    # This is equivalent to GRPO without the KL term in the advantage
    mean_reward = np.mean(rewards)
    advantages = rewards - mean_reward
    return advantages


def drgrpo_loss(
    model: LmHeadModel,
    reference_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute Dr. GRPO loss with importance sampling.
    
    Matches Flax implementation:
    - Advantages: Centered rewards (computed in compute_drgrpo_advantages)
    - Policy Gradient: Sum reduction
    - KL Penalty: Exact KL (exp(log_ratio) - 1 - log_ratio)
    """
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array
    
    batch_size, seq_len = batch.input_ids.array.shape
    
    # Get logits from current policy
    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )
    
    # Get logits from reference model (for KL)
    reference_output = reference_model(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )
    
    # Extract logits for next-token prediction
    logits_array = model_output.array.astype(jnp.float32)[:, :-1, :]
    reference_logits_array = reference_output.array.astype(jnp.float32)[:, :-1, :]
    target_ids_array = batch.input_ids.array[:, 1:]
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_array, axis=-1)
    reference_log_probs = jax.nn.log_softmax(reference_logits_array, axis=-1)
    
    batch_idx = jnp.arange(batch_size)[:, None]
    pos_idx = jnp.arange(seq_len - 1)
    
    # Extract logprobs for the actual tokens
    current_logprobs_shifted = log_probs[batch_idx, pos_idx, target_ids_array]
    reference_logprobs_shifted = reference_log_probs[batch_idx, pos_idx, target_ids_array]
    
    # Prepend zeros
    current_logprobs = jnp.concatenate([jnp.zeros((batch_size, 1)), current_logprobs_shifted], axis=1)
    reference_logprobs_array = jnp.concatenate([jnp.zeros((batch_size, 1)), reference_logprobs_shifted], axis=1)
    
    # Masking
    current_logprobs = current_logprobs * loss_masks_array
    reference_logprobs_array = reference_logprobs_array * loss_masks_array
    
    # Importance sampling ratio
    # ratio = exp(current_logprobs - policy_logprobs)
    # policy_logprobs are from the rollout generation
    log_ratio = jnp.subtract(current_logprobs, policy_logprobs_array)
    ratio = jnp.exp(log_ratio)
    
    # Policy Gradient Loss (Weighted by advantages)
    # Flax implementation uses SUM reduction for the policy gradient part
    # loss = -sum(ratio * advantages * mask)
    weighted_loss = -ratio * loss_weights_array * loss_masks_array
    reinforce_loss = jnp.sum(weighted_loss)
    
    # KL Penalty (Exact KL)
    # KL(p||q) = sum(p * (log(p) - log(q)))
    # Here we use the approximation: exp(log_ratio) - 1 - log_ratio
    # where log_ratio = log(p/q) = log(p) - log(q)
    # current_logprobs is log(p), reference_logprobs is log(q)
    ref_log_ratio = current_logprobs - reference_logprobs_array
    kl_penalty = jnp.exp(ref_log_ratio) - 1 - ref_log_ratio
    kl_loss = jnp.sum(kl_penalty * loss_masks_array)
    
    # Total loss (sum reduction as per Flax implementation)
    loss = reinforce_loss + kl_coef * kl_loss
    
    # Metrics
    action_mask = loss_masks_array > 0.0
    
    # KL metrics
    logprob_diff = policy_logprobs_array - current_logprobs # sample - train (approx)
    
    kl_sample_train_v1 = jnp.mean(logprob_diff, where=action_mask)
    kl_sample_train_v2 = 0.5 * jnp.mean(logprob_diff**2, where=action_mask)
    
    # Ratio difference
    ratio_diff = jnp.abs(jnp.exp(policy_logprobs_array) - jnp.exp(current_logprobs))
    mean_ratio_difference = jnp.mean(ratio_diff, where=action_mask)
    max_ratio_difference = jnp.max(ratio_diff, where=action_mask, initial=-jnp.inf)
    
    # Entropy
    entropy = -jnp.mean(policy_logprobs_array, where=action_mask)
    
    # Advantages
    mean_advantages_token_level = jnp.mean(loss_weights_array, where=action_mask)
    mean_advantages_sequence_level = jnp.mean(jnp.mean(loss_weights_array, axis=1, where=action_mask))

    return loss, {
        "optim/reinforce_loss": reinforce_loss,
        "optim/kl_loss": kl_loss,
        "optim/loss": loss,
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/mean_ratio_difference": mean_ratio_difference,
        "optim/max_ratio_difference": max_ratio_difference,
        "optim/entropy": entropy,
        "optim/mean_advantages_token_level": mean_advantages_token_level,
        "optim/mean_advantages_sequence_level": mean_advantages_sequence_level,
    }


@dataclass
class DrGRPOLoss:
    """Dr. GRPO loss (Centered Rewards + Exact KL + Sum Reduction)."""

    kl_coef: float = 0.0

    def build(self, reference_model: eqx.Module) -> eqx.Module:
        """Initialize any learned components (e.g., value heads)."""
        return self

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute advantages for a group of rollouts."""
        return compute_drgrpo_advantages(rollout_group)

    def create_loss_fn(self, reference_model: eqx.Module, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""

        def loss_fn(model, batch, key):
            return drgrpo_loss(
                model, reference_model, batch, key=key, kl_coef=self.kl_coef
            )

        return loss_fn
