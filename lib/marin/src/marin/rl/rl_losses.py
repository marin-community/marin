# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL loss functions."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionMask
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmHeadModel
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty
from marin.rl.kl_regularization import (
    KLConfig,
    kl_penalty_from_log_ratio,
    kl_statistics_from_log_ratio,
    masked_response_mean,
    token_log_ratio,
)
from marin.rl.types import Rollout, TrainingBatch

# TODO(power) - these should be refactored to accept the precomputed logits instead
# of computing outputs themselves.


class RLLossModule(Protocol):
    """Defines the interface used for computing RL loss & advantages."""

    def build(self, reference_model: eqx.Module) -> "RLLossModule":
        """Initialize any learned components (e.g., value heads)."""
        ...

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute advantages for a group of rollouts."""
        ...

    def needs_reference_model(self) -> bool:
        """Return whether this loss needs a separately retained reference model."""
        ...

    def create_loss_fn(self, reference_model: eqx.Module | None, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        ...


class PolicyGradientReducer(StrEnum):
    """How response-token policy-gradient objectives are reduced to a scalar."""

    PER_SEQUENCE_TOKEN_MEAN = "per_sequence_token_mean"
    ACTIVE_TOKEN_MEAN = "active_token_mean"
    CURRENT_BATCH_TOKEN_NORMALIZED = "current_batch_token_normalized"
    FIXED_RESPONSE_BUDGET = "fixed_response_budget"


def compute_metadata_metrics(
    current_logprobs: jax.Array,
    policy_logprobs_array: jax.Array,
    loss_weights_array: jax.Array,
    loss_masks_array: jax.Array,
) -> dict[str, Metric]:
    """Compute metadata metrics for the loss function."""
    batch_size, _ = policy_logprobs_array.shape
    response_token_counts = jnp.sum(loss_masks_array, axis=1)
    safe_response_token_counts = jnp.maximum(response_token_counts, 1.0)
    total_response_tokens = jnp.maximum(jnp.sum(loss_masks_array), 1.0)

    mean_ratio_difference = (
        jnp.sum((jnp.abs(jnp.exp(current_logprobs) - jnp.exp(policy_logprobs_array))) * loss_masks_array, axis=1)
        / safe_response_token_counts
    )
    mean_ratio_difference = jnp.mean(mean_ratio_difference)

    flattened_current_logprobs = current_logprobs.reshape(-1)
    flattened_policy_logprobs = policy_logprobs_array.reshape(-1)
    flattened_loss_masks = loss_masks_array.reshape(-1)

    policy_entropy = -jnp.sum(flattened_policy_logprobs * flattened_loss_masks) / total_response_tokens
    current_entropy = -jnp.sum(flattened_current_logprobs * flattened_loss_masks) / total_response_tokens

    mean_advantages = jnp.sum(loss_weights_array * loss_masks_array, axis=1) / safe_response_token_counts
    mean_advantages = jnp.mean(mean_advantages)

    max_ratio_diff = jnp.max(jnp.abs(jnp.exp(current_logprobs) - jnp.exp(policy_logprobs_array)) * loss_masks_array)

    metrics = {
        "trainer_sampler_prob_diff_max": Metric.from_value(max_ratio_diff.astype(jnp.float32), ReductionType.MAX),
        "trainer_sampler_prob_diff_mean": Metric.from_value(
            mean_ratio_difference.astype(jnp.float32), ReductionType.MEAN
        ),
        "current_entropy": Metric.from_value(current_entropy.astype(jnp.float32), ReductionType.MEAN),
        "max_advantages": Metric.from_value(jnp.max(loss_weights_array).astype(jnp.float32), ReductionType.MAX),
        "mean_advantages": Metric.from_value(mean_advantages.astype(jnp.float32), ReductionType.MEAN),
        "policy_entropy": Metric.from_value(policy_entropy.astype(jnp.float32), ReductionType.MEAN),
        "response_tokens_length": Metric.from_value(
            (jnp.sum(loss_masks_array) / batch_size).astype(jnp.float32), ReductionType.MEAN
        ),
    }

    return metrics


def compute_logprobs_and_entropy(
    model: LmHeadModel,
    batch: TrainingBatch,
    key: jax.Array | None,
    *,
    compute_entropy: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute selected-token logprobs and, optionally, full-vocabulary entropy.

    Args:
        model: The language model
        batch: Training batch containing input_ids, position_ids, temperature
        key: JAX random key for dropout
        compute_entropy: Whether to compute full-vocabulary entropy for each predicted token
    Returns:
        Tuple of selected-token logprobs and optional entropy arrays, each shaped [batch, seq_len].
    """
    batch_size, _seq_len = batch.input_ids.array.shape

    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )

    # logits[i] predicts token at position i+1
    # We want logprob[j] = P(token[j] | tokens[0:j])
    # This comes from logits[j-1] indexed by token[j]
    logits_array = model_output.array.astype(jnp.float32)[:, :-1, :]  # Drop last position [batch, seq_len-1, vocab]
    target_ids_array = batch.input_ids.array[:, 1:]  # Drop first position [batch, seq_len-1]

    safe_temperature = jnp.where(batch.temperature.array == 0, 1.0, batch.temperature.array).reshape(batch_size, 1, 1)
    logits_array = logits_array / safe_temperature.astype(jnp.float32)

    log_probs = jax.nn.log_softmax(logits_array, axis=-1)
    logprobs_shifted = jnp.take_along_axis(log_probs, target_ids_array[..., None], axis=-1).squeeze(
        -1
    )  # [batch, seq_len-1]

    logprobs = jnp.concatenate(
        [jnp.zeros((logprobs_shifted.shape[0], 1), dtype=logprobs_shifted.dtype), logprobs_shifted], axis=1
    )  # [batch, seq_len]

    if not compute_entropy:
        return logprobs, None

    probs = jax.nn.softmax(logits_array, axis=-1)
    entropy_shifted = jax.nn.logsumexp(logits_array, axis=-1) - jnp.sum(probs * logits_array, axis=-1)
    entropy = jnp.concatenate(
        [jnp.zeros((entropy_shifted.shape[0], 1), dtype=entropy_shifted.dtype), entropy_shifted],
        axis=1,
    )

    return logprobs, entropy


def compute_logprobs(
    model: LmHeadModel,
    batch: TrainingBatch,
    key: jax.Array | None,
) -> jax.Array:
    """Compute log probabilities of target tokens.

    Args:
        model: The language model
        batch: Training batch containing input_ids, position_ids, temperature
        key: JAX random key for dropout
    Returns:
        logprobs array of shape [batch, seq_len]
    """
    logprobs, _entropy = compute_logprobs_and_entropy(model, batch, key, compute_entropy=False)
    return logprobs


def chunked_compute_logprobs(
    model: LmHeadModel,
    batch: TrainingBatch,
    key: jax.Array | None,
    block_size: int,
) -> jax.Array:
    """Compute log probabilities of target tokens in a chunked manner to save memory.

    This avoids materializing the full [batch, seq, vocab] logits tensor by using
    the fused cross-entropy kernel with blockwise processing.
    """
    # Get activations
    activations = model.activations(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )
    if isinstance(activations, tuple):
        activations, _aux = activations

    pred_embeddings = activations.astype(jnp.float32)
    pred_lm_head = model.get_lm_head().astype(jnp.float32)

    safe_temperature = hax.where(batch.temperature == 0, 1.0, batch.temperature).astype(jnp.float32)
    pred_embeddings = pred_embeddings / safe_temperature

    pos_axis = batch.input_ids.resolve_axis("position")
    target_y = hax.roll(batch.input_ids, -1, pos_axis)

    loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        pred_lm_head,
        Contract=model.Embed,
        Label=model.Vocab,
        target_y=target_y,
        reduction=None,
        logsumexp_weight=0.0,
        block_size=block_size,
        dtype=jnp.float32,
    )

    logprobs_shifted = -loss.array[:, :-1]
    logprobs = jnp.concatenate(
        [jnp.zeros((logprobs_shifted.shape[0], 1), dtype=logprobs_shifted.dtype), logprobs_shifted],
        axis=1,
    )

    return logprobs


def compute_ppo_loss_objective(
    importance_sampling_ratio: jax.Array,
    loss_weights: jax.Array,
    loss_masks: jax.Array,
    *,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    reducer: PolicyGradientReducer,
    fixed_response_budget: int | None = None,
    trainer_inference_importance_sampling_ratio: jax.Array | None = None,
    response_truncated_array: jax.Array | None = None,  # [batch]
):
    """Compute PPO loss objective."""
    non_clipped_objective = importance_sampling_ratio * loss_weights * loss_masks
    clipped_objective = (
        jnp.clip(importance_sampling_ratio, min=1.0 - clip_epsilon_low, max=1.0 + clip_epsilon_high)
        * loss_weights
        * loss_masks
    )

    loss_objective = jnp.minimum(non_clipped_objective, clipped_objective)
    if trainer_inference_importance_sampling_ratio is not None:
        loss_objective = trainer_inference_importance_sampling_ratio * loss_objective

    effective_loss_masks = loss_masks
    if response_truncated_array is not None:
        batch_size, _ = loss_objective.shape
        response_truncated_array = jnp.asarray(response_truncated_array, dtype=loss_objective.dtype)
        response_keep_mask = 1.0 - response_truncated_array.reshape(batch_size, 1)
        loss_objective = loss_objective * response_keep_mask
        effective_loss_masks = loss_masks * response_keep_mask

    loss = reduce_policy_gradient_loss(
        loss_objective,
        effective_loss_masks,
        reducer,
        fixed_response_budget=fixed_response_budget,
    )

    per_batch_loss_denominator = jnp.maximum(jnp.sum(effective_loss_masks, axis=1), 1.0)
    per_batch_loss = jnp.sum(loss_objective * effective_loss_masks, axis=1) / per_batch_loss_denominator
    metadata = {
        "loss_max_over_batch": Metric.from_value((-jnp.max(per_batch_loss)).astype(jnp.float32), ReductionType.MAX),
        "loss_std_over_batch": Metric.from_value(jnp.std(per_batch_loss).astype(jnp.float32), ReductionType.MEAN),
    }
    return loss, metadata


def reduce_policy_gradient_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
    reducer: PolicyGradientReducer,
    *,
    fixed_response_budget: int | None = None,
) -> jax.Array:
    """Reduce a token-level policy-gradient objective to a scalar loss."""
    per_sequence_objective = jnp.sum(loss_objective * loss_masks, axis=1)

    if reducer == PolicyGradientReducer.PER_SEQUENCE_TOKEN_MEAN:
        return -1 * jnp.mean(per_sequence_objective / jnp.maximum(jnp.sum(loss_masks, axis=1), 1.0))

    if reducer == PolicyGradientReducer.ACTIVE_TOKEN_MEAN:
        return -1 * jnp.sum(per_sequence_objective) / jnp.maximum(jnp.sum(loss_masks), 1.0)

    if reducer == PolicyGradientReducer.CURRENT_BATCH_TOKEN_NORMALIZED:
        return -1 * jnp.mean(per_sequence_objective / jnp.maximum(jnp.sum(loss_masks), 1.0))

    if reducer == PolicyGradientReducer.FIXED_RESPONSE_BUDGET:
        if fixed_response_budget is None:
            raise ValueError("fixed_response_budget is required for FIXED_RESPONSE_BUDGET reduction")
        if fixed_response_budget <= 0:
            raise ValueError(f"fixed_response_budget must be positive, got {fixed_response_budget}")
        return -1 * jnp.mean(per_sequence_objective / fixed_response_budget)

    raise ValueError(f"Unknown policy-gradient reducer: {reducer}")


def importance_sampling_ratio(
    current_logprobs: jax.Array,
    policy_logprobs_array: jax.Array,
    loss_masks_array: jax.Array,
    *,
    stop_current_logprob_gradient: bool = False,
    stop_policy_logprob_gradient: bool = True,
) -> jax.Array:
    if stop_current_logprob_gradient:
        current_logprobs = jax.lax.stop_gradient(current_logprobs)

    if stop_policy_logprob_gradient:
        policy_logprobs_array = jax.lax.stop_gradient(policy_logprobs_array)

    prob_difference = jnp.exp(current_logprobs - policy_logprobs_array) * loss_masks_array
    return prob_difference


def policy_gradient_loss_with_importance_sampling(
    model: LmHeadModel,
    reference_model: LmHeadModel | None,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl: KLConfig,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    tis_importance_sampling_ratio_max: float,
    do_trainer_inference_mismatch_importance_sampling: bool = False,
    synchronous: bool = False,
    do_overlong_filtering: bool = False,
    log_policy_entropy: bool = False,
    reducer: PolicyGradientReducer,
    fixed_response_budget: int | None = None,
    compute_policy_stats_fn: Callable = compute_logprobs_and_entropy,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Compute clipped policy-gradient loss with importance sampling for off-policy data.

    Args:
        model: The language model
        batch: Training batch containing rollout data with precomputed advantages
        key: JAX random key for dropout
        kl: KL regularization configuration
        clip_epsilon_low: Lower clipping epsilon for importance sampling ratio
        clip_epsilon_high: Upper clipping epsilon for importance sampling ratio
        log_policy_entropy: Whether to log exact full-vocabulary policy entropy
        reducer: Reduction rule for the token-level policy-gradient objective
        fixed_response_budget: Fixed response budget for Dr.GRPO-style reduction
        compute_policy_stats_fn: Function to compute logprobs and optional entropy

    Returns:
        Tuple of (loss, aux_metrics)
    """
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array
    effective_loss_masks_array = loss_masks_array
    if do_overlong_filtering:
        batch_size, _ = loss_masks_array.shape
        truncated_array = jnp.asarray(batch.truncated, dtype=loss_masks_array.dtype)
        effective_loss_masks_array = loss_masks_array * (1.0 - truncated_array.reshape(batch_size, 1))

    # Get logits from current policy
    current_logprobs, current_policy_entropy = compute_policy_stats_fn(
        model, batch, key, compute_entropy=log_policy_entropy
    )
    current_logprobs = current_logprobs * loss_masks_array

    if synchronous:
        policy_logprobs_array_for_importance_sampling_calculation = current_logprobs
    else:
        policy_logprobs_array_for_importance_sampling_calculation = policy_logprobs_array

    ratio = importance_sampling_ratio(
        current_logprobs,
        policy_logprobs_array_for_importance_sampling_calculation,
        loss_masks_array,
        stop_current_logprob_gradient=False,
        stop_policy_logprob_gradient=True,
    )

    # N.B. This should be enabled, but we seem to be training far enough
    # off of policy that we're not learning anything when we clip.
    clipped_ratio = jnp.clip(ratio, min=1.0 - clip_epsilon_low, max=1.0 + clip_epsilon_high)

    # Compute fraction of ratios that were clipped
    is_clipped = jnp.logical_or(ratio > 1.0 + clip_epsilon_high, ratio < 1.0 - clip_epsilon_low)
    effective_response_token_counts = jnp.maximum(jnp.sum(effective_loss_masks_array, axis=1), 1.0)
    clip_fraction = jnp.mean(jnp.sum(is_clipped * effective_loss_masks_array, axis=1) / effective_response_token_counts)

    if do_trainer_inference_mismatch_importance_sampling:
        trainer_inference_importance_sampling_ratio = importance_sampling_ratio(
            current_logprobs,
            policy_logprobs_array,
            loss_masks_array,
            stop_current_logprob_gradient=True,
            stop_policy_logprob_gradient=True,
        )
        trainer_inference_importance_sampling_ratio = jnp.minimum(
            trainer_inference_importance_sampling_ratio, tis_importance_sampling_ratio_max
        )
    else:
        trainer_inference_importance_sampling_ratio = jnp.ones_like(current_logprobs)

    reinforce_loss, metadata = compute_ppo_loss_objective(
        importance_sampling_ratio=ratio,
        loss_weights=loss_weights_array,
        loss_masks=loss_masks_array,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
        reducer=reducer,
        fixed_response_budget=fixed_response_budget,
        trainer_inference_importance_sampling_ratio=trainer_inference_importance_sampling_ratio,
        response_truncated_array=batch.truncated if do_overlong_filtering else None,
    )

    # KL regularization is additive to the clipped policy-gradient objective.

    log_ratio = jnp.zeros_like(current_logprobs)
    kl_penalty = jnp.zeros_like(current_logprobs)
    kl_loss = jnp.asarray(0.0, dtype=current_logprobs.dtype)
    kl_statistics = kl_statistics_from_log_ratio(log_ratio, effective_loss_masks_array)
    if kl.enabled():
        if reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        reference_logprobs, _reference_entropy = compute_policy_stats_fn(
            reference_model, batch, key, compute_entropy=False
        )
        reference_logprobs = reference_logprobs * loss_masks_array
        log_ratio = token_log_ratio(current_logprobs, reference_logprobs)
        kl_penalty = kl_penalty_from_log_ratio(log_ratio, kl.mode)
        kl_statistics = kl_statistics_from_log_ratio(log_ratio, effective_loss_masks_array)
        kl_loss = kl.beta * masked_response_mean(kl_penalty, effective_loss_masks_array)

    loss = reinforce_loss + kl_loss

    trainer_inference_importance_sampling_ratio_mean = jnp.mean(
        jnp.sum(trainer_inference_importance_sampling_ratio * effective_loss_masks_array, axis=1)
        / effective_response_token_counts
    )
    ratio_mean_over_responses_only = jnp.mean(
        jnp.sum(ratio * effective_loss_masks_array, axis=1) / effective_response_token_counts
    )
    clipped_ratio_mean_over_responses_only = jnp.mean(
        jnp.sum(clipped_ratio * effective_loss_masks_array, axis=1) / effective_response_token_counts
    )

    metrics = {
        "ratio_mean": Metric.from_value(ratio_mean_over_responses_only.astype(jnp.float32), ReductionType.MEAN),
        "clipped_ratio_mean": Metric.from_value(
            clipped_ratio_mean_over_responses_only.astype(jnp.float32), ReductionType.MEAN
        ),
        "clip_fraction": Metric.from_value(clip_fraction.astype(jnp.float32), ReductionType.MEAN),
        "reinforce_loss": Metric.from_value(reinforce_loss.astype(jnp.float32), ReductionType.MEAN),
        "kl_loss": Metric.from_value(jnp.asarray(kl_loss, dtype=jnp.float32), ReductionType.MEAN),
        "kl_beta": Metric.from_value(jnp.asarray(kl.beta, dtype=jnp.float32), ReductionType.MEAN),
        "kl_k1_mean": Metric.from_value(jnp.asarray(kl_statistics.k1_mean, dtype=jnp.float32), ReductionType.MEAN),
        "kl_k2_mean": Metric.from_value(jnp.asarray(kl_statistics.k2_mean, dtype=jnp.float32), ReductionType.MEAN),
        "kl_k3_mean": Metric.from_value(jnp.asarray(kl_statistics.k3_mean, dtype=jnp.float32), ReductionType.MEAN),
        "trainer_inference_importance_sampling_ratio_mean": Metric.from_value(
            trainer_inference_importance_sampling_ratio_mean.astype(jnp.float32), ReductionType.MEAN
        ),
        "temperature": Metric.from_value(jnp.mean(batch.temperature.array).astype(jnp.float32), ReductionType.MEAN),
        "top_k": Metric.from_value(jnp.mean(batch.top_k.array).astype(jnp.float32), ReductionType.MEAN),
        **compute_metadata_metrics(
            current_logprobs, policy_logprobs_array, loss_weights_array, effective_loss_masks_array
        ),
        **metadata,
    }
    if do_overlong_filtering:
        metrics["hard_overlong_filtered_fraction"] = Metric.from_value(
            jnp.mean(jnp.asarray(batch.truncated, dtype=jnp.float32)), ReductionType.MEAN
        )

    if log_policy_entropy:
        if current_policy_entropy is None:
            raise ValueError("compute_policy_stats_fn must return entropy when log_policy_entropy=True")
        policy_entropy = masked_response_mean(current_policy_entropy, effective_loss_masks_array)
        metrics["current_policy_entropy"] = Metric.from_value(
            jax.lax.stop_gradient(policy_entropy).astype(jnp.float32), ReductionType.MEAN
        )

    return loss, metrics


def compute_rloo_advantages(rollouts: list[Rollout]) -> np.ndarray:
    """Compute RLOO (Reward Leave-One-Out) advantages for a group of rollouts."""
    rewards = np.asarray([r.episode_reward for r in rollouts], dtype=np.float32)

    n = len(rewards)
    if n <= 1:
        return np.zeros_like(rewards)

    total = rewards.sum()
    leave_one_out_baselines = (total - rewards) / (n - 1)
    advantages = rewards - leave_one_out_baselines
    return advantages


def compute_group_centered_advantages(
    rollouts: list[Rollout],
    *,
    normalize_by_std: bool,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute group-relative advantages for GRPO-style objectives."""
    rewards = np.asarray([r.episode_reward for r in rollouts], dtype=np.float32)
    if len(rewards) <= 1:
        return np.zeros_like(rewards)

    advantages = rewards - np.mean(rewards)
    if not normalize_by_std:
        return advantages

    reward_std = np.std(rewards, ddof=1)
    if reward_std <= 0.0:
        return np.zeros_like(rewards)

    return advantages / (reward_std + epsilon)


def _policy_stats_fn_for_loss(vocab_tile_size: int | None, log_policy_entropy: bool) -> Callable:
    if log_policy_entropy and vocab_tile_size is not None:
        raise ValueError(
            "Exact policy entropy is not supported with vocab_tile_size yet. "
            "Set vocab_tile_size=None or implement chunked entropy first."
        )

    if vocab_tile_size is None:
        return compute_logprobs_and_entropy

    def compute_policy_stats_fn(m, b, k, *, compute_entropy: bool):
        if compute_entropy:
            raise ValueError("Exact policy entropy is not supported with vocab_tile_size yet")
        return chunked_compute_logprobs(m, b, k, vocab_tile_size), None

    return compute_policy_stats_fn


def _validate_fixed_response_budget(batch: TrainingBatch, fixed_response_budget: int) -> None:
    if batch.max_output_tokens != fixed_response_budget:
        raise ValueError(
            "DrGRPOLoss.max_output_tokens must match TrainingBatch.max_output_tokens, "
            f"got {fixed_response_budget} and {batch.max_output_tokens}"
        )


@dataclass
class RLOOLoss(RLLossModule):
    """RLOO loss with importance sampling."""

    kl: KLConfig
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    tis_importance_sampling_ratio_max: float = 2.0
    synchronous: bool = False
    do_trainer_inference_mismatch_importance_sampling: bool = False
    do_overlong_filtering: bool = False
    vocab_tile_size: int | None = None
    log_policy_entropy: bool = False

    def build(self, reference_model: eqx.Module) -> RLLossModule:
        """Initialize any learned components (e.g., value heads)."""
        return self  # No learned parameters

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute advantages for a group of rollouts."""
        return compute_rloo_advantages(rollout_group)

    def needs_reference_model(self) -> bool:
        """Return whether KL regularization requires a reference model."""
        return self.kl.enabled()

    def create_loss_fn(self, reference_model: eqx.Module | None, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        if self.needs_reference_model() and reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        compute_policy_stats_fn = _policy_stats_fn_for_loss(self.vocab_tile_size, self.log_policy_entropy)

        def loss_fn(model, batch, key):
            return policy_gradient_loss_with_importance_sampling(
                model,
                reference_model,
                batch,
                key=key,
                kl=self.kl,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
                log_policy_entropy=self.log_policy_entropy,
                reducer=PolicyGradientReducer.CURRENT_BATCH_TOKEN_NORMALIZED,
                compute_policy_stats_fn=compute_policy_stats_fn,
            )

        return loss_fn


@dataclass
class GRPOLoss(RLLossModule):
    """GRPO loss with group-centered, optionally std-normalized advantages."""

    kl: KLConfig
    normalize_by_group_std: bool = True
    reducer: PolicyGradientReducer = PolicyGradientReducer.PER_SEQUENCE_TOKEN_MEAN
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    tis_importance_sampling_ratio_max: float = 2.0
    synchronous: bool = False
    do_trainer_inference_mismatch_importance_sampling: bool = False
    do_overlong_filtering: bool = False
    vocab_tile_size: int | None = None
    log_policy_entropy: bool = False

    def __post_init__(self):
        if self.reducer == PolicyGradientReducer.FIXED_RESPONSE_BUDGET:
            raise ValueError("Use DrGRPOLoss for fixed response-budget reduction")

    def build(self, reference_model: eqx.Module) -> RLLossModule:
        """Initialize any learned components (e.g., value heads)."""
        return self

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute group-centered advantages for a group of rollouts."""
        return compute_group_centered_advantages(
            rollout_group,
            normalize_by_std=self.normalize_by_group_std,
        )

    def needs_reference_model(self) -> bool:
        """Return whether KL regularization requires a reference model."""
        return self.kl.enabled()

    def create_loss_fn(self, reference_model: eqx.Module | None, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        if self.needs_reference_model() and reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        compute_policy_stats_fn = _policy_stats_fn_for_loss(self.vocab_tile_size, self.log_policy_entropy)

        def loss_fn(model, batch, key):
            return policy_gradient_loss_with_importance_sampling(
                model,
                reference_model,
                batch,
                key=key,
                kl=self.kl,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
                log_policy_entropy=self.log_policy_entropy,
                reducer=self.reducer,
                compute_policy_stats_fn=compute_policy_stats_fn,
            )

        return loss_fn


@dataclass
class DAPOLoss(RLLossModule):
    """DAPO loss: GRPO advantages with active-token reduction and clip-higher defaults."""

    kl: KLConfig
    normalize_by_group_std: bool = True
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.28
    tis_importance_sampling_ratio_max: float = 2.0
    synchronous: bool = False
    do_trainer_inference_mismatch_importance_sampling: bool = False
    do_overlong_filtering: bool = True
    vocab_tile_size: int | None = None
    log_policy_entropy: bool = False

    def __post_init__(self):
        if self.clip_epsilon_low < 0.0:
            raise ValueError(f"clip_epsilon_low must be non-negative, got {self.clip_epsilon_low}")
        if self.clip_epsilon_high < 0.0:
            raise ValueError(f"clip_epsilon_high must be non-negative, got {self.clip_epsilon_high}")
        if self.clip_epsilon_high < self.clip_epsilon_low:
            raise ValueError(
                "DAPOLoss.clip_epsilon_high must be greater than or equal to "
                f"clip_epsilon_low, got {self.clip_epsilon_high} and {self.clip_epsilon_low}"
            )

    def build(self, reference_model: eqx.Module) -> RLLossModule:
        """Initialize any learned components (e.g., value heads)."""
        return self

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute DAPO group-centered advantages for a group of rollouts."""
        return compute_group_centered_advantages(
            rollout_group,
            normalize_by_std=self.normalize_by_group_std,
        )

    def needs_reference_model(self) -> bool:
        """Return whether KL regularization requires a reference model."""
        return self.kl.enabled()

    def create_loss_fn(self, reference_model: eqx.Module | None, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        if self.needs_reference_model() and reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        compute_policy_stats_fn = _policy_stats_fn_for_loss(self.vocab_tile_size, self.log_policy_entropy)

        def loss_fn(model, batch, key):
            return policy_gradient_loss_with_importance_sampling(
                model,
                reference_model,
                batch,
                key=key,
                kl=self.kl,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
                log_policy_entropy=self.log_policy_entropy,
                reducer=PolicyGradientReducer.ACTIVE_TOKEN_MEAN,
                compute_policy_stats_fn=compute_policy_stats_fn,
            )

        return loss_fn


@dataclass
class DrGRPOLoss(RLLossModule):
    """Dr.GRPO loss with mean-centered advantages and fixed response-budget reduction."""

    max_output_tokens: int
    kl: KLConfig
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    tis_importance_sampling_ratio_max: float = 2.0
    synchronous: bool = False
    do_trainer_inference_mismatch_importance_sampling: bool = False
    do_overlong_filtering: bool = False
    vocab_tile_size: int | None = None
    log_policy_entropy: bool = False

    def __post_init__(self):
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")

    def build(self, reference_model: eqx.Module) -> RLLossModule:
        """Initialize any learned components (e.g., value heads)."""
        return self

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute mean-centered advantages for a group of rollouts."""
        return compute_group_centered_advantages(rollout_group, normalize_by_std=False)

    def needs_reference_model(self) -> bool:
        """Return whether KL regularization requires a reference model."""
        return self.kl.enabled()

    def create_loss_fn(self, reference_model: eqx.Module | None, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        if self.needs_reference_model() and reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        compute_policy_stats_fn = _policy_stats_fn_for_loss(self.vocab_tile_size, self.log_policy_entropy)

        def loss_fn(model, batch, key):
            _validate_fixed_response_budget(batch, self.max_output_tokens)
            return policy_gradient_loss_with_importance_sampling(
                model,
                reference_model,
                batch,
                key=key,
                kl=self.kl,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
                log_policy_entropy=self.log_policy_entropy,
                reducer=PolicyGradientReducer.FIXED_RESPONSE_BUDGET,
                fixed_response_budget=self.max_output_tokens,
                compute_policy_stats_fn=compute_policy_stats_fn,
            )

        return loss_fn
