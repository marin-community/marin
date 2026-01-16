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
import haliax as hax
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel
from levanter.models.loss import _blockwise_cross_entropy_loss
from levanter.metrics import Metric, ReductionType

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


def compute_metadata_metrics(
    current_logprobs: jax.Array,
    policy_logprobs_array: jax.Array,
    loss_weights_array: jax.Array,
    loss_masks_array: jax.Array,
) -> dict[str, jax.Array]:
    """Compute metadata metrics for the loss function."""
    batch_size, _ = policy_logprobs_array.shape

    mean_ratio_difference = jnp.sum(
        (jnp.abs(jnp.exp(current_logprobs) - jnp.exp(policy_logprobs_array))) * loss_masks_array, axis=1
    ) / jnp.sum(loss_masks_array, axis=1)
    mean_ratio_difference = jnp.mean(mean_ratio_difference)

    flattened_current_logprobs = current_logprobs.reshape(-1)
    flattened_policy_logprobs = policy_logprobs_array.reshape(-1)
    flattened_loss_masks = loss_masks_array.reshape(-1)

    policy_entropy = -jnp.sum(flattened_policy_logprobs * flattened_loss_masks) / jnp.sum(flattened_loss_masks)
    current_entropy = -jnp.sum(flattened_current_logprobs * flattened_loss_masks) / jnp.sum(flattened_loss_masks)

    mean_advantages = jnp.sum(loss_weights_array * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
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

    return logprobs


def chunked_compute_logprobs(
    model: LmHeadModel,
    batch: TrainingBatch,
    key: jax.Array | None,
    block_size: int,
) -> jax.Array:
    """Compute log probabilities of target tokens in a chunked manner to save memory.

    This avoids materializing the full [batch, seq, vocab] logits tensor by using
    the blockwise cross-entropy implementation with a custom VJP.
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

    loss, _log_z = _blockwise_cross_entropy_loss(
        (pred_embeddings, pred_lm_head),
        model.Embed,
        model.Vocab,
        target_y,
        block_size,
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
    max_output_tokens: int,
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

    if response_truncated_array is not None:
        batch_size, _ = loss_objective.shape
        loss_objective = loss_objective * (1 - response_truncated_array.reshape(batch_size, 1))

    # Default to DAPO loss (matches original active behavior)
    loss = compute_dapo_loss(loss_objective, loss_masks)

    per_batch_loss = jnp.sum(loss_objective * loss_masks, axis=1) / jnp.sum(loss_masks, axis=1)
    metadata = {
        "loss_max_over_batch": Metric.from_value((-jnp.max(per_batch_loss)).astype(jnp.float32), ReductionType.MAX),
        "loss_std_over_batch": Metric.from_value(jnp.std(per_batch_loss).astype(jnp.float32), ReductionType.MEAN),
    }
    return loss, metadata


def compute_ppo_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
) -> jax.Array:
    """Compute PPO loss (per-example normalization)."""
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / jnp.sum(loss_masks, axis=1))


def compute_dapo_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
) -> jax.Array:
    """Compute DAPO-like loss (per-example normalization)."""
    # Use per-example normalization (averaging the per-example means)
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / jnp.sum(loss_masks, axis=1))


def compute_grpo_loss(
    loss_objective: jax.Array,
    loss_masks: jax.Array,
    max_output_tokens: int,
) -> jax.Array:
    """Compute GRPO loss (token-level loss)."""
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / max_output_tokens)


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


def rloo_loss_with_importance_sampling(
    model: LmHeadModel,
    reference_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    tis_importance_sampling_ratio_max: float,
    do_trainer_inference_mismatch_importance_sampling: bool = False,
    synchronous: bool = False,
    do_overlong_filtering: bool = False,
    compute_logprobs_fn: Callable = compute_logprobs,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute RLOO (Reward Leave-One-Out) loss with importance sampling for off-policy data.

    Args:
        model: The language model
        batch: dict containing rollout data with RLOO advantages
        key: JAX random key for dropout
        kl_coef: Coefficient for KL regularization
        clip_epsilon_low: Lower clipping epsilon for importance sampling ratio
        clip_epsilon_high: Upper clipping epsilon for importance sampling ratio
        compute_logprobs_fn: Function to compute logprobs (can be chunked)

    Returns:
        Tuple of (loss, aux_metrics)
    """
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array

    # Get logits from current policy
    current_logprobs = compute_logprobs_fn(model, batch, key)
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
    clip_fraction = jnp.mean(jnp.sum(is_clipped * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1))

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
        max_output_tokens=batch.max_output_tokens,
        trainer_inference_importance_sampling_ratio=trainer_inference_importance_sampling_ratio,
        response_truncated_array=batch.truncated if do_overlong_filtering else None,
    )

    # RLOO loss with importance sampling
    # batch["loss_weights"] contains RLOO advantages: r_i - mean(r_j for j≠i)
    # weighted_loss = -clipped_ratio * loss_weights_array * loss_masks_array
    # KL regularization

    if kl_coef > 0:
        reference_logprobs = compute_logprobs_fn(reference_model, batch, key)
        reference_logprobs = reference_logprobs * loss_masks_array
        # log_ratio = (current_logprobs - reference_logprobs_array) * loss_masks_array
        kl_penalty = jnp.exp(reference_logprobs - current_logprobs) - (reference_logprobs - current_logprobs) - 1
        # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L151
        # kl_penalty = jnp.abs(log_ratio)
        kl_loss = jnp.mean(jnp.sum(kl_penalty * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1))
        kl_loss = kl_coef * kl_loss
    else:
        kl_penalty = 0
        kl_loss = 0

    loss = reinforce_loss + kl_loss

    trainer_inference_importance_sampling_ratio_mean = jnp.mean(
        jnp.sum(trainer_inference_importance_sampling_ratio * loss_masks_array, axis=1)
        / jnp.sum(loss_masks_array, axis=1)
    )
    ratio_mean_over_responses_only = jnp.mean(jnp.sum(ratio, axis=1) / jnp.sum(loss_masks_array, axis=1))
    clipped_ratio_mean_over_responses_only = jnp.mean(
        jnp.sum(clipped_ratio * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
    )
    kl_penalty_over_responses_only = jnp.mean(
        jnp.sum(kl_penalty * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
    )

    return loss, {
        "ratio_mean": Metric.from_value(ratio_mean_over_responses_only.astype(jnp.float32), ReductionType.MEAN),
        "clipped_ratio_mean": Metric.from_value(
            clipped_ratio_mean_over_responses_only.astype(jnp.float32), ReductionType.MEAN
        ),
        "clip_fraction": Metric.from_value(clip_fraction.astype(jnp.float32), ReductionType.MEAN),
        "reinforce_loss": Metric.from_value(reinforce_loss.astype(jnp.float32), ReductionType.MEAN),
        "kl_loss": Metric.from_value(jnp.asarray(kl_loss, dtype=jnp.float32), ReductionType.MEAN),
        "kl_penalty": Metric.from_value(
            jnp.asarray(kl_penalty_over_responses_only, dtype=jnp.float32), ReductionType.MEAN
        ),
        "trainer_inference_importance_sampling_ratio_mean": Metric.from_value(
            trainer_inference_importance_sampling_ratio_mean.astype(jnp.float32), ReductionType.MEAN
        ),
        "temperature": Metric.from_value(jnp.mean(batch.temperature.array).astype(jnp.float32), ReductionType.MEAN),
        "top_k": Metric.from_value(jnp.mean(batch.top_k.array).astype(jnp.float32), ReductionType.MEAN),
        **compute_metadata_metrics(current_logprobs, policy_logprobs_array, loss_weights_array, loss_masks_array),
        **metadata,
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
class RLOOLoss(RLLossModule):
    """RLOO loss with importance sampling."""

    kl_coef: float = 0.1
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    tis_importance_sampling_ratio_max: float = 2.0
    synchronous: bool = False
    do_trainer_inference_mismatch_importance_sampling: bool = False
    do_overlong_filtering: bool = False
    vocab_tile_size: int | None = None

    def build(self, reference_model: eqx.Module) -> eqx.Module:
        """Initialize any learned components (e.g., value heads)."""
        return self  # No learned parameters

    def compute_advantages(self, rollout_group: list[Rollout]) -> list[float]:
        """Compute advantages for a group of rollouts."""
        return compute_rloo_advantages(rollout_group)

    def create_loss_fn(self, reference_model: eqx.Module, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""

        def loss_fn(model, batch, key):
            if self.vocab_tile_size is not None:

                def compute_logprobs_fn(m, b, k):
                    return chunked_compute_logprobs(m, b, k, self.vocab_tile_size)

            else:
                compute_logprobs_fn = compute_logprobs

            return rloo_loss_with_importance_sampling(
                model,
                reference_model,
                batch,
                key=key,
                kl_coef=self.kl_coef,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
                compute_logprobs_fn=compute_logprobs_fn,
            )

        return loss_fn
