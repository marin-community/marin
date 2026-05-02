# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL loss functions."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmHeadModel
from marin.rl.objectives.reductions import compute_dapo_loss
from marin.rl.objectives.signals import compute_rloo_advantages_from_rewards
from marin.rl.objectives.terms import compute_metadata_metrics, importance_sampling_ratio
from marin.rl.scoring import LocalScoreSource, ModelRoles, ScoreRequirements, ScoreSource
from marin.rl.types import Rollout, TrainingBatch

logger = logging.getLogger(__name__)


class RLLossModule(Protocol):
    """Defines the interface used for computing RL loss & advantages."""

    def build(self, reference_model: eqx.Module) -> eqx.Module:
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


def rloo_loss_with_importance_sampling(
    model: LmHeadModel,
    reference_model: LmHeadModel | None,
    batch: TrainingBatch,
    score_source: ScoreSource,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    tis_importance_sampling_ratio_max: float,
    do_trainer_inference_mismatch_importance_sampling: bool = False,
    synchronous: bool = False,
    do_overlong_filtering: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute RLOO (Reward Leave-One-Out) loss with importance sampling for off-policy data.

    Args:
        model: The language model
        batch: dict containing rollout data with RLOO advantages
        key: JAX random key for dropout
        kl_coef: Coefficient for KL regularization
        clip_epsilon_low: Lower clipping epsilon for importance sampling ratio
        clip_epsilon_high: Upper clipping epsilon for importance sampling ratio
    Returns:
        Tuple of (loss, aux_metrics)
    """
    score_bundle = score_source.score(
        batch,
        info=None,
        roles=ModelRoles(student=model, reference=reference_model),
        key=key,
    )
    if score_bundle.student_logprobs is None:
        raise ValueError("score source did not produce student logprobs")
    if score_bundle.behavior_logprobs is None:
        raise ValueError("score source did not produce behavior logprobs")

    policy_logprobs_array = score_bundle.behavior_logprobs
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array

    current_logprobs = score_bundle.student_logprobs
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
        if reference_model is None:
            raise ValueError("reference_model is required when kl_coef > 0")
        if score_bundle.reference_logprobs is None:
            raise ValueError("score source did not produce reference logprobs")
        reference_logprobs = score_bundle.reference_logprobs
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
        "scoring/student_pass_count": Metric.from_value(
            jnp.asarray(score_bundle.student_pass_count, dtype=jnp.float32),
            ReductionType.MEAN,
        ),
        "scoring/reference_pass_count": Metric.from_value(
            jnp.asarray(score_bundle.reference_pass_count, dtype=jnp.float32),
            ReductionType.MEAN,
        ),
        "scoring/teacher_pass_count": Metric.from_value(
            jnp.asarray(score_bundle.teacher_pass_count, dtype=jnp.float32),
            ReductionType.MEAN,
        ),
        **compute_metadata_metrics(current_logprobs, policy_logprobs_array, loss_weights_array, loss_masks_array),
        **metadata,
    }


def compute_rloo_advantages(rollouts: list[Rollout]) -> np.ndarray:
    """Compute RLOO (Reward Leave-One-Out) advantages for a group of rollouts."""
    rewards = np.array([r.episode_reward for r in rollouts], dtype=np.float32)
    return compute_rloo_advantages_from_rewards(rewards)


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

    def needs_reference_model(self) -> bool:
        """Return whether KL regularization requires a reference model."""
        return self.kl_coef > 0

    def create_loss_fn(self, reference_model: eqx.Module | None, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        if self.needs_reference_model() and reference_model is None:
            raise ValueError("reference_model is required when kl_coef > 0")

        score_source = LocalScoreSource(
            score_requirements=ScoreRequirements(
                student_logprobs=True,
                behavior_logprobs=True,
                reference_logprobs=self.needs_reference_model(),
            ),
            vocab_tile_size=self.vocab_tile_size,
        )
        logger.info(
            "Using score source backend=%s requirements=%s vocab_tile_size=%s",
            score_source.backend_name,
            score_source.requirements(),
            self.vocab_tile_size,
        )

        def loss_fn(model, batch, key):
            return rloo_loss_with_importance_sampling(
                model,
                reference_model,
                batch,
                score_source,
                key=key,
                kl_coef=self.kl_coef,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
            )

        return loss_fn
