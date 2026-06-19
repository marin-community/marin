# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""On-policy distillation losses for RL training."""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmHeadModel
from marin.rl.kl_regularization import (
    KLConfig,
    kl_penalty_from_log_ratio,
    kl_statistics_from_log_ratio,
    masked_response_mean,
    token_log_ratio,
)
from marin.rl.rl_losses import (
    RLLossModule,
    chunked_compute_logprobs,
    compute_logprobs,
    compute_logprobs_and_entropy,
    compute_metadata_metrics,
    compute_ppo_loss_objective,
    compute_rloo_advantages,
    importance_sampling_ratio,
)
from marin.rl.types import Rollout, TrainingBatch


def sampled_token_reverse_kl_opd_loss(
    model: LmHeadModel,
    teacher_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    synchronous: bool,
    clip_epsilon_low: float | None,
    clip_epsilon_high: float | None,
    compute_policy_logprobs_fn: Callable[[LmHeadModel, TrainingBatch, jax.Array | None], jax.Array] = compute_logprobs,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Compute sampled-token reverse-KL OPD with an importance-sampling policy gradient.

    This is the train-time form used by the MVP: rollouts carry behavior-policy
    logprobs for sampled response tokens, and the teacher is scored locally on
    exactly those tokens.
    """
    behavior_logprobs = jax.lax.stop_gradient(batch.policy_logprobs.array)
    loss_masks = batch.loss_masks.array

    current_logprobs = compute_policy_logprobs_fn(model, batch, key) * loss_masks
    teacher_logprobs = jax.lax.stop_gradient(compute_policy_logprobs_fn(teacher_model, batch, None) * loss_masks)
    behavior_logprobs = behavior_logprobs * loss_masks

    teacher_advantage = jax.lax.stop_gradient((teacher_logprobs - behavior_logprobs) * loss_masks)
    behavior_logprobs_for_ratio = current_logprobs if synchronous else behavior_logprobs
    ratio = importance_sampling_ratio(
        current_logprobs,
        behavior_logprobs_for_ratio,
        loss_masks,
        stop_current_logprob_gradient=False,
        stop_policy_logprob_gradient=True,
    )

    unclipped_objective = ratio * teacher_advantage
    objective = unclipped_objective
    clipped_ratio = ratio
    clip_fraction = jnp.asarray(0.0, dtype=jnp.float32)
    if clip_epsilon_low is not None and clip_epsilon_high is not None:
        clipped_ratio = jnp.clip(ratio, min=1.0 - clip_epsilon_low, max=1.0 + clip_epsilon_high)
        clipped_objective = clipped_ratio * teacher_advantage
        objective = jnp.minimum(unclipped_objective, clipped_objective)
        is_clipped = jnp.logical_or(ratio > 1.0 + clip_epsilon_high, ratio < 1.0 - clip_epsilon_low)
        clip_fraction = masked_response_mean(is_clipped.astype(jnp.float32), loss_masks)

    loss = -masked_response_mean(objective, loss_masks)

    ratio_mean = masked_response_mean(ratio, loss_masks)
    clipped_ratio_mean = masked_response_mean(clipped_ratio, loss_masks)
    teacher_advantage_mean = masked_response_mean(teacher_advantage, loss_masks)
    current_teacher_gap_mean = masked_response_mean(current_logprobs - teacher_logprobs, loss_masks)
    behavior_teacher_gap_mean = masked_response_mean(behavior_logprobs - teacher_logprobs, loss_masks)

    metrics = {
        "opd/sampled_token_reverse_kl_loss": Metric.from_value(loss.astype(jnp.float32), ReductionType.MEAN),
        "opd/teacher_advantage_mean": Metric.from_value(teacher_advantage_mean.astype(jnp.float32), ReductionType.MEAN),
        "opd/current_teacher_gap_mean": Metric.from_value(
            current_teacher_gap_mean.astype(jnp.float32), ReductionType.MEAN
        ),
        "opd/behavior_teacher_gap_mean": Metric.from_value(
            behavior_teacher_gap_mean.astype(jnp.float32), ReductionType.MEAN
        ),
        "opd/ratio_mean": Metric.from_value(ratio_mean.astype(jnp.float32), ReductionType.MEAN),
        "opd/clipped_ratio_mean": Metric.from_value(clipped_ratio_mean.astype(jnp.float32), ReductionType.MEAN),
        "opd/clip_fraction": Metric.from_value(clip_fraction.astype(jnp.float32), ReductionType.MEAN),
        "opd/temperature": Metric.from_value(jnp.mean(batch.temperature.array).astype(jnp.float32), ReductionType.MEAN),
        "opd/top_k": Metric.from_value(jnp.mean(batch.top_k.array).astype(jnp.float32), ReductionType.MEAN),
    }

    return loss, metrics


@dataclass
class OPDSampledTokenReverseKLLoss(RLLossModule):
    """Sampled-token reverse-KL OPD loss with a local teacher model."""

    synchronous: bool = False
    clip_epsilon_low: float | None = None
    clip_epsilon_high: float | None = None
    vocab_tile_size: int | None = None

    def __post_init__(self) -> None:
        if (self.clip_epsilon_low is None) != (self.clip_epsilon_high is None):
            raise ValueError("clip_epsilon_low and clip_epsilon_high must be set together")
        if self.clip_epsilon_low is not None and self.clip_epsilon_low < 0.0:
            raise ValueError("clip_epsilon_low must be non-negative")
        if self.clip_epsilon_high is not None and self.clip_epsilon_high < 0.0:
            raise ValueError("clip_epsilon_high must be non-negative")

    def build(self, reference_model: eqx.Module) -> RLLossModule:
        """Initialize learned components."""
        return self

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Use unit advantages; teacher scoring supplies token-level OPD signal."""
        return np.ones(len(rollout_group), dtype=np.float32)

    def needs_reference_model(self) -> bool:
        """Return whether this loss needs a separately retained reference model."""
        return False

    def needs_teacher_model(self) -> bool:
        """Return whether this loss needs a separately retained teacher model."""
        return True

    def create_loss_fn(
        self,
        reference_model: eqx.Module | None,
        train_model: eqx.Module,
        *,
        teacher_model: eqx.Module | None = None,
    ) -> Callable:
        """Create the loss function for training."""
        del reference_model, train_model
        if teacher_model is None:
            raise ValueError("teacher_model is required for OPDSampledTokenReverseKLLoss")

        if self.vocab_tile_size is not None:

            def compute_policy_logprobs_fn(m, b, k):
                return chunked_compute_logprobs(m, b, k, self.vocab_tile_size)

        else:
            compute_policy_logprobs_fn = compute_logprobs

        def loss_fn(model, batch, key):
            return sampled_token_reverse_kl_opd_loss(
                model,
                teacher_model,
                batch,
                key=key,
                synchronous=self.synchronous,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                compute_policy_logprobs_fn=compute_policy_logprobs_fn,
            )

        return loss_fn


def hybrid_rloo_sampled_token_reverse_kl_opd_loss(
    model: LmHeadModel,
    reference_model: LmHeadModel | None,
    teacher_model: LmHeadModel,
    batch: TrainingBatch,
    *,
    key: jax.Array | None,
    kl: KLConfig,
    opd_coef: float,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    tis_importance_sampling_ratio_max: float,
    synchronous: bool = False,
    do_trainer_inference_mismatch_importance_sampling: bool = False,
    do_overlong_filtering: bool = False,
    log_policy_entropy: bool = False,
    compute_policy_stats_fn: Callable = compute_logprobs_and_entropy,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Compute hybrid RLOO reward and sampled-token reverse-KL OPD loss."""
    behavior_logprobs = jax.lax.stop_gradient(batch.policy_logprobs.array)
    reward_advantages = jax.lax.stop_gradient(batch.loss_weights.array)
    loss_masks = batch.loss_masks.array

    current_logprobs, current_policy_entropy = compute_policy_stats_fn(
        model, batch, key, compute_entropy=log_policy_entropy
    )
    current_logprobs = current_logprobs * loss_masks

    teacher_logprobs, _teacher_entropy = compute_policy_stats_fn(teacher_model, batch, None, compute_entropy=False)
    teacher_logprobs = jax.lax.stop_gradient(teacher_logprobs * loss_masks)
    behavior_logprobs = behavior_logprobs * loss_masks
    reward_advantages = reward_advantages * loss_masks

    opd_advantages = jax.lax.stop_gradient((teacher_logprobs - behavior_logprobs) * loss_masks)
    scaled_opd_advantages = opd_coef * opd_advantages
    combined_advantages = reward_advantages + scaled_opd_advantages

    behavior_logprobs_for_ratio = current_logprobs if synchronous else behavior_logprobs
    ratio = importance_sampling_ratio(
        current_logprobs,
        behavior_logprobs_for_ratio,
        loss_masks,
        stop_current_logprob_gradient=False,
        stop_policy_logprob_gradient=True,
    )

    clipped_ratio = jnp.clip(ratio, min=1.0 - clip_epsilon_low, max=1.0 + clip_epsilon_high)
    is_clipped = jnp.logical_or(ratio > 1.0 + clip_epsilon_high, ratio < 1.0 - clip_epsilon_low)
    clip_fraction = masked_response_mean(is_clipped.astype(jnp.float32), loss_masks)

    if do_trainer_inference_mismatch_importance_sampling:
        trainer_inference_importance_sampling_ratio = importance_sampling_ratio(
            current_logprobs,
            behavior_logprobs,
            loss_masks,
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
        loss_weights=combined_advantages,
        loss_masks=loss_masks,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
        trainer_inference_importance_sampling_ratio=trainer_inference_importance_sampling_ratio,
        response_truncated_array=batch.truncated if do_overlong_filtering else None,
    )

    component_multiplier = ratio * trainer_inference_importance_sampling_ratio
    if do_overlong_filtering:
        batch_size, _ = component_multiplier.shape
        component_multiplier = component_multiplier * (1 - batch.truncated.reshape(batch_size, 1))

    reward_loss = -masked_response_mean(component_multiplier * reward_advantages, loss_masks)
    opd_loss = -masked_response_mean(component_multiplier * scaled_opd_advantages, loss_masks)
    combined_unclipped_loss = -masked_response_mean(component_multiplier * combined_advantages, loss_masks)

    log_ratio = jnp.zeros_like(current_logprobs)
    kl_loss = jnp.asarray(0.0, dtype=current_logprobs.dtype)
    kl_statistics = kl_statistics_from_log_ratio(log_ratio, loss_masks)
    if kl.enabled():
        if reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        reference_logprobs, _reference_entropy = compute_policy_stats_fn(
            reference_model, batch, key, compute_entropy=False
        )
        reference_logprobs = reference_logprobs * loss_masks
        log_ratio = token_log_ratio(current_logprobs, reference_logprobs)
        kl_penalty = kl_penalty_from_log_ratio(log_ratio, kl.mode)
        kl_statistics = kl_statistics_from_log_ratio(log_ratio, loss_masks)
        kl_loss = kl.beta * masked_response_mean(kl_penalty, loss_masks)

    loss = reinforce_loss + kl_loss

    trainer_inference_importance_sampling_ratio_mean = masked_response_mean(
        trainer_inference_importance_sampling_ratio, loss_masks
    )
    ratio_mean = masked_response_mean(ratio, loss_masks)
    clipped_ratio_mean = masked_response_mean(clipped_ratio, loss_masks)
    reward_advantage_mean = masked_response_mean(reward_advantages, loss_masks)
    opd_advantage_mean = masked_response_mean(opd_advantages, loss_masks)
    combined_advantage_mean = masked_response_mean(combined_advantages, loss_masks)
    current_teacher_gap_mean = masked_response_mean(current_logprobs - teacher_logprobs, loss_masks)
    behavior_teacher_gap_mean = masked_response_mean(behavior_logprobs - teacher_logprobs, loss_masks)

    metrics = {
        "ratio_mean": Metric.from_value(ratio_mean.astype(jnp.float32), ReductionType.MEAN),
        "clipped_ratio_mean": Metric.from_value(clipped_ratio_mean.astype(jnp.float32), ReductionType.MEAN),
        "clip_fraction": Metric.from_value(clip_fraction.astype(jnp.float32), ReductionType.MEAN),
        "reinforce_loss": Metric.from_value(reinforce_loss.astype(jnp.float32), ReductionType.MEAN),
        "hybrid/combined_reinforce_loss": Metric.from_value(reinforce_loss.astype(jnp.float32), ReductionType.MEAN),
        "hybrid/reward_loss": Metric.from_value(reward_loss.astype(jnp.float32), ReductionType.MEAN),
        "hybrid/opd_loss": Metric.from_value(opd_loss.astype(jnp.float32), ReductionType.MEAN),
        "hybrid/combined_unclipped_loss": Metric.from_value(
            combined_unclipped_loss.astype(jnp.float32), ReductionType.MEAN
        ),
        "hybrid/opd_coef": Metric.from_value(jnp.asarray(opd_coef, dtype=jnp.float32), ReductionType.MEAN),
        "hybrid/reward_advantage_mean": Metric.from_value(reward_advantage_mean.astype(jnp.float32), ReductionType.MEAN),
        "hybrid/opd_advantage_mean": Metric.from_value(opd_advantage_mean.astype(jnp.float32), ReductionType.MEAN),
        "hybrid/combined_advantage_mean": Metric.from_value(
            combined_advantage_mean.astype(jnp.float32), ReductionType.MEAN
        ),
        "hybrid/current_teacher_gap_mean": Metric.from_value(
            current_teacher_gap_mean.astype(jnp.float32), ReductionType.MEAN
        ),
        "hybrid/behavior_teacher_gap_mean": Metric.from_value(
            behavior_teacher_gap_mean.astype(jnp.float32), ReductionType.MEAN
        ),
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
        **compute_metadata_metrics(current_logprobs, behavior_logprobs, combined_advantages, loss_masks),
        **metadata,
    }

    if log_policy_entropy:
        if current_policy_entropy is None:
            raise ValueError("compute_policy_stats_fn must return entropy when log_policy_entropy=True")
        policy_entropy = masked_response_mean(current_policy_entropy, loss_masks)
        metrics["current_policy_entropy"] = Metric.from_value(
            jax.lax.stop_gradient(policy_entropy).astype(jnp.float32), ReductionType.MEAN
        )

    return loss, metrics


@dataclass
class HybridRLOOOPDSampledTokenReverseKLLoss(RLLossModule):
    """RLOO reward loss plus sampled-token reverse-KL OPD from a local teacher."""

    kl: KLConfig
    opd_coef: float
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    tis_importance_sampling_ratio_max: float = 2.0
    synchronous: bool = False
    do_trainer_inference_mismatch_importance_sampling: bool = False
    do_overlong_filtering: bool = False
    vocab_tile_size: int | None = None
    log_policy_entropy: bool = False

    def __post_init__(self) -> None:
        if self.opd_coef < 0.0:
            raise ValueError("opd_coef must be non-negative")
        if self.clip_epsilon_low < 0.0:
            raise ValueError("clip_epsilon_low must be non-negative")
        if self.clip_epsilon_high < 0.0:
            raise ValueError("clip_epsilon_high must be non-negative")
        if self.tis_importance_sampling_ratio_max <= 0.0:
            raise ValueError("tis_importance_sampling_ratio_max must be positive")

    def build(self, reference_model: eqx.Module) -> RLLossModule:
        """Initialize learned components."""
        return self

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        """Compute reward advantages with RLOO; OPD adds a token-level advantage at loss time."""
        return compute_rloo_advantages(rollout_group)

    def needs_reference_model(self) -> bool:
        """Return whether KL regularization requires a reference model."""
        return self.kl.enabled()

    def needs_teacher_model(self) -> bool:
        """Return whether this loss needs a separately retained teacher model."""
        return True

    def create_loss_fn(
        self,
        reference_model: eqx.Module | None,
        train_model: eqx.Module,
        *,
        teacher_model: eqx.Module | None = None,
    ) -> Callable:
        """Create the loss function for training."""
        del train_model
        if self.needs_reference_model() and reference_model is None:
            raise ValueError("reference_model is required when KL regularization is enabled")
        if teacher_model is None:
            raise ValueError("teacher_model is required for HybridRLOOOPDSampledTokenReverseKLLoss")
        if self.log_policy_entropy and self.vocab_tile_size is not None:
            raise ValueError(
                "Exact policy entropy is not supported with vocab_tile_size yet. "
                "Set vocab_tile_size=None or implement chunked entropy first."
            )

        if self.vocab_tile_size is not None:

            def compute_policy_stats_fn(m, b, k, *, compute_entropy: bool):
                if compute_entropy:
                    raise ValueError("Exact policy entropy is not supported with vocab_tile_size yet")
                return chunked_compute_logprobs(m, b, k, self.vocab_tile_size), None

        else:
            compute_policy_stats_fn = compute_logprobs_and_entropy

        def loss_fn(model, batch, key):
            return hybrid_rloo_sampled_token_reverse_kl_opd_loss(
                model,
                reference_model,
                teacher_model,
                batch,
                key=key,
                kl=self.kl,
                opd_coef=self.opd_coef,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                tis_importance_sampling_ratio_max=self.tis_importance_sampling_ratio_max,
                synchronous=self.synchronous,
                do_trainer_inference_mismatch_importance_sampling=self.do_trainer_inference_mismatch_importance_sampling,
                do_overlong_filtering=self.do_overlong_filtering,
                log_policy_entropy=self.log_policy_entropy,
                compute_policy_stats_fn=compute_policy_stats_fn,
            )

        return loss_fn
