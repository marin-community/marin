# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable loss terms for RL/post-training objectives."""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from levanter.metrics import Metric, ReductionType

from marin.rl.scoring import ScoreBundle, ScoreRequirements
from marin.rl.types import SequenceBatch

from .reductions import reduce_loss_objective
from .signals import PreparedSignals
from .spec import PolicyGradientTermConfig, ReductionConfig, ReferenceKLTermConfig, TruncationPolicy


@dataclass(frozen=True)
class LossTermResult:
    """Scalar loss contribution plus observability metrics."""

    loss: jax.Array
    metrics: dict[str, Metric]


class LossTerm(Protocol):
    """Composable scalar loss term."""

    def score_requirements(self) -> ScoreRequirements:
        """Return the score requirements for this term."""
        ...

    def compute(self, batch: SequenceBatch, signals: PreparedSignals, scores: ScoreBundle) -> LossTermResult:
        """Compute the scalar loss contribution for one prepared batch."""
        ...


def importance_sampling_ratio(
    current_logprobs: jax.Array,
    policy_logprobs_array: jax.Array,
    loss_masks_array: jax.Array,
    *,
    stop_current_logprob_gradient: bool = False,
    stop_policy_logprob_gradient: bool = True,
) -> jax.Array:
    """Compute per-token policy-ratio weights."""
    if stop_current_logprob_gradient:
        current_logprobs = jax.lax.stop_gradient(current_logprobs)

    if stop_policy_logprob_gradient:
        policy_logprobs_array = jax.lax.stop_gradient(policy_logprobs_array)

    return jnp.exp(current_logprobs - policy_logprobs_array) * loss_masks_array


def compute_metadata_metrics(
    current_logprobs: jax.Array,
    policy_logprobs_array: jax.Array,
    loss_weights_array: jax.Array,
    loss_masks_array: jax.Array,
) -> dict[str, Metric]:
    """Compute legacy RLOO metadata metrics from token-level tensors."""
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

    return {
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


@dataclass(frozen=True)
class PolicyGradientTerm:
    """Clipped policy-gradient term over token-level sequence signals."""

    config: PolicyGradientTermConfig
    reduction: ReductionConfig
    truncation_policy: TruncationPolicy

    def score_requirements(self) -> ScoreRequirements:
        return ScoreRequirements(
            student_logprobs=True,
            behavior_logprobs=True,
        )

    def compute(self, batch: SequenceBatch, signals: PreparedSignals, scores: ScoreBundle) -> LossTermResult:
        if scores.student_logprobs is None:
            raise ValueError("student logprobs are required for PolicyGradientTerm")
        if scores.behavior_logprobs is None:
            raise ValueError("behavior logprobs are required for PolicyGradientTerm")

        current_logprobs = scores.student_logprobs * signals.loss_mask.array
        behavior_logprobs = scores.behavior_logprobs
        loss_weights_array = signals.token_weights.array
        loss_masks_array = signals.loss_mask.array

        if self.config.synchronous:
            policy_logprobs_for_ratio = current_logprobs
        else:
            policy_logprobs_for_ratio = behavior_logprobs

        ratio = importance_sampling_ratio(
            current_logprobs,
            policy_logprobs_for_ratio,
            loss_masks_array,
            stop_current_logprob_gradient=False,
            stop_policy_logprob_gradient=True,
        )
        clipped_ratio = jnp.clip(
            ratio,
            min=1.0 - self.config.clip_epsilon_low,
            max=1.0 + self.config.clip_epsilon_high,
        )

        is_clipped = jnp.logical_or(
            ratio > 1.0 + self.config.clip_epsilon_high,
            ratio < 1.0 - self.config.clip_epsilon_low,
        )
        clip_fraction = jnp.mean(jnp.sum(is_clipped * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1))

        if self.config.do_trainer_inference_mismatch_importance_sampling:
            trainer_inference_ratio = importance_sampling_ratio(
                current_logprobs,
                behavior_logprobs,
                loss_masks_array,
                stop_current_logprob_gradient=True,
                stop_policy_logprob_gradient=True,
            )
            trainer_inference_ratio = jnp.minimum(
                trainer_inference_ratio,
                self.config.tis_importance_sampling_ratio_max,
            )
        else:
            trainer_inference_ratio = jnp.ones_like(current_logprobs)

        non_clipped_objective = ratio * loss_weights_array * loss_masks_array
        clipped_objective = clipped_ratio * loss_weights_array * loss_masks_array
        loss_objective = jnp.minimum(non_clipped_objective, clipped_objective)
        loss_objective = trainer_inference_ratio * loss_objective

        if self.truncation_policy is TruncationPolicy.FILTER_ENTIRE_RESPONSE:
            batch_size, _ = loss_objective.shape
            loss_objective = loss_objective * (1 - batch.truncated.reshape(batch_size, 1))

        reinforce_loss = reduce_loss_objective(
            loss_objective,
            loss_masks_array,
            self.reduction,
            max_output_tokens=batch.max_output_tokens,
        )

        per_batch_loss = jnp.sum(loss_objective * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
        trainer_inference_ratio_mean = jnp.mean(
            jnp.sum(trainer_inference_ratio * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
        )
        ratio_mean_over_responses_only = jnp.mean(jnp.sum(ratio, axis=1) / jnp.sum(loss_masks_array, axis=1))
        clipped_ratio_mean_over_responses_only = jnp.mean(
            jnp.sum(clipped_ratio * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
        )

        metrics = {
            "ratio_mean": Metric.from_value(ratio_mean_over_responses_only.astype(jnp.float32), ReductionType.MEAN),
            "clipped_ratio_mean": Metric.from_value(
                clipped_ratio_mean_over_responses_only.astype(jnp.float32),
                ReductionType.MEAN,
            ),
            "clip_fraction": Metric.from_value(clip_fraction.astype(jnp.float32), ReductionType.MEAN),
            "reinforce_loss": Metric.from_value(reinforce_loss.astype(jnp.float32), ReductionType.MEAN),
            "trainer_inference_importance_sampling_ratio_mean": Metric.from_value(
                trainer_inference_ratio_mean.astype(jnp.float32),
                ReductionType.MEAN,
            ),
            "temperature": Metric.from_value(
                jnp.mean(batch.sampling_temperature.array).astype(jnp.float32),
                ReductionType.MEAN,
            ),
            "top_k": Metric.from_value(jnp.mean(batch.sampling_top_k.array).astype(jnp.float32), ReductionType.MEAN),
            "loss_max_over_batch": Metric.from_value((-jnp.max(per_batch_loss)).astype(jnp.float32), ReductionType.MAX),
            "loss_std_over_batch": Metric.from_value(jnp.std(per_batch_loss).astype(jnp.float32), ReductionType.MEAN),
            **compute_metadata_metrics(current_logprobs, behavior_logprobs, loss_weights_array, loss_masks_array),
        }
        return LossTermResult(loss=reinforce_loss, metrics=metrics)


@dataclass(frozen=True)
class ReferenceKLTerm:
    """Reference-model KL regularizer."""

    config: ReferenceKLTermConfig

    def score_requirements(self) -> ScoreRequirements:
        return ScoreRequirements(
            student_logprobs=self.config.kl_coef > 0,
            reference_logprobs=self.config.kl_coef > 0,
        )

    def compute(self, batch: SequenceBatch, signals: PreparedSignals, scores: ScoreBundle) -> LossTermResult:
        del batch

        if self.config.kl_coef <= 0:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return LossTermResult(
                loss=zero,
                metrics={
                    "kl_loss": Metric.from_value(zero, ReductionType.MEAN),
                    "kl_penalty": Metric.from_value(zero, ReductionType.MEAN),
                },
            )

        if scores.student_logprobs is None:
            raise ValueError("student logprobs are required for ReferenceKLTerm")
        if scores.reference_logprobs is None:
            raise ValueError("reference logprobs are required for ReferenceKLTerm")

        loss_masks_array = signals.loss_mask.array
        current_logprobs = scores.student_logprobs * loss_masks_array
        reference_logprobs = scores.reference_logprobs * loss_masks_array

        kl_penalty = jnp.exp(reference_logprobs - current_logprobs) - (reference_logprobs - current_logprobs) - 1
        kl_loss = jnp.mean(jnp.sum(kl_penalty * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1))
        kl_loss = self.config.kl_coef * kl_loss

        kl_penalty_over_responses_only = jnp.mean(
            jnp.sum(kl_penalty * loss_masks_array, axis=1) / jnp.sum(loss_masks_array, axis=1)
        )
        return LossTermResult(
            loss=kl_loss,
            metrics={
                "kl_loss": Metric.from_value(jnp.asarray(kl_loss, dtype=jnp.float32), ReductionType.MEAN),
                "kl_penalty": Metric.from_value(
                    jnp.asarray(kl_penalty_over_responses_only, dtype=jnp.float32),
                    ReductionType.MEAN,
                ),
            },
        )


def build_loss_terms(
    term_configs: tuple[PolicyGradientTermConfig | ReferenceKLTermConfig, ...],
    *,
    reduction: ReductionConfig,
    truncation_policy: TruncationPolicy,
) -> tuple[LossTerm, ...]:
    """Instantiate concrete loss terms from the public objective spec."""
    terms: list[LossTerm] = []
    for config in term_configs:
        if isinstance(config, PolicyGradientTermConfig):
            terms.append(
                PolicyGradientTerm(
                    config=config,
                    reduction=reduction,
                    truncation_policy=truncation_policy,
                )
            )
            continue
        if isinstance(config, ReferenceKLTermConfig):
            terms.append(ReferenceKLTerm(config=config))
            continue

        raise TypeError(f"Unsupported loss term config: {type(config)!r}")
    return tuple(terms)
