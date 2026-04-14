# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Objective runtime for neutral RL/post-training batches."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmHeadModel

from marin.rl.scoring import LocalScoreSource, ModelRoles, ScoreRequirements, ScoreSource
from marin.rl.types import BatchInfo, SequenceBatch

from .signals import PreparedSignals, SignalBuilder, build_signal_builder
from .spec import BatchView, ObjectiveSpec
from .terms import LossTerm, build_loss_terms


class ObjectiveBatch(eqx.Module):
    """Prepared device-ready batch for an objective runtime."""

    sequence_batch: SequenceBatch
    signals: PreparedSignals


@dataclass(frozen=True)
class ObjectiveRuntimeConfig:
    """Runtime configuration for compiling one objective recipe."""

    objective: ObjectiveSpec
    metric_namespace: str = ""
    vocab_tile_size: int | None = None


@dataclass(frozen=True)
class ObjectiveRuntime:
    """Compiled runtime for a concrete objective recipe."""

    config: ObjectiveRuntimeConfig
    score_source: ScoreSource
    signal_builder: SignalBuilder
    terms: tuple[LossTerm, ...]
    score_requirements: ScoreRequirements

    def prepare_batch(self, batch: SequenceBatch, info: BatchInfo) -> ObjectiveBatch:
        """Prepare one neutral batch for device execution."""
        if self.config.objective.batch_view is not BatchView.SEQUENCE:
            raise NotImplementedError(f"Batch view {self.config.objective.batch_view!r} is not implemented yet")

        return ObjectiveBatch(
            sequence_batch=batch,
            signals=self.signal_builder.build(batch, info),
        )

    def create_loss_fn(
        self,
        *,
        reference_model: LmHeadModel | None = None,
        teacher_model: LmHeadModel | None = None,
    ):
        """Create the objective loss function used by the trainer."""

        def loss_fn(model: LmHeadModel, batch: ObjectiveBatch, key: jax.Array | None):
            score_bundle = self.score_source.score(
                batch.sequence_batch,
                info=None,
                roles=ModelRoles(student=model, reference=reference_model, teacher=teacher_model),
                key=key,
            )

            total_loss = jnp.asarray(0.0, dtype=jnp.float32)
            metrics: dict[str, Metric] = {}
            for term in self.terms:
                result = term.compute(batch.sequence_batch, batch.signals, score_bundle)
                total_loss = total_loss + result.loss
                metrics.update(result.metrics)

            metrics.update(
                {
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
                }
            )

            if self.config.metric_namespace:
                metrics = {f"{self.config.metric_namespace}/{name}": metric for name, metric in metrics.items()}

            return total_loss, metrics

        return loss_fn


def _merge_score_requirements(requirements: tuple[ScoreRequirements, ...]) -> ScoreRequirements:
    teacher_topk_support: int | None = None
    for requirement in requirements:
        if requirement.teacher_topk_support is None:
            continue
        if teacher_topk_support is None:
            teacher_topk_support = requirement.teacher_topk_support
        else:
            teacher_topk_support = max(teacher_topk_support, requirement.teacher_topk_support)

    return ScoreRequirements(
        student_logprobs=any(requirement.student_logprobs for requirement in requirements),
        behavior_logprobs=any(requirement.behavior_logprobs for requirement in requirements),
        reference_logprobs=any(requirement.reference_logprobs for requirement in requirements),
        teacher_token_logprobs=any(requirement.teacher_token_logprobs for requirement in requirements),
        teacher_entropy=any(requirement.teacher_entropy for requirement in requirements),
        teacher_topk_support=teacher_topk_support,
    )


def _validate_score_source_requirements(provided: ScoreRequirements, required: ScoreRequirements) -> None:
    missing: list[str] = []
    if required.student_logprobs and not provided.student_logprobs:
        missing.append("student_logprobs")
    if required.behavior_logprobs and not provided.behavior_logprobs:
        missing.append("behavior_logprobs")
    if required.reference_logprobs and not provided.reference_logprobs:
        missing.append("reference_logprobs")
    if required.teacher_token_logprobs and not provided.teacher_token_logprobs:
        missing.append("teacher_token_logprobs")
    if required.teacher_entropy and not provided.teacher_entropy:
        missing.append("teacher_entropy")
    if required.teacher_topk_support is not None:
        if provided.teacher_topk_support is None or provided.teacher_topk_support < required.teacher_topk_support:
            missing.append(f"teacher_topk_support>={required.teacher_topk_support}")

    if missing:
        raise ValueError(f"score source is missing required scores: {', '.join(missing)}")


def build_objective_runtime(
    config: ObjectiveRuntimeConfig,
    *,
    score_source: ScoreSource | None = None,
) -> ObjectiveRuntime:
    """Build a compiled objective runtime from the public objective spec."""
    signal_builder = build_signal_builder(config.objective.signal_builder)
    terms = build_loss_terms(
        config.objective.terms,
        reduction=config.objective.reduction,
        truncation_policy=config.objective.truncation_policy,
    )
    score_requirements = _merge_score_requirements(tuple(term.score_requirements() for term in terms))

    resolved_score_source = score_source
    if resolved_score_source is None:
        resolved_score_source = LocalScoreSource(
            score_requirements=score_requirements,
            vocab_tile_size=config.vocab_tile_size,
        )
    _validate_score_source_requirements(resolved_score_source.requirements(), score_requirements)

    return ObjectiveRuntime(
        config=config,
        score_source=resolved_score_source,
        signal_builder=signal_builder,
        terms=terms,
        score_requirements=score_requirements,
    )
