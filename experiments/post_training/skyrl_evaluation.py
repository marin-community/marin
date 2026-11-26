# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind model artifacts to the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from evalrecords.records import RLAttemptRef
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLModelSource, SkyRLPolicyModel

if TYPE_CHECKING:
    from marin.evaluation.model_config import ModelConfig

    from experiments.evaluation.evals import EvaluationDefinition
    from experiments.evaluation.pipeline import EvaluationResult


class EvaluationRecipe(Protocol):
    """Evaluate an exported model and publish its records."""

    def evaluated(self, model: SkyRLModelSource, *, version: str) -> ArtifactStep[EvaluationResult]: ...


@dataclass(frozen=True)
class EvaluationPreset:
    name: str
    recipe: EvaluationRecipe

    def evaluated(self, model: SkyRLModelSource, *, version: str) -> ArtifactStep[EvaluationResult]:
        return replace(self.recipe.evaluated(model, version=version), name=f"{model.step.name}/{self.name}")


@dataclass(frozen=True)
class CuratedEvaluation:
    evals: tuple[EvaluationDefinition, ...]
    serving: ModelConfig
    accelerator: str | None
    cluster: str
    instances: int = 1
    max_concurrent: int = 1

    def evaluated(self, model: SkyRLModelSource, *, version: str) -> ArtifactStep[EvaluationResult]:
        return evaluate_model(
            model,
            self.serving,
            self.evals,
            version=version,
            accelerator=self.accelerator,
            submission_cluster=self.cluster,
            federated_cluster=self.cluster,
            instances=self.instances,
            max_concurrent=self.max_concurrent,
        )


def model_rl_attempt(model: SkyRLModelSource, context: StepContext) -> RLAttemptRef | None:
    if not isinstance(model, SkyRLPolicyModel) or context.is_fingerprint:
        return None
    run = context.resolved(model.step)
    if run.run_id is None or run.attempt_id is None:
        return None
    return RLAttemptRef(run_id=run.run_id, attempt_id=run.attempt_id)


def evaluate_model(
    model: SkyRLModelSource,
    serving: ModelConfig,
    evals: tuple[EvaluationDefinition, ...],
    *,
    version: str,
    accelerator: str | None,
    submission_cluster: str,
    federated_cluster: str | None,
    instances: int = 1,
    max_concurrent: int | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate an artifact using a serving config with an empty location."""
    from experiments.evaluation.pipeline import DEFAULT_EVAL_CONCURRENCY, eval_step  # noqa: PLC0415

    if serving.location or any(
        value is not None
        for value in (serving.revision, serving.identity, serving.tokenizer, serving.tokenizer_revision)
    ):
        raise ValueError("Evaluation location, revision, identity, tokenizer and tokenizer_revision come from the model")

    def resolve_model(ctx: StepContext) -> ModelConfig:
        resolved = model.resolve(ctx)
        return replace(
            serving,
            location=resolved.uri,
            identity=resolved.identity,
            tokenizer=resolved.tokenizer_uri or resolved.uri,
            tokenizer_revision=resolved.tokenizer_revision,
        )

    return eval_step(
        serving,
        evals,
        version=version,
        deps=model.deps(),
        resolve_model=resolve_model,
        resolve_rl_attempt=lambda context: model_rl_attempt(model, context),
        accelerator=accelerator,
        submission_cluster=submission_cluster,
        federated_cluster=federated_cluster,
        instances=instances,
        max_concurrent=DEFAULT_EVAL_CONCURRENCY if max_concurrent is None else max_concurrent,
    )
