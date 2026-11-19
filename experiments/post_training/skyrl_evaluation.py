# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind model artifacts to the evaluation pipeline."""

from dataclasses import replace

from evalrecords.records import RLRunRef
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLModelSource, SkyRLPolicyModel

from experiments.evaluation.evals import EvaluationDefinition
from experiments.evaluation.pipeline import EvaluationResult, eval_step


def evaluate(
    model: SkyRLModelSource,
    serving: ModelConfig,
    evals: tuple[EvaluationDefinition, ...],
    *,
    version: str,
    accelerator: str | None,
    submission_cluster: str,
    federated_cluster: str | None,
    job_timeout_seconds: int | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate a trained model with serving settings and replace its location with the resolved export."""
    if any(
        value is not None
        for value in (serving.revision, serving.identity, serving.tokenizer, serving.tokenizer_revision)
    ):
        raise ValueError("Evaluation revision, identity, tokenizer and tokenizer_revision come from the model")

    def resolve_model(ctx: StepContext) -> ModelConfig:
        resolved = model.resolve(ctx)
        return replace(
            serving,
            location=resolved.uri,
            identity=resolved.identity,
            tokenizer=resolved.tokenizer_uri or resolved.uri,
            tokenizer_revision=resolved.tokenizer_revision,
        )

    def resolve_rl_run(ctx: StepContext) -> RLRunRef | None:
        if not isinstance(model, SkyRLPolicyModel):
            return None
        run = ctx.resolved(model.step)
        if run.run_id is None or run.attempt_id is None:
            return None
        return RLRunRef(run_id=run.run_id, attempt_id=run.attempt_id)

    return eval_step(
        serving,
        evals,
        version=version,
        deps=model.deps(),
        resolve_model=resolve_model,
        resolve_rl_run=resolve_rl_run,
        accelerator=accelerator,
        submission_cluster=submission_cluster,
        federated_cluster=federated_cluster,
        job_timeout_seconds=job_timeout_seconds,
    )
