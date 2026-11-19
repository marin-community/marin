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
    judge: ModelConfig | None = None,
    instances: int = 1,
    task_timeout_seconds: int | None = None,
) -> ArtifactStep[EvaluationResult]:
    """Evaluate the HF export and tokenizer identified by a trained model artifact."""

    def resolve_model(ctx: StepContext) -> ModelConfig:
        resolved = model.resolve(ctx)
        return replace(
            serving,
            location=resolved.uri,
            identity=resolved.identity,
            tokenizer=resolved.tokenizer_uri or resolved.uri,
            tokenizer_revision=resolved.tokenizer_revision,
        )

    step = eval_step(
        serving,
        evals,
        version=version,
        deps=model.deps(),
        resolve_model=resolve_model,
        accelerator=accelerator,
        submission_cluster=submission_cluster,
        federated_cluster=federated_cluster,
        judge=judge,
        instances=instances,
        task_timeout_seconds=task_timeout_seconds,
    )
    if not isinstance(model, SkyRLPolicyModel):
        return step

    def with_run_ref(ctx: StepContext):
        config = step.build_config(ctx)
        if ctx.is_fingerprint:
            return config
        run = ctx.resolved(model.step)
        if run.run_id is None or run.attempt_id is None:
            return config
        return replace(config, rl_run=RLRunRef(run_id=run.run_id, attempt_id=run.attempt_id))

    return replace(step, build_config=with_run_ref)
