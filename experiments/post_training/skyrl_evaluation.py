# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve completed SkyRL policies for experiment evaluations."""

from dataclasses import replace

from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLRun, resolve_skyrl_policy

from experiments.evaluation.evals import EvaluationDefinition
from experiments.evaluation.pipeline import EvaluationResult, eval_step

SKYRL_POLICY_LOCATION = "<skyrl-policy>"


def resolve_skyrl_model(ctx: StepContext, source: ArtifactStep[SkyRLRun], model: ModelConfig) -> ModelConfig:
    """Resolve a terminal policy's export URI and tokenizer in an evaluation step."""
    if model.location != SKYRL_POLICY_LOCATION:
        raise ValueError(f"SkyRL evaluation model location must be {SKYRL_POLICY_LOCATION!r}")
    if model.tokenizer is None:
        raise ValueError("SkyRL evaluation models require an explicit Hugging Face tokenizer")

    resolved = resolve_skyrl_policy(ctx, source)
    return replace(
        model,
        location=resolved.uri,
        identity=resolved.identity,
        tokenizer=resolved.tokenizer_uri,
        tokenizer_revision=resolved.tokenizer_revision,
    )


def skyrl_eval_step(
    source: ArtifactStep[SkyRLRun],
    model: ModelConfig,
    evals: tuple[EvaluationDefinition, ...],
    *,
    version: str,
    accelerator: str | None,
    submission_cluster: str,
    federated_cluster: str | None,
) -> ArtifactStep[EvaluationResult]:
    """Build an evaluation that resolves a completed SkyRL policy export."""
    return eval_step(
        model,
        evals,
        version=version,
        deps=(source,),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, source, model),
        accelerator=accelerator,
        submission_cluster=submission_cluster,
        federated_cluster=federated_cluster,
    )
