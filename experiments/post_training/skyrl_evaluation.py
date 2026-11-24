# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve completed SkyRL policies for experiment evaluations."""

from dataclasses import replace

from evalrecords.records import RLAttemptRef
from finestore.rl.run_records import read_attempt
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLRun, resolve_skyrl_policy
from marin.rl.skyrl_records import EXPORTS_SUBDIRECTORY
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.evals import EvaluationDefinition
from experiments.evaluation.pipeline import EvaluationResult, eval_step

SKYRL_POLICY_LOCATION = "<skyrl-policy>"


def resolve_skyrl_model(ctx: StepContext, source: ArtifactStep[SkyRLRun], model: ModelConfig) -> ModelConfig:
    """Resolve a terminal policy's export URI and tokenizer in an evaluation step."""
    if model.location != SKYRL_POLICY_LOCATION:
        raise ValueError(f"SkyRL evaluation model location must be {SKYRL_POLICY_LOCATION!r}")

    resolved = resolve_skyrl_policy(ctx, source)
    return replace(
        model,
        location=resolved.uri,
        identity=resolved.identity,
        tokenizer=resolved.tokenizer_uri or resolved.uri,
        tokenizer_revision=resolved.tokenizer_revision,
    )


def resolve_rl_attempt(ctx: StepContext, source: ArtifactStep[SkyRLRun]) -> RLAttemptRef | None:
    """Identify the exact RL attempt and checkpoint evaluated by this step."""
    run = ctx.resolved(source)
    if run.run_id is None or run.attempt_id is None:
        return None
    record = read_attempt(prefix_join(ctx.artifact_path(source), EXPORTS_SUBDIRECTORY), run.attempt_id)
    run_id = record.index_run_id if record is not None and record.run_id == run.run_id else run.run_id
    return RLAttemptRef(run_id=run_id, attempt_id=run.attempt_id, step=run.global_step)


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
        resolve_rl_attempt=lambda ctx: resolve_rl_attempt(ctx, source),
        accelerator=accelerator,
        submission_cluster=submission_cluster,
        federated_cluster=federated_cluster,
    )
