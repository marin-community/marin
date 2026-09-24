# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt a completed SkyRL policy to the typed target model used by evaluation steps."""

from dataclasses import replace

from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.rl.skyrl import SkyRLModel

from experiments.evaluation.pipeline import TargetModelArtifact, TargetModelStepConfig, run_target_model_step

SKYRL_POLICY_LOCATION = "<skyrl-policy>"


def skyrl_target_model_step(
    source: ArtifactStep[SkyRLModel], model: ModelConfig, *, name: str, version: str
) -> ArtifactStep[TargetModelArtifact]:
    """Record a terminal SkyRL policy's export URI and validated tokenizer for evaluation."""
    if model.location != SKYRL_POLICY_LOCATION:
        raise ValueError(f"SkyRL evaluation model location must be {SKYRL_POLICY_LOCATION!r}")
    if model.tokenizer is None:
        raise ValueError("SkyRL evaluation models require an explicit Hugging Face tokenizer")

    def build_config(ctx: StepContext) -> TargetModelStepConfig:
        if ctx.is_fingerprint:
            resolved = replace(model, location=f"{artifact_identity(source)}/policy", identity=artifact_identity(source))
        else:
            terminal = ctx.resolved(source)
            resolved = replace(
                model,
                location=terminal.policy_export_uri,
                identity=artifact_identity(source),
                tokenizer=terminal.tokenizer_uri,
                tokenizer_revision=terminal.tokenizer_revision,
            )
        return TargetModelStepConfig(artifact_path=ctx.output_path, model=resolved)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TargetModelArtifact,
        run=run_target_model_step,
        build_config=build_config,
        deps=(source,),
    )
