# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shared assessment work downstream of a source's curation artifact."""

import os
from dataclasses import dataclass, replace
from functools import partial

from finestore.rl.curation import AssessmentPaths
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from taskcompendium.pipeline.assessment import AssessmentSampling, prepare_assessment
from taskcompendium.pipeline.assessment_review import AssessmentModel, grade_assessment
from taskcompendium.pipeline.chat_requests import chat_output
from taskcompendium.pipeline.review_requests import DEFAULT_BATCH_POLL_SECONDS, batch_output
from taskcompendium.pipeline.stages import ReviewMode

from experiments.post_training.glm import (
    GLM_BULK_TOKEN_ENV,
    GLM_REQUEST_ATTEMPTS,
    GLM_REQUEST_TIMEOUT,
    resolve_glm_base_url,
)
from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime, PipelineResult
from experiments.post_training.task_curation.config import RecipeSettings

INPUTS_VERSION = "2026.10.13.7"
JUDGMENTS_VERSION = "2026.10.13.1"


@dataclass(frozen=True)
class AssessmentSettings:
    sampling: AssessmentSampling
    model: AssessmentModel
    rubric: str


@dataclass(frozen=True)
class AssessmentStages:
    curated: ArtifactStep[CampaignArtifact]
    inputs: ArtifactStep
    judgments: ArtifactStep[CampaignArtifact]


def assessment_paths(stages: AssessmentStages) -> AssessmentPaths:
    return AssessmentPaths(stages.curated.path(), stages.inputs.path(), stages.judgments.path())


def _prepare(
    runtime: CampaignRuntime,
    *,
    curated_root: str,
    sampling: AssessmentSampling,
    output_root: str,
) -> None:
    prepare_assessment(
        runtime.context,
        curated_root=curated_root,
        sampling=sampling,
        output_root=output_root,
    )


def _judge(
    runtime: CampaignRuntime,
    recipe_settings: RecipeSettings,
    curated_type: type[CampaignArtifact],
    *,
    curated_root: str,
    inputs_root: str,
    rubric: str,
    model: AssessmentModel,
    output_root: str,
) -> CampaignArtifact:
    assert recipe_settings.review is not None and recipe_settings.review_cache is not None
    base_url = recipe_settings.base_url or resolve_glm_base_url(recipe_settings.relay_job)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    if recipe_settings.review.mode == ReviewMode.CHAT:
        submit = partial(
            chat_output,
            OpenAIChatClient(base_url, token, timeout=GLM_REQUEST_TIMEOUT),
            max_concurrent=recipe_settings.review_concurrency,
        )
    else:
        submit = partial(
            batch_output,
            OpenAIBatchClient(base_url, token, timeout=GLM_REQUEST_TIMEOUT, request_attempts=GLM_REQUEST_ATTEMPTS),
            poll_seconds=DEFAULT_BATCH_POLL_SECONDS,
        )
    grade_assessment(
        runtime.context,
        curated_root=curated_root,
        inputs_root=inputs_root,
        rubric=rubric,
        model=model,
        submit=submit,
        cache_root=recipe_settings.review_cache,
        output_root=output_root,
    )

    curated = curated_type.raw_load(curated_root)
    assert curated.result is not None
    result = PipelineResult(
        curated.result.status,
        curated.result.outputs,
        curated.result.evidence,
        (*curated.result.stages, "assessment_inputs", "assessment_judgments"),
    )
    return CampaignArtifact(path=output_root, status=curated.status, result=result)


def assessment_pipeline(
    curated: ArtifactStep[CampaignArtifact],
    *,
    source: str,
    settings: AssessmentSettings,
    runtime: CampaignRuntime,
    recipe_settings: RecipeSettings,
) -> AssessmentStages:
    """Return the evidence preparation and judgment stages for a source."""
    inputs = apply(
        f"rl-assessment/{source}/inputs",
        partial(_prepare, runtime),
        version=INPUTS_VERSION,
        curated_root=curated,
        sampling=settings.sampling,
        output_root=OUT,
    )
    inputs = replace(inputs, name=f"{inputs.name}-{inputs.fingerprint()[:16]}")
    judgments = apply(
        f"rl-assessment/{source}/judgments",
        partial(_judge, runtime, recipe_settings, curated.artifact_type),
        version=JUDGMENTS_VERSION,
        artifact_type=CampaignArtifact,
        curated_root=curated,
        inputs_root=inputs,
        rubric=settings.rubric,
        model=settings.model,
        output_root=OUT,
    )
    judgments = replace(judgments, name=f"{judgments.name}-{judgments.fingerprint()[:16]}")
    return AssessmentStages(curated, inputs, judgments)
