# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shared assessment work downstream of a source's curation artifact."""

import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.assessment import AssessmentSampling, assessment_summary, prepare_assessment, write_json
from taskcompendium.pipeline.assessment_review import AssessmentModel, grade_assessment
from taskcompendium.pipeline.chat_requests import chat_output
from taskcompendium.pipeline.inputs import SourceFiles
from taskcompendium.pipeline.review_requests import batch_output
from taskcompendium.pipeline.sources import source_files_identity
from taskcompendium.pipeline.stages import ReviewMode

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url
from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime, PipelineResult
from experiments.post_training.task_curation.config import RecipeSettings

INPUTS_VERSION = "2026.10.13.4"
JUDGMENTS_VERSION = "2026.10.12.5"
REPORT_VERSION = "2026.10.12.3"
REQUEST_TIMEOUT = 60
REQUEST_ATTEMPTS = 3


@dataclass(frozen=True)
class AssessmentSettings:
    sampling: AssessmentSampling
    model: AssessmentModel
    rubric: str


@dataclass(frozen=True)
class AssessmentStages:
    curated: ArtifactStep[CampaignArtifact]
    inputs: ArtifactStep
    judgments: ArtifactStep
    report: ArtifactStep[CampaignArtifact]


def assessment_evidence(stages: AssessmentStages) -> dict[str, str]:
    return {
        "manifest": str(StoragePath(stages.curated.path()) / "manifest.json"),
        "assessment_inputs": stages.inputs.path(),
        "assessment_judgments": stages.judgments.path(),
        "assessment_report": str(StoragePath(stages.report.path()) / "report.json"),
    }


def _prepare(
    runtime: CampaignRuntime,
    *,
    curated_root: str,
    sampling: AssessmentSampling,
    raw_source: SourceFiles | None,
    raw_source_identity: dict[str, Any] | None,
    output_root: str,
) -> None:
    prepare_assessment(
        runtime.context,
        curated_root=curated_root,
        sampling=sampling,
        raw_source=raw_source,
        raw_source_identity=raw_source_identity,
        output_root=output_root,
    )


def _judge(
    runtime: CampaignRuntime,
    recipe_settings: RecipeSettings,
    *,
    inputs_root: str,
    rubric: str,
    model: AssessmentModel,
    output_root: str,
) -> None:
    assert recipe_settings.review is not None and recipe_settings.review_cache is not None
    base_url = recipe_settings.base_url or resolve_glm_base_url(recipe_settings.relay_job)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    if recipe_settings.review.mode == ReviewMode.CHAT:
        submit = partial(
            chat_output,
            OpenAIChatClient(base_url, token, timeout=REQUEST_TIMEOUT),
            max_concurrent=recipe_settings.review_concurrency,
        )
    else:
        submit = partial(
            batch_output,
            OpenAIBatchClient(base_url, token, timeout=REQUEST_TIMEOUT, request_attempts=REQUEST_ATTEMPTS),
            poll_seconds=5.0,
        )
    grade_assessment(
        runtime.context,
        inputs_root=inputs_root,
        rubric=rubric,
        model=model,
        submit=submit,
        cache_root=recipe_settings.review_cache,
        output_root=output_root,
    )


def _summarize(
    curated_type: type[CampaignArtifact],
    *,
    curated_root: str,
    inputs_root: str,
    judgments_root: str,
    output_root: str,
) -> CampaignArtifact:
    curated = curated_type.raw_load(curated_root)
    assert curated.result is not None
    report = assessment_summary(inputs_root, judgments_root)
    report_path = str(StoragePath(output_root) / "report.json")
    write_json(report_path, report)
    result = PipelineResult(
        curated.result.status,
        curated.result.outputs,
        {
            **curated.result.evidence,
            "assessment_inputs": inputs_root,
            "assessment_judgments": judgments_root,
            "assessment_report": report_path,
        },
        (*curated.result.stages, "assessment_inputs", "assessment_judgments", "assessment_report"),
    )
    return CampaignArtifact(path=output_root, status=curated.status, result=result)


def assessment_pipeline(
    curated: ArtifactStep[CampaignArtifact],
    *,
    source: str,
    raw_source: SourceFiles | None,
    settings: AssessmentSettings,
    runtime: CampaignRuntime,
    recipe_settings: RecipeSettings,
) -> AssessmentStages:
    """Return the evidence preparation, judgment and report stages for a source."""
    inputs = apply(
        f"rl-assessment/{source}/inputs",
        partial(_prepare, runtime, raw_source=raw_source),
        version=INPUTS_VERSION,
        curated_root=curated,
        sampling=settings.sampling,
        raw_source_identity=source_files_identity(raw_source) if raw_source is not None else None,
        output_root=OUT,
    )
    inputs = replace(inputs, name=f"{inputs.name}-{inputs.fingerprint()[:16]}")
    judgments = apply(
        f"rl-assessment/{source}/judgments",
        partial(_judge, runtime, recipe_settings),
        version=JUDGMENTS_VERSION,
        inputs_root=inputs,
        rubric=settings.rubric,
        model=settings.model,
        output_root=OUT,
    )
    judgments = replace(judgments, name=f"{judgments.name}-{judgments.fingerprint()[:16]}")
    report = apply(
        f"rl-assessment/{source}/report",
        partial(_summarize, curated.artifact_type),
        version=REPORT_VERSION,
        artifact_type=CampaignArtifact,
        curated_root=curated,
        inputs_root=inputs,
        judgments_root=judgments,
        output_root=OUT,
    )
    report = replace(report, name=f"{report.name}-{report.fingerprint()[:16]}")
    return AssessmentStages(curated, inputs, judgments, report)
