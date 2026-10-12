# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind shared assessment work downstream of a source's curation artifact."""

import os
from dataclasses import dataclass, replace
from functools import partial

from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.assessment import AssessmentSampling, assessment_summary, prepare_assessment, write_json
from taskcompendium.pipeline.assessment_review import AssessmentModel, grade_assessment
from taskcompendium.pipeline.chat_requests import chat_output
from taskcompendium.pipeline.review_requests import batch_output
from taskcompendium.pipeline.stages import ReviewMode

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url
from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime, PipelineResult
from experiments.post_training.task_curation.config import RecipeSettings

INPUTS_VERSION = "2026.10.12.8"
JUDGMENTS_VERSION = "2026.10.12.5"
REPORT_VERSION = "2026.10.12.3"


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


def _prepare(runtime: CampaignRuntime, *, curated_root: str, sampling: AssessmentSampling, output_root: str) -> None:
    prepare_assessment(runtime.context, curated_root=curated_root, sampling=sampling, output_root=output_root)


def _judge(
    runtime: CampaignRuntime,
    execution: RecipeSettings,
    *,
    inputs_root: str,
    rubric: str,
    model: AssessmentModel,
    output_root: str,
) -> None:
    assert execution.review is not None and execution.review_cache is not None
    base_url = execution.base_url or resolve_glm_base_url(execution.relay_job)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    if execution.review.mode == ReviewMode.CHAT:
        submit = partial(
            chat_output,
            OpenAIChatClient(base_url, token, timeout=60),
            max_concurrent=execution.review_concurrency,
        )
    else:
        submit = partial(
            batch_output, OpenAIBatchClient(base_url, token, timeout=60, request_attempts=3), poll_seconds=5.0
        )
    grade_assessment(
        runtime.context,
        inputs_root=inputs_root,
        rubric=rubric,
        model=model,
        submit=submit,
        cache_root=execution.review_cache,
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
    settings: AssessmentSettings,
    runtime: CampaignRuntime,
    execution: RecipeSettings,
) -> AssessmentStages:
    """Keep preparation reusable when only the rubric or model changes."""
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
        partial(_judge, runtime, execution),
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
