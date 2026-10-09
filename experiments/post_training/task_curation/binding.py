# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind dataset-owned invocation to campaign artifacts."""

import hashlib
from dataclasses import asdict, dataclass, field
from functools import partial

from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from taskcompendium.pipeline.fingerprints import function_code_identity
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime
from experiments.post_training.task_curation.invocation import PipelineRun
from experiments.post_training.task_curation.pipeline import PIPELINE_VERSION, RlDataPipeline, source_step
from experiments.post_training.task_curation.source import RlDataSource


@dataclass(frozen=True)
class CustomRun:
    identity: dict[str, object]
    output_path: str
    source_input: str | None
    inputs: dict[str, str] = field(default_factory=dict)


def _custom_run(
    identity: dict[str, object], source_input: str | None, inputs: dict[str, str], ctx: StepContext
) -> CustomRun:
    return CustomRun(identity, ctx.output_path, source_input, inputs)


def _invoke(
    source: RlDataSource, mode: SourceProcessingMode, run: CustomRun, *, campaign: CampaignRuntime
) -> CampaignArtifact:
    assert source.pipeline is not None
    result = source.pipeline(source, PipelineRun(mode, campaign.context, run.output_path, run.source_input, run.inputs))
    return CampaignArtifact(path=run.output_path, status=result.status, result=result)


def pipeline_step(
    source: RlDataSource,
    *,
    mode: SourceProcessingMode,
    campaign: CampaignRuntime,
    standard_config: SourcePipelineConfig | None = None,
    source_input: str | None = None,
    inputs: dict[str, str] | None = None,
) -> ArtifactStep[CampaignArtifact]:
    """Bind only the chosen implementation's dependencies; custom ingestion runs in its callable."""
    implementation = source.pipeline
    if implementation is None:
        raise ValueError(f"Source has no pipeline: {source.name}")
    if isinstance(implementation, RlDataPipeline):
        if standard_config is None:
            raise ValueError("Standard campaign binding requires reviewed settings")
        if standard_config.mode != mode:
            raise ValueError("Standard settings must use the invocation mode")
        return source_step(implementation, standard_config, campaign, source=source)
    identity = {
        "name": source.name,
        "version": source.version,
        "dataset": asdict(source.dataset) if source.dataset is not None else None,
        "files": source.files,
        "mode": mode,
        "implementation": function_code_identity(implementation),
        "source_input": source_input,
        "inputs": inputs or {},
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]
    return ArtifactStep(
        name=f"data/rl/{source.name}-{digest}",
        version=PIPELINE_VERSION,
        artifact_type=CampaignArtifact,
        run=partial(_invoke, source, mode, campaign=campaign),
        build_config=partial(_custom_run, identity, source_input, inputs or {}),
    )
