# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run authoritative source declarations against staged data."""

from collections.abc import Mapping

from taskcompendium.models import EnvironmentRequirements
from taskcompendium.pipeline.conversion import ConversionResult
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourcePipelineResult, SourceProcessingMode
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.pipeline import convert_pipeline
from experiments.post_training.task_curation.source import RlDataSource


def convert_source(
    source: RlDataSource,
    *,
    mode: SourceProcessingMode,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    inputs: Mapping[str, str],
    config: SourcePipelineConfig | None = None,
    grader_environment: EnvironmentRequirements | None = None,
) -> ConversionResult | SourcePipelineResult:
    """Convert a declared source in quick, sample, or full mode using its bound recipe."""
    if source.pipeline is None:
        raise ValueError(f"Source has no conversion pipeline: {source.metadata.id}")
    return convert_pipeline(
        source.pipeline,
        mode=mode,
        context=context,
        source_input=source_input,
        output_path=output_path,
        inputs=inputs,
        config=config,
        grader_environment=grader_environment,
    )
