# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invocation options and concrete configuration for process_rows."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.controls import GradingMachines
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB
from experiments.post_training.task_curation.campaign import CampaignRuntime


@dataclass(frozen=True)
class RecipeSettings:
    """Options for process_rows; service construction belongs to its execution path."""

    config: SourcePipelineConfig | None = None
    review: ReviewConfig | None = None
    review_cache: str | None = None
    base_url: str | None = None
    relay_job: str = DEFAULT_GLM_RELAY_JOB
    review_concurrency: int = MAX_DIRECT_CONCURRENT_REQUESTS
    normalized_shards: int | None = None
    machines: GradingMachines | None = None
    seed: int = 0
    verification_sample_size: int = 20
    execution: AuditExecution = field(default_factory=AuditExecution)


@dataclass(frozen=True)
class InputOverrides:
    root: str | None = None
    files: Mapping[str, Path] = field(default_factory=dict)
    auxiliary: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineOptions:
    """Invocation choices; a dataset owns its dependencies and optional stages."""

    mode: SourceProcessingMode
    runtime: CampaignRuntime
    inputs: InputOverrides = field(default_factory=InputOverrides)
    recipe_settings: RecipeSettings | None = None
