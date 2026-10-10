# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The graph-building interface shared by dataset-owned curation pipelines."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.settings import RecipeSettings


@dataclass(frozen=True)
class InputOverrides:
    root: str | None = None
    files: Mapping[str, Path] = field(default_factory=dict)
    auxiliary: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalPaths:
    output_root: Path
    download_cache: Path


@dataclass(frozen=True)
class PipelineOptions:
    """Invocation choices; a dataset owns its dependencies and optional stages."""

    mode: SourceProcessingMode
    runtime: CampaignRuntime
    inputs: InputOverrides = field(default_factory=InputOverrides)
    local: LocalPaths | None = None
    recipe_settings: RecipeSettings | None = None
