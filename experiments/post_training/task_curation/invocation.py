# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invocation and results shared by dataset-owned curation implementations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from taskcompendium.pipeline.inputs import SourceFileOverride
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext


class CurationSource(Protocol):
    """Catalog identity available to a pipeline without depending on its recipe."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...


@dataclass(frozen=True)
class PipelineRun:
    """One invocation; implementations choose ingestion and interpret input overrides."""

    mode: SourceProcessingMode
    context: ZephyrContext
    output_path: str
    source_input: str | None = None
    inputs: Mapping[str, str] = field(default_factory=dict)
    source_overrides: Mapping[str, SourceFileOverride] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Named products and evidence, with only the stages the implementation chose."""

    status: SourceStatus
    outputs: dict[str, str]
    evidence: dict[str, str]
    stages: tuple[str, ...]


class CurationPipeline(Protocol):
    def __call__(self, source: CurationSource, run: PipelineRun) -> PipelineResult: ...
