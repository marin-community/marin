# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Named products and evidence reported by dataset-owned pipelines."""

from dataclasses import dataclass

from taskcompendium.pipeline.models import SourceStatus


@dataclass(frozen=True)
class PipelineResult:
    """An execution outcome and the stages that actually ran."""

    status: SourceStatus
    outputs: dict[str, str]
    evidence: dict[str, str]
    stages: tuple[str, ...]
