# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen sample panels and their informational content-review artifacts."""

from typing import Literal

from finestore.rl.curation import SourceQualityResult
from marin.execution.artifact import Artifact
from pydantic import BaseModel, ConfigDict, Field

SAMPLE_SIZE = 100
SAMPLE_SEED = 0


class SamplePanel(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    dataset_id: str
    dataset_revision: str
    converter_version: str
    input_artifact: str
    task_specs_sha256: str
    scanned_task_specs_sha256: str
    retained_tasks: int = Field(gt=0)
    sample_seed: int
    sample_size: int = Field(gt=0, le=SAMPLE_SIZE)
    panel_path: str
    panel_sha256: str
    evidence_url: str


class SampleQCReview(Artifact):
    panel: SamplePanel
    reviewer: dict
    rubric: dict
    scope: Literal["sample_content"] = "sample_content"
    qualification_status: Literal["pending"] = "pending"
    panel_coverage_complete: bool
    attempted_count: int
    result: SourceQualityResult
    records_path: str
    records_sha256: str
    batch_receipts: list[str]
