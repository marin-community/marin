# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen sample panels and their informational content-review artifacts."""

from typing import Literal

from finestore.rl.curation import SourceQualityResult
from marin.execution.artifact import Artifact
from pydantic import BaseModel, ConfigDict, Field

SAMPLE_SIZE = 100
SAMPLE_SEED = 0
PINNED_REVIEWER_SHA256 = "a3e17ee0607a1bf59bdbb8e0b9bffda2e8788390aaf41c7ab73c20b68325c9ea"


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


class SampleInventory(SamplePanel):
    """Population inventory with diagnostics kept outside the frozen QC panel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    oracle_tasks: int = Field(ge=0)
    execution_qualification: Literal["not_assessed"]
    runnable_grader_tasks: int | None = None
    no_grader_tasks: int | None = None
    grader_kinds: dict[str, int] | None = None
    missing_grader_reasons: dict[str, int] | None = None

    def qc_panel(self) -> SamplePanel:
        return SamplePanel(**{name: getattr(self, name) for name in SamplePanel.model_fields})


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
