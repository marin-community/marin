# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen sample panels and their informational content-review artifacts."""

from typing import Literal

from finestore.rl.curation import SourceQualityResult
from marin.execution.artifact import Artifact
from pydantic import BaseModel, ConfigDict, Field

SAMPLE_SIZE = 100
SAMPLE_SEED = 0
PINNED_REVIEWER_SHA256 = "cad081d2e6c013e083ee1737ea42d65a175117ebaae9b56dfacbc8fb1cd2489a"
PINNED_REVIEWER_V4_SHA256 = "9c0d83d65755be99ec382ac5c061c05db36ade8a2a800d3d48ddd82f7a88f3dc"
PINNED_REVIEWER_V5_SHA256 = "2fb4ff7d1134d63db62b8a186fa2e841f284dca7dc60c38681a07c96c6333c64"
# The pinned reviewer identities each payload revision may produce. Payload
# Revision 6 admits the V3, V4 and V5 profiles at their recorded identities.
SAMPLE_REVIEWER_SHA256_BY_PAYLOAD_REVISION = {
    "4": frozenset({"a3e17ee0607a1bf59bdbb8e0b9bffda2e8788390aaf41c7ab73c20b68325c9ea"}),
    "5": frozenset({"0a00cdd8682fe13ae169d724cb630c73ebab9be1dab3b4d6d971eca04fa0e308"}),
    "6": frozenset({PINNED_REVIEWER_SHA256, PINNED_REVIEWER_V4_SHA256, PINNED_REVIEWER_V5_SHA256}),
}
SAMPLE_REVIEWER_SHAS = {
    "v3": PINNED_REVIEWER_SHA256,
    "v4": PINNED_REVIEWER_V4_SHA256,
    "v5": PINNED_REVIEWER_V5_SHA256,
}
"""The pinned reviewer identity per SAMPLE profile version."""


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
    # Provenance when the panel was not measured at this converter identity: the receipt
    # was carried forward from an earlier converter version after a scorer-only change,
    # with the reason and invariance evidence. None means a fresh qualification.
    carry_forward: str | None = None


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
