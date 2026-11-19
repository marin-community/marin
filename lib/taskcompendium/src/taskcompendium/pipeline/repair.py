# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair-candidate lane records: matching, comparative review and per-task decisions.

A repair candidate is a "cleaned subset" of a source the Atlas already tracks, offered by a
third party (for example FWeindel/validated-tasks). The lane never trusts task IDs and never
admits the candidate as a parallel source: every candidate task is matched to its tracked
counterpart by content, reviewed comparatively against that counterpart, and either replaces
it, or the original is kept. Tracked tasks the candidate's filter leaves out become advisory
drop flags, never exclusions.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.pipeline.models import Confidence, Defect, Quality, ReferenceStatus, ReviewStatus

MATCH_SCHEMA_VERSION = Literal["repair-match-v1"]


class RepairMatchMethod(StrEnum):
    """How the matcher paired a candidate with a tracked task."""

    EXACT = "exact"
    """Canonical content identity: normalized instruction, state and grading contract agree."""

    EQUIVALENT = "equivalent"
    """Normalized instruction matches; the state, reference or grading contract differs."""

    NEIGHBOR = "neighbor"
    """Approximate nearest neighbor over normalized content at or above the repair threshold."""


class RepairMatchClass(StrEnum):
    """What a match means for the candidate task."""

    IDENTICAL = "identical"
    """The candidate is the tracked task; nothing to review or replace."""

    REPAIRED = "repaired"
    """Same task with different content; the pair enters comparative review."""

    NEW = "new"
    """No tracked counterpart; the candidate follows the normal admission pipeline."""

    CONFLICTING = "conflicting"
    """The match is ambiguous; a manual ruling is required before anything is replaced."""


class MatchReceipt(BaseModel):
    """One candidate-to-tracked content match, recorded as immutable lane evidence.

    Identifiers are carried for display only; matching is content-only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: MATCH_SCHEMA_VERSION = "repair-match-v1"
    candidate_source_id: str
    candidate_task_id: str
    tracked_source_id: str
    tracked_task_id: str | None = None
    """``None`` for NEW candidates and for CONFLICTING candidates with no unique best match."""
    instruction_match: bool
    state_match: bool | None = None
    reference_match: bool | None = None
    tests_match: bool | None = None
    similarity: float = Field(ge=0.0, le=1.0)
    method: RepairMatchMethod
    match_class: RepairMatchClass
    reason: str


class MatchCensus(BaseModel):
    """Aggregate match outcome for one candidate folder against one tracked source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_source_id: str
    tracked_source_id: str
    candidates: int
    identical: int
    repaired: int
    new: int
    conflicting: int
    dropped_advisories: int
    overlapped: int = 0
    """Candidates with a tracked counterpart at or above the drop floor.

    A candidate folder with zero overlap is a parallel derivation, not a cleaned subset:
    its absence proves nothing about the tracked tasks, so the lane emits no drop advisories.
    """


class ComparativeOutcome(StrEnum):
    """Which version of a matched pair is the better task."""

    BETTER = "better"
    TIE = "tie"
    WORSE = "worse"
    INCOMPARABLE = "incomparable"
    """The two versions are not the same task; the match or a version is defective."""


class ComparativeVerdict(BaseModel):
    """The judge's comparative finding for one candidate-original pair."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    task_id: str = Field(min_length=1)
    """The candidate task's ID."""
    outcome: ComparativeOutcome
    defects_fixed: list[Defect]
    defects_introduced: list[Defect]
    confidence: Confidence
    evidence: str = Field(min_length=1, json_schema_extra={"maxLength": 1000})


class ComparativeReviewRecord(BaseModel):
    """One comparative judge attempt outcome for a pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    status: ReviewStatus
    verdict: ComparativeVerdict | None = None
    detail: str


class RepairProvenance(StrEnum):
    """Why a task is in the merged release, relative to the tracked source.

    The merge step that publishes the new PRIVATE tracked-source version stamps this on
    every task; the Atlas renders it through the catalog snapshot's per-task provenance.
    """

    ORIGINAL = "original"
    """The tracked task is retained unchanged (possibly re-confirmed by an identical candidate)."""

    REPLACED = "replaced"
    """A matched candidate task replaced the tracked task after passing every gate."""

    ADDED = "added"
    """A new candidate task entered through the normal admission pipeline."""


class RepairDecision(StrEnum):
    """What the lane does with a matched-repaired candidate."""

    REPLACED = "replaced"
    KEPT_ORIGINAL = "kept_original"


class ComparativeRepairReceipt(BaseModel):
    """Typed per-task evidence for one matched-repaired pair's final decision.

    The merge step builds one receipt per reviewed pair once the oracle/NoP gates have run,
    and the published release manifest references it as the task's replacement evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_source_id: str
    candidate_task_id: str
    candidate_task_sha256: str
    tracked_source_id: str
    tracked_task_id: str
    tracked_task_sha256: str
    verdict: ComparativeVerdict | None = None
    oracle_qualified: bool | None = None
    nop_qualified: bool | None = None
    candidate_quality: Quality | None = None
    candidate_reference_status: ReferenceStatus | None = None
    decision: RepairDecision
    reasons: list[str]


def repair_decision(
    *,
    verdict: ComparativeVerdict | None,
    oracle_qualified: bool | None,
    nop_qualified: bool | None,
    candidate_quality: Quality | None,
    candidate_reference_status: ReferenceStatus | None,
) -> tuple[RepairDecision, list[str]]:
    """Apply the replacement rule to one matched-repaired pair.

    A candidate replaces its tracked counterpart only when it is executable (oracle and NoP
    qualified), the comparison is not worse, it introduces no defects, and its own static
    quality is acceptable without a reference conflict. Every failed gate is returned as a
    reason, so a kept original always carries its typed evidence.
    """
    reasons: list[str] = []
    if verdict is None:
        reasons.append("comparative_review_missing")
    else:
        if verdict.outcome is ComparativeOutcome.WORSE:
            reasons.append("comparative_outcome_worse")
        if verdict.outcome is ComparativeOutcome.INCOMPARABLE:
            reasons.append("comparative_outcome_incomparable")
        if verdict.defects_introduced:
            reasons.append("defects_introduced")
    if oracle_qualified is not True:
        reasons.append("oracle_not_qualified")
    if nop_qualified is not True:
        reasons.append("nop_not_qualified")
    if candidate_quality not in (Quality.GOOD, Quality.ISSUES, None):
        reasons.append("candidate_quality_rejected")
    if candidate_reference_status is ReferenceStatus.CONFLICT:
        reasons.append("candidate_reference_conflict")
    if reasons:
        return RepairDecision.KEPT_ORIGINAL, reasons
    return RepairDecision.REPLACED, []


class CandidateDropAdvisory(BaseModel):
    """An advisory per-task flag on a tracked task whose candidate counterpart was filtered out.

    The candidate's own filter (for example pass@k) is a signal, not a mandate: the flag feeds
    the tracked task's QC evidence and never excludes the task by itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_source_id: str
    tracked_task_id: str
    tracked_task_sha256: str
    signal: str
    reason: str
