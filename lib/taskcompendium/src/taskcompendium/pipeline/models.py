# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset recipes and persisted curation evidence."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import RecipeInputs
from taskcompendium.runtime.models import RolloutRecord


@dataclass(frozen=True)
class HFSource:
    dataset: str
    revision: str
    config: str
    split: str


@dataclass(frozen=True)
class GeneratedSource:
    dataset: str
    revision: str
    config: str
    split: str
    module: str


@dataclass(frozen=True)
class RawRow:
    id: str
    source: Source
    data: Mapping[str, Any]


class ImportRejection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    reason: str
    detail: str


@dataclass(frozen=True)
class EnvironmentInventory:
    """Reviewer-only evidence about a pinned environment and an explicit path scope."""

    environment_id: str
    origin: str
    roots: tuple[str, ...]
    paths: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class ReviewRubric:
    id: str
    version: str
    criteria: tuple[str, ...]
    environment_inventory: EnvironmentInventory | None = None


class IntendedUse(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class NormalizationChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    field: str
    reason: str
    original: str
    replacement: str


@dataclass(frozen=True)
class NormalizedTask:
    task: TaskSpec
    changes: tuple[NormalizationChange, ...]


@dataclass(frozen=True)
class DatasetRecipe:
    name: str
    version: str
    source: HFSource | GeneratedSource
    normalize: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection]
    rubric: ReviewRubric
    intended_use: IntendedUse
    inputs: RecipeInputs
    check_suite: "CheckSuite | None" = None


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"
    INFRA_ERROR = "infra_error"


class GraderReadiness(StrEnum):
    READY = "ready"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    check: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    checks: list[CheckResult]
    rollouts: tuple[RolloutRecord, ...] = ()


@dataclass(frozen=True)
class CheckSuite:
    id: str
    revision: str
    parameters: Mapping[str, Any]
    run: Callable[[TaskSpec], VerificationReport]


class Quality(StrEnum):
    GOOD = "good"
    ISSUES = "some_issues"
    BAD = "bad"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReferenceStatus(StrEnum):
    CONSISTENT = "consistent"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class Defect(StrEnum):
    MISSING_CONTEXT = "missing_context"
    AMBIGUITY = "ambiguity"
    WRONG_REFERENCE = "wrong_reference"
    ANSWER_LEAKAGE = "answer_leakage"
    MALFORMED = "malformed"
    RUBRIC_MISMATCH = "rubric_mismatch"


class ReviewVerdict(BaseModel):
    """Quality findings; a model's key assessment is advisory evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    task_id: str = Field(min_length=1)
    quality: Quality
    confidence: Confidence
    reference_status: ReferenceStatus
    defects: list[Defect]
    evidence: str = Field(min_length=1, max_length=1000)


class ReviewStatus(StrEnum):
    REVIEWED = "reviewed"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class ReviewRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    status: ReviewStatus
    verdict: ReviewVerdict | None
    detail: str


class Disposition(StrEnum):
    KEEP = "keep"
    REJECT = "reject"


@dataclass(frozen=True)
class FilterPolicy:
    id: str = "binary-static-v1"
    minimum_confidence: Confidence = Confidence.MEDIUM


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    disposition: Disposition
    reasons: list[str]
    duplicate_of: str | None = None


class RewriteAction(StrEnum):
    REWRITE = "rewrite"
    UNCHANGED = "unchanged"
    UNREPAIRABLE = "unrepairable"


class InstructionEdit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    old_text: str = Field(min_length=1)
    replacement: str


class RewriteProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    task_id: str
    action: RewriteAction
    edits: list[InstructionEdit]
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_replacement(self) -> "RewriteProposal":
        if (self.action == RewriteAction.REWRITE) != bool(self.edits):
            raise ValueError("Only a rewrite must supply nonempty edits")
        return self


class RewriteRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    status: ReviewStatus
    proposal: RewriteProposal | None
    detail: str


class TaskAudit(BaseModel):
    """One source row and all observations retained before the accepted export."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    source: Source
    raw: dict[str, Any] | None
    normalized: TaskSpec | None
    normalization_rejection: ImportRejection | None
    checks: list[CheckResult]
    review: ReviewRecord | None
    decision: Decision | None
    original: TaskSpec | None = None
    cleanup: RewriteRecord | None = None
    lineage: dict[str, Any] | None = None
    normalization_changes: tuple[NormalizationChange, ...] = ()
    intended_use: IntendedUse | None = None
