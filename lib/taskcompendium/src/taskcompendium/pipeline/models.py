# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source recipes and persisted curation evidence."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import AssistantToolCalls, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles
from taskcompendium.runtime.models import RolloutRecord

RESOURCE_BUDGET_BYTES = 1_000_000
"""Default limit on one task's decoded resource bytes; larger rows are deferred at normalization."""
RESOURCES_OVER_BUDGET = "resources_over_budget"


@dataclass(frozen=True)
class RawRow:
    id: str
    source: Source
    data: Mapping[str, Any]


class ImportFailureKind(StrEnum):
    SOURCE_DEFECT = "source_defect"
    UNSUPPORTED = "unsupported"
    CONVERTER_ERROR = "converter_error"


class ImportRejection(BaseModel):
    """An import failure; only demonstrated source defects warrant rejection."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: ImportFailureKind
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


type Converter = Callable[[RawRow, ConversionContext], TaskSpec | NormalizedTask | ImportRejection]
"""Convert one raw row into a task with its grader fixed, or reject it."""


@dataclass(frozen=True)
class Reply:
    """A final assistant event, graded after the task's context."""

    event: TextMessage | AssistantToolCalls


@dataclass(frozen=True)
class WorkspaceFiles:
    """Agent output files, graded as a file submission."""

    files: Mapping[str, bytes]


@dataclass(frozen=True)
class OracleCommand:
    """A shell command run with the task's worker and oracle files; its output is the submission.

    It runs in a machine of the task's agent image, or of the grader image when the task has none.
    File-answer tasks submit their captured output files. Conversation-answer tasks submit the
    contents of ``answer_file``, resolved in the grader workspace, as the final assistant message.
    """

    command: str
    answer_file: str | None = None


type ControlSubmission = Reply | WorkspaceFiles | OracleCommand


@dataclass(frozen=True)
class Controls:
    """The one control submission that tests a source's grader on each sampled task.

    ``golden(task)`` is a known-correct submission, which must score one. When ``golden`` is
    absent or returns ``None`` because the source supplies no known-correct answer, the task is
    graded on an empty submission instead, which must score zero; a sandbox grader grades it in a
    fresh grading machine. ``memory_mb`` sizes each fresh grading machine.
    """

    golden: Callable[[TaskSpec], ControlSubmission | None] | None = None
    memory_mb: int = 512


@dataclass(frozen=True)
class SourceRecipe:
    """One staged source and how its rows become reviewed, verified tasks.

    ``rubric=None`` skips model review. ``controls=None`` skips grader verification, so sandbox
    graders other than judges remain unverified. ``inputs`` holds staged auxiliary input paths by name, and
    ``grader_environment`` the source's grader image; both reach the source callables through
    their ``ConversionContext``. A task whose decoded resources exceed ``resource_budget_bytes``
    is deferred as ``resources_over_budget``.
    """

    name: str
    version: str
    source: SourceFiles
    convert: Converter
    rubric: ReviewRubric | None
    controls: Controls | None
    intended_use: IntendedUse
    inputs: Mapping[str, str] = field(default_factory=dict)
    grader_environment: EnvironmentRequirements | None = None
    resource_budget_bytes: int = RESOURCE_BUDGET_BYTES


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    INFRA_ERROR = "infra_error"


class GraderReadiness(StrEnum):
    READY = "ready"
    SOURCE_SAMPLED = "source_sampled"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class SourceStatus(StrEnum):
    """The terminal status of one source pipeline run, recorded in its manifest.

    ``SAMPLED``: a sample-mode run processed only its panel. ``COMPLETED``: every row was processed,
    by full expansion or because the panel is a census. ``GATED``: a quality or verification gate
    rejected the source. ``INCOMPLETE``: the quality gate could not decide or a control trial hit an
    infrastructure error; evidence is retained for a retry.
    """

    SAMPLED = "sampled"
    COMPLETED = "completed"
    GATED = "gated"
    INCOMPLETE = "incomplete"


class Admission(StrEnum):
    """Whether a row reaches the final export, and why not."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    NO_GRADER = "no_grader"
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
    # Keep the provider's brevity guidance and exact request identity, but retain
    # longer explanations rather than invalidate an otherwise usable verdict.
    evidence: str = Field(min_length=1, json_schema_extra={"maxLength": 1000})


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
    DEFER = "defer"


class QualityBasis(StrEnum):
    UNREVIEWED = "unreviewed"
    DIRECT_REVIEW = "direct_review"
    INFERRED_FROM_SOURCE = "inferred_from_source"
    SOURCE_REJECTED = "source_rejected"
    SOURCE_INCOMPLETE = "source_incomplete"


@dataclass(frozen=True)
class FilterPolicy:
    id: str = "evidence-static-v2"
    minimum_confidence: Confidence = Confidence.MEDIUM


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    disposition: Disposition
    reasons: list[str]
    duplicate_of: str | None = None


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
    normalization_changes: tuple[NormalizationChange, ...] = ()
    intended_use: IntendedUse | None = None
    quality_basis: QualityBasis | None = None
    source_quality_report: str | None = None
