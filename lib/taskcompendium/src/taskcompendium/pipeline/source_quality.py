# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample source quality without imputing missing model observations."""

import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from finestore.rl.curation import SourceQualityResult
from pydantic import BaseModel, ConfigDict

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import NoGrader, ScriptGrader, VerifyitGrader
from taskcompendium.pipeline.models import (
    REJECTING_CHECK_STATUSES,
    Confidence,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    TaskAudit,
)
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order, seeded_sample
from taskcompendium.runtime.resources import resource_bytes

SOURCE_QUALITY_REVISION = "9"
MAX_SOURCE_QUALITY_SAMPLE_SIZE = 100
SOURCE_QUALITY_SAMPLE_SIZE = MAX_SOURCE_QUALITY_SAMPLE_SIZE


@dataclass(frozen=True)
class SourceQualityPolicy:
    sample_size: int = SOURCE_QUALITY_SAMPLE_SIZE
    seed: int = 0

    def __post_init__(self):
        if self.sample_size < 1:
            raise ValueError("Sampling requires a positive size")


class SourceQualityStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CENSUS = "census"
    TRUST = "trust"
    INCOMPLETE = "incomplete"


class QualitySampleCoverage(StrEnum):
    CENSUS = "census"
    SAMPLE = "sample"
    RAW_SAMPLE = "raw_sample"


class Assessment(StrEnum):
    GOOD = "good"
    DEFECT = "defect"
    UNCERTAIN = "uncertain"
    UNAVAILABLE = "unavailable"
    UNUSABLE = "unusable"


@dataclass(frozen=True)
class QualitySample:
    input_count: int
    eligible_count: int
    exclusions: dict[str, int]
    contracts: dict[str, int]
    task_ids: tuple[str, ...]


class SourceQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    policy: SourceQualityPolicy
    population: QualitySample
    coverage: QualitySampleCoverage
    assessments: dict[Assessment, int]
    defect_fraction: float | None
    status: SourceQualityStatus
    reason: str
    result: SourceQualityResult


def assessment(record: ReviewRecord) -> Assessment:
    """Separate demonstrated defects from uncertainty and request failures."""
    if record.status != ReviewStatus.REVIEWED or record.verdict is None:
        return Assessment.UNAVAILABLE
    verdict = record.verdict
    if verdict.confidence == Confidence.LOW:
        return Assessment.UNCERTAIN
    if verdict.quality == Quality.BAD or verdict.defects or verdict.reference_status == ReferenceStatus.CONFLICT:
        return Assessment.DEFECT
    if verdict.quality == Quality.GOOD:
        return Assessment.GOOD
    return Assessment.UNCERTAIN


def quality_exclusion(audit: TaskAudit) -> str | None:
    """Identify rows outside the unique, structurally usable review population."""
    if audit.normalization_rejection is not None:
        return f"normalization:{audit.normalization_rejection.kind}"
    # A failed preparation check is a source defect, whatever decision it produced;
    # other decisions (duplicates, unsupported conversions) are neutral exclusions.
    if any(check.status in REJECTING_CHECK_STATUSES for check in audit.checks):
        return "check:failed"
    if audit.decision is not None:
        return audit.decision.reasons[0]
    return None


def contract_signature(audit: TaskAudit) -> str:
    """Return a runtime/answer contract signature, excluding task-specific references."""
    task = audit.normalized
    assert task is not None
    grader = task.grader
    execution_contract = None
    entrypoint_digest = None
    if isinstance(grader, ScriptGrader):
        execution_contract = grader.model_dump(mode="json", exclude={"environment"})
    elif isinstance(grader, VerifyitGrader) and grader.mode in {"script", "stdio", "pytest", "junit", "gotest"}:
        execution_contract = dict(grader.parameters)
        if grader.mode == "script":
            for resource in task.resources.verifier:
                if resource.path == grader.parameters["path"]:
                    entrypoint_digest = hashlib.sha256(resource_bytes(resource)).hexdigest()
    elif isinstance(grader, NoGrader):
        execution_contract = {
            "reason": grader.reason,
            **{
                key: grader.contract[key]
                for key in ("evaluator", "source_revision", "runtime_requirements")
                if key in grader.contract
            },
        }
    environment = grader.environment if isinstance(grader, VerifyitGrader | ScriptGrader) else None
    return canonical_sha256(
        {
            "grader_kind": grader.kind,
            "verifyit_mode": grader.mode if isinstance(grader, VerifyitGrader) else None,
            "execution_contract": execution_contract,
            "entrypoint_digest": entrypoint_digest,
            "answer_type": task.answer_type,
            "answer_format": task.answer_format.model_dump(mode="json"),
            "environment": task.environment_requirements.model_dump(mode="json"),
            "grader_environment": None if environment is None else environment.model_dump(mode="json"),
            "output_paths": task.output_paths,
        }
    )


def sample_quality_rows(records: Iterator[dict], *, policy: SourceQualityPolicy) -> QualitySample:
    """Sample IDs with bounded memory while counting every population and exclusion."""
    inputs = 0
    exclusions: Counter[str] = Counter()
    contracts: Counter[str] = Counter()

    def eligible_ids() -> Iterator[str]:
        nonlocal inputs
        for record in records:
            inputs += 1
            audit = TaskAudit.model_validate(record)
            if reason := quality_exclusion(audit):
                exclusions[reason] += 1
                continue
            contracts[contract_signature(audit)] += 1
            yield audit.task_id

    eligible, selected = seeded_sample(
        eligible_ids(), size=policy.sample_size, key=partial(seeded_order, seed=policy.seed)
    )
    return QualitySample(inputs, eligible, dict(exclusions), dict(contracts), tuple(selected))


def merge_quality_samples(samples: Iterator[QualitySample], *, policy: SourceQualityPolicy) -> QualitySample:
    inputs = 0
    exclusions: Counter[str] = Counter()
    contracts: Counter[str] = Counter()

    def shard_samples() -> Iterator[tuple[int, tuple[str, ...]]]:
        nonlocal inputs
        for sample in samples:
            inputs += sample.input_count
            exclusions.update(sample.exclusions)
            contracts.update(sample.contracts)
            yield sample.eligible_count, sample.task_ids

    eligible, selected = merge_sample_rows(
        shard_samples(), size=policy.sample_size, key=partial(seeded_order, seed=policy.seed)
    )
    return QualitySample(inputs, eligible, dict(exclusions), dict(contracts), tuple(selected))


def source_quality_report(
    sample: QualitySample,
    reviews: Sequence[ReviewRecord],
    policy: SourceQualityPolicy,
    *,
    coverage: QualitySampleCoverage,
) -> SourceQualityReport:
    """Report reviewed-sample issues without changing task admission or expansion."""
    if (
        len(sample.task_ids) != min(sample.eligible_count, policy.sample_size)
        or len(reviews) != len(sample.task_ids)
        or {record.task_id for record in reviews} != set(sample.task_ids)
    ):
        raise ValueError("Quality observations must cover the sampled tasks exactly once")
    counts = Counter(assessment(record) for record in reviews)
    reviewed = counts[Assessment.GOOD] + counts[Assessment.DEFECT]
    issue_count = counts[Assessment.DEFECT]
    issues = [
        f"{record.task_id}: {record.verdict.evidence}"
        for record in reviews
        if assessment(record) is Assessment.DEFECT and record.verdict is not None
    ]
    result = SourceQualityResult(
        reviewed=reviewed,
        issue_count=issue_count,
        unavailable_count=counts[Assessment.UNAVAILABLE],
        uncertain_count=counts[Assessment.UNCERTAIN],
        issues=issues,
        repair_recommendation=(
            "Review the recorded issues and qualify any repaired release with SAMPLE or FULL QA."
            if issue_count
            else "No repair indicated by the reviewed sample."
        ),
    )
    return SourceQualityReport(
        policy=policy,
        population=sample,
        coverage=coverage,
        assessments=dict(counts),
        defect_fraction=result.issue_rate,
        status=SourceQualityStatus.CENSUS if coverage == QualitySampleCoverage.CENSUS else SourceQualityStatus.TRUST,
        reason="Sample QC is informational; mechanical exclusions and execution qualification remain separate.",
        result=result,
    )


def unreviewed_quality_report(
    sample: QualitySample, policy: SourceQualityPolicy, *, coverage: QualitySampleCoverage
) -> SourceQualityReport:
    """Record mechanical exclusions without inventing reviewed quality observations."""
    return SourceQualityReport(
        policy=policy,
        population=sample,
        coverage=coverage,
        assessments={},
        defect_fraction=None,
        status=SourceQualityStatus.UNREVIEWED,
        reason="The source declares no rubric; model review is skipped",
        result=SourceQualityResult(
            reviewed=0, issue_count=0, issues=[], repair_recommendation="Run SAMPLE or FULL review to assess quality."
        ),
    )
