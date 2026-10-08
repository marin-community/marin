# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample source quality without imputing missing model observations."""

import hashlib
import heapq
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import NoGrader, ScriptGrader, VerifyitGrader
from taskcompendium.pipeline.models import (
    CheckStatus,
    Confidence,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    TaskAudit,
)
from taskcompendium.runtime.resources import resource_bytes

SOURCE_QUALITY_REVISION = "6"


@dataclass(frozen=True)
class SourceQualityPolicy:
    sample_size: int = 100
    seed: int = 0
    reject_above: float = 0.50
    trust_below: float = 0.10

    def __post_init__(self):
        if self.sample_size < 1:
            raise ValueError("Sampling requires a positive size")
        if not 0 <= self.trust_below < self.reject_above <= 1:
            raise ValueError("Source quality thresholds must satisfy 0 <= trust < reject <= 1")


class SourceQualityStatus(StrEnum):
    CENSUS = "census"
    FULL_REVIEW = "full_review"
    TRUST = "trust"
    REJECT = "reject"
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


def assessment(record: ReviewRecord) -> Assessment:
    """Separate demonstrated defects from uncertainty and transport failures."""
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
    if audit.decision is not None:
        return audit.decision.reasons[0]
    if any(check.status == CheckStatus.FAIL for check in audit.checks):
        return "check:failed"
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
            "output_directories": [directory.model_dump(mode="json") for directory in task.output_directories],
        }
    )


def sample_order(task_id: str, seed: int) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), task_id


def sample_quality_rows(records: Iterator[dict], *, policy: SourceQualityPolicy) -> QualitySample:
    """Sample IDs with bounded memory while counting every population and exclusion."""
    inputs = 0
    eligible = 0
    exclusions: Counter[str] = Counter()
    contracts: Counter[str] = Counter()

    def candidates():
        nonlocal inputs, eligible
        for record in records:
            inputs += 1
            audit = TaskAudit.model_validate(record)
            if reason := quality_exclusion(audit):
                exclusions[reason] += 1
                continue
            eligible += 1
            contracts[contract_signature(audit)] += 1
            yield audit.task_id

    selected = heapq.nsmallest(policy.sample_size, candidates(), key=lambda task_id: sample_order(task_id, policy.seed))
    return QualitySample(inputs, eligible, dict(exclusions), dict(contracts), tuple(selected))


def merge_quality_samples(samples: Iterator[QualitySample], *, policy: SourceQualityPolicy) -> QualitySample:
    inputs = 0
    eligible = 0
    exclusions: Counter[str] = Counter()
    contracts: Counter[str] = Counter()

    def candidates():
        nonlocal inputs, eligible
        for sample in samples:
            inputs += sample.input_count
            eligible += sample.eligible_count
            exclusions.update(sample.exclusions)
            contracts.update(sample.contracts)
            yield from sample.task_ids

    selected = heapq.nsmallest(policy.sample_size, candidates(), key=lambda task_id: sample_order(task_id, policy.seed))
    return QualitySample(inputs, eligible, dict(exclusions), dict(contracts), tuple(selected))


def source_quality_report(
    sample: QualitySample,
    reviews: Sequence[ReviewRecord],
    policy: SourceQualityPolicy,
    *,
    coverage: QualitySampleCoverage,
) -> SourceQualityReport:
    """Decide once on the fixed panel; missing observations never support extrapolation."""
    if (
        len(sample.task_ids) != min(sample.eligible_count, policy.sample_size)
        or len(reviews) != len(sample.task_ids)
        or {record.task_id for record in reviews} != set(sample.task_ids)
    ):
        raise ValueError("Quality observations must cover the sampled tasks exactly once")
    counts = Counter(assessment(record) for record in reviews)
    census = coverage == QualitySampleCoverage.CENSUS
    raw_panel = coverage in {QualitySampleCoverage.RAW_SAMPLE, QualitySampleCoverage.CENSUS}
    panel_size = sample.input_count if raw_panel else len(reviews)
    if raw_panel:
        source_defects = sample.exclusions.get("normalization:source_defect", 0) + sample.exclusions.get(
            "check:failed", 0
        )
        counts[Assessment.DEFECT] += source_defects
        counts[Assessment.UNUSABLE] += sum(sample.exclusions.values()) - source_defects
    defect_fraction = None
    unresolved = counts[Assessment.UNAVAILABLE] + counts[Assessment.UNCERTAIN] + counts[Assessment.UNUSABLE]
    if not unresolved and panel_size:
        defect_fraction = counts[Assessment.DEFECT] / panel_size
    # The raw draw supplies the denominator even when conversion excludes rows.
    # Source defects are observed failures; unsupported conversion and duplicates
    # provide neither good judgments nor evidence of bad source content.
    known_defects = counts[Assessment.DEFECT] / panel_size if panel_size else 0.0
    known_good = counts[Assessment.GOOD] / panel_size if panel_size else 0.0
    possible_good = (counts[Assessment.GOOD] + unresolved) / panel_size if panel_size else 0.0
    possible_defects = (counts[Assessment.DEFECT] + unresolved) / panel_size if panel_size else 0.0
    if known_defects > policy.reject_above:
        status, reason = SourceQualityStatus.REJECT, "Known defects exceed the rejection threshold over the whole panel"
    elif not reviews:
        status, reason = SourceQualityStatus.INCOMPLETE, "No usable tasks in the fixed raw panel; no source inference"
    elif census:
        status, reason = SourceQualityStatus.CENSUS, "Review attempted for every eligible unique task; no extrapolation"
    elif known_good > 1 - policy.trust_below:
        status, reason = (
            SourceQualityStatus.TRUST,
            "Known good judgments exceed the trust threshold over the whole panel",
        )
    elif (
        counts[Assessment.UNAVAILABLE]
        and possible_good <= 1 - policy.trust_below
        and possible_defects <= policy.reject_above
    ):
        status, reason = (
            SourceQualityStatus.FULL_REVIEW,
            "Resolving missing observations cannot cross either source decision threshold",
        )
    elif counts[Assessment.UNAVAILABLE]:
        status, reason = SourceQualityStatus.INCOMPLETE, "Missing or invalid model responses; resume the same sample"
    elif counts[Assessment.UNCERTAIN] or counts[Assessment.UNUSABLE]:
        status, reason = (
            SourceQualityStatus.FULL_REVIEW,
            "Unassessed raw rows or semantic uncertainty prevent source extrapolation",
        )
    else:
        status, reason = SourceQualityStatus.FULL_REVIEW, "Sample quality falls between the decision thresholds"
    return SourceQualityReport(
        policy=policy,
        population=sample,
        coverage=coverage,
        assessments=dict(counts),
        defect_fraction=defect_fraction,
        status=status,
        reason=reason,
    )
