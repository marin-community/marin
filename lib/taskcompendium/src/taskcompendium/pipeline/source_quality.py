# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample source quality without imputing missing model observations."""

import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from pydantic import BaseModel, ConfigDict

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
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

SOURCE_QUALITY_REVISION = "8"


@dataclass(frozen=True)
class SourceQualityPolicy:
    sample_size: int = 100
    seed: int = 0
    reject_above: float = 0.50

    def __post_init__(self):
        if self.sample_size < 1:
            raise ValueError("Sampling requires a positive size")
        if not 0 < self.reject_above <= 1:
            raise ValueError("The source rejection threshold must satisfy 0 < reject <= 1")


class SourceQualityStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CENSUS = "census"
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
            "output_directories": [directory.model_dump(mode="json") for directory in task.output_directories],
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
    """Gate the whole source once on the fixed panel; rows outside the panel are never reviewed.

    Known defects above ``reject_above`` reject the source. Unavailable responses leave it
    incomplete only while resolving them could still push defects above that threshold.
    """
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
    # Only unavailable responses change on resume; uncertain judgments and unusable rows stay as they are.
    resolvable = counts[Assessment.UNAVAILABLE]
    if known_defects > policy.reject_above:
        status, reason = SourceQualityStatus.REJECT, "Known defects exceed the rejection threshold over the whole panel"
    elif not reviews:
        status, reason = SourceQualityStatus.INCOMPLETE, "No usable tasks in the fixed raw panel; no source inference"
    elif census:
        status, reason = SourceQualityStatus.CENSUS, "Review attempted for every eligible unique task; no extrapolation"
    elif (counts[Assessment.DEFECT] + resolvable) / panel_size > policy.reject_above:
        status, reason = (
            SourceQualityStatus.INCOMPLETE,
            "Missing or invalid model responses could still reject the source; resume the same sample",
        )
    else:
        status, reason = (
            SourceQualityStatus.TRUST,
            "Known defects stay within the rejection threshold over the whole panel; "
            "uncertain judgments and unusable rows are not defects",
        )
    return SourceQualityReport(
        policy=policy,
        population=sample,
        coverage=coverage,
        assessments=dict(counts),
        defect_fraction=defect_fraction,
        status=status,
        reason=reason,
    )


def unreviewed_quality_report(
    sample: QualitySample, policy: SourceQualityPolicy, *, coverage: QualitySampleCoverage
) -> SourceQualityReport:
    """Decide on a source without a rubric from observed conversion and check failures alone."""
    defects = sample.exclusions.get("normalization:source_defect", 0) + sample.exclusions.get("check:failed", 0)
    defect_fraction = defects / sample.input_count if sample.input_count else None
    if defect_fraction is not None and defect_fraction > policy.reject_above:
        status, reason = SourceQualityStatus.REJECT, "Known defects exceed the rejection threshold over the whole panel"
    elif not sample.eligible_count:
        status, reason = SourceQualityStatus.INCOMPLETE, "No usable tasks in the fixed raw panel; no source inference"
    else:
        status, reason = SourceQualityStatus.UNREVIEWED, "The source declares no rubric; model review is skipped"
    return SourceQualityReport(
        policy=policy,
        population=sample,
        coverage=coverage,
        assessments={Assessment.DEFECT: defects} if defects else {},
        defect_fraction=defect_fraction,
        status=status,
        reason=reason,
    )
