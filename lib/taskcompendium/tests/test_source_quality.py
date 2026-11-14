# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A source sample separates quality evidence, missingness, and population coverage."""

from dataclasses import replace

import pytest
from verifyit.spec import ScriptSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ResourceGroups, Source
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    Confidence,
    Decision,
    Disposition,
    ImportFailureKind,
    ImportRejection,
    Quality,
    RawRow,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    ReviewVerdict,
    TaskAudit,
)
from taskcompendium.pipeline.source_quality import (
    Assessment,
    QualitySample,
    QualitySampleCoverage,
    SourceQualityPolicy,
    SourceQualityStatus,
    contract_signature,
    merge_quality_samples,
    sample_quality_rows,
    source_quality_report,
)
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import svamp_row_task


def reviewed(task_id, quality=Quality.GOOD, confidence=Confidence.HIGH):
    return ReviewRecord(
        task_id=task_id,
        status=ReviewStatus.REVIEWED,
        detail="complete",
        verdict=ReviewVerdict(
            task_id=task_id,
            quality=quality,
            confidence=confidence,
            reference_status=ReferenceStatus.CONSISTENT,
            defects=[],
            evidence="Assessment of the task's provided context and answer.",
        ),
    )


@pytest.mark.parametrize(
    "issues,verdict",
    [
        (0, "good"),
        (9, "good"),
        (10, "good"),
        (11, "some_issues"),
        (49, "some_issues"),
        (50, "bad"),
        (51, "bad"),
        (100, "bad"),
    ],
)
def test_reviewed_issue_boundaries_are_informational(issues, verdict):
    ids = tuple(str(i) for i in range(100))
    reviews = [reviewed(task_id, Quality.BAD if i < issues else Quality.GOOD) for i, task_id in enumerate(ids)]
    report = source_quality_report(
        QualitySample(10000, 10000, {}, {"numeric": 10000}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.SAMPLE,
    )
    assert report.status == SourceQualityStatus.TRUST
    assert report.result.verdict == verdict
    assert report.result.reviewed == 100
    assert report.result.issue_count == issues
    assert report.result.issue_rate == issues / 100
    assert len(report.result.issues) == issues
    assert report.result.repair_recommendation


def test_result_separates_missing_reviews_and_mechanical_exclusions():
    ids = ("clean", "broken", "missing", "uncertain")
    reviews = [
        reviewed("clean"),
        reviewed("broken", Quality.BAD),
        ReviewRecord(task_id="missing", status=ReviewStatus.UNAVAILABLE, verdict=None, detail="Timeout"),
        reviewed("uncertain", Quality.UNKNOWN),
    ]
    report = source_quality_report(
        QualitySample(100, 4, {"normalization:source_defect": 96}, {"numeric": 4}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.RAW_SAMPLE,
    )
    assert report.result.reviewed == 2
    assert report.result.sample_count == 4
    assert report.result.unavailable_count == 1
    assert report.result.uncertain_count == 1
    assert report.result.issue_count == 1
    assert report.result.issue_rate == 0.5
    assert report.result.verdict == "bad"
    assert report.population.exclusions == {"normalization:source_defect": 96}
    assert report.assessments[Assessment.UNAVAILABLE] == 1
    assert report.assessments[Assessment.UNCERTAIN] == 1
    assert report.status == SourceQualityStatus.TRUST


def test_sampling_preserves_population_exclusions_and_ignores_partition_order():
    audits = []
    for index in range(80):
        source = Source(dataset="fixture", revision="1", row=str(index), importer_revision="1")
        task = svamp_row_task(
            RawRow(
                str(index),
                source,
                {
                    "Body": f"Person {index} has two apples.",
                    "Question": "How many apples?",
                    "Answer": "2",
                    "Equation": "2",
                },
            )
        )
        audit = TaskAudit(
            task_id=str(index),
            source=source,
            raw=None,
            normalized=task,
            normalization_rejection=None,
            checks=[],
            review=None,
            decision=None,
        )
        if index < 10:
            audit = audit.model_copy(
                update={
                    "decision": Decision(
                        task_id=str(index),
                        disposition=Disposition.REJECT,
                        reasons=["exact_semantic_duplicate"],
                        duplicate_of="10",
                    )
                }
            )
        elif index < 15:
            audit = audit.model_copy(
                update={
                    "normalized": None,
                    "normalization_rejection": ImportRejection(
                        kind=ImportFailureKind.CONVERTER_ERROR, reason="fixture", detail="Unsupported conversion"
                    ),
                    "decision": Decision(
                        task_id=str(index), disposition=Disposition.DEFER, reasons=["normalize:fixture"]
                    ),
                }
            )
        elif index < 20:
            # A failed check carries the decision it produced, as _with_checks records it.
            audit = audit.model_copy(
                update={
                    "checks": [CheckResult(check="contract", status=CheckStatus.FAIL, detail="Broken")],
                    "decision": Decision(task_id=str(index), disposition=Disposition.REJECT, reasons=["check:contract"]),
                }
            )
        else:
            audit = audit.model_copy(
                update={
                    "checks": [
                        CheckResult(
                            check="source_sample", status=CheckStatus.UNSUPPORTED, detail="Deferred runtime checks"
                        ),
                        CheckResult(check="oracle", status=CheckStatus.SKIPPED, detail="Absent"),
                    ]
                }
            )
        audits.append(audit.model_dump(mode="json"))
    policy = SourceQualityPolicy(sample_size=25)
    whole = sample_quality_rows(iter(audits), policy=policy)
    split = merge_quality_samples(
        (sample_quality_rows(iter(audits[i::7]), policy=policy) for i in range(7)), policy=policy
    )
    assert whole == split == sample_quality_rows(iter(reversed(audits)), policy=policy)
    assert whole.input_count == 80 and whole.eligible_count == 60
    assert whole.exclusions == {"exact_semantic_duplicate": 10, "normalization:converter_error": 5, "check:failed": 5}
    assert all(int(task_id) >= 20 for task_id in whole.task_ids)
    assert whole.task_ids != sample_quality_rows(iter(audits), policy=replace(policy, seed=1)).task_ids


def test_contract_signature_separates_grader_code_but_not_reference_values():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow(
            "task",
            source,
            {
                "Body": "Two apples.",
                "Question": "How many?",
                "Answer": "2",
                "Equation": "2",
            },
        )
    )
    assert not isinstance(task, ImportRejection)
    task = task.model_copy(
        update={
            "grader": (
                verifyit_package(
                    ScriptSpec(path="grade.py"),
                    environment=EnvironmentRequirements(
                        docker_image="fixture@sha256:" + "0" * 64, command_semantics=CommandSemantics.LINUX_PROCESS
                    ),
                ).grader
            ),
            "resources": ResourceGroups(
                verifier=(
                    inline_resource("grade.py", b"read_and_grade_reference()"),
                    inline_resource("answer.txt", b"2"),
                )
            ),
        }
    )
    audit = TaskAudit(
        task_id="task",
        source=source,
        raw=None,
        normalized=task,
        normalization_rejection=None,
        checks=[],
        review=None,
        decision=None,
    )
    different_reference = task.model_copy(
        update={
            "resources": ResourceGroups(
                verifier=(
                    task.resources.verifier[0],
                    inline_resource("answer.txt", b"3"),
                )
            )
        }
    )
    different_grader = task.model_copy(
        update={
            "resources": ResourceGroups(
                verifier=(
                    inline_resource("grade.py", b"always_pass()"),
                    task.resources.verifier[1],
                )
            )
        }
    )
    assert contract_signature(audit) == contract_signature(audit.model_copy(update={"normalized": different_reference}))
    assert contract_signature(audit) != contract_signature(audit.model_copy(update={"normalized": different_grader}))


@pytest.mark.parametrize(
    "exclusion", ["normalization:source_defect", "check:failed", "normalization:unsupported", "exact_semantic_duplicate"]
)
def test_raw_exclusions_do_not_change_reviewed_sample_result(exclusion):
    ids = ("retained",)
    report = source_quality_report(
        QualitySample(100, 1, {exclusion: 99}, {"numeric": 1}, ids),
        [reviewed("retained")],
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.RAW_SAMPLE,
    )
    assert report.result.reviewed == 1
    assert report.result.issue_count == 0
    assert report.result.verdict == "good"
    assert report.population.exclusions == {exclusion: 99}


def test_transport_outage_does_not_certify_qa_or_invent_a_bad_result():
    ids = ("a", "b")
    report = source_quality_report(
        QualitySample(10, 10, {}, {"numeric": 10}, ids),
        [
            ReviewRecord(task_id=i, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="Transport unavailable")
            for i in ids
        ],
        SourceQualityPolicy(sample_size=2),
        coverage=QualitySampleCoverage.SAMPLE,
    )
    assert report.status == SourceQualityStatus.INCOMPLETE
    assert report.result.verdict == "unknown"
    assert report.result.reviewed == 0
    assert report.result.unavailable_count == 2
