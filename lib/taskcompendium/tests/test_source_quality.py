# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A source sample separates quality evidence, missingness, and population coverage."""

from dataclasses import replace

import pytest
from shellbox.machine import Backend
from verifyit.spec import ScriptSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.models import EnvironmentRequirements, ResourceGroups, Source
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


@pytest.mark.parametrize("good,expected", [(49, "reject"), (50, "full_review"), (90, "full_review"), (91, "trust")])
def test_fixed_sample_success_fraction_controls_extrapolation(good, expected):
    ids = tuple(str(i) for i in range(100))
    sample = QualitySample(10000, 10000, {}, {"numeric": 10000}, ids)
    reviews = [reviewed(task_id, Quality.GOOD if i < good else Quality.BAD) for i, task_id in enumerate(ids)]
    report = source_quality_report(sample, reviews, SourceQualityPolicy(), coverage=QualitySampleCoverage.SAMPLE)
    assert report.status == expected
    assert report.assessments[Assessment.GOOD] == good
    assert report.defect_fraction == (100 - good) / 100


@pytest.mark.parametrize(
    "unresolved",
    ["unavailable", "invalid", "unknown", "low"],
)
@pytest.mark.parametrize("quality,expected", [(Quality.GOOD, "trust"), (Quality.BAD, "reject")])
def test_unresolved_response_preserves_decisive_observed_quality(unresolved, expected, quality):
    ids = tuple(str(i) for i in range(100))
    reviews = [reviewed(task_id, quality) for task_id in ids]
    if unresolved in ("unavailable", "invalid"):
        reviews[0] = ReviewRecord(
            task_id="0", status=ReviewStatus(unresolved), verdict=None, detail="No usable response"
        )
    else:
        reviews[0] = reviewed(
            "0",
            Quality.UNKNOWN if unresolved == "unknown" else quality,
            Confidence.LOW if unresolved == "low" else Confidence.HIGH,
        )
    report = source_quality_report(
        QualitySample(10000, 10000, {}, {"numeric": 10000}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.SAMPLE,
    )
    assert report.status == expected
    assert report.defect_fraction is None


def test_contract_counts_are_diagnostic_and_census_does_not_impute_reviews():
    ids = tuple(str(i) for i in range(100))
    reviews = [reviewed(task_id) for task_id in ids]
    sample = QualitySample(10000, 10000, {}, {"numeric": 9999, "script": 1}, ids)
    policy = SourceQualityPolicy()
    report = source_quality_report(sample, reviews, policy, coverage=QualitySampleCoverage.SAMPLE)
    assert report.status == SourceQualityStatus.TRUST
    assert report.population.contracts == {"numeric": 9999, "script": 1}
    census = replace(sample, input_count=100, eligible_count=100, contracts={"numeric": 99, "script": 1})
    assert (
        source_quality_report(census, reviews, policy, coverage=QualitySampleCoverage.CENSUS).status
        == SourceQualityStatus.CENSUS
    )


@pytest.mark.parametrize("unresolved", [None, "unknown", "unavailable"])
def test_census_rejects_when_known_defects_alone_exceed_threshold(unresolved):
    ids = tuple(str(i) for i in range(10))
    reviews = [reviewed(task_id, Quality.GOOD if i == 0 else Quality.BAD) for i, task_id in enumerate(ids)]
    if unresolved == "unknown":
        reviews[-1] = reviewed(ids[-1], Quality.UNKNOWN)
    elif unresolved == "unavailable":
        reviews[-1] = ReviewRecord(task_id=ids[-1], status=ReviewStatus.UNAVAILABLE, verdict=None, detail="Timeout")
    report = source_quality_report(
        QualitySample(10, 10, {}, {"numeric": 10}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.CENSUS,
    )
    assert report.status == SourceQualityStatus.REJECT
    if unresolved is None:
        assert report.defect_fraction == 0.9
    else:
        assert report.defect_fraction is None


@pytest.mark.parametrize("known_good,expected", [(90, "incomplete"), (91, "trust"), (0, "incomplete")])
def test_missing_responses_do_not_count_as_defects_or_replace_trust_threshold(known_good, expected):
    ids = tuple(str(i) for i in range(100))
    reviews = [
        (
            reviewed(task_id)
            if i < known_good
            else ReviewRecord(task_id=task_id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="Upload timed out")
        )
        for i, task_id in enumerate(ids)
    ]
    report = source_quality_report(
        QualitySample(10000, 10000, {}, {"numeric": 10000}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.SAMPLE,
    )
    assert report.status == expected
    assert report.assessments.get(Assessment.DEFECT, 0) == 0
    assert report.assessments[Assessment.UNAVAILABLE] == 100 - known_good


@pytest.mark.parametrize(
    "good,defects,uncertain,expected",
    [
        (80, 19, 0, "full_review"),
        (89, 10, 0, "full_review"),
        (90, 9, 0, "incomplete"),
        (50, 49, 0, "full_review"),
        (49, 50, 0, "incomplete"),
        (50, 10, 0, "full_review"),
        (49, 11, 0, "incomplete"),
        (51, 9, 0, "incomplete"),
        (50, 48, 1, "full_review"),
        (49, 49, 1, "full_review"),
    ],
)
def test_missing_reviews_cannot_block_a_certain_middle_band(good, defects, uncertain, expected):
    ids = tuple(str(i) for i in range(100))
    reviews = (
        [reviewed(task_id) for task_id in ids[:good]]
        + [reviewed(task_id, Quality.BAD) for task_id in ids[good : good + defects]]
        + [reviewed(task_id, Quality.UNKNOWN) for task_id in ids[good + defects : good + defects + uncertain]]
        + [
            ReviewRecord(task_id=task_id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="Download timed out")
            for task_id in ids[good + defects + uncertain :]
        ]
    )
    report = source_quality_report(
        QualitySample(10000, 10000, {}, {"numeric": 10000}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.SAMPLE,
    )
    assert report.status == expected
    assert report.assessments.get(Assessment.GOOD, 0) == good
    assert report.assessments.get(Assessment.DEFECT, 0) == defects
    assert report.assessments[Assessment.UNAVAILABLE] == 100 - good - defects - uncertain
    assert report.defect_fraction is None


@pytest.mark.parametrize("unavailable,expected", [(2, "full_review"), (3, "incomplete")])
def test_unusable_raw_rows_do_not_count_as_resolvable_when_bounding_the_decision(unavailable, expected):
    """A raw panel with 48 defects, 20 unsupported rows and 2 missing reviews is decided: even two more
    defects reach only the 50% threshold, so resuming the sample cannot change the outcome."""
    usable = 80
    ids = tuple(str(i) for i in range(usable))
    good = usable - 48 - unavailable
    reviews = (
        [reviewed(task_id) for task_id in ids[:good]]
        + [reviewed(task_id, Quality.BAD) for task_id in ids[good : good + 48]]
        + [
            ReviewRecord(task_id=task_id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="HTTP 400")
            for task_id in ids[good + 48 :]
        ]
    )
    report = source_quality_report(
        QualitySample(100, usable, {"normalization:unsupported": 20}, {"numeric": usable}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.RAW_SAMPLE,
    )
    assert report.assessments[Assessment.UNUSABLE] == 20
    assert report.status == expected


@pytest.mark.parametrize("good,expected", [(4, SourceQualityStatus.REJECT), (5, SourceQualityStatus.CENSUS)])
def test_census_rejection_uses_exact_population_success_fraction(good, expected):
    ids = tuple(str(i) for i in range(10))
    reviews = [reviewed(task_id, Quality.GOOD if i < good else Quality.BAD) for i, task_id in enumerate(ids)]
    report = source_quality_report(
        QualitySample(10, 10, {}, {"numeric": 10}, ids),
        reviews,
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.CENSUS,
    )
    assert report.status == expected
    assert report.defect_fraction == (10 - good) / 10


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
            audit = audit.model_copy(
                update={"checks": [CheckResult(check="contract", status=CheckStatus.FAIL, detail="Broken")]}
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
                        docker_image="fixture@sha256:" + "0" * 64, compatible_backends=(Backend.DOCKER,)
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
    "source_defects,expected",
    [(1, SourceQualityStatus.TRUST), (51, SourceQualityStatus.REJECT), (100, SourceQualityStatus.REJECT)],
)
def test_raw_panel_counts_broken_apps_contracts_against_all_drawn_rows(source_defects, expected):
    ids = tuple(str(index) for index in range(source_defects, 100))
    sample = QualitySample(100, len(ids), {"normalization:source_defect": source_defects}, {"apps": len(ids)}, ids)
    report = source_quality_report(
        sample, [reviewed(task_id) for task_id in ids], SourceQualityPolicy(), coverage=QualitySampleCoverage.RAW_SAMPLE
    )
    assert report.status == expected
    assert report.assessments[Assessment.DEFECT] == source_defects
    assert report.defect_fraction == source_defects / 100


@pytest.mark.parametrize(
    "exclusion", ["normalization:unsupported", "normalization:converter_error", "exact_semantic_duplicate"]
)
@pytest.mark.parametrize("usable", [0, 49, 99])
def test_raw_conversion_gaps_and_duplicates_are_neutral(exclusion, usable):
    ids = tuple(str(index) for index in range(usable))
    report = source_quality_report(
        QualitySample(100, usable, {exclusion: 100 - usable}, {"numeric": usable}, ids),
        [reviewed(task_id) for task_id in ids],
        SourceQualityPolicy(),
        coverage=QualitySampleCoverage.RAW_SAMPLE,
    )
    assert report.assessments.get(Assessment.DEFECT, 0) == 0
    assert report.assessments[Assessment.UNUSABLE] == 100 - usable
    assert report.defect_fraction is None
    assert (
        report.status
        == {0: SourceQualityStatus.INCOMPLETE, 49: SourceQualityStatus.FULL_REVIEW, 99: SourceQualityStatus.TRUST}[
            usable
        ]
    )
