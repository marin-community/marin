# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A filtered native pin selects only its own measured population and content panel."""

from dataclasses import replace

import pytest
from finestore.rl.curation import (
    ContentEvidenceReference,
    DatasetRelease,
    DaytonaMeasurement,
    OracleFiltrationEvidence,
    OracleValidation,
    QualityResultFinalization,
    SourceQualityResult,
)

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.oracle_admitted_release import (
    OracleAdmittedReleaseReceipt,
    bind_oracle_admitted_release,
    bind_selected_source_content,
)
from experiments.post_training.task_curation.prebuilt import (
    import_oracle_harbor_prebuilt,
    import_taskcompendium_prebuilt,
)
from experiments.post_training.task_curation.retained_grading import RetainedGradingReceipt
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.source import TaskFormat
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input, grading_evidence
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence


@pytest.fixture
def admitted_release(tmp_path):
    source = number_source(tmp_path / "original")
    source = replace(
        source,
        info=replace(
            source.info,
            count=2,
            within_source_dedup=source.info.within_source_dedup.model_copy(update={"raw_count": 2, "unique_count": 2}),
            quality_result=SourceQualityResult(
                reviewed=2, issue_count=1, issues=["Old population"], repair_recommendation="Old"
            ),
        ),
    )
    release = DatasetRelease(
        repo="open-athena/native-filtered",
        revision="f" * 40,
        url="https://huggingface.co/datasets/open-athena/native-filtered/tree/" + "f" * 40,
    )
    proof = ContentEvidenceReference(uri="memory://fixture/evidence", sha256="a" * 64)
    filtration = OracleFiltrationEvidence(
        original_release=source.current_release,
        original_cohort=proof,
        original_count=2,
        oracle_filter=proof,
        oracle_filter_version=4,
        filtered_release=release,
        private=True,
        tasks_parquet=proof,
        admitted_count=1,
        exclusion_counts={"negative_control_passed": 1},
        validation=proof,
    )
    census = actor_census_input(release.repo, release.revision, "2026.11.14.2-final-private", 1)
    actor = ActorCensusReceipt(
        source_id=source.info.id,
        input=census,
        measurement=DaytonaMeasurement(
            dataset_id=release.repo,
            dataset_revision=release.revision,
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Synthetic metadata census",
            daytona_unsafe_reasons=[],
            daytona_unsafe_details={},
        ),
        integrity_contracts=None,
    )
    panel = SamplePanel(
        source_id=source.info.id,
        dataset_id=release.repo,
        dataset_revision=release.revision,
        converter_version="2026.11.14.2-final-private",
        input_artifact="memory://fixture/population",
        task_specs_sha256=census.task_specs_sha256,
        scanned_task_specs_sha256=census.task_specs_sha256,
        retained_tasks=1,
        sample_seed=0,
        sample_size=1,
        panel_path="memory://fixture/panel.parquet",
        panel_sha256="b" * 64,
        evidence_url="memory://fixture/panel.json",
    )
    review = SampleQCReview(
        path="memory://fixture/review",
        panel=panel,
        reviewer={},
        rubric={},
        panel_coverage_complete=True,
        attempted_count=1,
        result=SourceQualityResult(reviewed=1, issue_count=0, issues=[], repair_recommendation="None"),
        records_path="memory://fixture/reviews.parquet",
        records_sha256="c" * 64,
        batch_receipts=[],
    )
    receipt = OracleAdmittedReleaseReceipt(
        source_id=source.info.id,
        task_format=TaskFormat.HARBOR,
        original_pipeline=source.pipeline_reference,
        original_info=source.info,
        filtration=filtration,
        output_dedup=source.info.within_source_dedup.model_copy(
            update={
                "source_dataset_id": release.repo,
                "source_revision": release.revision,
                "converter_revision": census.converter_revision,
                "converter_version": census.converter_version,
                "raw_count": 1,
                "unique_count": 1,
            }
        ),
        actor=actor,
        grading=RetainedGradingReceipt(
            source_id=source.info.id,
            input=census,
            normalized_path="memory://fixture/normalized",
            evidence=grading_evidence(1),
        ),
        lexical=decontamination_evidence(release.repo, release.revision, 1).model_copy(
            update={"converter_revision": census.converter_revision}
        ),
        sample=SampleQCReceipt(source_id=source.info.id, review=review, artifact_sha256="d" * 64),
        repair_evidence=(proof,),
        consumer=proof,
    )
    return source, receipt


def test_native_filtered_pin_is_selectable_with_fresh_qualification(admitted_release):
    source, receipt = admitted_release
    bound = bind_oracle_admitted_release(source, receipt)

    assert bound.pipeline is import_oracle_harbor_prebuilt
    assert bound.current_release == receipt.filtration.filtered_release
    assert bound.declaration.task_count == 1 and bound.declaration.status == "Available"
    assert bound.info.actor_census_input == receipt.actor.input
    assert bound.info.oracle_validation is OracleValidation.EXECUTED
    assert bound.info.quality_result.reviewed == 1 and bound.info.quality_result.issue_count == 0
    assert source.info.quality_result.issue_count == 1


def test_old_pin_sample_cannot_qualify_filtered_native_population(admitted_release):
    source, receipt = admitted_release
    record = receipt.model_dump(mode="json")
    record["sample"]["review"]["panel"]["dataset_revision"] = source.current_release.revision

    with pytest.raises(ValueError, match="SAMPLE content review belongs to another filtered population"):
        OracleAdmittedReleaseReceipt.model_validate(record)


def test_shared_native_oracles_cannot_replace_missing_integrity_contracts(admitted_release):
    _, receipt = admitted_release
    record = receipt.model_dump(mode="json")
    record["grading"]["evidence"]["required_counts"] = {"shared": 1}
    record["grading"]["evidence"]["declared_counts"] = {"shared": 1}

    with pytest.raises(ValueError, match="SHARED integrity census does not qualify"):
        OracleAdmittedReleaseReceipt.model_validate(record)


def test_canonical_filtered_pin_uses_the_fresh_output_version_and_consumer(admitted_release):
    source, native = admitted_release
    receipt = native.model_copy(update={"task_format": TaskFormat.TASKCOMPENDIUM})
    bound = bind_oracle_admitted_release(source, receipt)

    assert bound.pipeline is import_taskcompendium_prebuilt
    assert bound.version == receipt.actor.input.converter_version == "2026.11.14.2-final-private"
    assert bound.info.task_format is TaskFormat.TASKCOMPENDIUM
    assert bound.declaration.task_count == 1


def test_current_content_attaches_after_the_filtered_pin_is_selected(admitted_release):
    source, receipt = admitted_release
    selected = bind_oracle_admitted_release(source, receipt)
    bound = bind_selected_source_content(source, selected, receipt.sample, None)

    assert bound.current_release == receipt.filtration.filtered_release
    assert bound.info.quality_result == receipt.sample.review.result

    unrelated = receipt.sample.model_copy(
        update={
            "review": receipt.sample.review.model_copy(
                update={"panel": receipt.sample.review.panel.model_copy(update={"dataset_revision": "d" * 40})}
            )
        }
    )
    with pytest.raises(ValueError, match="Sample Result belongs to another current population"):
        bind_selected_source_content(source, selected, unrelated, None)


def test_archived_original_sample_does_not_replace_filtered_sample(admitted_release):
    source, receipt = admitted_release
    original_actor = actor_census_input(source.dataset.name, source.dataset.revision, source.version, 2)
    source = replace(source, info=replace(source.info, actor_census_input=original_actor))
    archived_panel = receipt.sample.review.panel.model_copy(
        update={
            "dataset_id": source.dataset.name,
            "dataset_revision": source.dataset.revision,
            "converter_version": source.version,
            "task_specs_sha256": original_actor.task_specs_sha256,
            "retained_tasks": 2,
            "sample_size": 2,
        }
    )
    archived = receipt.sample.model_copy(
        update={
            "review": receipt.sample.review.model_copy(
                update={"panel": archived_panel, "attempted_count": 2, "result": source.info.quality_result}
            )
        }
    )
    selected = bind_oracle_admitted_release(source, receipt)
    bound = bind_selected_source_content(source, selected, archived, None)

    assert bound.info.quality_result == receipt.sample.review.result
    assert bound.info.quality_result.reviewed == 1


def test_fresh_full_result_attaches_after_filtered_sample(admitted_release):
    source, receipt = admitted_release
    current_result = SourceQualityResult(
        reviewed=1, issue_count=1, issues=["Current FULL finding"], repair_recommendation="Repair"
    )
    full = FullContentResultBinding(
        source_id=source.info.id,
        input=receipt.actor.input,
        contract_index=receipt.consumer,
        review_receipt=receipt.consumer,
        records=receipt.consumer,
        result=current_result,
        coverage_complete=True,
        finalization=QualityResultFinalization.COMPLETE,
    )
    selected = bind_oracle_admitted_release(source, receipt)
    bound = bind_selected_source_content(source, selected, receipt.sample, full)

    assert bound.info.quality_result == current_result
    assert bound.info.full_content_result_binding == full
    assert bound.info.actor_census_input == receipt.actor.input
    assert bound.info.oracle_validation is OracleValidation.EXECUTED
