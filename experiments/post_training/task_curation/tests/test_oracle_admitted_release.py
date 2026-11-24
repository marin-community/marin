# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A filtered native pin selects only its own measured population and content panel."""

from dataclasses import replace

import pytest
from finestore.rl.curation import (
    ContentEvidenceReference,
    DatasetRelease,
    DaytonaMeasurement,
    LexicalContaminationEvidence,
    LexicalUse,
    OracleFiltrationEvidence,
    OraclePendingExclusion,
    OraclePendingExclusionReason,
    OracleValidation,
    QualityResultFinalization,
    SamplePanelCurrency,
    SourceQualityResult,
    SourceUse,
)
from finestore.rl.status_taxonomy import ExclusionReason
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.lexical_policy import measurement_for_use
from experiments.post_training.task_curation.oracle_admitted_release import (
    OracleAdmittedReleaseReceipt,
    bind_oracle_admitted_release,
    bind_selected_source_content,
)
from experiments.post_training.task_curation.oversight_binding import bind_derived_oversight
from experiments.post_training.task_curation.prebuilt import (
    import_oracle_harbor_prebuilt,
    import_taskcompendium_prebuilt,
)
from experiments.post_training.task_curation.retained_grading import RetainedGradingReceipt
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.source import TaskFormat
from experiments.post_training.task_curation.tests.admission_fixtures import (
    actor_census_input,
    grading_evidence,
    with_consumer_receipt,
)
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from infra.marina.apps.rl_data_catalog.release_evidence import projected_sources, task_totals
from infra.marina.apps.rl_data_catalog.server.filtration import filtration
from infra.marina.apps.rl_data_catalog.server.source_counts import SourceCounts
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence
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
        intended_use=source.intended_use,
        task_format=TaskFormat.TASKSPEC,
        original_pipeline=source.pipeline_reference,
        original_info=source.info,
        filtration=filtration,
        execution_release=None,
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
    bound = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))
    bound = bind_derived_oversight(bound, receipt.sample, receipt.full_content)

    assert bound.pipeline is import_taskcompendium_prebuilt
    assert bound.current_release == receipt.filtration.filtered_release
    assert bound.declaration.task_count == 1 and bound.declaration.status == "Available"
    assert bound.info.actor_census_input == receipt.actor.input
    assert bound.info.oracle_validation is OracleValidation.EXECUTED
    assert bound.info.quality_result.reviewed == 1 and bound.info.quality_result.issue_count == 0
    assert source.info.quality_result.issue_count == 1
    projected = filtration(bound.declaration.model_dump(mode="json"))
    assert projected["state"] == "complete" and projected["example_count"] == 1
    assert projected["admitted_count"] == projected["population_count"] == 1
    assert projected["excluded_count"] == 0
    assert filtration(bound.declaration.model_dump(mode="json"), count_evidence(bound.declaration, 1, None)) == projected

    unqualified = replace(bound, info=replace(bound.info, actor_census_input=None))
    assert filtration(unqualified.declaration.model_dump(mode="json"))["example_count"] is None


def test_harbor_oracle_admitted_receipt_binds_but_cannot_admit(admitted_release):
    """TaskSpec-only admission: a Harbor release receipt binds for provenance and refuses admission.

    The receipt itself stays valid (its census, controls and SAMPLE evidence are
    real); the bound source's declared shape is not a TaskSpec population, so
    the fail-closed conversion gate refuses it with NOT_TASKSPEC_CONVERTED until
    the conversion chain produces a TaskSpec release (ruling 2026-11-24).
    """
    source, receipt = admitted_release
    harbor = OracleAdmittedReleaseReceipt.model_validate(
        {**receipt.model_dump(mode="json"), "task_format": TaskFormat.HARBOR}
    )
    bound = with_consumer_receipt(bind_oracle_admitted_release(source, harbor))

    assert bound.pipeline is import_oracle_harbor_prebuilt
    assert "admission requires a verified TaskSpec conversion receipt for the admitted revision" in (
        bound.admission_blockers
    )
    reasons = {claim.reason for claim in bound.admission_claims}
    assert ExclusionReason.NOT_TASKSPEC_CONVERTED in reasons
    assert bound.declaration.status == "Excluded"


@pytest.mark.parametrize("missing_grading", [False, True])
def test_served_oracle_rows_keep_real_qualification_independent_of_display_fields(admitted_release, missing_grading):
    source, receipt = admitted_release
    bound = bind_derived_oversight(
        with_consumer_receipt(bind_oracle_admitted_release(source, receipt)), receipt.sample, None
    )
    declaration = bound.declaration.model_dump(mode="json")
    served = {
        **declaration,
        "quality": "QUICK",  # A display label cannot grant or remove admission.
        "curation": None,
        "filtration": {"state": "pending"},
        "derived_contract_receipt": {"uri": "memory://fixture/derived-contract"},
    }
    if missing_grading:
        served["grading_environment"] = None
    projected = projected_sources({"sources": [served]}, SourceCounts(sources=()))
    totals = task_totals(projected["sources"])
    assert totals.population == 1
    assert totals.admitted == (0 if missing_grading else 1)
    assert totals.unmeasured == (1 if missing_grading else 0)
    assert projected["sources"][0]["filtration"]["state"] == ("pending" if missing_grading else "complete")


def test_old_pin_sample_cannot_qualify_filtered_native_population(admitted_release):
    source, receipt = admitted_release
    record = receipt.model_dump(mode="json")
    record["sample"]["review"]["panel"]["dataset_revision"] = source.current_release.revision

    with pytest.raises(ValueError, match="SAMPLE content review belongs to another filtered population"):
        OracleAdmittedReleaseReceipt.model_validate(record)


@pytest.mark.parametrize(
    "missing", ["actor_census_input", "grading_environment", "quality_result", "lexical_contamination"]
)
def test_available_claim_without_fresh_qualification_cannot_project_admitted_counts(admitted_release, missing):
    source, receipt = admitted_release
    bound = bind_derived_oversight(
        with_consumer_receipt(bind_oracle_admitted_release(source, receipt)), receipt.sample, None
    )
    declaration = bound.declaration.model_dump(mode="json")
    declaration[missing] = None
    projected = filtration(declaration)
    assert projected["example_count"] is None and projected["admitted_count"] is None
    assert projected["state"] == "pending"


def test_shared_native_oracles_cannot_replace_missing_integrity_contracts(admitted_release):
    _, receipt = admitted_release
    record = receipt.model_dump(mode="json")
    record["grading"]["evidence"]["required_counts"] = {"shared": 1}
    record["grading"]["evidence"]["declared_counts"] = {"shared": 1}

    with pytest.raises(ValueError, match="SHARED integrity census does not qualify"):
        OracleAdmittedReleaseReceipt.model_validate(record)


def test_canonical_filtered_pin_uses_the_fresh_output_version_and_consumer(admitted_release):
    source, native = admitted_release
    receipt = native.model_copy(update={"task_format": TaskFormat.TASKSPEC})
    bound = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))

    assert bound.pipeline is import_taskcompendium_prebuilt
    assert bound.version == receipt.actor.input.converter_version == "2026.11.14.2-final-private"
    assert bound.info.task_format is TaskFormat.TASKSPEC
    assert bound.declaration.task_count == 1


def test_answer_graded_release_does_not_claim_workspace_controls(admitted_release):
    source, native = admitted_release
    record = native.model_dump(mode="json")
    record["task_format"] = TaskFormat.TASKSPEC
    record["filtration"]["exclusion_counts"] = {"oracle_failed": 1}
    receipt = OracleAdmittedReleaseReceipt.model_validate(record)

    bound = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))

    assert bound.info.daytona_machine_slots == 0
    assert bound.info.recipe_qa_history.verification.startswith("Full original-oracle outcomes and applicable controls:")
    assert "workspace no-op controls" not in bound.info.recipe_qa_history.verification


def test_current_content_attaches_after_the_filtered_pin_is_selected(admitted_release):
    source, receipt = admitted_release
    original_actor = actor_census_input(source.dataset.name, source.dataset.revision, source.version, 2)
    source = replace(source, info=replace(source.info, actor_census_input=original_actor))
    selected = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))
    bound = bind_selected_source_content(source, selected, receipt.sample, None)

    assert bound.current_release == receipt.filtration.filtered_release
    assert bound.info.quality_result == receipt.sample.review.result

    unrelated = receipt.sample.model_copy(
        update={
            "review": receipt.sample.review.model_copy(
                update={
                    "panel": receipt.sample.review.panel.model_copy(
                        update={
                            "dataset_revision": "d" * 40,
                            "task_specs_sha256": "changed-" + receipt.sample.review.panel.task_specs_sha256,
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="Sample Result belongs to another current population"):
        bind_selected_source_content(source, selected, unrelated, None)


def test_byte_identical_repin_panel_binds_the_selected_pin_without_a_lineage(admitted_release):
    """A cross-pin panel whose bytes equal the selected census binds the selected source.

    The retained release dropped tasks, so no zero-drop lineage proof exists; the panel's
    digest and task count still equal the selected pin's actor census, so the reviewed
    bytes are the served population's and the panel carries forward at the current pin.
    """
    source, receipt = admitted_release
    selected = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))
    repinned = receipt.sample.model_copy(
        update={
            "review": receipt.sample.review.model_copy(
                update={"panel": receipt.sample.review.panel.model_copy(update={"dataset_revision": "d" * 40})}
            )
        }
    )
    bound = bind_selected_source_content(source, selected, repinned, None)

    assert bound.info.quality_result == repinned.review.result
    assert bound.info.independent_qc.panel_currency is SamplePanelCurrency.CURRENT_PIN
    assert "re-pin" in bound.info.independent_qc.method


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
    selected = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))
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
    selected = with_consumer_receipt(bind_oracle_admitted_release(source, receipt))
    bound = bind_selected_source_content(source, selected, receipt.sample, full)

    assert bound.info.quality_result == current_result
    assert bound.info.full_content_result_binding == full
    assert bound.info.actor_census_input == receipt.actor.input
    assert bound.info.oracle_validation is OracleValidation.EXECUTED


def test_filtered_evaluation_retains_measured_overlap_without_training_remediation(admitted_release):
    source, receipt = admitted_release
    source = replace(source, config=replace(source.config, intended_use=IntendedUse.EVAL))
    raw = receipt.lexical.model_dump(mode="json")
    raw.update(receipt=None, contaminated_tasks=1, contaminated_fraction=1.0, matched_evals={"HumanEval": 1})
    measured = measurement_for_use(LexicalContaminationEvidence.model_validate(raw), LexicalUse.EVAL)
    record = receipt.model_dump(mode="json")
    record.update(intended_use="eval", lexical=measured.model_dump(mode="json"))
    evaluation = OracleAdmittedReleaseReceipt.model_validate(record)
    bound = bind_derived_oversight(
        with_consumer_receipt(bind_oracle_admitted_release(source, evaluation)), evaluation.sample, None
    )

    assert not bound.admission_blockers
    assert bound.declaration.intended_use is SourceUse.EVAL
    assert bound.declaration.task_count == 1 and bound.declaration.status == "Available"
    assert bound.info.lexical_contamination.contaminated_tasks == 1
    assert bound.info.lexical_contamination.receipt.release is None
    assert bound.info.lexical_contamination.receipt.dropped_tasks == 0
    projected = filtration(bound.declaration.model_dump(mode="json"))
    assert projected["state"] == "complete" and projected["example_count"] == 1

    record["intended_use"] = "train"
    with pytest.raises(ValueError, match="benchmark measurement does not certify a training release"):
        OracleAdmittedReleaseReceipt.model_validate(record)


def test_evaluation_receipt_cannot_change_a_configured_training_source(admitted_release):
    source, receipt = admitted_release
    raw = receipt.lexical.model_dump(mode="json")
    raw["receipt"] = None
    measured = measurement_for_use(LexicalContaminationEvidence.model_validate(raw), LexicalUse.EVAL)
    record = receipt.model_dump(mode="json")
    record.update(intended_use="eval", lexical=measured.model_dump(mode="json"))
    evaluation = OracleAdmittedReleaseReceipt.model_validate(record)

    with pytest.raises(ValueError, match="another configured original population"):
        bind_oracle_admitted_release(source, evaluation)


@pytest.mark.parametrize("missing_count", [5, 6])
def test_filtered_release_sample_missing_tolerance_keeps_measured_counts(admitted_release, missing_count):
    source, original = admitted_release
    source = replace(
        source,
        info=replace(
            source.info,
            count=101,
            within_source_dedup=source.info.within_source_dedup.model_copy(
                update={"raw_count": 101, "unique_count": 101}
            ),
        ),
    )
    raw = original.model_dump(mode="json")
    raw["original_info"]["count"] = 101
    raw["original_info"]["within_source_dedup"].update(raw_count=101, unique_count=101)
    raw["filtration"].update(original_count=101, admitted_count=100)
    raw["actor"]["input"]["measured_tasks"] = 100
    raw["grading"]["input"]["measured_tasks"] = 100
    raw["grading"]["evidence"] = grading_evidence(100).model_dump(mode="json")
    raw["output_dedup"].update(raw_count=100, unique_count=100)
    lexical = decontamination_evidence(
        original.filtration.filtered_release.repo, original.filtration.filtered_release.revision, 100
    )
    raw["lexical"] = lexical.model_copy(
        update={"converter_revision": original.actor.input.converter_revision}
    ).model_dump(mode="json")
    raw["sample"]["review"]["panel"].update(retained_tasks=100, sample_size=100)
    raw["sample"]["review"].update(
        panel_coverage_complete=False,
        attempted_count=100,
        result=SourceQualityResult(
            reviewed=100 - missing_count,
            unavailable_count=missing_count,
            issue_count=19,
            issues=[],
            repair_recommendation="Inspect findings",
        ).model_dump(mode="json"),
    )
    raw["sample"]["missing_observations"] = [
        {
            "review": {
                "task_id": f"missing-{index}",
                "status": "invalid",
                "verdict": None,
                "detail": "Truncated response after bounded attempts",
            },
            "batch_receipt": {"uri": "memory://fixture/batch.json", "sha256": "e" * 64},
        }
        for index in range(missing_count)
    ]
    if missing_count == 6:
        with pytest.raises(ValueError, match="at most 5% missing"):
            OracleAdmittedReleaseReceipt.model_validate(raw)
        return
    receipt = OracleAdmittedReleaseReceipt.model_validate(raw)
    bound = bind_derived_oversight(
        with_consumer_receipt(bind_oracle_admitted_release(source, receipt)), receipt.sample, None
    )
    assert bound.declaration.status == "Available" and bound.declaration.task_count == 100
    assert bound.declaration.oversight.value == "sample"
    assert bound.info.quality_result.reviewed == bound.info.independent_qc.reviewed == 95
    assert bound.info.recipe_qa_history.reviewed == 95
    assert bound.info.quality_result.unavailable_count == 5
    assert bound.info.quality_result.issue_rate == 19 / 95
    assert not receipt.sample.review.panel_coverage_complete
    with pytest.raises(ValueError, match="Complete finalization requires complete coverage"):
        FullContentResultBinding(
            source_id=source.info.id,
            input=receipt.actor.input,
            contract_index=receipt.consumer,
            review_receipt=receipt.consumer,
            records=receipt.consumer,
            result=receipt.sample.review.result,
            coverage_complete=False,
            finalization=QualityResultFinalization.COMPLETE,
        )


@pytest.mark.parametrize("reason", ["resource_calibration:infrastructure", "negative_control_pending"])
def test_passing_population_is_selectable_with_pending_originals(admitted_release, reason):
    source, receipt = admitted_release
    filtration = OracleFiltrationEvidence.model_validate(
        {
            **receipt.filtration.model_dump(),
            "exclusion_counts": {reason: 1},
            "pending_exclusions": (
                [OraclePendingExclusion(task_id="pending-original", reason=OraclePendingExclusionReason.INFRASTRUCTURE)]
                if reason == "resource_calibration:infrastructure"
                else []
            ),
        }
    )
    qualified = OracleAdmittedReleaseReceipt.model_validate({**receipt.model_dump(), "filtration": filtration})
    bound = bind_derived_oversight(
        with_consumer_receipt(bind_oracle_admitted_release(source, qualified)), qualified.sample, None
    )
    assert bound.declaration.status == "Available" and bound.declaration.task_count == 1
    assert bound.current_release == filtration.filtered_release
    assert bound.info.oracle_filtration == filtration
    assert filtration.exclusion_counts == {reason: 1}
