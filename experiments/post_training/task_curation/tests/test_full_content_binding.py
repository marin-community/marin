# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whole-source finality survives SAMPLE composition only for the exact current input."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    OversightEvidence,
    OversightLevel,
    QualityResultFinalization,
    SourceDeclaration,
    SourceQualityResult,
)
from finestore.schema import arrow_schema
from marin.execution.artifact import RECORD_FILENAME, read_record, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.models import (
    Confidence,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    ReviewVerdict,
)

from experiments.post_training.task_curation.full_content_binding import bind_full_content_result
from experiments.post_training.task_curation.full_content_receipts import full_content_result_binding
from experiments.post_training.task_curation.full_content_results import (
    ContentEvidenceReference,
    FullContentResultBinding,
)
from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.full_qc_inputs import prepare_inputs
from experiments.post_training.task_curation.full_qc_review import contract_batches, write_review_result
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt, bind_sample_result
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from infra.marina.apps.rl_data_catalog.load import review_completeness
from infra.marina.apps.rl_data_catalog.server.quality import quality
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


@pytest.fixture
def content_source(tmp_path):
    source = number_source(tmp_path / "input")
    actor = actor_census_input(source.dataset.name, source.dataset.revision, source.version, 201)
    return replace(
        source,
        info=replace(
            source.info,
            actor_census_input=actor,
            within_source_dedup=deduplication_evidence(actor.dataset_id, actor.dataset_revision, source.version, 201),
            oversight=OversightEvidence(level=OversightLevel.SAMPLE, campaign="fixture", verification="qualified"),
        ),
    )


def whole_source_binding(source, finalization):
    incomplete = finalization is QualityResultFinalization.ACCEPTED_INCOMPLETE
    reference = ContentEvidenceReference(uri="memory://verified/evidence", sha256="a" * 64)
    return FullContentResultBinding(
        source_id=source.info.id,
        input=source.info.actor_census_input,
        contract_index=reference,
        review_receipt=reference,
        records=reference,
        result=SourceQualityResult(
            reviewed=190 if incomplete else 194,
            uncertain_count=7,
            unavailable_count=4 if incomplete else 0,
            issue_count=42,
            issues=[],
            repair_recommendation="Inspect pinned task evidence",
        ),
        coverage_complete=not incomplete,
        finalization=finalization,
        waiver=reference if incomplete else None,
    )


def sample_receipt(source):
    actor = source.info.actor_census_input
    panel = SamplePanel(
        source_id=source.info.id,
        dataset_id=actor.dataset_id,
        dataset_revision=actor.dataset_revision,
        converter_version=actor.converter_version,
        input_artifact="memory://sample/input",
        task_specs_sha256=actor.task_specs_sha256,
        scanned_task_specs_sha256=actor.task_specs_sha256,
        retained_tasks=actor.measured_tasks,
        sample_seed=0,
        sample_size=100,
        panel_path="memory://sample/panel.parquet",
        panel_sha256="b" * 64,
        evidence_url="memory://sample/panel.json",
    )
    return SampleQCReceipt(
        source_id=source.info.id,
        artifact_sha256="c" * 64,
        review=SampleQCReview(
            path="memory://sample/review",
            panel=panel,
            reviewer={},
            rubric={},
            panel_coverage_complete=True,
            attempted_count=100,
            result=SourceQualityResult(reviewed=100, issue_count=0, issues=[], repair_recommendation="None"),
            records_path="memory://sample/reviews.parquet",
            records_sha256="d" * 64,
            batch_receipts=[],
        ),
    )


@pytest.mark.parametrize("finalization", list(QualityResultFinalization))
def test_same_input_sample_cannot_replace_explicit_whole_source_finality(content_source, finalization):
    binding = whole_source_binding(content_source, finalization)
    first = bind_full_content_result(bind_sample_result(content_source, sample_receipt(content_source)), binding)
    repeated = bind_full_content_result(bind_sample_result(first, sample_receipt(first)), None)
    assert repeated.info.quality_result == binding.result
    assert repeated.info.quality_result_finalization is finalization
    assert repeated.info.actor_census_input == content_source.info.actor_census_input
    assert repeated.info.oversight == content_source.info.oversight
    assert repeated.info.within_source_dedup == content_source.info.within_source_dedup
    assert repeated.info.grading_environment == content_source.info.grading_environment
    table = pa.Table.from_pylist([repeated.declaration.model_dump(mode="json")], schema=arrow_schema(SourceDeclaration))
    restored = SourceDeclaration.model_validate(table.to_pylist()[0])
    assert restored.quality_result_finalization is finalization
    completeness = review_completeness(restored, [], [])
    assert completeness == {
        "reviewed": binding.result.reviewed,
        "required": 201,
        "coverage_complete": binding.coverage_complete,
        "provisional": False,
    }
    assert quality(restored.model_dump(mode="json"), None) == ("FULL" if binding.coverage_complete else "SAMPLE")


@pytest.mark.parametrize("change", ["dataset_revision", "task_specs_sha256", "missing_binding"])
def test_changed_input_or_missing_binding_clears_stale_finality_before_fresh_sample(content_source, change):
    binding = whole_source_binding(content_source, QualityResultFinalization.ACCEPTED_INCOMPLETE)
    final = bind_full_content_result(content_source, binding)
    actor = final.info.actor_census_input
    if change == "missing_binding":
        changed = replace(final, info=replace(final.info, full_content_result_binding=None))
    else:
        actor = actor.model_copy(update={change: "b" * (40 if change == "dataset_revision" else 64)})
        changed = replace(
            final,
            config=replace(
                final.config,
                dataset=replace(
                    final.dataset,
                    revision=actor.dataset_revision,
                    url=f"https://huggingface.co/datasets/{actor.dataset_id}/tree/{actor.dataset_revision}",
                ),
            ),
            info=replace(
                final.info,
                actor_census_input=actor,
                within_source_dedup=deduplication_evidence(actor.dataset_id, actor.dataset_revision, final.version, 201),
            ),
        )
    fresh = sample_receipt(changed)
    bound = bind_full_content_result(bind_sample_result(changed, fresh), None)
    assert bound.info.quality_result == fresh.review.result
    assert bound.info.quality_result_finalization is None and bound.info.full_content_result_binding is None
    assert review_completeness(bound.declaration, [], []) == {
        "reviewed": 100,
        "required": 100,
        "coverage_complete": True,
        "provisional": False,
    }
    if change != "missing_binding":
        with pytest.raises(ValueError, match="another current retained population"):
            bind_full_content_result(bound, binding)


def test_retained_gym_3939_result_cannot_bind_to_raw_3941_population(content_source):
    actor = content_source.info.actor_census_input.model_copy(update={"measured_tasks": 3939})
    reference = ContentEvidenceReference(uri="memory://verified/gym", sha256="e" * 64)
    binding = FullContentResultBinding(
        source_id=content_source.info.id,
        input=actor,
        contract_index=reference,
        review_receipt=reference,
        records=reference,
        result=SourceQualityResult(
            reviewed=3884,
            uncertain_count=35,
            unavailable_count=20,
            issue_count=372,
            issues=[],
            repair_recommendation="Inspect issues",
        ),
        coverage_complete=False,
        finalization=QualityResultFinalization.ACCEPTED_INCOMPLETE,
        waiver=reference,
    )
    raw = replace(
        content_source,
        info=replace(
            content_source.info,
            actor_census_input=actor.model_copy(update={"measured_tasks": 3941}),
            within_source_dedup=deduplication_evidence(
                actor.dataset_id, actor.dataset_revision, content_source.version, 3941
            ),
        ),
    )
    with pytest.raises(ValueError, match="another current retained population"):
        bind_full_content_result(raw, binding)
    assert raw.info.quality_result is None


@pytest.mark.parametrize("finalization", list(QualityResultFinalization))
def test_final_binding_verifies_records_and_preserves_exact_retained_runtime(qc_inputs, tmp_path, finalization):
    config, documents = qc_inputs
    inputs = prepare_inputs(config)
    batches = list(contract_batches(inputs))
    incomplete = finalization is QualityResultFinalization.ACCEPTED_INCOMPLETE
    observations = [
        (
            ReviewRecord(task_id=contract["task_id"], status=ReviewStatus.INVALID, verdict=None, detail="Truncated")
            if incomplete and index == 1
            else ReviewRecord(
                task_id=contract["task_id"],
                status=ReviewStatus.REVIEWED,
                detail="",
                verdict=ReviewVerdict(
                    task_id=contract["task_id"],
                    quality=Quality.BAD,
                    confidence=Confidence.HIGH,
                    reference_status=ReferenceStatus.CONFLICT,
                    defects=[],
                    evidence="Reference conflicts with the task",
                ),
            )
        )
        for index, contract in enumerate(batches[0].contracts)
    ]
    reviewer = local_reviewer(ReviewerVersion.V2)
    review = write_review_result(
        inputs,
        {"mode": "chat", **reviewer.identity},
        REVIEW_RUBRIC,
        str(tmp_path / "reviews"),
        [(batches[0], observations, "memory://batch")],
    )
    files = write_text_population(
        ((doc, {"deployment": 1}) for doc in (documents[0], documents[2])), str(tmp_path / "retained")
    )
    population = inputs.population.model_copy(
        update={
            "path": str(tmp_path / "retained"),
            "task_count": 2,
            "task_specs_sha256": files.task_specs_sha256,
            "specs_output_dir": files.specs_output_dir,
            "main_output_dir": files.main_output_dir,
        }
    )
    receipt_path = StoragePath(review.path) / RECORD_FILENAME
    receipt = ContentEvidenceReference(uri=str(receipt_path), sha256=file_sha256(receipt_path))
    waiver = None
    if incomplete:
        path = tmp_path / "waiver.json"
        path.write_text(
            json.dumps(
                {
                    "source_id": review.source_id,
                    "retained_count": 2,
                    "contract_index_sha256": inputs.index_sha256,
                    "review_artifact_sha256": receipt.sha256,
                    "records_sha256": review.records_sha256,
                }
            )
        )
        waiver = ContentEvidenceReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))
    binding = full_content_result_binding(receipt, finalization, waiver, reviewer, population)
    assert binding.input.measured_tasks == 2 and binding.input.task_specs_sha256 == files.task_specs_sha256
    assert binding.result.issue_count == (1 if incomplete else 2)
    assert binding.result.unavailable_count == int(incomplete)
    assert binding.records.sha256 == review.records_sha256 and binding.contract_index.sha256 == inputs.index_sha256
    assert binding.result.issues == []
    table = pq.read_table(review.records_path)
    rows = table.to_pylist()
    rows[0]["runtime_sha256"] = "f" * 64
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), review.records_path)
    record = read_record(review.path)
    write_record(
        record.model_copy(
            update={"result": {**record.result, "records_sha256": file_sha256(StoragePath(review.records_path))}}
        )
    )
    changed_receipt = ContentEvidenceReference(uri=receipt.uri, sha256=file_sha256(receipt_path))
    with pytest.raises(ValueError, match="exact retained contract"):
        full_content_result_binding(changed_receipt, finalization, waiver, reviewer, population)
