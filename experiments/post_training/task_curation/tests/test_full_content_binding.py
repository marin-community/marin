# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whole-source finality survives SAMPLE composition only for the exact current input."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    ContentEvidenceReference,
    OversightEvidence,
    OversightLevel,
    QualityResultFinalization,
    SamplePanelCurrency,
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

from experiments.post_training.task_curation.full_content_binding import (
    RetainedLineageProof,
    bind_full_content_result,
    full_content_matches_lineage,
)
from experiments.post_training.task_curation.full_content_receipts import full_content_result_binding
from experiments.post_training.task_curation.full_content_results import (
    FullContentResultBinding,
)
from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.full_qc_inputs import prepare_inputs
from experiments.post_training.task_curation.full_qc_review import contract_batches, write_review_result
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.oversight_binding import derived_oversight_level
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import (
    SampleQCReceipt,
    bind_sample_result,
)
from experiments.post_training.task_curation.source import SourceReference
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input
from experiments.post_training.task_curation.tests.conftest import fixture_qc_inputs
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from infra.marina.apps.rl_data_catalog.server.quality import quality, review_completeness
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
            reviewed=192 if incomplete else 194,
            uncertain_count=7,
            unavailable_count=2 if incomplete else 0,
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
    assert quality(restored.model_dump(mode="json"), None) == "FULL"
    if finalization is QualityResultFinalization.ACCEPTED_INCOMPLETE:
        without_waiver = restored.model_copy(update={"quality_result_waiver": None})
        assert quality(without_waiver.model_dump(mode="json"), None) == "SAMPLE"
        assert review_completeness(without_waiver, [], [])["provisional"] is True


@pytest.mark.parametrize("change", ["dataset_revision", "task_specs_sha256", "evidence_url", "missing_binding"])
def test_changed_input_or_missing_binding_clears_stale_finality_before_fresh_sample(content_source, change):
    binding = whole_source_binding(content_source, QualityResultFinalization.ACCEPTED_INCOMPLETE)
    final = bind_full_content_result(content_source, binding)
    actor = final.info.actor_census_input
    if change == "missing_binding":
        changed = replace(final, info=replace(final.info, full_content_result_binding=None))
    else:
        replacement = {
            "dataset_revision": "b" * 40,
            "task_specs_sha256": "b" * 64,
            "evidence_url": "memory://actor-census/replaced/.artifact.json",
        }[change]
        actor = actor.model_copy(update={change: replacement})
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
    assert bound.info.quality_result_waiver is None
    assert review_completeness(bound.declaration, [], []) == {
        "reviewed": 100,
        "required": 100,
        "coverage_complete": True,
        "provisional": False,
    }
    if change != "missing_binding":
        with pytest.raises(ValueError, match="another current retained population"):
            bind_full_content_result(bound, binding)


def waived_binding(unavailable: int) -> FullContentResultBinding:
    """A waiver binding over a 100-task retained population with one uncertain row."""
    reference = ContentEvidenceReference(uri="memory://verified/waiver", sha256="e" * 64)
    return FullContentResultBinding(
        source_id="Marin:fixture",
        input=actor_census_input("open-athena/fixture", "a" * 40, "1", 100),
        contract_index=reference,
        review_receipt=reference,
        records=reference,
        result=SourceQualityResult(
            reviewed=99 - unavailable,
            uncertain_count=1,
            unavailable_count=unavailable,
            issue_count=0,
            issues=[],
            repair_recommendation="No repairs indicated by conclusive reviews.",
        ),
        coverage_complete=False,
        finalization=QualityResultFinalization.ACCEPTED_INCOMPLETE,
        waiver=reference,
    )


def test_full_waiver_at_the_one_percent_bound_is_accepted():
    binding = waived_binding(1)
    assert binding.result.sample_count == 100
    assert binding.waiver is not None


def test_full_waiver_above_the_one_percent_bound_is_rejected():
    with pytest.raises(ValueError, match="1% unreviewed bound"):
        waived_binding(2)


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


@pytest.mark.parametrize("finalization", list(QualityResultFinalization))
def test_final_binding_verifies_records_and_preserves_exact_retained_runtime(tmp_path, finalization):
    # One waived task must stay within the 1% unreviewed bound, so the retained
    # population here is large enough that its single invalid row qualifies.
    # Within-source deduplication drops identical contracts, so every extra task
    # carries its own prompt text to survive as a distinct retained task.
    extra_documents = []
    for index in range(119):
        document = json.loads(prebuilt_task(f"extra-{index}")["lowered_task_spec"])["task"]
        document["context"]["events"][0]["content"] = f"Do the task {index}."
        extra_documents.append(document)
    config, documents = fixture_qc_inputs(tmp_path, extra_documents=extra_documents)
    retained_documents = [document for index, document in enumerate(documents) if index != 1]
    retained_count = len(retained_documents)
    incomplete = finalization is QualityResultFinalization.ACCEPTED_INCOMPLETE
    inputs = prepare_inputs(config)
    batches = list(contract_batches(inputs))
    observations = []
    retained_index = 0
    for batch in batches:
        batch_observations = []
        for contract in batch.contracts:
            if incomplete and retained_index == 1:
                batch_observations.append(
                    ReviewRecord(
                        task_id=contract["task_id"], status=ReviewStatus.INVALID, verdict=None, detail="Truncated"
                    )
                )
            else:
                batch_observations.append(
                    ReviewRecord(
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
            retained_index += 1
        observations.append(batch_observations)
    assert retained_index == retained_count
    reviewer = local_reviewer(ReviewerVersion.V2)
    review = write_review_result(
        inputs,
        {"mode": "chat", **reviewer.identity},
        REVIEW_RUBRIC,
        str(tmp_path / "reviews"),
        [
            (batch, batch_observations, "memory://batch")
            for batch, batch_observations in zip(batches, observations, strict=True)
        ],
    )
    files = write_text_population(((doc, {"deployment": 1}) for doc in retained_documents), str(tmp_path / "retained"))
    population = inputs.population.model_copy(
        update={
            "path": str(tmp_path / "retained"),
            "task_count": retained_count,
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
                    "retained_count": retained_count,
                    "contract_index_sha256": inputs.index_sha256,
                    "review_artifact_sha256": receipt.sha256,
                    "records_sha256": review.records_sha256,
                    "qc_unreviewed": [
                        {"task_id": row.task_id, "reason": f"QC unreviewed: {row.detail}"}
                        for row in observations[0]
                        if row.status is not ReviewStatus.REVIEWED
                    ],
                    "result": review.result.model_dump(mode="json"),
                }
            )
        )
        waiver = ContentEvidenceReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))
    actor_input = ActorCensusInput(
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        converter_revision=population.converter_revision,
        measured_tasks=population.task_count,
        task_specs_sha256=population.task_specs_sha256,
        evidence_url="memory://actor-census/immutable/.artifact.json",
    )
    binding = full_content_result_binding(receipt, finalization, waiver, reviewer, actor_input, population)
    assert binding.input.measured_tasks == retained_count
    assert binding.input.task_specs_sha256 == files.task_specs_sha256
    assert binding.result.issue_count == (retained_count - 1 if incomplete else retained_count)
    assert binding.result.unavailable_count == int(incomplete)
    assert binding.records.sha256 == review.records_sha256 and binding.contract_index.sha256 == inputs.index_sha256
    assert binding.result.issues == []
    assert review.result.issues
    assert binding.result.model_dump(exclude={"issues"}) == review.result.model_dump(exclude={"issues"})
    if incomplete:
        original = path.read_text()
        omitted = json.loads(original)
        omitted["qc_unreviewed"] = []
        path.write_text(json.dumps(omitted))
        missing_task_waiver = ContentEvidenceReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))
        with pytest.raises(ValueError, match="enumerate every unreviewed task"):
            full_content_result_binding(receipt, finalization, missing_task_waiver, reviewer, actor_input, population)
        changed_result = json.loads(original)
        changed_result["result"]["uncertain_count"] += 1
        path.write_text(json.dumps(changed_result))
        changed_result_waiver = ContentEvidenceReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))
        with pytest.raises(ValueError, match="Result and uncertainty counts"):
            full_content_result_binding(receipt, finalization, changed_result_waiver, reviewer, actor_input, population)
        path.write_text(original)
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
        full_content_result_binding(changed_receipt, finalization, waiver, reviewer, actor_input, population)


def retained_view(source):
    """The same source re-pinned to a zero-drop retained release with its own census.

    The retained census carries its own artifact-level digest: the retained release
    re-serializes the same tasks, so the digest differs from the original pin's while
    the task count is unchanged (the ContentIdentityReceipt case).
    """
    release = SourceReference(
        "open-athena/numbers-retained",
        "b" * 40,
        "https://huggingface.co/datasets/open-athena/numbers-retained/tree/" + "b" * 40,
    )
    config = replace(source.config, name="numbers-retained", version="2026.11.13.1", dataset=release)
    actor = actor_census_input(release.name, release.revision, config.version, 201).model_copy(
        update={"task_specs_sha256": "e" * 64}
    )
    return replace(
        source,
        config=config,
        info=replace(
            source.info,
            actor_census_input=actor,
            within_source_dedup=deduplication_evidence(release.name, release.revision, config.version, 201),
            quality_result=SourceQualityResult(reviewed=100, issue_count=0, issues=[], repair_recommendation="None"),
        ),
    )


def retained_lineage(source):
    actor = source.info.actor_census_input
    return RetainedLineageProof(
        source_id=source.info.id,
        original_dataset_id=source.dataset.name,
        original_dataset_revision=source.dataset.revision,
        original_input=actor,
        original_unique_count=source.info.within_source_dedup.unique_count,
        retained_content_sha256="f" * 64,
    )


def test_lineage_validated_original_panel_is_bound_and_recorded(content_source):
    """A panel byte-different from the current pin binds through the zero-drop lineage label."""
    original, retained = content_source, retained_view(content_source)
    lineage = retained_lineage(original)
    receipt = sample_receipt(original)
    bound = bind_sample_result(retained, receipt, retained_lineage=lineage)
    assert "validated against the original pin" in bound.info.independent_qc.method
    assert lineage.retained_content_sha256[:12] in bound.info.independent_qc.method
    assert bound.info.quality_result == receipt.review.result
    assert bound.info.independent_qc.evidence_url == receipt.review.path


def test_byte_identical_retained_panel_binds_current_pin_over_the_lineage_label(content_source):
    """A retained panel whose digest equals the retained pin's census binds CURRENT_PIN.

    The lineage label is the fallback for panels whose bytes differ from the served
    population; when the digests match, the reviewed bytes ARE the served bytes and
    the stronger direct evidence wins (coordinator carry-forward rule).
    """
    original, retained = content_source, retained_view(content_source)
    lineage = retained_lineage(original)
    actor = retained.info.actor_census_input
    receipt = sample_receipt(original).model_copy(
        update={
            "review": (
                sample_receipt(original).review.model_copy(
                    update={
                        "panel": (
                            sample_receipt(original).review.panel.model_copy(
                                update={
                                    "dataset_id": actor.dataset_id,
                                    "dataset_revision": actor.dataset_revision,
                                    "task_specs_sha256": actor.task_specs_sha256,
                                }
                            )
                        )
                    }
                )
            )
        }
    )
    bound = bind_sample_result(retained, receipt, retained_lineage=lineage)
    assert bound.info.independent_qc.panel_currency is SamplePanelCurrency.CURRENT_PIN
    assert "the reviewed bytes are the served population's" in bound.info.independent_qc.method
    assert bound.info.quality_result == receipt.review.result


def test_lineage_derives_full_from_an_archived_whole_source_result(content_source):
    original, retained = content_source, retained_view(content_source)
    lineage = retained_lineage(original)
    binding = whole_source_binding(original, QualityResultFinalization.COMPLETE)
    assert derived_oversight_level(retained, None, binding, lineage) is OversightLevel.FULL
    assert derived_oversight_level(retained, None, binding) is OversightLevel.QUICK


def test_lineage_mismatch_still_raises(content_source):
    """A lineage whose original pin differs AND whose panel bytes differ from the served
    population cannot bind: neither the lineage label nor the byte-identity path applies."""
    original, retained = content_source, retained_view(content_source)
    moved = retained_lineage(original).model_copy(update={"original_dataset_revision": "c" * 40})
    with pytest.raises(ValueError, match="belongs to another current population"):
        bind_sample_result(retained, sample_receipt(original), retained_lineage=moved)


def test_lineage_validated_whole_source_result_is_bound_and_recorded(content_source):
    original, retained = content_source, retained_view(content_source)
    lineage = retained_lineage(original)
    binding = whole_source_binding(original, QualityResultFinalization.COMPLETE)
    bound = bind_full_content_result(retained, binding, retained_lineage=lineage)
    assert "validated against the original pin" in bound.info.independent_qc.method
    assert bound.info.quality_result == binding.result
    assert bound.info.quality_result_finalization is QualityResultFinalization.COMPLETE
    assert bound.info.independent_qc.evidence_url == binding.review_receipt.uri
    assert full_content_matches_lineage(binding, lineage)
    with pytest.raises(ValueError, match="belongs to another current retained population"):
        bind_full_content_result(retained, binding)
