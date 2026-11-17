# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grandfathered old-campaign FULL reviews carry only content-matching completed rows."""

import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ActorCensusInput, ContentEvidenceReference, QualityResultFinalization
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name, write_record
from marin.inference.openai_chat import OpenAIChatClient
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
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.full_content_receipts import full_content_result_binding
from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.full_qc_artifacts import FullQCReview
from experiments.post_training.task_curation.full_qc_grandfather import grandfathered_review
from experiments.post_training.task_curation.full_qc_inputs import prepare_inputs
from experiments.post_training.task_curation.full_qc_review import contract_batches, write_review_result
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.tests.conftest import fixture_qc_inputs


def reviewed_record(task_id: str) -> ReviewRecord:
    return ReviewRecord(
        task_id=task_id,
        status=ReviewStatus.REVIEWED,
        detail="",
        verdict=ReviewVerdict(
            task_id=task_id,
            quality=Quality.BAD,
            confidence=Confidence.HIGH,
            reference_status=ReferenceStatus.CONFLICT,
            defects=[],
            evidence="Reference conflicts with the task",
        ),
    )


def receipt_reference(review: FullQCReview) -> ContentEvidenceReference:
    path = StoragePath(review.path) / RECORD_FILENAME
    return ContentEvidenceReference(uri=str(path), sha256=file_sha256(path))


def record_evidence(tmp_path: Path, payload: dict) -> ContentEvidenceReference:
    path = tmp_path / "grandfathering-evidence.json"
    path.write_text(json.dumps(payload))
    return ContentEvidenceReference(uri=str(path), sha256=file_sha256(StoragePath(str(path))))


def rewrite_receipt(review: FullQCReview, tmp_path: Path, name: str) -> ContentEvidenceReference:
    """Persist a modified receipt copy so binding checks can be exercised on storage."""
    target = tmp_path / name
    write_record(
        ArtifactRecord(
            output_path=str(target), result_type=result_type_name(FullQCReview), result=review.result_payload()
        )
    )
    path = StoragePath(str(target)) / RECORD_FILENAME
    return ContentEvidenceReference(uri=str(path), sha256=file_sha256(path))


def _receipt_rows(records_path: str) -> dict[str, dict]:
    with StoragePath(records_path).open("rb") as stream:
        return {row["task_id"]: row for row in pq.read_table(stream).to_pylist()}


@pytest.fixture
def prepared_inputs(tmp_path):
    config, documents = fixture_qc_inputs(tmp_path)
    inputs = prepare_inputs(config)
    batches = list(contract_batches(inputs))
    return inputs, batches, documents


def test_grandfathered_merge_carries_matching_rows_and_marks_the_old_reviewer(prepared_inputs, review_server, tmp_path):
    inputs, batches, _ = prepared_inputs
    current = local_reviewer(ReviewerVersion.V2)
    old_spec = replace(current, model_revision="aca966e4-old-campaign")
    observations = [
        reviewed_record(batches[0].contracts[0]["task_id"]),
        ReviewRecord(
            task_id=batches[0].contracts[1]["task_id"], status=ReviewStatus.INVALID, verdict=None, detail="Truncated"
        ),
    ]
    old = write_review_result(
        inputs,
        {"mode": "chat", **old_spec.identity},
        REVIEW_RUBRIC,
        str(tmp_path / "old"),
        [(batches[0], observations, "memory://batch")],
    )
    evidence = record_evidence(tmp_path, {"ruling": "grandfather earlier FULL QC reviews", "date": "2026-11-15"})
    client = OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2)
    reviewer = ChatReviewer(client, current, query_cache_root=str(tmp_path / "cache"))

    merged = grandfathered_review(
        inputs, receipt_reference(old), reviewer, REVIEW_RUBRIC, evidence, str(tmp_path / "merged")
    )

    assert merged.reviewer == {"mode": "chat", **current.identity}
    assert merged.grandfathered is not None
    assert merged.grandfathered.reviewer == {"mode": "chat", **old_spec.identity}
    assert merged.grandfathered.accepted_count == 1
    assert merged.coverage_complete
    assert merged.attempted_count == inputs.retained_count
    assert merged.result.reviewed == inputs.retained_count
    assert merged.result.issue_count == 1

    old_rows = _receipt_rows(old.records_path)
    rows = _receipt_rows(merged.records_path)
    carried_id = batches[0].contracts[0]["task_id"]
    fresh_id = batches[0].contracts[1]["task_id"]
    # The carried row keeps its original review payload and old-campaign batch receipt.
    assert rows[carried_id]["review"] == old_rows[carried_id]["review"]
    assert rows[carried_id]["batch_receipt"] == old_rows[carried_id]["batch_receipt"]
    assert rows[carried_id]["batch_receipt"] not in set(merged.batch_receipts)
    # The invalid row was not grandfathered; the current reviewer re-reviewed it.
    assert json.loads(rows[fresh_id]["review"])["status"] == "reviewed"
    assert rows[fresh_id]["batch_receipt"] in set(merged.batch_receipts)


def test_grandfathered_merge_rejects_foreign_sources_and_empty_carryovers(prepared_inputs, review_server, tmp_path):
    inputs, batches, _ = prepared_inputs
    current = local_reviewer(ReviewerVersion.V2)
    old_spec = replace(current, model_revision="aca966e4-old-campaign")
    observations = [reviewed_record(row["task_id"]) for row in batches[0].contracts]
    old = write_review_result(
        inputs,
        {"mode": "chat", **old_spec.identity},
        REVIEW_RUBRIC,
        str(tmp_path / "old"),
        [(batches[0], observations, "memory://batch")],
    )
    evidence = record_evidence(tmp_path, {"ruling": "grandfather earlier FULL QC reviews"})
    client = OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2)
    reviewer = ChatReviewer(client, current, query_cache_root=str(tmp_path / "cache"))

    foreign = old.model_copy(update={"source_id": "Marin:another-source"})
    with pytest.raises(ValueError, match="belongs to another source"):
        grandfathered_review(
            inputs, rewrite_receipt(foreign, tmp_path, "foreign"), reviewer, REVIEW_RUBRIC, evidence, str(tmp_path / "a")
        )

    incomplete = write_review_result(
        inputs,
        {"mode": "chat", **old_spec.identity},
        REVIEW_RUBRIC,
        str(tmp_path / "incomplete"),
        [
            (
                batches[0],
                [
                    ReviewRecord(task_id=row["task_id"], status=ReviewStatus.INVALID, verdict=None, detail="Truncated")
                    for row in batches[0].contracts
                ],
                "memory://batch",
            )
        ],
    )
    with pytest.raises(ValueError, match="no completed rows to carry"):
        grandfathered_review(
            inputs,
            receipt_reference(incomplete),
            reviewer,
            REVIEW_RUBRIC,
            evidence,
            str(tmp_path / "b"),
        )


def test_binding_validates_the_grandfathered_marker(prepared_inputs, review_server, tmp_path):
    inputs, batches, documents = prepared_inputs
    current = local_reviewer(ReviewerVersion.V2)
    old_spec = replace(current, model_revision="aca966e4-old-campaign")
    observations = [reviewed_record(row["task_id"]) for row in batches[0].contracts]
    old = write_review_result(
        inputs,
        {"mode": "chat", **old_spec.identity},
        REVIEW_RUBRIC,
        str(tmp_path / "old"),
        [(batches[0], observations, "memory://batch")],
    )
    evidence = record_evidence(tmp_path, {"ruling": "grandfather earlier FULL QC reviews"})
    client = OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2)
    reviewer = ChatReviewer(client, current, query_cache_root=str(tmp_path / "cache"))
    merged = grandfathered_review(
        inputs, receipt_reference(old), reviewer, REVIEW_RUBRIC, evidence, str(tmp_path / "merged")
    )
    # The binding binds to the exact retained population: every document but the dropped one.
    retained = [document for index, document in enumerate(documents) if index != 1]
    files = write_text_population(((doc, {"deployment": 1}) for doc in retained), str(tmp_path / "retained"))
    population = inputs.population.model_copy(
        update={
            "path": str(tmp_path / "retained"),
            "task_count": len(retained),
            "task_specs_sha256": files.task_specs_sha256,
            "specs_output_dir": files.specs_output_dir,
            "main_output_dir": files.main_output_dir,
        }
    )
    actor_input = ActorCensusInput(
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        converter_revision=population.converter_revision,
        measured_tasks=population.task_count,
        task_specs_sha256=population.task_specs_sha256,
        evidence_url="memory://actor-census/immutable/.artifact.json",
    )

    binding = full_content_result_binding(
        receipt_reference(merged), QualityResultFinalization.COMPLETE, None, current, actor_input, population
    )
    assert binding.finalization is QualityResultFinalization.COMPLETE
    assert binding.result.model_dump(exclude={"issues"}) == merged.result.model_dump(exclude={"issues"})

    relabeled = merged.model_copy(
        update={"grandfathered": merged.grandfathered.model_copy(update={"reviewer": merged.reviewer})}
    )
    with pytest.raises(ValueError, match="another reviewer identity"):
        full_content_result_binding(
            rewrite_receipt(relabeled, tmp_path, "relabeled"),
            QualityResultFinalization.COMPLETE,
            None,
            current,
            actor_input,
            population,
        )

    miscounted = merged.model_copy(
        update={"grandfathered": merged.grandfathered.model_copy(update={"accepted_count": 3})}
    )
    with pytest.raises(ValueError, match="does not match the carried observations"):
        full_content_result_binding(
            rewrite_receipt(miscounted, tmp_path, "miscounted"),
            QualityResultFinalization.COMPLETE,
            None,
            current,
            actor_input,
            population,
        )

    changed = record_evidence(tmp_path, {"ruling": "a different decision"})
    stale = merged.model_copy(update={"grandfathered": merged.grandfathered.model_copy(update={"evidence": changed})})
    rewrite_receipt(stale, tmp_path, "stale-evidence")
    Path(changed.uri).write_text(json.dumps({"ruling": "rewritten after the receipt"}))
    with pytest.raises(ValueError, match="Grandfathering evidence changed"):
        full_content_result_binding(
            ContentEvidenceReference(
                uri=str(StoragePath(str(tmp_path / "stale-evidence")) / RECORD_FILENAME),
                sha256=file_sha256(StoragePath(str(tmp_path / "stale-evidence")) / RECORD_FILENAME),
            ),
            QualityResultFinalization.COMPLETE,
            None,
            current,
            actor_input,
            population,
        )
