# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The recovery runner groups superseded receipts and recovers them typed."""

import json

import pytest
from finestore.rl.curation import OversightLevel

from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.sample_review_recovery import (
    ReceiptReference,
    RecoveryTarget,
    collect_receipt_history,
    document_sha256,
    recover_target,
)
from experiments.post_training.task_curation.tests.test_review_carry_forward import (
    CURRENT_PIN,
    CURRENT_REVISION,
    EARLIER_REVISION,
    SOURCE_ID,
    good_review,
    write_population,
    write_receipt,
)

FIELD = "execution_requirements"


def earlier_tasks() -> dict[str, dict]:
    return {f"task-{index:03d}": {"id": f"task-{index:03d}", "prompt": "x" * index} for index in range(12)}


def current_tasks() -> dict[str, dict]:
    return {f"task-{index:03d}": {"id": f"task-{index:03d}", "prompt": "x" * index, FIELD: None} for index in range(12)}


def receipt_document(receipt: SampleQCReceipt) -> dict:
    return json.loads(receipt.model_dump_json())


def test_collect_receipt_history_keeps_distinct_superseded_pins():
    newest, second, oldest = "f" * 40, "e" * 40, "d" * 40
    current = {"source_id": SOURCE_ID, "review": {"panel": {"dataset_revision": CURRENT_REVISION}}}
    stale_a = {"source_id": SOURCE_ID, "review": {"panel": {"dataset_revision": EARLIER_REVISION}}}
    stale_b = {"source_id": SOURCE_ID, "review": {"panel": {"dataset_revision": "c" * 40}}}
    documents = {
        newest: {"part-000.json": current},
        second: {"part-000.json": stale_a, "part-001.json": current},
        oldest: {"part-002.json": stale_b},
    }
    history = collect_receipt_history(documents)
    assert sorted(history) == [SOURCE_ID]
    references = history[SOURCE_ID]
    assert len(references) == 2
    assert {reference.revision for reference in references} == {second, oldest}
    assert {reference.sha256 for reference in references} == {document_sha256(stale_a), document_sha256(stale_b)}


def _write(path, document) -> str:
    # Match canonical_sha256's exact serialization so the reference pins these bytes.
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return str(path)


def _target(tmp_path, monkeypatch, stale_receipt, current_receipt):
    stale_document = receipt_document(stale_receipt)
    current_document = receipt_document(current_receipt)
    stale_document_path = _write(tmp_path / "stale-receipt.json", stale_document)
    current_document_path = _write(tmp_path / "current-receipt.json", current_document)

    def fake_download(repo_id, repo_type, revision, filename):
        assert repo_id == "fixture/registry" and repo_type == "dataset"
        if filename == "part-000.json":
            return stale_document_path
        return current_document_path

    monkeypatch.setattr("experiments.post_training.task_curation.registry.hf_hub_download", fake_download)
    return RecoveryTarget(
        source_id=SOURCE_ID,
        current=ReceiptReference(
            repo="fixture/registry",
            revision="b" * 40,
            path="current-receipt.json",
            sha256=document_sha256(current_document),
        ),
        prior=(
            ReceiptReference(
                repo="fixture/registry", revision="a" * 40, path="part-000.json", sha256=document_sha256(stale_document)
            ),
        ),
        current_population=current_receipt.review.panel.input_artifact,
    )


def test_recover_target_carries_non_panel_members_with_normalization(tmp_path, monkeypatch):
    earlier_population = write_population(tmp_path, "earlier", earlier_tasks())
    current_population = write_population(
        tmp_path, "current", current_tasks(), dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    stale_reviews = {task_id: good_review(task_id) for task_id in list(earlier_tasks())[:6]}
    stale_receipt = write_receipt(
        tmp_path, earlier_population, earlier_tasks(), stale_reviews, panel_ids=list(earlier_tasks())[:6], name="stale"
    )
    current_reviews = {task_id: good_review(task_id) for task_id in list(current_tasks())[6:]}
    current_receipt = write_receipt(
        tmp_path,
        current_population,
        current_tasks(),
        current_reviews,
        panel_ids=list(current_tasks())[6:],
        name="current",
    )
    target = _target(tmp_path, monkeypatch, stale_receipt, current_receipt)
    recovery, priors = recover_target(target, tmp_path.as_posix(), (FIELD,))
    assert priors[0].carried == 6
    assert priors[0].normalization is not None and priors[0].normalization.field == FIELD
    assert recovery.carried_reviews == 6
    assert recovery.coverage_conclusive == 12
    assert recovery.panel_conclusive == 6
    assert recovery.reviewed_fraction == 1.0
    assert recovery.level is OversightLevel.FULL and recovery.full is True


def test_recover_target_without_a_verified_normalization_carries_exact_identities_only(tmp_path, monkeypatch):
    earlier_population = write_population(tmp_path, "earlier", earlier_tasks())
    current_population = write_population(
        tmp_path, "current", current_tasks(), dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    stale_reviews = {task_id: good_review(task_id) for task_id in list(earlier_tasks())[:6]}
    stale_receipt = write_receipt(
        tmp_path, earlier_population, earlier_tasks(), stale_reviews, panel_ids=list(earlier_tasks())[:6], name="stale"
    )
    current_reviews = {task_id: good_review(task_id) for task_id in list(current_tasks())[6:]}
    current_receipt = write_receipt(
        tmp_path,
        current_population,
        current_tasks(),
        current_reviews,
        panel_ids=list(current_tasks())[6:],
        name="current",
    )
    target = _target(tmp_path, monkeypatch, stale_receipt, current_receipt)
    recovery, priors = recover_target(target, tmp_path.as_posix(), ())
    assert priors[0].carried == 0
    assert priors[0].normalization is None
    assert recovery.carried_reviews == 0
    assert recovery.coverage_conclusive == 6
    assert recovery.level is OversightLevel.SAMPLE and recovery.full is False


def test_recover_target_refuses_a_receipt_from_another_source(tmp_path, monkeypatch):
    earlier_population = write_population(tmp_path, "earlier", earlier_tasks())
    current_population = write_population(
        tmp_path, "current", current_tasks(), dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    stale_reviews = {task_id: good_review(task_id) for task_id in list(earlier_tasks())[:6]}
    stale_receipt = write_receipt(
        tmp_path, earlier_population, earlier_tasks(), stale_reviews, panel_ids=list(earlier_tasks())[:6], name="stale"
    )
    current_reviews = {task_id: good_review(task_id) for task_id in list(current_tasks())[6:]}
    current_receipt = write_receipt(
        tmp_path,
        current_population,
        current_tasks(),
        current_reviews,
        panel_ids=list(current_tasks())[6:],
        name="current",
    )
    target = _target(tmp_path, monkeypatch, stale_receipt, current_receipt)
    alien = target.model_copy(update={"source_id": "OpenAthena:another-source"})
    with pytest.raises(ValueError, match="another source"):
        recover_target(alien, tmp_path.as_posix(), (FIELD,))
