# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The recovery runner groups superseded receipts and recovers them typed."""

import hashlib
import json
from pathlib import Path

import pytest
from finestore.rl.curation import OversightLevel
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus

import experiments.post_training.task_curation.sample_review_recovery as sample_review_recovery
from experiments.post_training.task_curation.sample_qc_binding import SampleMissingObservation, SampleQCReceipt
from experiments.post_training.task_curation.sample_review_recovery import (
    REGISTRY_REPO,
    ReceiptReference,
    RecoveryTarget,
    collect_receipt_history,
    completion_queue,
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

    def part_reference(revision: str, path: str) -> ReceiptReference:
        sha256 = hashlib.sha256(revision.encode()).hexdigest()
        return ReceiptReference(repo=REGISTRY_REPO, revision=revision, path=path, sha256=sha256)

    current = {"source_id": SOURCE_ID, "review": {"panel": {"dataset_revision": CURRENT_REVISION}}}
    stale_a = {"source_id": SOURCE_ID, "review": {"panel": {"dataset_revision": EARLIER_REVISION}}}
    stale_b = {"source_id": SOURCE_ID, "review": {"panel": {"dataset_revision": "c" * 40}}}
    documents = {
        newest: {SOURCE_ID: (part_reference(newest, "part-000.json"), current)},
        second: {SOURCE_ID: (part_reference(second, "part-000.json"), stale_a)},
        oldest: {SOURCE_ID: (part_reference(oldest, "part-002.json"), stale_b)},
    }
    history = collect_receipt_history(documents)
    assert sorted(history) == [SOURCE_ID]
    references = history[SOURCE_ID]
    assert len(references) == 2
    assert {reference.revision for reference in references} == {second, oldest}


def _write_part(path, document) -> str:
    part = {SOURCE_ID: document}
    path.write_text(json.dumps(part, sort_keys=True))
    return str(path)


def _target(tmp_path, monkeypatch, stale_receipt, current_receipt):
    stale_part_path = _write_part(tmp_path / "stale-part.json", receipt_document(stale_receipt))
    current_part_path = _write_part(tmp_path / "current-part.json", receipt_document(current_receipt))
    stale_part_sha = hashlib.sha256(Path(stale_part_path).read_bytes()).hexdigest()
    current_part_sha = hashlib.sha256(Path(current_part_path).read_bytes()).hexdigest()

    def fake_download(repo_id, repo_type, revision, filename):
        assert repo_id == "fixture/registry" and repo_type == "dataset"
        if filename == "part-000.json":
            return stale_part_path
        return current_part_path

    monkeypatch.setattr("experiments.post_training.task_curation.registry.hf_hub_download", fake_download)
    return RecoveryTarget(
        source_id=SOURCE_ID,
        current=ReceiptReference(
            repo="fixture/registry",
            revision="b" * 40,
            path="current-part.json",
            sha256=current_part_sha,
        ),
        prior=(
            ReceiptReference(repo="fixture/registry", revision="a" * 40, path="part-000.json", sha256=stale_part_sha),
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
    with pytest.raises(ValueError, match="no entry for this source"):
        recover_target(alien, tmp_path.as_posix(), (FIELD,))


def test_completion_queue_targets_a_receipts_missing_observations(tmp_path, monkeypatch):
    """The completion queue carries exactly the tasks a receipt's own missing
    observations enumerate, beside a derivation record that resolves."""
    population = write_population(tmp_path, "current", current_tasks())
    reviews = {task_id: good_review(task_id) for task_id in list(current_tasks())}
    missing_id = "task-004"
    reviews[missing_id] = ReviewRecord(task_id=missing_id, status=ReviewStatus.REVIEWED, verdict=None, detail="held")
    receipt = write_receipt(tmp_path, population, current_tasks(), reviews)
    receipt = receipt.model_copy(
        update={
            "missing_observations": (
                SampleMissingObservation(
                    review=reviews[missing_id],
                    batch_receipt={"uri": "memory://batch", "sha256": "e" * 64},
                ),
            )
        }
    )
    reference = ReceiptReference(repo="fixture/registry", revision="b" * 40, path="part-000.json", sha256="a" * 64)
    monkeypatch.setattr(sample_review_recovery, "_newest_sample_receipts", lambda: {SOURCE_ID: (reference, receipt)})
    records_root = tmp_path / "records"
    queue = completion_queue(records_root.as_posix(), (SOURCE_ID,))
    assert len(queue.targets) == 1
    target = queue.targets[0]
    assert target.source_id == SOURCE_ID
    assert target.task_ids == (missing_id,)
    assert target.population == receipt.review.panel.input_artifact
    record = json.loads(Path(target.topup_evidence).read_text())
    assert record["missing_task_ids"] == [missing_id]
    assert record["receipt"]["path"] == "part-000.json"


def test_completion_queue_skips_complete_receipts(tmp_path, monkeypatch):
    population = write_population(tmp_path, "current", current_tasks())
    reviews = {task_id: good_review(task_id) for task_id in current_tasks()}
    receipt = write_receipt(tmp_path, population, current_tasks(), reviews)
    reference = ReceiptReference(repo="fixture/registry", revision="b" * 40, path="part-000.json", sha256="a" * 64)
    monkeypatch.setattr(sample_review_recovery, "_newest_sample_receipts", lambda: {SOURCE_ID: (reference, receipt)})
    assert completion_queue((tmp_path / "records").as_posix(), (SOURCE_ID,)).targets == ()
