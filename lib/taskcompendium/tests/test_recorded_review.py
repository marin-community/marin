# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from taskcompendium.datasets.numeric_answers import normalize_svamp, svamp_policy
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow, ReviewRecord, ReviewStatus
from taskcompendium.pipeline.recorded_review import RecordedReviewer, load_recorded_reviews


@dataclass(frozen=True)
class UnavailableReviewer:
    @property
    def identity(self):
        return {"model": "fixture-provider", "transport": "direct"}

    def review(self, tasks, rubric, output_path, *, originals=None):
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "submitted.json").write_text(
            json.dumps(
                {
                    "tasks": {task.id: canonical_sha256(task.model_dump(mode="json")) for task in tasks},
                    "originals": sorted(originals or {}),
                }
            )
        )
        return [
            ReviewRecord(task_id=task.id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="Provider unresolved")
            for task in tasks
        ]


def task(task_id="one", answer="1"):
    return normalize_svamp(
        RawRow(
            task_id,
            Source(dataset="fixture/tasks", revision="1", row=task_id, importer_revision="1"),
            {"Body": "Aya has one apple.", "Question": "How many apples?", "Answer": answer, "Equation": answer},
        )
    )


def judgment(candidate: TaskSpec, rubric):
    return {
        "task_sha256": canonical_sha256(candidate.model_dump(mode="json")),
        "rubric_sha256": canonical_sha256(asdict(rubric)),
        "review": {
            "task_id": candidate.id,
            "status": "reviewed",
            "verdict": {
                "task_id": candidate.id,
                "quality": "good",
                "confidence": "medium",
                "reference_status": "unknown",
                "defects": [],
                "evidence": "Inspected only the supplied bounded preview; omitted resource bytes were not inspected.",
            },
            "detail": "Manual judgment; no program execution or provider completion.",
        },
        "provenance": {"source_revision": "1", "limitations": ["Projected content only; oracle was not executed"]},
    }


def write_bundle(path: Path, records):
    payload = {
        "schema_version": "recorded-review-v1",
        "provenance": {"reviewer_model": "manual-fixture", "reviewer_session": "recorded-session"},
        "records": records,
    }
    path.write_text(json.dumps(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorded_review_merges_exact_judgment_and_retains_distinct_provenance(tmp_path):
    first, second = task(), task("two")
    rubric = svamp_policy().rubric
    evidence = judgment(first, rubric)
    bundle = tmp_path / "bundle.json"
    digest = write_bundle(bundle, [evidence])
    reviewer = RecordedReviewer(UnavailableReviewer(), str(bundle), digest)
    output = tmp_path / "review"
    results = reviewer.review([second, first], rubric, output)
    assert [record.task_id for record in results] == [second.id, first.id]
    assert results[0].status == ReviewStatus.UNAVAILABLE
    assert results[1].verdict.model_dump(mode="json") == evidence["review"]["verdict"]
    detail = json.loads(results[1].detail)["recorded_review"]
    assert detail["evidence_sha256"] == digest
    assert detail["original_detail"] == evidence["review"]["detail"]
    assert detail["review_provenance"] == evidence["provenance"]
    submitted = json.loads((output / "fallback/submitted.json").read_text())
    assert submitted["tasks"] == {second.id: canonical_sha256(second.model_dump(mode="json"))}
    sidecar = json.loads((output / "recorded-reviews.json").read_text())
    assert sidecar["reviewer"]["model"] == "fixture-provider"
    assert sidecar["reviewer"]["recorded_evidence"]["sha256"] == digest
    assert sidecar["records"][0]["bundle_provenance"]["reviewer_model"] == "manual-fixture"
    assert load_recorded_reviews(str(bundle), digest).records[0].review.detail == evidence["review"]["detail"]
    all_recorded = tmp_path / "all-recorded"
    assert reviewer.review([first], rubric, all_recorded)[0].verdict == results[1].verdict
    assert not (all_recorded / "fallback").exists()


@pytest.mark.parametrize("mismatch", ["private_contract", "rubric", "rewrite_original"])
def test_same_task_id_does_not_authorize_recorded_review_for_another_request(tmp_path, mismatch):
    original = task()
    rubric = svamp_policy().rubric
    bundle = tmp_path / "bundle.json"
    digest = write_bundle(bundle, [judgment(original, rubric)])
    candidate = task(answer="2") if mismatch == "private_contract" else original
    current_rubric = (
        replace(rubric, criteria=(*rubric.criteria, "Compare another requirement")) if mismatch == "rubric" else rubric
    )
    originals = {original.id: original} if mismatch == "rewrite_original" else None
    output = tmp_path / "review"
    result = RecordedReviewer(UnavailableReviewer(), str(bundle), digest).review(
        [candidate], current_rubric, output, originals=originals
    )[0]
    assert result.status == ReviewStatus.UNAVAILABLE
    assert result.verdict is None
    assert json.loads((output / "recorded-reviews.json").read_text())["records"] == []
    submitted = json.loads((output / "fallback/submitted.json").read_text())
    assert submitted["tasks"] == {candidate.id: canonical_sha256(candidate.model_dump(mode="json"))}
    assert submitted["originals"] == ([original.id] if originals is not None else [])


@pytest.mark.parametrize(
    "defect", ["conflicting_judgments", "missing_verdict", "wrong_verdict_id", "wrong_claimed_task"]
)
def test_ambiguous_or_malformed_recorded_evidence_cannot_supply_a_judgment(tmp_path, defect):
    candidate = task()
    rubric = svamp_policy().rubric
    evidence = judgment(candidate, rubric)
    records = [evidence]
    if defect == "conflicting_judgments":
        conflicting = judgment(candidate, rubric)
        conflicting["review"]["verdict"]["quality"] = "bad"
        records.append(conflicting)
    elif defect == "missing_verdict":
        evidence["review"]["verdict"] = None
    elif defect == "wrong_verdict_id":
        evidence["review"]["verdict"]["task_id"] = "another-task"
    else:
        evidence["review"]["task_id"] = evidence["review"]["verdict"]["task_id"] = "another-task"
    bundle = tmp_path / "bundle.json"
    digest = write_bundle(bundle, records)
    output = tmp_path / "review"
    with pytest.raises(ValueError):
        RecordedReviewer(UnavailableReviewer(), str(bundle), digest).review([candidate], rubric, output)
    assert not output.exists()


def test_changed_evidence_bytes_cannot_reuse_a_previous_good_judgment(tmp_path):
    candidate = task()
    rubric = svamp_policy().rubric
    evidence = judgment(candidate, rubric)
    bundle = tmp_path / "bundle.json"
    digest = write_bundle(bundle, [evidence])
    evidence["review"]["verdict"]["quality"] = "bad"
    write_bundle(bundle, [evidence])
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="SHA256"):
        RecordedReviewer(UnavailableReviewer(), str(bundle), digest).review([candidate], rubric, output)
    assert not output.exists()
