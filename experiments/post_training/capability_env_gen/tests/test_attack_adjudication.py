import copy
import json
from types import SimpleNamespace

import pytest

from capability_pipeline.attack_adjudication import (
    SCHEMA,
    prepare_packet,
    run_adjudication,
    validate_receipt,
    validate_resolution,
)
from capability_pipeline.quality import sha256


def fixture(root, *, reward=0.75, shortcut_error=False):
    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    put(
        "contract/accepted.json",
        {"rubric": "four independent equally weighted criteria"},
    )
    put(
        "workspace/task/specification.json", {"fixture": "not a runtime-certified task"}
    )
    cases = []
    for strategy in ("injection", "shortcut", "boundary"):
        grade = {"status": "graded", "reward": reward if strategy == "boundary" else 0}
        grading = put(f"runtime-trials/{strategy}/grading.json", grade)
        transcript = put(
            f"runtime-trials/{strategy}/transcript.json",
            [{"role": "assistant", "content": "fixture candidate"}],
        )
        case = {
            "strategy": strategy,
            "steps": [
                {
                    "step_index": 0,
                    "grading_artifact": str(grading.relative_to(root)),
                    "grading_sha256": sha256(grading),
                    "transcript_artifact": str(transcript.relative_to(root)),
                    "transcript_sha256": sha256(transcript),
                    "result": grade,
                }
            ],
        }
        if shortcut_error and strategy == "shortcut":
            case["error"] = "model_output_truncated"
        cases.append(case)
    put(
        "independent-adversary.json",
        {"independent": True, "max_false_reward": 0.2, "cases": cases},
    )


def receipt(manifest):
    return {
        "schema_version": SCHEMA,
        "snapshot_hash": manifest["snapshot_hash"],
        "cases": [
            {
                "strategy": "boundary",
                "step_index": 0,
                "disposition": "legitimate_partial",
                "rubric_basis": "Three correct criteria out of four; fixture only.",
                "candidate_analysis": "Fourth criterion fails; first three pass.",
                "critical_gate_analysis": "No critical gate exists in this fixture.",
                "expected_reward": {"min": 0.75, "max": 0.75},
                "citations": [
                    {"path": name, "sha256": value, "supports": "Fixture evidence only"}
                    for name, value in manifest["files"].items()
                ],
            }
        ],
    }


def test_legitimate_partial_clears_only_reward_issue_and_preserves_raw(tmp_path):
    item, review = tmp_path / "item", tmp_path / "review"
    fixture(item, shortcut_error=True)
    original = (item / "independent-adversary.json").read_bytes()
    manifest = prepare_packet(item, review)
    assert validate_receipt(receipt(manifest), manifest, review / "input") == [
        "shortcut: attack execution failed"
    ]
    assert (item / "independent-adversary.json").read_bytes() == original


def test_exploit_uncertain_missing_duplicate_and_false_full_credit_stay_closed(
    tmp_path,
):
    item, review = tmp_path / "item", tmp_path / "review"
    fixture(item)
    manifest = prepare_packet(item, review)
    for disposition in ("exploit", "uncertain"):
        value = receipt(manifest)
        value["cases"][0]["disposition"] = disposition
        assert validate_receipt(value, manifest, review / "input") == [
            "boundary:0: rewarded attack requires independent adjudication"
        ]
    for change in ([], receipt(manifest)["cases"] * 2):
        value = receipt(manifest)
        value["cases"] = change
        with pytest.raises(ValueError, match="every rewarded"):
            validate_receipt(value, manifest, review / "input")
    value = receipt(manifest)
    value["cases"][0]["expected_reward"]["max"] = 1
    with pytest.raises(ValueError, match="partial credit"):
        validate_receipt(value, manifest, review / "input")


def test_stale_snapshot_missing_candidate_and_mismatched_raw_are_rejected(tmp_path):
    item, review = tmp_path / "item", tmp_path / "review"
    fixture(item)
    manifest = prepare_packet(item, review)
    value = receipt(manifest)
    value["snapshot_hash"] = "stale"
    with pytest.raises(ValueError, match="snapshot"):
        validate_receipt(value, manifest, review / "input")
    value = receipt(manifest)
    value["cases"][0]["citations"] = [
        c
        for c in value["cases"][0]["citations"]
        if not c["path"].endswith("transcript.json")
    ]
    with pytest.raises(ValueError, match="actual raw grade"):
        validate_receipt(value, manifest, review / "input")
    grading = review / "input/runtime-trials/boundary/grading.json"
    grading.chmod(0o644)
    grading.write_text('{"status":"graded","reward":1}')
    with pytest.raises(ValueError, match="input changed"):
        validate_receipt(receipt(manifest), manifest, review / "input")
    altered = copy.deepcopy(manifest)
    altered["files"]["runtime-trials/boundary/grading.json"] = sha256(grading)
    value = receipt(altered)
    with pytest.raises(ValueError, match="actual raw grade"):
        validate_receipt(value, altered, review / "input")


def test_fresh_agent_receipt_resolves_without_certifying_task(tmp_path):
    item, review = tmp_path / "item", tmp_path / "review"
    fixture(item)

    def invoke(cwd, transcript, prompt, index):
        manifest = json.loads((cwd / "input-manifest.json").read_text())
        (cwd / "receipt.json").write_text(json.dumps(receipt(manifest)))
        return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}

    result = run_adjudication(
        item, review, SimpleNamespace(model="fixture-only", invoke=invoke)
    )
    assert result["state"] == "resolved"
    assert result["runtime_certified"] is False
    assert result["quality_certified"] is False
    assert result["receipt_sha256"] == sha256(review / "receipt.json")
    assert validate_resolution(item, review) == result
    diagnostics = item / "diagnostics/attempt-1/evaluation"
    diagnostics.mkdir(parents=True)
    (diagnostics / "matrix.json").write_text('{"state":"repeated_runtime_passed"}')
    assert validate_resolution(item, review) == result
    (item / "workspace/task/specification.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed after adjudication"):
        validate_resolution(item, review)
    with pytest.raises(ValueError, match="fresh directory"):
        prepare_packet(item, review)


def test_invalid_citation_paths_get_bounded_same_session_correction(tmp_path):
    item, review = tmp_path / "item", tmp_path / "review"
    fixture(item)
    calls = []

    def invoke(cwd, transcript, prompt, index):
        calls.append((transcript, index))
        manifest = json.loads((cwd / "input-manifest.json").read_text())
        value = receipt(manifest)
        if index == 0:
            for citation in value["cases"][0]["citations"]:
                citation["path"] = "input/" + citation["path"]
        (cwd / "receipt.json").write_text(json.dumps(value))
        return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}

    result = run_adjudication(
        item, review,
        SimpleNamespace(model="fixture-only", invoke=invoke, max_continuations=8),
    )
    assert result["state"] == "resolved"
    assert calls == [(review / "transcript", 0), (review / "transcript", 1)]
    assert result["invalid_receipts"][0]["validation_error"] == (
        "adjudication citation is not bound to evidence"
    )
    assert result["invalid_receipts"][0]["sha256"] == sha256(
        review / "receipt-attempt-0.json"
    )
    assert validate_resolution(item, review) == result


def test_invalid_receipt_correction_exhaustion_never_clears_attack(tmp_path):
    item, review = tmp_path / "item", tmp_path / "review"
    fixture(item)
    calls = []

    def invoke(cwd, transcript, prompt, index):
        calls.append(index)
        manifest = json.loads((cwd / "input-manifest.json").read_text())
        value = receipt(manifest)
        for citation in value["cases"][0]["citations"]:
            citation["path"] = "input/" + citation["path"]
        (cwd / "receipt.json").write_text(json.dumps(value))
        return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}

    result = run_adjudication(
        item, review,
        SimpleNamespace(model="fixture-only", invoke=invoke, max_continuations=8),
    )
    assert calls == [0, 1, 2]
    assert result["state"] == "pending"
    assert len(result["invalid_receipts"]) == 3
    with pytest.raises(ValueError, match="no completed independent resolution"):
        validate_resolution(item, review)
