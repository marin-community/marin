import json

import pytest

from capability_pipeline import captured_grading_diagnostics as diagnostic


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return diagnostic.sha256(path)


def fixture(tmp_path, monkeypatch, mutation):
    output = tmp_path / "regrade"
    inputs = tmp_path / "plan-bundle/input/bundle"
    source_spec = write(inputs / "specification.json", {})
    source_renderings = write(inputs / "renderings.json", [])
    fingerprint = {"grading_input_sha256": "f" * 64}
    captured = {
        "response": None,
        "transcript": [],
        "manifest_sha256": "a" * 64,
        "manifest": {
            "fingerprint": fingerprint,
            "source_specification_sha256": source_spec,
            "source_renderings_sha256": source_renderings,
        },
    }
    monkeypatch.setattr(diagnostic, "load_capture", lambda _: captured)

    def candidate(path, root, seen):
        seen.add("candidate")
        return {"sandbox_id": "candidate"}

    def private(grade, seen, candidates, **kwargs):
        identifier = grade["detail"]["verifier_sandbox_id"]
        if identifier in seen or identifier in candidates:
            raise ValueError("reused sandbox")
        seen.add(identifier)
        return {"sandbox_id": identifier}

    monkeypatch.setattr(diagnostic, "provider_isolation_record", candidate)
    monkeypatch.setattr(diagnostic, "verifier_isolation_record", private)
    capture_root = output / "runtime-trials/capture-positive"
    original_grade = {
        "status": "graded",
        "reward": 1,
        "detail": {
            "grading_input_fingerprint": fingerprint,
            "verifier_sandbox_id": "original-private",
        },
    }
    capture = {
        "case_id": "positive",
        "trial_name": "capture-positive",
        "capture_path": "runtime-trials/capture-positive/verifier/fixed-grading-capture",
        "capture_manifest_sha256": "a" * 64,
        "status": "graded",
        "reward": 1,
        "exception": None,
        "verifier_result_present": True,
        "grading_input_fingerprint": fingerprint,
        "candidate_environment": {"sandbox_id": "candidate"},
        "private_verifier": {"sandbox_id": "original-private"},
        "trial_sha256": write(
            capture_root / "result.json",
            {"exception_info": None, "verifier_result": {"rewards": {"reward": 1}}},
        ),
        "grading_sha256": write(
            capture_root / "verifier/taskcompendium-result.json", original_grade
        ),
    }
    plan = {
        "binding_kind": "docker",
        "grading_strategy": "capture_once",
        "runtime": {"image": "fixture", "supervisor_python": "python"},
        "cells": [],
    }
    rows = []
    for repeat in range(1, 11):
        cell = {
            "ordinal": repeat,
            "case_id": "positive",
            "repeat": repeat,
            "trial_name": f"regrade-positive-run-{repeat:02d}",
        }
        plan["cells"].append(cell)
        root = output / "runtime-trials" / cell["trial_name"]
        identifier = f"private-{repeat}"
        if mutation == "reuse_original" and repeat == 1:
            identifier = "original-private"
        grade = {
            "schema_version": "capability-captured-private-grade-v1",
            "cell": cell,
            "capture_manifest_sha256": "a" * 64,
            "status": "graded",
            "reward": 1,
            "detail": {
                "grading_input_fingerprint": fingerprint,
                "fixed_grading_capture_manifest_sha256": "a" * 64,
                "verifier_sandbox_id": identifier,
            },
        }
        if mutation == "wrong_cell" and repeat == 1:
            grade["cell"] = {**cell, "repeat": 2}
        if mutation == "fabricated_trial" and repeat == 1:
            write(root / "result.json", {})
        row = {
            **cell,
            "transport": "captured_private_grade",
            "trial_sha256": None,
            "verifier_result_present": None,
            "exception": None,
            "outcome_class": "graded",
            "status": "graded",
            "reward": 1,
            "capture_manifest_sha256": "a" * 64,
            "grading_input_fingerprint": fingerprint,
            "private_verifier": {"sandbox_id": identifier},
            "grading_sha256": write(root / "private-grade.json", grade),
        }
        rows.append(row)
    if mutation == "source_drift":
        captured["manifest"]["source_specification_sha256"] = "0" * 64
    if mutation == "missing_capture":
        capture = None
    return (
        plan,
        {"captures": [capture] if capture else [], "cells": rows},
        output,
        {"cases": [{"id": "positive"}]},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "reuse_original",
        "wrong_cell",
        "fabricated_trial",
        "source_drift",
        "missing_capture",
    ],
)
def test_original_capture_and_direct_grades_are_distinct_bound_evidence(
    tmp_path, monkeypatch, mutation
):
    args = fixture(tmp_path, monkeypatch, mutation)
    issues = diagnostic.artifact_issues(*args)
    assert bool(issues) is (mutation is not None), issues
