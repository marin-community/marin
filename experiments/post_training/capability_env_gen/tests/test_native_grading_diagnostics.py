"""Native grading and extraction receipts use distinct, frozen contracts."""

import hashlib
import json

import pytest

from capability_pipeline.grading_diagnostics import _classify


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(tmp_path, *, mutation=None):
    controls = {
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "response": "42",
                "expect": {"status": "graded", "reward_min": 1, "reward_max": 1},
            },
            {
                "id": "malformed",
                "class": "malformed",
                "response": "",
                "expect": {"status": "extraction_error"},
            },
        ]
    }
    plan = {
        "binding_kind": "none",
        "verifier_surface": "native_deterministic",
        "repeats": 10,
        "cell_count": 20,
        "require_fixed_grading_input": True,
        "identities": {
            "native_verifier": {"sha256": "a" * 64},
            "native_semantic_verifier": {"sha256": "c" * 64},
            "native_semantic_verifier_runtime_sha256": "b" * 64,
        },
        "runtime": {"image": None, "supervisor_python": None},
        "cells": [],
    }
    receipt = {
        "schema_version": "capability-native-verifier-receipt-v1",
        "adapter_sha256": "a" * 64,
        "semantic_verifier_sha256": "b" * 64,
    }
    rows = []
    output = tmp_path / "regrade"
    write(tmp_path / "plan-bundle/input/bundle/controls.json", controls)
    for case in controls["cases"]:
        for repeat in range(1, 11):
            cell = {
                "ordinal": len(rows) + 1,
                "case_id": case["id"],
                "repeat": repeat,
                "trial_name": f"regrade-{case['id']}-run-{repeat:02d}",
            }
            plan["cells"].append(cell)
            extraction = case["id"] == "malformed"
            if mutation == "graded_instead_of_extraction" and extraction:
                extraction = False
            if mutation == "extraction_instead_of_graded" and not extraction:
                extraction = True
            status, reward = ("extraction_error", None) if extraction else ("graded", 1)
            fingerprint = {
                "schema_version": "capability-grading-input-fingerprint-v1",
                **{
                    key: "d" * 64
                    for key in (
                        "submission_sha256",
                        "grading_input_sha256",
                        "specification_sha256",
                        "protocol_sha256",
                        "transcript_sha256",
                        "payload_sha256",
                    )
                },
                "workspace_file_count": 0,
                "step_index": 0,
            }
            if mutation == "missing_fingerprint" and extraction:
                fingerprint = None
            actual_receipt = dict(receipt)
            if mutation == "wrong_source" and len(rows) == 0:
                actual_receipt["semantic_verifier_sha256"] = "c" * 64
            grade = {
                "status": status,
                "reward": reward,
                "detail": {
                    "grading_input_fingerprint": fingerprint,
                    "native_verifier_receipt": actual_receipt,
                },
            }
            trial = {
                "exception_info": (
                    {
                        "exception_type": "ExtractionError",
                        "exception_message": "fixture",
                    }
                    if extraction
                    else None
                ),
                "verifier_result": None
                if extraction
                else {"rewards": {"reward": reward}},
            }
            root = output / "runtime-trials" / cell["trial_name"]
            row = {
                **cell,
                "status": status,
                "reward": reward,
                "outcome_class": status,
                "exception": {"type": "ExtractionError", "message": "fixture"}
                if extraction
                else None,
                "verifier_result_present": not extraction,
                "grading_input_fingerprint": fingerprint,
                "native_verifier_receipt": actual_receipt,
                "trial_sha256": write(root / "result.json", trial),
                "grading_sha256": write(
                    root / "verifier/taskcompendium-result.json", grade
                ),
            }
            rows.append(row)
    return plan, {"cells": rows}, output


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (None, "ready"),
        ("wrong_source", "pending"),
        ("missing_fingerprint", "pending"),
        ("graded_instead_of_extraction", "semantic_failed"),
        ("extraction_instead_of_graded", "semantic_failed"),
    ],
)
def test_native_full_receipts_distinguish_contract_failure_and_missing_evidence(
    tmp_path,
    monkeypatch,
    mutation,
    expected,
):
    def unexpected(*args, **kwargs):
        raise AssertionError("native no-tool grade must not claim Daytona isolation")

    monkeypatch.setattr(
        "capability_pipeline.grading_diagnostics.provider_isolation_record", unexpected
    )
    monkeypatch.setattr(
        "capability_pipeline.grading_diagnostics.verifier_isolation_record", unexpected
    )
    plan, report, output = matrix(tmp_path, mutation=mutation)
    state, issues, summary = _classify(plan, report, output)
    assert state == expected, issues
    assert summary["recorded_cells"] == summary["denominator"] == 20
