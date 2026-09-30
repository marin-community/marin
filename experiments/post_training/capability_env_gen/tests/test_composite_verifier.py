import json

import pytest

from capability_pipeline.composite_policy import composite_evidence_detail
from capability_pipeline.composite_probe import (
    capture_case_artifacts,
    validate_case_result,
)


def test_ungraded_judge_detail_retains_completed_private_machine_evidence():
    machine = [
        {
            "id": "gate",
            "status": "graded",
            "reward": 1.0,
            "detail": {
                "verifier_isolation": "daytona-network-block-all",
                "verifier_sandbox_id": "sandbox-a",
            },
        }
    ]
    detail = composite_evidence_detail(
        {"error": "Judge request failed: timed out"},
        machine,
        adapter_sha256="a" * 64,
        policy_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    assert detail["error"] == "Judge request failed: timed out"
    assert detail["machine_results"] == machine
    assert detail["composite_adapter_sha256"] == "a" * 64
    assert detail["composite_policy_sha256"] == "b" * 64
    assert detail["composite_config_sha256"] == "c" * 64


def test_probe_reports_judge_infrastructure_error_before_isolation_gate():
    result = {
        "status": "infra_error",
        "reward": None,
        "detail": {"error": "Judge request failed: timed out"},
    }
    with pytest.raises(
        RuntimeError,
        match=(
            "grading outcome infra_error with null reward: "
            "Judge request failed: timed out"
        ),
    ):
        validate_case_result(
            "disagreement-one-timeout", result, {"network_block_all": True}
        )


def test_probe_requires_private_isolation_only_after_a_graded_result():
    with pytest.raises(RuntimeError, match="lacks network-blocked verifier evidence"):
        validate_case_result(
            "graded-without-private-evidence",
            {"status": "graded", "reward": 1.0, "detail": {}},
            {"network_block_all": True},
        )

    machine = [
        {
            "id": "gate",
            "detail": {"verifier_isolation": "daytona-network-block-all"},
        }
    ]
    assert (
        validate_case_result(
            "graded",
            {
                "status": "graded",
                "reward": 1.0,
                "detail": {"machine_results": machine},
            },
            {"network_block_all": True},
        )
        == machine
    )


def test_probe_retains_missing_verifier_result_as_infrastructure_evidence(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps({"exception_info": {"type": "RuntimeError"}})
    )
    (tmp_path / "daytona-environment.json").write_text(
        json.dumps({"network_block_all": True})
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "agreement-high trial lacks verifier/taskcompendium-result.json; "
            "retained probe-case-evidence.json"
        ),
    ):
        capture_case_artifacts("agreement-high", tmp_path)

    receipt = json.loads((tmp_path / "probe-case-evidence.json").read_text())
    assert receipt["state"] == "infra_error"
    assert receipt["reason"] == "missing_taskcompendium_result"
    assert set(receipt["artifacts"]) == {"harbor_result", "candidate_provider"}
    assert receipt["artifacts"]["harbor_result"]["sha256"]


def test_probe_captures_complete_case_before_semantic_validation(tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    expected_result = {"status": "graded", "reward": 1.0, "detail": {}}
    expected_provider = {"network_block_all": True}
    (verifier / "taskcompendium-result.json").write_text(json.dumps(expected_result))
    (tmp_path / "daytona-environment.json").write_text(json.dumps(expected_provider))

    result, provider, receipt_path = capture_case_artifacts("complete", tmp_path)

    assert result == expected_result
    assert provider == expected_provider
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state"] == "captured"
    assert set(receipt["artifacts"]) == {
        "taskcompendium_result",
        "candidate_provider",
    }
