import json
from types import SimpleNamespace

import pytest

from capability_pipeline.quality import sha256
from capability_pipeline.repair import run_repair


def fixture(tmp_path, *, tamper=False, missing_check=False, calibration_change=None):
    item, out = tmp_path / "item", tmp_path / "repair-1"
    (item / "contract").mkdir(parents=True)
    (item / "contract/accepted.json").write_text('{"proposal_hash":"fixture"}')
    (item / "workspace/task").mkdir(parents=True)
    (item / "workspace/task/specification.json").write_text('{"fixture":"before"}')
    if calibration_change is not None:
        (item / "workspace/task/judge-calibration.json").write_text(json.dumps({
            "specification_sha256": "old", "cases": [{"id": "measured", "candidate": "answer", "expected_reward_range": [0, 0.2]}],
        }))

    def invoke(workspace, session_dir, prompt, attempt):
        manifest = json.loads((out / "input-manifest.json").read_text())
        (workspace / "task/specification.json").write_text('{"fixture":"after"}')
        if calibration_change is not None:
            calibration = json.loads((workspace / "task/judge-calibration.json").read_text())
            calibration["specification_sha256"] = "new"
            if calibration_change == "change":
                calibration["cases"][0]["expected_reward_range"] = [0, 1]
            elif calibration_change == "add":
                calibration["cases"].append({"id": "new", "candidate": "another", "expected_reward_range": [0, 0.2]})
            (workspace / "task/judge-calibration.json").write_text(json.dumps(calibration))
        log = workspace / "check.txt"
        log.write_text("fake unit-test check, not runtime evidence")
        if tamper:
            (item / "contract/accepted.json").write_text("changed contract")
        (out / "receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": "capability-repair-receipt-v1",
                    "snapshot_hash": manifest["snapshot_hash"],
                    "status": "ready_for_validation",
                    "changes": ["fixture edit"],
                    "remaining_issues": [],
                    "checks": []
                    if missing_check
                    else [
                        {
                            "path": "check.txt",
                            "sha256": sha256(log),
                            "claim": "fixture check",
                        }
                    ],
                }
            )
        )
        return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}

    return item, out, SimpleNamespace(invoke=invoke)


def test_repair_preserves_before_state_and_requires_fresh_validation(tmp_path):
    item, out, agent = fixture(tmp_path)
    result = run_repair(item, out, agent, {"issues": ["fixture defect"]})
    assert result["state"] == "ready_for_validation"
    assert result["runtime_certified"] is False
    assert result["quality_certified"] is False
    assert "workspace/task/specification.json" in result["changed_files"]
    assert json.loads(
        (out / "before/workspace/task/specification.json").read_text()
    ) == {"fixture": "before"}
    with pytest.raises(ValueError, match="fresh evidence"):
        run_repair(item, out, agent, {"issues": ["again"]})


def test_repair_continues_same_omp_session_until_receipt(tmp_path):
    item, out, agent = fixture(tmp_path)
    original = agent.invoke
    calls = []

    def invoke(workspace, session_dir, prompt, attempt):
        calls.append((session_dir, attempt, prompt))
        result = original(workspace, session_dir, prompt, attempt)
        if attempt == 0:
            (out / "receipt.json").unlink()
        return result

    agent.invoke = invoke
    agent.max_continuations = 2
    result = run_repair(item, out, agent, {"issues": ["fixture defect"]})
    assert result["state"] == "ready_for_validation"
    assert [attempt for _, attempt, _ in calls] == [0, 1]
    assert calls[0][0] == calls[1][0] == out / "transcript"
    assert calls[1][2] == out / "continuation-1.md"
    assert len(result["attempts"]) == 2
    assert (out / "attempt-0.log").is_file()
    assert (out / "attempt-1.log").is_file()


def test_repair_missing_receipt_exhausts_continuations_without_acceptance(tmp_path):
    item, out, agent = fixture(tmp_path)
    original = agent.invoke
    calls = []

    def invoke(workspace, session_dir, prompt, attempt):
        calls.append(attempt)
        result = original(workspace, session_dir, prompt, attempt)
        (out / "receipt.json").unlink()
        return result

    agent.invoke = invoke
    agent.max_continuations = 1
    result = run_repair(item, out, agent, {"issues": ["fixture defect"]})
    assert calls == [0, 1]
    assert result["state"] == "pending"
    assert result["runtime_certified"] is False
    assert result["quality_certified"] is False
    assert result["issues"] == [
        "repair continuation budget exhausted without the required receipt"
    ]


def test_repair_does_not_continue_a_present_invalid_receipt(tmp_path):
    item, out, agent = fixture(tmp_path)
    original = agent.invoke
    calls = []

    def invoke(workspace, session_dir, prompt, attempt):
        calls.append(attempt)
        result = original(workspace, session_dir, prompt, attempt)
        (out / "receipt.json").write_text('{"status":"ready_for_validation"}')
        return result

    agent.invoke = invoke
    agent.max_continuations = 2
    result = run_repair(item, out, agent, {"issues": ["fixture defect"]})
    assert calls == [0]
    assert result["state"] == "pending"
    assert any("identity or status" in issue for issue in result["issues"])


def test_repair_cannot_rewrite_repeated_diagnostic_history(tmp_path):
    item, out, agent = fixture(tmp_path)
    history = item / "diagnostics/attempt-1/evaluation/matrix.json"
    history.parent.mkdir(parents=True)
    history.write_text('{"fixture":"failed diagnostic"}')
    invoke = agent.invoke

    def tamper(*args):
        outcome = invoke(*args)
        history.write_text('{"fixture":"rewritten as passed"}')
        return outcome

    agent.invoke = tamper
    result = run_repair(item, out, agent, {"issues": ["fixture defect"]})
    assert result["state"] == "pending"
    assert json.loads((out / "before" / history.relative_to(item)).read_text()) == {
        "fixture": "failed diagnostic"
    }
    assert any("protected" in issue for issue in result["issues"])


@pytest.mark.parametrize("tamper,missing_check", [(True, False), (False, True)])
def test_repair_cannot_change_admission_or_claim_unmeasured_readiness(
    tmp_path, tamper, missing_check
):
    item, out, agent = fixture(tmp_path, tamper=tamper, missing_check=missing_check)
    result = run_repair(item, out, agent, {"issues": ["fixture defect"]})
    assert result["state"] == "pending"
    assert result["issues"]


@pytest.mark.parametrize("change,expected", [("change", "needs_readmission"), ("add", "ready_for_validation")])
def test_measured_calibration_cases_are_immutable_after_failure(tmp_path, change, expected):
    item, out, agent = fixture(tmp_path, calibration_change=change)
    result = run_repair(item, out, agent, {"judge_calibration_failure": {"artifact": "measured"}})
    assert result["state"] == expected
