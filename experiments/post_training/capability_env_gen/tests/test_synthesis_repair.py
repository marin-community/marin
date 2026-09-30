import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import repair, synthesis
from capability_pipeline.attack_adjudication import SCHEMA, prepare_packet
from capability_pipeline.inference import digest
from capability_pipeline.judge import _assess_calibration
from capability_pipeline.quality import sha256


def _attack_status(tmp_path, disposition="exploit", *, tamper=False, graded=True):
    root = item_root(tmp_path)
    (root / "contract").mkdir(parents=True)
    (root / "workspace/task").mkdir(parents=True)
    (root / "runtime-trials").mkdir()
    (root / "contract/accepted.json").write_text("{}")
    (root / "workspace/task/specification.json").write_text("{}")
    grade = {"status": "graded" if graded else "error", "reward": 0.75}
    cases = []
    for strategy in ("injection", "shortcut", "boundary"):
        result = grade if strategy == "boundary" else {"status": "graded", "reward": 0}
        grade_path = root / f"runtime-trials/{strategy}-grade.json"
        transcript_path = root / f"runtime-trials/{strategy}-transcript.json"
        grade_path.write_text(json.dumps(result))
        transcript_path.write_text("[]")
        cases.append({"strategy": strategy, "steps": [{
            "step_index": 0, "result": result,
            "grading_artifact": str(grade_path.relative_to(root)),
            "grading_sha256": sha256(grade_path),
            "transcript_artifact": str(transcript_path.relative_to(root)),
            "transcript_sha256": sha256(transcript_path),
        }]})
    report = {"independent": True, "max_false_reward": 0.2, "cases": cases}
    (root / "independent-adversary.json").write_text(json.dumps(report))
    review = tmp_path / "attack-adjudication" / root.name / "attempt-1"
    if graded:
        manifest = prepare_packet(root, review)
        receipt = {
            "schema_version": SCHEMA, "snapshot_hash": manifest["snapshot_hash"],
            "cases": [{
                "strategy": "boundary", "step_index": 0, "disposition": disposition,
                "rubric_basis": "fixture rubric", "candidate_analysis": "fixture candidate",
                "critical_gate_analysis": "fixture gate",
                "citations": [
                    {"path": name, "sha256": value, "supports": "fixture evidence"}
                    for name, value in manifest["files"].items()
                ],
            }],
        }
        (review / "receipt.json").write_text(json.dumps(receipt))
        from capability_pipeline.attack_adjudication import validate_receipt
        issues = validate_receipt(receipt, manifest, review / "input")
        adjudication = {
            "schema_version": SCHEMA + "-result",
            "snapshot_hash": manifest["snapshot_hash"],
            "state": "needs_repair_or_retry",
            "issues": issues,
            "receipt_sha256": sha256(review / "receipt.json"),
            "execution": {"returncode": 0, "timed_out": False},
            "reviewer_policy": {"independent_session": True, "model": "glm-5.3"},
        }
        (review / "result.json").write_text(json.dumps(adjudication))
    else:
        review.mkdir(parents=True)
        (review / "result.json").write_text("{}")
        manifest = {"snapshot_hash": "x"}
    status = {
        "state": "pending_attack_adjudication",
        "item_root": str(root),
        "issues": ["boundary:0: rewarded attack requires independent adjudication"],
        "attack_adjudication": {
            "state": "needs_repair_or_retry",
            "artifact": str(review / "result.json"),
            "artifact_sha256": sha256(review / "result.json"),
            "snapshot_hash": manifest["snapshot_hash"],
        },
    }
    if tamper:
        (root / "workspace/task/specification.json").write_text('{"changed":true}')
    (root / "status.json").write_text(json.dumps(status))
    return status


def test_proven_exploit_repairs_once_then_fresh_validation(tmp_path, monkeypatch):
    status = _attack_status(tmp_path)
    assert synthesis._fresh_construction_repair_allowed(status)
    calls = []
    def repair_once(_root, out, _agent, feedback, *, source=None):
        calls.append(("repair", feedback["attack_adjudication_receipt"]["artifact_sha256"]))
        out.mkdir(parents=True)
        (out / "result.json").write_text('{"state":"ready_for_validation"}')
        return {"state": "ready_for_validation", "changed_files": []}
    def fresh(*_args):
        calls.append(("fresh", None))
        return {"state": "quality_accepted", "item_root": str(item_root(tmp_path))}
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(repair, "run_repair", repair_once)
    monkeypatch.setattr(synthesis, "_synthesize_attempt", fresh)
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)
    assert [name for name, _ in calls] == ["repair", "fresh"]
    assert result["repair_budget"] == {"used": 1, "max": 1, "exhausted": True}
    assert "terminal_disposition" not in result


def test_unproven_attack_never_edits_task(tmp_path, monkeypatch):
    for disposition, tamper, graded in [
        ("uncertain", False, True), ("exploit", True, True), ("exploit", False, False)
    ]:
        case_root = tmp_path / f"{disposition}-{tamper}-{graded}"
        status = _attack_status(case_root, disposition, tamper=tamper, graded=graded)
        assert not synthesis._fresh_construction_repair_allowed(status)
        monkeypatch.setattr(repair, "run_repair", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("repair ran")))
        result = synthesis.synthesize_one(item(), case_root, object(), None, None, 60)
        assert result["state"] == "pending_attack_adjudication"
        assert "terminal_disposition" not in result


def test_actionable_failure_with_exhausted_budget_is_terminal(tmp_path, monkeypatch):
    root = item_root(tmp_path)
    root.mkdir(parents=True)
    prior = {
        "state": "failed",
        "issues": ["invalid task bundle: bad verifier"],
        "item_root": str(root),
    }
    (root / "status.json").write_text(json.dumps(prior))
    (tmp_path / "repair-budget" / root.name / "attempt-2").mkdir(parents=True)
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "2")
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)
    assert result["state"] == "failed"
    assert result["repair_budget"] == {"used": 2, "max": 2, "exhausted": True}
    assert result["terminal_disposition"] == "rejected"


def test_graded_extraction_expectation_mismatch_enters_glm_repair(tmp_path, monkeypatch):
    root = item_root(tmp_path)
    root.mkdir(parents=True)
    prior = {
        "state": "failed",
        "issues": ["M1: status 'graded', expected 'extraction_error'"],
        "item_root": str(root),
    }
    (root / "status.json").write_text(json.dumps(prior))
    calls = []

    def repair_once(_root, out, _agent, feedback, *, source=None):
        calls.append(("repair", feedback["issues"]))
        out.mkdir(parents=True)
        (out / "result.json").write_text('{"state":"ready_for_validation"}')
        return {"state": "ready_for_validation", "changed_files": []}

    def fresh(*_args):
        calls.append(("fresh", None))
        return {"state": "quality_accepted", "item_root": str(root)}

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(repair, "run_repair", repair_once)
    monkeypatch.setattr(synthesis, "_synthesize_attempt", fresh)
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)
    assert [name for name, _ in calls] == ["repair", "fresh"]
    assert result["state"] == "quality_accepted"
    assert not synthesis._fresh_construction_repair_allowed({
        "state": "failed",
        "issues": ["M1: status 'infra_error', expected 'extraction_error'"],
    })


def _calibration_status(tmp_path, *, ungraded=False, wrong_path=False, tamper=False):
    root = item_root(tmp_path)
    bundle = root / "workspace/task"
    bundle.mkdir(parents=True)
    (bundle / "specification.json").write_text("{}")
    spec_hash = sha256(bundle / "specification.json")
    cases = []
    for positive in (True, False):
        for index in range(40):
            label = f"{'positive' if positive else 'negative'}-{index}"
            cases.append({
                "id": label,
                "kind": "oracle" if positive else
                    ("plausible_wrong" if index == 0 else "empty" if index == 1 else "prompt_injection"),
                "source_family": "fixture",
                "variant_group": label,
                "design_label": label,
                "candidate": label,
                "expected_judge_path": "model",
                "expected_reward_range": [0.8, 1.0] if positive else [0.0, 0.2],
            })
    fixture = {
        "schema_version": "taskcompendium-judge-calibration-v1",
        "specification_sha256": spec_hash,
        "cases": cases,
    }
    fixture_path = bundle / "judge-calibration.json"
    fixture_path.write_text(json.dumps(fixture))
    results = {}
    for case in cases:
        for repeat in range(3):
            key = f"{case['id']}:{repeat}"
            reward = 1.0 if case["kind"] == "oracle" else 0.0
            if key == "negative-0:0":
                reward = 1.0
            results[key] = {
                "status": "graded",
                "reward": reward,
                "detail": {"judgments": [{"criterion": 0, "score": reward}]},
            }
    if ungraded:
        results["negative-0:0"]["status"] = "error"
    if wrong_path:
        results["negative-0:0"]["detail"] = {"gate": "exact"}
    issues, metrics = _assess_calibration(
        cases, results, {}, 3, 0.15, require_model_judgment=True,
    )
    calibration_root = root / "judge-calibration"
    calibration_root.mkdir()
    raw_path = calibration_root / "taskcompendium-judge-results.json"
    raw_path.write_text(json.dumps({"results": results, "failures": {}}))
    report_path = calibration_root / "judge-calibration.json"
    report_path.write_text(json.dumps({
        "state": "failed", "mode": "taskcompendium-native-judge",
        "issues": issues, "results": results, "failures": {},
        "fixture_hash": digest(fixture), "specification_sha256": spec_hash,
        "composite_config_sha256": None, "repeats": 3, "max_spread": 0.15,
        "metrics": metrics,
    }))
    status = {
        "state": "pending_judge_calibration",
        "key": "c17.browser:3",
        "proposal_hash": "a" * 64,
        "issues": ["native judge calibration is incomplete: native judge calibration did not pass"],
        "item_root": str(root),
        "judge_calibration_failure": {
            "artifact": str(report_path), "artifact_sha256": sha256(report_path),
            "raw_artifact": str(raw_path), "raw_artifact_sha256": sha256(raw_path),
            "fixture_artifact": str(fixture_path),
            "fixture_artifact_sha256": sha256(fixture_path),
        },
    }
    if tamper:
        raw_path.write_text('{"results":{},"failures":{}}')
    (root / "status.json").write_text(json.dumps(status))
    return status


def test_complete_judge_misgrading_enters_repair_and_feedback(tmp_path, monkeypatch):
    status = _calibration_status(tmp_path)
    assert synthesis._fresh_construction_repair_allowed(status)
    observed = []
    def repair_once(_root, out, _agent, feedback, *, source=None):
        observed.append(feedback["judge_calibration_failure"])
        out.mkdir(parents=True)
        (out / "result.json").write_text('{"state":"needs_continuation"}')
        return {"state": "needs_continuation", "changed_files": []}
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(repair, "run_repair", repair_once)
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)
    assert len(observed) == 1
    assert observed[0]["issues"] and observed[0]["metrics"]["balanced_accuracy"] < 1
    assert result["terminal_disposition"] == "rejected"


def test_calibration_repair_requires_current_item_identity(tmp_path):
    status = _calibration_status(tmp_path)
    expected = {
        "expected_item_root": item_root(tmp_path),
        "expected_key": "c17.browser:3",
        "expected_proposal_hash": "a" * 64,
    }
    assert synthesis._fresh_construction_repair_allowed(status, **expected)
    for field, wrong in (
        ("item_root", str(tmp_path / "other-item")),
        ("key", "c17.browser:4"),
        ("proposal_hash", "b" * 64),
    ):
        changed = dict(status)
        changed[field] = wrong
        assert not synthesis._fresh_construction_repair_allowed(changed, **expected)


def test_unsupported_composite_lowering_is_an_infrastructure_hold(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    error = ValueError(synthesis.UNSUPPORTED_COMPOSITE_FINAL_STATE)
    assert synthesis._lowering_failure_state(bundle, error) == "failed"
    (bundle / "composite-verifier.json").write_text("{}")
    assert synthesis._lowering_failure_state(bundle, error) == "pending_schema_validation"
    assert not synthesis._fresh_construction_repair_allowed({
        "state": "pending_schema_validation",
        "issues": [str(error)],
    })


@pytest.mark.parametrize("field,wrong", [("repeats", 4), ("max_spread", 0.2)])
def test_calibration_repair_requires_current_controller_policy(tmp_path, field, wrong):
    status = _calibration_status(tmp_path)
    report_path = Path(status["judge_calibration_failure"]["artifact"])
    report = json.loads(report_path.read_text())
    report[field] = wrong
    report_path.write_text(json.dumps(report))
    status["judge_calibration_failure"]["artifact_sha256"] = sha256(report_path)
    assert not synthesis._fresh_construction_repair_allowed(
        status,
        expected_item_root=item_root(tmp_path),
        expected_key="c17.browser:3",
        expected_proposal_hash="a" * 64,
    )


def test_incomplete_or_tampered_calibration_never_repairs(tmp_path, monkeypatch):
    for name, kwargs in {
        "ungraded": {"ungraded": True},
        "wrong-path": {"wrong_path": True},
        "tampered": {"tamper": True},
    }.items():
        root = tmp_path / name
        status = _calibration_status(root, **kwargs)
        assert not synthesis._fresh_construction_repair_allowed(status)
        monkeypatch.setattr(repair, "run_repair", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("repair ran")))
        result = synthesis.synthesize_one(item(), root, object(), None, None, 60)
        assert result["state"] == "pending_judge_calibration"
        assert "terminal_disposition" not in result


def item():
    return {
        "proposal_hash": "a" * 64,
        "proposal": {"capability_id": "c17.browser", "slot": 3},
    }


def item_root(root: Path) -> Path:
    return root / "items" / f"c17.browser-3-{'a' * 12}"


def test_new_failure_enters_same_worker_repair_with_retained_status(tmp_path, monkeypatch):
    expected = {
        "state": "pending_build_acceptance",
        "issues": ["fix all bundle checks together"],
        "item_root": str(item_root(tmp_path)),
    }
    def first_attempt(*_args):
        root = item_root(tmp_path)
        root.mkdir(parents=True)
        (root / "status.json").write_text(json.dumps(expected))
        return expected

    calls = []
    def repair_once(_item_path, repair_root, _agent, feedback, *, source=None):
        calls.append(feedback["failed_state"])
        repair_root.mkdir(parents=True)
        (repair_root / "result.json").write_text('{"state":"needs_continuation"}')
        return {"state": "needs_continuation", "changed_files": []}

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(synthesis, "_synthesize_attempt", first_attempt)
    monkeypatch.setattr(synthesis, "_repair_feedback", lambda *_: {"failed_state": "pending_build_acceptance"})
    monkeypatch.setattr(repair, "run_repair", repair_once)
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)
    assert result["state"] == "pending_build_acceptance"
    assert calls == ["pending_build_acceptance"]
    source_status = json.loads((tmp_path / "repair-budget" / item_root(tmp_path).name / "attempt-1/source-status.json").read_text())
    assert source_status["issues"] == ["fix all bundle checks together"]
    assert "repairs" not in source_status


def test_resume_repairs_before_rerunning_all_gates_and_archives_evidence(
    tmp_path, monkeypatch
):
    root = item_root(tmp_path)
    (root / "workspace").mkdir(parents=True)
    (root / "contract").mkdir()
    (root / "contract/accepted.json").write_text("{}")
    (root / "workspace/task.txt").write_text("before")
    (root / "harbor").mkdir()
    (root / "harbor/old.txt").write_text("old package")
    prior = {
        "state": "pending_build_acceptance",
        "issues": ["invalid control plus browser boundary"],
        "item_root": str(root),
    }
    (root / "status.json").write_text(json.dumps(prior))
    calls = []

    def fake_repair(item_path, repair_root, agent, feedback, *, source=None):
        calls.append(("repair", feedback["failed_state"]))
        repair_root.mkdir(parents=True)
        (repair_root / "result.json").write_text('{"state":"ready_for_validation"}')
        (root / "workspace/task.txt").write_text("repaired")
        return {
            "state": "ready_for_validation",
            "changed_files": ["workspace/task.txt"],
        }

    def fake_attempt(*args):
        calls.append(("validate", None))
        return {
            "state": "quality_accepted",
            "issues": [],
            "item_root": str(root),
        }

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(repair, "run_repair", fake_repair)
    monkeypatch.setattr(synthesis, "_synthesize_attempt", fake_attempt)
    result = synthesis.synthesize_one(
        item(), tmp_path, object(), SimpleNamespace(package_root=tmp_path), None, 60
    )

    assert calls == [("repair", "pending_build_acceptance"), ("validate", None)]
    assert result["state"] == "quality_accepted"
    assert result["repairs"][0]["state"] == "ready_for_validation"
    history = Path(result["repairs"][0]["prior_attempt"])
    assert (history / "harbor/old.txt").read_text() == "old package"
    assert not (root / "harbor").exists()


def test_resume_keeps_prior_repair_record_and_uses_next_global_round(tmp_path, monkeypatch):
    root = item_root(tmp_path)
    root.mkdir(parents=True)
    prior_record = {"round": 1, "state": "needs_continuation", "artifact": "historical"}
    prior = {
        "state": "failed",
        "issues": ["invalid task bundle: bad grader"],
        "item_root": str(root),
        "repairs": [prior_record],
    }
    (root / "status.json").write_text(json.dumps(prior))
    (tmp_path / "repair-budget" / root.name / "attempt-1").mkdir(parents=True)
    calls = []

    def repaired(_item_path, repair_root, _agent, _feedback, *, source=None):
        calls.append(repair_root.name)
        repair_root.mkdir(parents=True)
        (repair_root / "result.json").write_text('{"state":"ready_for_validation"}')
        return {"state": "ready_for_validation", "changed_files": ["workspace/task/grader.py"]}

    def validated(*_args):
        result = {"state": "quality_accepted", "issues": [], "item_root": str(root)}
        (root / "status.json").write_text(json.dumps(result))
        return result

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "2")
    monkeypatch.setattr(synthesis, "_repair_feedback", lambda *_: {"issues": ["bad grader"]})
    monkeypatch.setattr(synthesis, "_synthesize_attempt", validated)
    monkeypatch.setattr(repair, "run_repair", repaired)
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)
    assert calls == ["attempt-2"]
    assert [record["round"] for record in result["repairs"]] == [1, 2]
    assert result["repairs"][0] == prior_record


def test_repair_activity_is_visible_while_repair_runs_and_closes_on_success(
    tmp_path, monkeypatch
):
    root = item_root(tmp_path)
    (root / "workspace").mkdir(parents=True)
    (root / "contract").mkdir()
    (root / "contract/accepted.json").write_text("{}")
    prior = {
        "state": "failed",
        "issues": ["invalid task bundle: measured grader failure"],
        "item_root": str(root),
    }
    status = root / "status.json"
    status.write_text(json.dumps(prior))
    entered = threading.Event()
    release = threading.Event()

    def blocked_repair(item_path, repair_root, agent, feedback, *, source=None):
        entered.set()
        assert release.wait(timeout=5)
        repair_root.mkdir(parents=True)
        (repair_root / "result.json").write_text('{"state":"needs_continuation"}')
        return {"state": "needs_continuation", "changed_files": []}

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(repair, "run_repair", blocked_repair)
    outcome = {}

    def run():
        outcome["result"] = synthesis.synthesize_one(
            item(), tmp_path, object(), None, None, 60
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(timeout=5)

    activity_path = root / "controller/active-operation.json"
    activity = json.loads(activity_path.read_text())
    assert activity["state"] == "active"
    assert activity["attempt"] == 1
    assert activity["source_prior_state"] == "failed"
    assert activity["completed_at"] is None
    assert activity["outcome"] is None
    assert activity["transcript_directory"].endswith("attempt-1/transcript")
    assert json.loads(status.read_text()) == prior

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome["result"]["state"] == "failed"
    closed = json.loads(activity_path.read_text())
    assert closed["state"] == "completed"
    assert closed["started_at"] == activity["started_at"]
    assert closed["completed_at"] is not None
    assert closed["outcome"]["repair_state"] == "needs_continuation"
    assert closed["outcome"]["result_artifact_sha256"]
    history = root / "controller/operations/repair-attempt-1.json"
    assert json.loads(history.read_text()) == closed


def test_repair_activity_closes_on_error_before_existing_error_handling(
    tmp_path, monkeypatch
):
    root = item_root(tmp_path)
    (root / "workspace").mkdir(parents=True)
    (root / "contract").mkdir()
    (root / "contract/accepted.json").write_text("{}")
    prior = {
        "state": "pending_quality_review",
        "issues": ["review requested repair"],
        "quality_review": {"state": "repair"},
        "item_root": str(root),
    }
    (root / "status.json").write_text(json.dumps(prior))

    def failed_repair(*args, **kwargs):
        raise RuntimeError("repair transport stopped")

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr(repair, "run_repair", failed_repair)
    result = synthesis.synthesize_one(item(), tmp_path, object(), None, None, 60)

    assert result["state"] == "pending_quality_review"
    assert result["repair_issue"].endswith("repair transport stopped")
    activity = json.loads((root / "controller/active-operation.json").read_text())
    assert activity["state"] == "error"
    assert activity["source_prior_state"] == "pending_quality_review"
    assert activity["completed_at"] is not None
    assert activity["outcome"] == {
        "exception_type": "RuntimeError",
        "message": "repair transport stopped",
    }
