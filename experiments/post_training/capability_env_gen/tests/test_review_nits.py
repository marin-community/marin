"""Integration-review nits (2026-09-29): acceptance resume robustness, capture ledger
numbering, and the pre-review gate receipt."""

import json
import time
from pathlib import Path

from test_acceptance_resume import (
    NAME,
    _accept,
    _demote,
    _forbid_rebuild,
    _item,
    _relaunch,
    _results,
    _run,
)
from test_synthesis import FakeAgent, FakeToolchain, accepted

from capability_pipeline import acceptance, synthesis
from capability_pipeline.inference import atomic_json
from tests.test_capture_failure_handling import _controller
from tests.test_capture_failure_handling import _run as _capture_run


def _demoted_relaunch(tmp_path, *, issues=("KC2: reward is below reward_min",)):
    old = _results(tmp_path, "cap-construct-003-shard-071-c6")
    _accept(old)
    _demote(old, list(issues))
    new = _results(tmp_path, "cap-construct-003-shard-071-k3")
    _relaunch(old, new)
    return new


# -- acceptance: nothing escapes resume_accepted ----------------------------------------------


def test_an_exception_while_verifying_a_claim_does_not_escape(tmp_path, monkeypatch):
    root = _results(tmp_path, "run-a")
    _accept(root)

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected evidence shape")

    monkeypatch.setattr(acceptance, "verify_acceptance", boom)
    item_root = root / "items" / NAME
    status = json.loads((item_root / "status.json").read_text())
    assert acceptance.resume_accepted(_item(), status["key"], root, item_root, status) is None


def test_an_export_failure_falls_through_instead_of_raising(tmp_path, monkeypatch):
    root = _results(tmp_path, "run-a")
    _accept(root)

    def full_disk(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(acceptance, "_export", full_disk)
    item_root = root / "items" / NAME
    status = json.loads((item_root / "status.json").read_text())
    assert acceptance.resume_accepted(_item(), status["key"], root, item_root, status) is None


def test_a_gate_proof_failure_skips_only_that_review(tmp_path, monkeypatch):
    new = _demoted_relaunch(tmp_path)
    _forbid_rebuild(monkeypatch)

    def boom(attempt):
        raise KeyError("diagnostics")

    monkeypatch.setattr(acceptance, "post_review_gate_proof", boom)
    result = _run(new)  # the construction path, not a traceback
    assert result["state"] == "failed"


def test_review_record_claims_are_lazy_when_the_status_claim_verifies(tmp_path, monkeypatch):
    root = _results(tmp_path, "run-a")
    _accept(root)
    monkeypatch.setattr(acceptance, "post_review_gate_proof",
                        lambda attempt: (_ for _ in ()).throw(AssertionError("evaluated eagerly")))
    _forbid_rebuild(monkeypatch)
    result = _run(root)
    assert result["state"] == "quality_accepted" and result["acceptance_resumed"]["source"] == "status"


def test_gate_proofs_are_memoised_per_review_attempt(tmp_path, monkeypatch):
    new = _demoted_relaunch(tmp_path)
    # The task changed after the review: the claim is rejected every time, so each resume
    # evaluates the review-record claims again.
    (new / "items" / NAME / "harbor/task.toml").write_text("name = 'changed'\n")
    calls = []
    real = acceptance.post_review_gate_proof
    monkeypatch.setattr(acceptance, "post_review_gate_proof", lambda attempt: calls.append(attempt) or real(attempt))
    _forbid_rebuild(monkeypatch)
    assert _run(new)["state"] == "failed"
    assert _run(new)["state"] == "failed"
    assert len(calls) == 1
    # A changed receipt is a different key: evaluated afresh.
    time.sleep(0.01)
    atomic_json(new / "quality" / NAME / "attempt-1" / acceptance.GATE_RECEIPT,
                {"repeated_diagnostics_state": "pending"})
    assert _run(new)["state"] == "failed"
    assert len(calls) == 2


def test_recovered_acceptance_is_stamped_and_drops_stale_conveyor_fields(tmp_path, monkeypatch):
    new = _demoted_relaunch(tmp_path)
    status_path = new / "items" / NAME / "status.json"
    demoted = json.loads(status_path.read_text())
    before = time.time()
    atomic_json(status_path, {
        **demoted, "state": "failed", "state_since": before - 3600, "updated_at": before - 3600,
        "transitions": [{"state": "failed", "at": before - 3600, "reason": "wait_budget_exhausted"}],
        "wait_exhausted": {"kind": "image_publication", "cause": "deadline"},
        "unclassified_state": "pending_x", "failure_stage": "image_publication",
    })
    _forbid_rebuild(monkeypatch)
    result = _run(new)
    assert result["state"] == "quality_accepted"
    for field in ("wait_exhausted", "unclassified_state", "failure_stage"):
        assert field not in result
    assert [row["state"] for row in result["transitions"]] == ["failed", "quality_accepted"]
    assert result["state_since"] >= before
    assert json.loads(status_path.read_text()) == result


# -- capture control: attempt numbering survives an unreadable ledger record -------------------


def test_next_capture_attempt_is_numbered_from_the_record_file_not_its_contents(tmp_path, monkeypatch):
    item, attempt, _, review_base = _controller(tmp_path, monkeypatch)
    (attempt / "capture-candidate.attempt-1.json").write_text("{torn write")  # consumed attempt 1

    def failing(command):
        from types import SimpleNamespace

        return SimpleNamespace(returncode=1, stdout="", stderr="sandbox create failed")

    result = _capture_run(item, review_base, failing)  # KeyError 'attempt' before the fix
    assert result["state"] == "pending_capture" and result["retryable"] is True
    assert result["attempts"] == 2
    assert (attempt / "capture-candidate.attempt-2.json").is_file()
    assert result["step_timeout_seconds"] > 0


# -- gate receipt: written before the review, so a failed post-review write is not "legacy" ----


def test_gate_receipt_exists_before_the_review_and_survives_a_failed_post_review_write(tmp_path, monkeypatch):
    from capability_pipeline.acceptance import (
        GATE_PROOF_RECEIPT,
        GATE_RECEIPT,
        post_review_gate_proof,
    )

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "0")
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n")
    runner.chmod(0o755)
    cases = [
        ("positive", "known_correct", "independent_solver", {"status": "graded", "reward": 1.0}),
        ("malformed", "empty_or_malformed", "authored_adversarial_control", {"status": "extraction_error", "reward": None}),
        ("plausible-wrong", "plausible_wrong", "authored_adversarial_control", {"status": "graded", "reward": 0.0}),
        ("shortcut", "task_specific_shortcut", "authored_adversarial_control", {"status": "graded", "reward": 0.0}),
    ]
    monkeypatch.setattr(
        "capability_pipeline.synthesis._external_controls",
        lambda *args, **kwargs: {"cases": [
            {"id": case, "source_author": "builder", "category": category, "control_type": kind, "result": graded}
            for case, category, kind, graded in cases]},
    )
    monkeypatch.setattr("capability_pipeline.synthesis._attestation_issues", lambda *args: [])
    monkeypatch.setattr("capability_pipeline.synthesis._repeated_quality_diagnostics",
                        lambda *args: {"state": "ready", "reviewable": True, "extra_files": {}})
    seen = {}

    def review(item_root, review_root, agent):
        seen["pre"] = json.loads((review_root / GATE_RECEIPT).read_text())
        review_root.mkdir(parents=True, exist_ok=True)
        result = {"schema_version": "capability-quality-result-v1", "snapshot_hash": "s", "state": "accept"}
        (review_root / "result.json").write_text(json.dumps(result))
        seen["root"] = review_root
        return result

    monkeypatch.setattr("capability_pipeline.quality.run_review", review)
    real_atomic = synthesis.atomic_json

    def failing_post_review_write(path, value):
        if Path(path).name == GATE_RECEIPT and value.get("phase") == "post_review":
            raise OSError(5, "Input/output error")
        return real_atomic(path, value)

    monkeypatch.setattr(synthesis, "atomic_json", failing_post_review_write)
    result = synthesis.synthesize_one(accepted(), tmp_path / "run", FakeAgent(), FakeToolchain(), runner, 30)
    assert seen["pre"]["phase"] == "pre_review" and seen["pre"]["repeated_diagnostics_state"] == "ready"
    assert result["state"] == "pending_quality_review"  # the post-review write failed
    # The accepting review is judged by the controller's own receipt, never the legacy reconstruction.
    assert post_review_gate_proof(seen["root"]) == (GATE_PROOF_RECEIPT, None)
