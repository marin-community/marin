import copy
import json
from types import SimpleNamespace

import pytest
from test_pipeline import _portfolio_review, _proposed_plan, _valid_proposal

from capability_pipeline import cli
from capability_pipeline.seed import load_seed


def seed_fixture(root):
    root.mkdir()
    capability = {"capability_id": "cap.one", "capability": {"id": "cap.one"}}
    pilot = {"source": {"sha256": "a" * 64}, "capabilities": [capability]}
    documents = {
        "input_pilot.json": pilot,
        "plans.json": {"cap.one": _proposed_plan("cap.one")},
        "proposals.json": [_valid_proposal("cap.one", slot) for slot in range(1, 11)],
        "report.json": {"stage": "proposal_review", "state": "needs_iteration"},
    }
    for name, value in documents.items():
        (root / name).write_text(json.dumps(value))
    return pilot


def test_seed_preserves_proposals_and_hashes_without_inheriting_acceptance(tmp_path):
    source = tmp_path / "source"
    pilot = seed_fixture(source)
    seed = load_seed(source, pilot, pilot["capabilities"])
    assert len(seed["proposals"]) == 10
    assert len(seed["provenance"]["source_files"]) == 4
    assert seed["provenance"]["acceptance_inherited"] is False
    changed = copy.deepcopy(pilot)
    changed["source"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="pilot differs"):
        load_seed(source, changed, changed["capabilities"])


def test_seed_rejects_running_report_and_duplicate_identity(tmp_path):
    source = tmp_path / "source"
    pilot = seed_fixture(source)
    path = source / "report.json"
    path.write_text('{"stage":"proposal_review","state":"running"}')
    with pytest.raises(ValueError, match="terminal"):
        load_seed(source, pilot, pilot["capabilities"])
    path.write_text('{"stage":"proposal_review","state":"needs_iteration"}')
    path = source / "proposals.json"
    values = json.loads(path.read_text())
    path.write_text(json.dumps(values + values[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        load_seed(source, pilot, pilot["capabilities"])


def test_refinement_runs_fresh_review_without_initial_generation(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    pilot = seed_fixture(source)
    historical = {
        "proposal": {
            "cap.one:5": {
                "error_type": "InvalidArtifact",
                "error": "old missing validation_plan",
            }
        }
    }
    (source / "report.json").write_text(
        json.dumps(
            {
                "stage": "proposal_review",
                "state": "needs_iteration",
                "failures": historical,
            }
        )
    )
    output.mkdir()
    stages = []

    class Store:
        def generate(self, stage, identity, system, prompt, validator, **kwargs):
            stages.append(stage)
            assert stage == "review"
            review = _portfolio_review(identity, range(1, 11), "repair", [])
            review["portfolio_issues"] = ["Plan diversity rationale needs revision."]
            validator(review)
            return review

    monkeypatch.setattr(cli, "GLMClient", lambda **kwargs: None)
    monkeypatch.setattr(cli, "StageStore", lambda *args: Store())
    args = SimpleNamespace(
        seed_run=str(source),
        tier="interactive",
        hold_seconds=1,
        concurrency=4,
        repair_rounds=0,
    )
    assert cli._propose(args, pilot, pilot["capabilities"], output) == 2
    assert stages == ["review"]
    assert len(json.loads((output / "accepted.json").read_text())) == 10
    assert json.loads((output / "proposals.json").read_text()) == json.loads(
        (source / "proposals.json").read_text()
    )
    report = json.loads((output / "report.json").read_text())
    assert report["partial_portfolio_admissions"] == 10
    assert report["missing_slots"] == []
    assert report["failures"]["proposal"] == {}
    assert report["inherited_failures"][0]["failures"] == historical
    # Another seeded pass retains the historical failure as repair context,
    # without attributing it to either fresh review-only run.
    next_seed = load_seed(output, pilot, pilot["capabilities"])
    assert next_seed["failure_history"] == report["inherited_failures"]
    assert next_seed["prior_failures"] == historical["proposal"]


def test_legacy_exclusions_can_be_reviewed_but_not_accepted(tmp_path):
    source = tmp_path / "source"
    pilot = seed_fixture(source)
    path = source / "plans.json"
    plans = json.loads(path.read_text())
    plans["cap.one"]["excluded_combinations"] = [
        {
            "environment": "reasoning",
            "verification": "simple",
            "reason": "legacy contradiction",
        }
    ]
    path.write_text(json.dumps(plans))
    imported = load_seed(source, pilot, pilot["capabilities"])
    review = _portfolio_review("cap.one", range(1, 11), "accept", [])
    with pytest.raises(ValueError, match="excluded combination is used"):
        cli._validate_portfolio_review(
            review,
            "cap.one",
            list(range(1, 11)),
            {slot: "proposed" for slot in range(1, 11)},
            imported["plans"]["cap.one"],
        )
