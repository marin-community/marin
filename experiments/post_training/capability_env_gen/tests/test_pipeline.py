import json
from types import SimpleNamespace

import pytest

import capability_pipeline.cli as pipeline_cli
from capability_pipeline.cli import _propose, _source_provenance, _stage_response_format
from capability_pipeline.inference import StageStore, digest, parallel_map, parse_json
from capability_pipeline.prompts import capability_prompt_record
from capability_pipeline.validation import (
    AXES,
    InvalidArtifact,
    validate_plan,
    validate_review,
)


class FakeGLM:
    tier = "interactive"

    def __init__(self):
        self.calls = 0
        self.requests = []

    def complete(self, request, events_path):
        self.calls += 1
        self.requests.append(request)
        return {
            "content": '{"value": 1}',
            "usage": {"completion_tokens": 4},
            "finish_reason": "stop",
        }


def test_request_uses_high_reasoning_structured_output_and_64k_repair(tmp_path):
    client = FakeGLM()
    responses = iter(["not json", '{"value": 1}'])

    def complete(request, events_path):
        client.calls += 1
        client.requests.append(request)
        return {"content": next(responses), "usage": {}, "finish_reason": "stop"}

    client.complete = complete
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "artifact",
            "strict": True,
            "schema": {"type": "object"},
        },
    }
    artifact = StageStore(tmp_path, client).generate(
        "s",
        "id",
        "system",
        "prompt",
        lambda obj: None,
        response_format=response_format,
    )

    assert artifact == {"value": 1}
    assert [request["max_tokens"] for request in client.requests] == [32000, 64000]
    assert all(
        request["chat_template_kwargs"] == {"reasoning_effort": "high"}
        for request in client.requests
    )
    assert all("reasoning_effort" not in request for request in client.requests)
    assert all(
        request["response_format"] == response_format for request in client.requests
    )


def test_structured_output_defaults_on_and_has_explicit_diagnostic_off():
    schema = {"type": "object"}
    default = _stage_response_format(SimpleNamespace(), "artifact", schema)
    assert default == {
        "type": "json_schema",
        "json_schema": {"name": "artifact", "strict": True, "schema": schema},
    }
    assert (
        _stage_response_format(
            SimpleNamespace(structured_output="off"), "artifact", schema
        )
        is None
    )


def test_accepted_source_provenance_retains_exact_catalog_record():
    capability = {
        "capability_id": "cap.one",
        "capability_sha256": "a" * 64,
        "capability": {"id": "cap.one", "includes": ["exact source"]},
    }
    pilot = {
        "source": {"path": "catalog.json", "sha256": "b" * 64},
        "catalog_audit": {"source_sha256": "b" * 64},
    }
    provenance = _source_provenance(pilot, capability)
    assert provenance == {
        "catalog_source": pilot["source"],
        "capability_record": capability,
        "capability_record_hash": digest(capability),
    }


def test_progression_prompt_context_keeps_incoming_edges_and_raw_provenance():
    capability = {
        "capability_id": "cap.one",
        "capability": {"id": "cap.one", "sampling_facets": ["diagnosis", "repair"]},
    }
    incoming = {"dependent_id": "cap.one", "prerequisite_id": "foundation"}
    unrelated = {"dependent_id": "cap.two", "prerequisite_id": "cap.one"}
    pilot = {
        "source": {"sha256": "a" * 64},
        "learning_progression": {
            "catalog_version": "new",
            "prompt_version": "v1",
            "source_sha256": "b" * 64,
            "edges_sha256": digest([incoming, unrelated]),
            "edges": [incoming, unrelated],
        },
    }
    enriched = capability_prompt_record(capability, pilot)
    assert enriched["learning_progression"]["edges"] == [incoming]
    assert enriched["capability"] == capability["capability"]
    provenance = _source_provenance(pilot, capability)
    assert provenance["capability_record"] == capability
    assert provenance["capability_record_hash"] == digest(capability)
    assert provenance["learning_progression_hash"] == digest(
        enriched["learning_progression"]
    )
    assert "learning_progression" not in capability


def test_progression_reaches_initial_and_repair_planning_and_review(
    tmp_path, monkeypatch
):
    store = NullPortfolioStore(repair_plan=True)
    original_generate = store.generate
    prompts = []

    def capture(stage, identity, system, prompt, validator, **kwargs):
        prompts.append((stage, prompt))
        return original_generate(stage, identity, system, prompt, validator, **kwargs)

    store.generate = capture
    monkeypatch.setattr(pipeline_cli, "GLMClient", lambda **kwargs: object())
    monkeypatch.setattr(pipeline_cli, "StageStore", lambda root, client: store)
    cap = {"capability_id": "cap.one", "capability": {"id": "cap.one"}}
    pilot = {
        "catalog_audit": {"status": "complete"},
        "learning_progression": {
            "catalog_version": "new",
            "prompt_version": "v1",
            "source_sha256": "a" * 64,
            "edges_sha256": "b" * 64,
            "edges": [
                {
                    "dependent_id": "cap.one",
                    "prerequisite_id": "foundation",
                    "transfer_basis": "unique-transfer-witness",
                }
            ],
        },
    }
    args = SimpleNamespace(
        tier="interactive", hold_seconds=1, concurrency=2, repair_rounds=1
    )
    assert _propose(args, pilot, [cap], tmp_path) == 0
    assert [stage for stage, _ in prompts] == [
        "plan",
        "review",
        "plan-repair",
        "review",
    ]
    assert all("unique-transfer-witness" in prompt for _, prompt in prompts)


def test_cache_invalidates_changed_prompt_and_revalidates_outputs(tmp_path):
    client = FakeGLM()
    store = StageStore(tmp_path, client)
    def validate(obj):
        return obj["value"] == 1
    store.generate("s", "id", "system", "prompt v1", validate)
    store.generate("s", "id", "system", "prompt v1", validate)
    assert client.calls == 1
    store.generate("s", "id", "system", "prompt v2", validate)
    assert client.calls == 2

    def reject(obj):
        raise InvalidArtifact("new schema rejects old cached output")

    with pytest.raises(InvalidArtifact):
        store.generate("s", "id", "system", "prompt v2", reject)


def test_stale_cache_is_preserved_and_regenerated(tmp_path):
    client = FakeGLM()
    responses = iter(['{"value": 1}', '{"value": 2}'])

    def complete(*args):
        client.calls += 1
        return {"content": next(responses), "usage": {}, "finish_reason": "stop"}

    client.complete = complete
    store = StageStore(tmp_path, client)

    store.generate("s", "id", "system", "prompt", lambda obj: None)

    def require_two(obj):
        if obj["value"] != 2:
            raise InvalidArtifact("validator advanced")

    assert store.generate("s", "id", "system", "prompt", require_two) == {"value": 2}
    assert client.calls == 2
    assert list(tmp_path.rglob("rejected-result.json"))
    assert list(tmp_path.rglob("cache-rejection.json"))


def test_incomplete_model_response_cannot_be_cached_as_success(tmp_path):
    client = FakeGLM()
    client.complete = lambda *args: {
        "content": '{"value": 1}',
        "usage": {},
        "finish_reason": "length",
    }
    with pytest.raises(ValueError, match="incomplete model output"):
        StageStore(tmp_path, client).generate(
            "s", "id", "system", "prompt", lambda obj: None
        )
    assert not list(tmp_path.rglob("result.json"))
    status = json.loads(next(tmp_path.rglob("status.json")).read_text())
    assert status["state"] == "failed"


def test_parallel_errors_do_not_discard_successful_siblings():
    def fail():
        raise RuntimeError("bad item")

    results, failures = parallel_map(
        [("a", lambda: 1), ("b", fail), ("c", lambda: 3)], 3
    )
    assert results == {"a": 1, "c": 3}
    assert failures["b"]["error_type"] == "RuntimeError"


def test_review_rejects_false_green_portfolio():
    review = {
        "capability_id": "x",
        "portfolio_verdict": "accept",
        "portfolio_issues": [],
        "missing_slots": [],
        "reviews": [
            {
                "slot": i,
                "verdict": "accept",
                "scores": dict.fromkeys(AXES, 4),
                "critical_failures": [],
                "issues": [],
                "required_changes": [],
            }
            for i in range(1, 11)
        ],
    }
    validate_review(review, "x", range(1, 11))
    review["reviews"][0]["scores"]["reward_validity"] = 2
    with pytest.raises(InvalidArtifact, match="contradicts"):
        validate_review(review, "x", range(1, 11))
    review["reviews"] = review["reviews"][1:]
    with pytest.raises(InvalidArtifact, match="missing_slots"):
        validate_review(review, "x", range(2, 11))


def test_plan_exclusions_cannot_overlap_proposed_slot_pairings():
    plan = _null_plan("x")
    plan["slots"][0].update(
        status="propose",
        title="real task",
        workflow="do work",
        distinctive_challenge="edge cases",
        environment="reasoning",
        verification="code",
        reason=None,
    )
    plan["excluded_combinations"] = [
        {
            "environment": "reasoning",
            "verification": "code",
            "reason": "Not excluded generally, but unsuitable for another slot.",
        }
    ]
    with pytest.raises(InvalidArtifact, match="excluded combination is used"):
        validate_plan(plan, "x")


def test_review_acceptance_cannot_hide_required_changes_or_null_mismatch():
    review = {
        "capability_id": "x",
        "portfolio_verdict": "accept",
        "portfolio_issues": [],
        "missing_slots": list(range(2, 11)),
        "reviews": [
            {
                "slot": 1,
                "verdict": "accept",
                "scores": dict.fromkeys(AXES, 4),
                "critical_failures": [],
                "issues": [],
                "required_changes": ["still required"],
            }
        ],
    }
    with pytest.raises(InvalidArtifact, match="still requires changes"):
        validate_review(review, "x", [1], {1: "proposed"})
    review["reviews"][0]["required_changes"] = []
    review["reviews"][0]["verdict"] = "null"
    with pytest.raises(InvalidArtifact, match="contradicts proposal status"):
        validate_review(review, "x", [1], {1: "proposed"})


def test_parser_does_not_salvage_truncated_generation():
    with pytest.raises(ValueError):
        parse_json('{"a": "unfinished')
    assert parse_json('```json\n{"a":1}\n```') == {"a": 1}
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def _null_plan(capability_id, rationale="initial"):
    return {
        "capability_id": capability_id,
        "coverage_rationale": rationale,
        "excluded_combinations": [],
        "research_priorities": [],
        "slots": [
            {
                "slot": slot,
                "title": "unused",
                "workflow": "unused",
                "environment": "reasoning",
                "verification": "simple",
                "distinctive_challenge": "unused",
                "status": "null",
                "reason": "No sound task for this slot.",
            }
            for slot in range(1, 11)
        ],
    }


def _null_review(capability_id, verdict, issues):
    return {
        "capability_id": capability_id,
        "portfolio_verdict": verdict,
        "portfolio_issues": issues,
        "missing_slots": [],
        "reviews": [
            {
                "slot": slot,
                "verdict": "null",
                "scores": dict.fromkeys(AXES, 4),
                "critical_failures": [],
                "issues": [],
                "required_changes": [],
            }
            for slot in range(1, 11)
        ],
    }


@pytest.mark.parametrize("portfolio_verdict", ["repair", "reject"])
@pytest.mark.parametrize("slot_verdict", ["accept", "null"])
def test_nonaccepting_review_requires_actionable_feedback(portfolio_verdict, slot_verdict):
    review = _null_review("x", portfolio_verdict, [])
    for item in review["reviews"]:
        item["verdict"] = slot_verdict
    with pytest.raises(InvalidArtifact, match="no actionable feedback"):
        validate_review(review, "x", range(1, 11))
    review["reviews"][0]["verdict"] = "repair"
    review["reviews"][0]["required_changes"] = ["Correct the reference answer."]
    validate_review(review, "x", range(1, 11))
    review["reviews"] = review["reviews"][1:]
    review["missing_slots"] = [1]
    validate_review(review, "x", range(2, 11))


class NullPortfolioStore:
    def __init__(self, repair_plan=False):
        self.repair_plan = repair_plan
        self.review_calls = 0
        self.stages = []

    def generate(self, stage, identity, system, prompt, validator, **kwargs):
        self.stages.append(stage)
        if stage == "plan":
            artifact = _null_plan(identity)
        elif stage == "plan-repair":
            artifact = _null_plan(identity, "corrected")
        elif stage == "review":
            self.review_calls += 1
            if self.repair_plan and self.review_calls == 1:
                artifact = _null_review(
                    identity, "repair", ["plan rationale is invalid"]
                )
            else:
                artifact = _null_review(
                    identity,
                    "accept" if self.repair_plan else "reject",
                    [] if self.repair_plan else ["No credible portfolio coverage."],
                )
        else:  # pragma: no cover - test fixture should expose unexpected stage work
            raise AssertionError(stage)
        validator(artifact)
        return artifact


def _run_null_portfolio(tmp_path, monkeypatch, *, repair_rounds, repair_plan):
    store = NullPortfolioStore(repair_plan=repair_plan)
    monkeypatch.setattr(pipeline_cli, "GLMClient", lambda **kwargs: object())
    monkeypatch.setattr(pipeline_cli, "StageStore", lambda root, client: store)
    capability = {
        "capability_id": "cap.one",
        "subject_id": "S1",
        "subject_name": "Subject",
        "capability_sha256": "0" * 64,
        "selection_rationale": "test",
        "capability": {"id": "cap.one", "kind": "capability", "name": "Test"},
    }
    args = SimpleNamespace(
        tier="interactive",
        hold_seconds=1,
        concurrency=2,
        repair_rounds=repair_rounds,
    )
    pilot = {"catalog_audit": {"status": "complete"}}
    return _propose(args, pilot, [capability], tmp_path), store


def test_nonaccepting_all_null_portfolio_cannot_report_complete(tmp_path, monkeypatch):
    (tmp_path / "accepted.json").write_text('[{"stale": true}]')

    return_code, _ = _run_null_portfolio(
        tmp_path, monkeypatch, repair_rounds=0, repair_plan=False
    )

    report = json.loads((tmp_path / "report.json").read_text())
    assert return_code == 2
    assert report["state"] == "needs_iteration"
    assert report["nonaccepting_portfolios"] == {"cap.one": "reject"}
    assert json.loads((tmp_path / "accepted.json").read_text()) == []


def test_portfolio_repair_admits_only_individually_accepted_siblings(
    tmp_path, monkeypatch
):
    class Store:
        def generate(self, stage, identity, _system, _prompt, validator, **_kwargs):
            if stage == "plan":
                artifact = _proposed_plan(identity)
            elif stage == "proposal":
                artifact = _valid_proposal("cap.one", int(identity.rsplit(":", 1)[1]))
            elif stage == "review":
                artifact = _portfolio_review("cap.one", range(1, 11), "repair", [])
                artifact["portfolio_issues"] = ["Slot 1 still needs an anchor correction."]
                artifact["reviews"][0].update(
                    verdict="repair", required_changes=["Correct the anchor."]
                )
            else:
                raise AssertionError(stage)
            validator(artifact)
            return artifact

    monkeypatch.setattr(pipeline_cli, "GLMClient", lambda **_kwargs: object())
    monkeypatch.setattr(pipeline_cli, "StageStore", lambda _root, _client: Store())
    capability = {
        "capability_id": "cap.one", "subject_id": "S1", "subject_name": "Subject",
        "capability_sha256": "0" * 64, "selection_rationale": "test",
        "capability": {"id": "cap.one", "kind": "capability", "name": "Test"},
    }
    pilot = {
        "source": {"path": "catalog.json", "sha256": "a" * 64},
        "catalog_audit": {"status": "complete"},
    }
    args = SimpleNamespace(
        tier="interactive", hold_seconds=1, concurrency=4, repair_rounds=0
    )
    assert _propose(args, pilot, [capability], tmp_path) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    accepted = json.loads((tmp_path / "accepted.json").read_text())
    rejected = json.loads((tmp_path / "rejected.json").read_text())
    assert report["accepted"] == report["partial_portfolio_admissions"] == 9
    assert report["nonaccepting_portfolios"] == {"cap.one": "repair"}
    assert report["state"] == "needs_iteration"
    assert {row["proposal"]["slot"] for row in accepted} == set(range(2, 11))
    assert [row["proposal"]["slot"] for row in rejected] == [1]


def test_portfolio_issue_repairs_plan_and_requires_fresh_review(tmp_path, monkeypatch):
    return_code, store = _run_null_portfolio(
        tmp_path, monkeypatch, repair_rounds=1, repair_plan=True
    )

    report = json.loads((tmp_path / "report.json").read_text())
    plans = json.loads((tmp_path / "plans.json").read_text())
    assert return_code == 0
    assert report["state"] == "complete"
    assert store.stages == ["plan", "review", "plan-repair", "review"]
    assert plans["cap.one"]["coverage_rationale"] == "corrected"
    validate_plan(plans["cap.one"], "cap.one")


def _proposed_plan(capability_id):
    plan = _null_plan(capability_id)
    for slot in plan["slots"]:
        slot.update(
            title=f"Task {slot['slot']}",
            workflow="Analyze a distinct supplied case.",
            distinctive_challenge=f"edge pattern {slot['slot']}",
            status="propose",
            reason=None,
        )
    return plan


def _valid_proposal(capability_id, slot):
    return {
        "capability_id": capability_id,
        "slot": slot,
        "status": "proposed",
        "null_reason": None,
        "title": f"Task {slot}",
        "task_family": "analysis",
        "environment": "reasoning",
        "verification": "simple",
        "capability_alignment": "Directly exercises the source capability.",
        "environment_rationale": "All evidence is present in the prompt.",
        "workflow": "Inspect the case and derive the answer.",
        "task_brief": "Analyze the supplied case and return the requested answer.",
        "inputs": ["case record"],
        "deliverables": ["answer"],
        "constraints": ["cite the supplied evidence"],
        "difficulty_drivers": [f"edge pattern {slot}"],
        "grounding": {"known_facts": [], "research_needed": [], "sources": []},
        "environment_spec": {
            "initial_state": "prompt only",
            "tools": [],
            "reset": "new prompt",
            "dependency_strategy": "none",
            "resource_estimate": "one short response",
        },
        "verification_spec": {
            "observable_success": "answer matches the private reference",
            "grader_design": "normalized exact comparison",
            "positive_controls": ["reference answer"],
            "negative_controls": ["plausible wrong answer"],
            "anti_shortcuts": ["hold out case values"],
            "rubric": ["not applicable: exact comparison"],
        },
        "builder_plan": [
            {
                "session": "build",
                "depends_on": [],
                "goal": "construct and test the task",
                "handoff_artifacts": ["task bundle"],
                "acceptance_checks": ["positive and negative controls run"],
            }
        ],
        "validation_plan": ["Run the reference and plausible-wrong controls."],
        "risks": [
            {
                "risk": "ambiguous answer",
                "mitigation": "pin normalization",
                "abandon_if": "two incompatible answers remain valid",
            }
        ],
        "data_policy": {
            "provenance": "synthetic case",
            "license": "CC0",
            "split_group": f"case-{slot}",
            "contamination_check": "hash exact text",
            "private_evaluator_data": ["reference answer"],
        },
    }


def _portfolio_review(capability_id, slots, verdict, missing_slots):
    return {
        "capability_id": capability_id,
        "portfolio_verdict": verdict,
        "portfolio_issues": ["regenerate the missing slot"] if missing_slots else [],
        "missing_slots": missing_slots,
        "reviews": [
            {
                "slot": slot,
                "verdict": "accept",
                "scores": dict.fromkeys(AXES, 4),
                "critical_failures": [],
                "issues": [],
                "required_changes": [],
            }
            for slot in slots
        ],
    }


class MissingSlotStore:
    def __init__(self):
        self.review_calls = 0
        self.missing_repair_prompt = None
        self.repaired_slots = []

    def generate(self, stage, identity, system, prompt, validator, **kwargs):
        if stage == "plan":
            artifact = _proposed_plan(identity)
        elif stage == "proposal":
            slot = int(identity.rsplit(":", 1)[1])
            if slot == 5:
                raise InvalidArtifact("empty validation_plan")
            artifact = _valid_proposal("cap.one", slot)
        elif stage == "plan-repair":
            artifact = _proposed_plan(identity)
        elif stage == "repair":
            slot = int(identity.rsplit(":", 1)[1])
            self.repaired_slots.append(slot)
            if slot == 5:
                self.missing_repair_prompt = prompt
            artifact = _valid_proposal("cap.one", slot)
        elif stage == "review":
            self.review_calls += 1
            slots = list(range(1, 11))
            if self.review_calls == 1:
                slots.remove(5)
                artifact = _portfolio_review("cap.one", slots, "repair", [5])
            else:
                artifact = _portfolio_review("cap.one", slots, "accept", [])
        else:  # pragma: no cover
            raise AssertionError(stage)
        validator(artifact)
        return artifact


def test_missing_initial_slot_is_regenerated_and_freshly_reviewed(
    tmp_path, monkeypatch
):
    store = MissingSlotStore()
    monkeypatch.setattr(pipeline_cli, "GLMClient", lambda **kwargs: object())
    monkeypatch.setattr(pipeline_cli, "StageStore", lambda root, client: store)
    capability = {
        "capability_id": "cap.one",
        "subject_id": "S1",
        "subject_name": "Subject",
        "capability_sha256": "0" * 64,
        "selection_rationale": "test",
        "capability": {"id": "cap.one", "kind": "capability", "name": "Test"},
    }
    args = SimpleNamespace(
        tier="interactive",
        hold_seconds=1,
        concurrency=4,
        repair_rounds=1,
    )
    pilot = {
        "source": {"path": "catalog.json", "sha256": "a" * 64},
        "catalog_audit": {"status": "complete"},
    }

    assert _propose(args, pilot, [capability], tmp_path) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["accepted"] == 10
    assert report["missing_slots"] == []
    assert "cap.one:5" in report["failures"]["proposal"]
    assert report["unresolved_repairs"] == {}
    assert store.review_calls == 2
    assert store.repaired_slots == [5]
    assert '"missing_slot"' in store.missing_repair_prompt
    assert '"prior_failure"' in store.missing_repair_prompt
    assert '"previous_proposal"' not in store.missing_repair_prompt


@pytest.mark.parametrize("affected", [(5,), (2, 5), ()])
def test_portfolio_repair_preserves_accepted_siblings(tmp_path, monkeypatch, affected):
    class TargetedStore:
        def __init__(self):
            self.review_calls = 0
            self.repairs = []
            self.initial = {}
            self.plan_repairs = 0

        def generate(self, stage, identity, system, prompt, validator, **kwargs):
            if stage in {"plan", "plan-repair"}:
                artifact = _proposed_plan(identity)
                if stage == "plan-repair":
                    self.plan_repairs += 1
                    artifact["coverage_rationale"] = "Corrected portfolio rationale."
            elif stage in {"proposal", "repair"}:
                slot = int(identity.rsplit(":", 1)[1])
                artifact = _valid_proposal("cap.one", slot)
                if stage == "proposal":
                    self.initial[slot] = artifact
                else:
                    self.repairs.append(slot)
                    artifact["task_brief"] += " Clarified the required evidence."
            elif stage == "review":
                self.review_calls += 1
                artifact = _portfolio_review("cap.one", range(1, 11), "accept", [])
                if self.review_calls == 1:
                    artifact["portfolio_verdict"] = "repair"
                    artifact["portfolio_issues"] = ["Clarify the plan and affected slots."]
                    for row in artifact["reviews"]:
                        if row["slot"] in affected:
                            row.update(verdict="repair", required_changes=["Clarify evidence."])
            else:
                raise AssertionError(stage)
            validator(artifact)
            return artifact

    store = TargetedStore()
    monkeypatch.setattr(pipeline_cli, "GLMClient", lambda **kwargs: object())
    monkeypatch.setattr(pipeline_cli, "StageStore", lambda root, client: store)
    capability = {
        "capability_id": "cap.one", "capability_sha256": "0" * 64,
        "capability": {"id": "cap.one", "kind": "capability", "name": "Test"},
    }
    pilot = {
        "source": {"path": "catalog.json", "sha256": "a" * 64},
        "catalog_audit": {"status": "complete"},
    }
    args = SimpleNamespace(tier="interactive", hold_seconds=1, concurrency=4, repair_rounds=1)
    assert _propose(args, pilot, [capability], tmp_path) == 0
    assert sorted(store.repairs) == list(affected)
    assert store.plan_repairs == 1
    assert store.review_calls == 2
    accepted = json.loads((tmp_path / "accepted.json").read_text())
    assert len(accepted) == 10
    for record in accepted:
        proposal = record["proposal"]
        slot = proposal["slot"]
        assert (proposal == store.initial[slot]) is (slot not in affected)
