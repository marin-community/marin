import argparse
import copy

import pytest
from test_synthesis import accepted

from capability_pipeline.admission import add_parser, admit_one, validate_individual
from capability_pipeline.inference import digest
from capability_pipeline.validation import InvalidArtifact


def candidate():
    value = accepted()
    value["construction_context"] = {
        "portfolio_issues": ["The shared fixture may expose the answer."],
        "plan": {"slots": [{"slot": 1}]},
    }
    return value


def review(verdict="accept", changes=None):
    value = accepted()["review"]
    return {
        **value,
        "slot": 1,
        "verdict": verdict,
        "issues": [],
        "required_changes": changes or [],
    }


class Store:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def generate(self, stage, identity, system, prompt, validator, **kwargs):
        self.calls.append((stage, prompt))
        value = next(self.values)
        validator(value)
        return value


def test_repair_rejected_cli_policy_defaults_off_and_is_explicitly_enabled():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser(subparsers)
    default = parser.parse_args(["admit", "--candidates", "in", "--out", "out"])
    enabled = parser.parse_args(
        ["admit", "--candidates", "in", "--out", "out", "--repair-rejected"]
    )
    assert default.repair_rejected is False
    assert enabled.repair_rejected is True
    assert default.require_changed_hash is False


def test_changed_hash_guard_repairs_an_initial_accept_then_requires_fresh_accept():
    source = candidate()
    repaired = copy.deepcopy(source["proposal"])
    repaired["title"] = "Clarified fresh admission proposal"
    store = Store([review(), repaired, review()])

    result = admit_one(source, store, 1, require_changed_hash=True)

    assert result["state"] == "accepted"
    assert result["item"]["proposal_hash"] == digest(repaired)
    assert [stage for stage, _ in store.calls] == [
        "construction-review",
        "construction-repair",
        "construction-review",
    ]


def test_changed_hash_guard_preserves_unchanged_accept_as_non_success():
    source = candidate()
    result = admit_one(
        source,
        Store([review(), copy.deepcopy(source["proposal"])]),
        1,
        require_changed_hash=True,
    )

    assert result["state"] == "rejected"
    assert "new proposal hash" in result["controller_issue"]


def test_changed_hash_guard_cannot_accept_unchanged_when_repair_budget_is_zero():
    source = candidate()
    result = admit_one(source, Store([review()]), 0, require_changed_hash=True)

    assert result["state"] == "rejected"
    assert result["history"][0]["review"]["verdict"] == "accept"
    assert result["history"][0]["controller_block"]["reason_code"] == (
        "clarification_requires_changed_hash"
    )
    assert "fresh accepting review" in result["controller_issue"]


def test_changed_hash_guard_rejects_return_to_original_after_prior_repair():
    source = candidate()
    repaired = copy.deepcopy(source["proposal"])
    repaired["title"] = "Intermediate clarification"
    store = Store([
        review(), repaired, review("repair", ["Still needs clarification"]),
        copy.deepcopy(source["proposal"]), review(),
    ])

    result = admit_one(source, store, 2, require_changed_hash=True)

    assert result["state"] == "rejected"
    assert result["history"][-1]["proposal_hash"] == source["proposal_hash"]
    assert result["history"][-1]["review"]["verdict"] == "accept"
    assert "substantive proposal change" in store.calls[1][1]


def test_candidate_cannot_inherit_acceptance_and_global_feedback_is_visible():
    source = candidate()
    store = Store([review("reject", ["Unrecoverable answer exposure"])])
    result = admit_one(source, store, 1)
    assert result["state"] == "rejected"
    assert "shared fixture may expose the answer" in store.calls[0][1]


def test_rejected_candidate_is_repaired_only_when_explicitly_enabled():
    source = candidate()
    repaired = copy.deepcopy(source["proposal"])
    repaired["title"] = "Salvaged after independent rejection"
    store = Store(
        [
            review("reject", ["Repair the salvageable score contract"]),
            repaired,
            review(),
        ]
    )
    result = admit_one(source, store, 1, repair_rejected=True)
    assert [stage for stage, _ in store.calls] == [
        "construction-review",
        "construction-repair",
        "construction-review",
    ]
    assert result["state"] == "accepted"
    assert result["item"]["proposal_hash"] == digest(repaired)
    assert [
        entry["review"]["verdict"] for entry in result["item"]["admission"]["history"]
    ] == [
        "reject",
        "accept",
    ]


def test_rejected_candidate_remains_rejected_after_final_fresh_review():
    source = candidate()
    repaired = copy.deepcopy(source["proposal"])
    repaired["title"] = "Attempted salvage"
    store = Store(
        [
            review("reject", ["Repair the premise"]),
            repaired,
            review("reject", ["Premise remains invalid"]),
        ]
    )
    result = admit_one(source, store, 1, repair_rejected=True)
    assert result["state"] == "rejected"
    assert [entry["review"]["verdict"] for entry in result["history"]] == [
        "reject",
        "reject",
    ]
    assert len(store.calls) == 3


def test_rejected_repair_must_produce_a_new_hash():
    source = candidate()
    store = Store(
        [
            review("reject", ["Substantive repair required"]),
            copy.deepcopy(source["proposal"]),
        ]
    )
    result = admit_one(source, store, 1, repair_rejected=True)
    assert result["state"] == "rejected"
    assert "new proposal hash" in result["controller_issue"]
    assert [stage for stage, _ in store.calls] == [
        "construction-review",
        "construction-repair",
    ]


def test_opted_rejected_repair_can_honestly_return_null():
    source = candidate()
    null = {
        "capability_id": "cap.test",
        "slot": 1,
        "status": "null",
        "null_reason": "The original anchors cannot be preserved faithfully",
    }
    result = admit_one(
        source,
        Store([review("reject", ["Repair or abstain"]), null]),
        1,
        repair_rejected=True,
    )
    assert result["state"] == "null"
    assert result["history"][0]["review"]["verdict"] == "reject"


def test_repair_requires_fresh_review_and_retains_original_hash():
    source = candidate()
    repaired = copy.deepcopy(source["proposal"])
    repaired["title"] = "Repaired fixture boundary"
    store = Store([review("repair", ["Remove visible gold"]), repaired, review()])
    result = admit_one(source, store, 1)
    assert [call[0] for call in store.calls] == [
        "construction-review",
        "construction-repair",
        "construction-review",
    ]
    item = result["item"]
    assert item["proposal_hash"] == digest(repaired)
    assert item["admission"]["source_proposal_hash"] == source["proposal_hash"]
    assert item["admission"]["state"] == "accepted"
    assert item["admission"]["portfolio_certified"] is False
    assert item["admission"]["runtime_certified"] is False
    assert "Repaired fixture boundary" in store.calls[-1][1]


def test_scoped_acceptance_cannot_leave_required_changes():
    with pytest.raises(InvalidArtifact, match="still requires"):
        validate_individual(review("accept", ["Fix grader"]), candidate()["proposal"])


def test_repair_can_honestly_abstain_without_admitting_an_unreviewed_task():
    source = candidate()
    null = {
        "capability_id": "cap.test",
        "slot": 1,
        "status": "null",
        "null_reason": "No observable ground truth",
    }
    result = admit_one(
        source, Store([review("repair", ["Find ground truth"]), null]), 1
    )
    assert result["state"] == "null"
    assert "item" not in result


def test_revoked_hash_cannot_be_readmitted_by_accepting_model(monkeypatch):
    source = candidate()
    revocation = {"reason": "Reference contradicts the declared arithmetic"}
    monkeypatch.setattr(
        "capability_pipeline.admission._revoked_proposals",
        lambda: {source["proposal_hash"]: revocation},
    )
    result = admit_one(source, Store([review()]), 0)
    assert result["state"] == "rejected"
    assert "item" not in result
    assert result["history"][0]["review"]["verdict"] == "accept"
    assert result["history"][0]["controller_block"]["revocation"] == revocation
    assert "cannot be readmitted unchanged" in result["controller_issue"]


def test_revoked_repair_repeats_gate_and_preserves_raw_accepting_reviews(monkeypatch):
    source = candidate()
    monkeypatch.setattr(
        "capability_pipeline.admission._revoked_proposals",
        lambda: {source["proposal_hash"]: {"reason": "Invalid source key"}},
    )
    store = Store([review(), copy.deepcopy(source["proposal"]), review()])
    result = admit_one(source, store, 1)
    assert [stage for stage, _ in store.calls] == [
        "construction-review",
        "construction-repair",
        "construction-review",
    ]
    assert "Invalid source key" in store.calls[1][1]
    assert "Invalid source key" in store.calls[-1][1]
    assert "cosmetic edit" in store.calls[1][1]
    assert result["state"] == "rejected"
    assert all(h["review"]["verdict"] == "accept" for h in result["history"])
    assert all("controller_block" in h for h in result["history"])


def test_revoked_rejected_noop_repair_remains_blocked(monkeypatch):
    source = candidate()
    revocation = {"reason": "Original threshold semantics are stricter"}
    monkeypatch.setattr(
        "capability_pipeline.admission._revoked_proposals",
        lambda: {source["proposal_hash"]: revocation},
    )
    store = Store(
        [
            review("reject", ["Restore the original anchors"]),
            copy.deepcopy(source["proposal"]),
        ]
    )
    result = admit_one(source, store, 1, repair_rejected=True)
    assert result["state"] == "rejected"
    assert result["history"][0]["controller_block"]["revocation"] == revocation
    assert "new proposal hash" in result["controller_issue"]
