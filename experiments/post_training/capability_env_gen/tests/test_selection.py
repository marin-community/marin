import pytest

from capability_pipeline.inference import digest
from capability_pipeline.select_builds import attach_provenance, select
from capability_pipeline.validation import AXES


def item(environment, verification, index):
    proposal = {
        "capability_id": f"cap-{index}",
        "environment": environment,
        "verification": verification,
        "builder_plan": [{"session": "s1"}],
    }
    return {
        "proposal": proposal,
        "proposal_hash": digest(proposal),
        "review": {
            "verdict": "accept",
            "scores": dict.fromkeys(AXES, 4),
            "critical_failures": [],
        },
    }


def test_selection_is_order_independent_and_covers_existing_modes():
    pool = [
        item("reasoning", "simple", 1),
        item("shellsim", "code", 2),
        item("container", "judge", 3),
        item("reasoning", "judge", 4),
        item("container", "code", 5),
    ]
    chosen, report = select(pool, 3)
    reversed_chosen, _ = select(list(reversed(pool)), 3)
    assert chosen == reversed_chosen
    assert not report["missing_environments"]
    assert not report["missing_verifiers"]


def test_missing_modes_are_reported_not_manufactured():
    chosen, report = select([item("reasoning", "simple", 1)], 9)
    assert len(chosen) == 1
    assert report["missing_environments"] == ["container", "shellsim"]
    assert report["missing_verifiers"] == ["code", "judge"]


def test_selection_does_not_bypass_unresolved_review():
    candidate = item("reasoning", "simple", 1)
    candidate["review"]["required_changes"] = ["Fix ambiguous answer contract"]
    with pytest.raises(ValueError, match="passing reviews"):
        select([candidate])


def test_provenance_backfill_binds_original_record_and_rejects_mismatch():
    candidate = item("reasoning", "simple", 1)
    record = {
        "capability_id": "cap-1",
        "capability": {"id": "cap-1", "name": "Example"},
    }
    pilot = {"source": {"sha256": "catalog-hash"}, "capabilities": [record]}
    enriched = attach_provenance([candidate], pilot)[0]
    assert enriched["proposal_hash"] == candidate["proposal_hash"]
    assert enriched["provenance"]["capability_record"] == record
    assert enriched["provenance"]["capability_record_hash"] == digest(record)
    assert "provenance" not in candidate
    enriched["provenance"]["capability_record_hash"] = "changed"
    with pytest.raises(ValueError, match="differs"):
        attach_provenance([enriched], pilot)
