# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from pydantic import ValidationError
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.proposal.model import Environment, Verification, parse, render
from taskforge.proposal.source import ProposalBatch, SlotFailure, SlotOutcome, SlotProposal
from taskforge.proposal.sources.capability import (
    CapabilitySource,
    PlannedSlot,
    SlotStatus,
    author_proposal,
    capability_idea_record,
    capability_prompt_record,
    checked_proposal,
    document_template,
    load_capability_ideas,
    null_slot_proposal,
    proposal_prefix,
    slot_plan_type,
)

CAPABILITY = {
    "id": "d01.algebra.linear-transformations",
    "kind": "capability",
    "name": "Linear transformations",
    "includes": ["kernels"],
    "excludes": [],
    "outcome": "Analyze maps.",
    "parent_id": "d01.algebra",
    "prerequisites": [],
    "sample_tasks": [],
    "sampling_facets": [],
}
EDGE = {"dependent_id": "d01.algebra.linear-transformations", "prerequisite_id": "d01.algebra.exact-linear-systems"}


@pytest.fixture
def ideas(tmp_path):
    catalog = {
        "catalog_version": "v3",
        "curricula": [
            {
                "curriculum": {
                    "subject_id": "D01",
                    "subject_name": "Mathematics",
                    "sections": [{"id": "d01.algebra", "kind": "group"}, CAPABILITY],
                }
            }
        ],
        "learning_progression": {"catalog_version": "v3", "edges": [EDGE, {**EDGE, "dependent_id": "other"}]},
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    return load_capability_ideas(path)


def test_catalog_loads_capabilities_with_their_incoming_edges(ideas):
    assert list(ideas) == ["d01.algebra.linear-transformations"]
    idea = ideas["d01.algebra.linear-transformations"]
    assert idea.subject_id == "D01"
    assert idea.prerequisite_edges == (EDGE,)


def test_the_idea_record_carries_the_identifiers_and_the_prompt_record(ideas):
    idea = ideas["d01.algebra.linear-transformations"]

    record = capability_idea_record(idea)

    assert record == {
        "capability_id": "d01.algebra.linear-transformations",
        "subject_id": "D01",
        "subject_name": "Mathematics",
        "catalog_version": "v3",
        "capability_hash": idea.capability_hash,
        "record": capability_prompt_record(idea),
    }
    assert record["record"]["learning_progression"] == {"catalog_version": "v3", "edges": [EDGE]}


def slot(number: int, environment: str, verification: str, status: str = "propose") -> dict:
    return {
        "slot": number,
        "title": f"t{number}",
        "workflow": "w",
        "environment": environment,
        "verification": verification,
        "distinctive_challenge": "c",
        "status": status,
        "reason": "no grader" if status == "null" else None,
    }


def plan(slots: list[dict], excluded: list[dict]) -> dict:
    return {
        "coverage_rationale": "r",
        "excluded_combinations": excluded,
        "slots": slots,
        "research_priorities": [],
    }


def test_slot_plan_requires_n_slots_and_rejects_using_an_excluded_pairing():
    plan_type = slot_plan_type(2)
    ok = plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "container", "judge", "null")], []))
    assert [s.status for s in ok.slots] == [SlotStatus.PROPOSE, SlotStatus.NULL]

    with pytest.raises(ValidationError):
        plan_type.model_validate(plan([slot(1, "reasoning", "code")], []))
    excluded = [{"environment": "container", "verification": "judge", "reason": "x"}]
    with pytest.raises(ValidationError, match="excluded_combinations"):
        plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "container", "judge")], excluded))


def test_slot_plan_caps_how_many_proposed_slots_share_a_pairing():
    plan_type = slot_plan_type(4)
    crowded = [slot(n, "reasoning", "code") for n in (1, 2, 3)] + [slot(4, "container", "judge")]
    with pytest.raises(ValidationError, match="reasoning x code"):
        plan_type.model_validate(plan(crowded, []))
    crowded[2] = slot(3, "reasoning", "code", "null")
    assert plan_type.model_validate(plan(crowded, [])).slots[2].status is SlotStatus.NULL


def filled_document(idea, number: int) -> str:
    return (
        document_template(f"d01.algebra.linear-transformations/{number}", idea)
        .replace("<reasoning | shellsim | container>", "shellsim")
        .replace("<simple | code | judge | composite>", "code")
        .replace("<web | github>", "github")
    )


def continuation(idea, number: int, document: str) -> str:
    """What the model writes after the prefilled start of slot ``number``'s ``document``."""
    prefix = proposal_prefix(f"d01.algebra.linear-transformations/{number}", idea)
    assert document.startswith(prefix)
    return document.removeprefix(prefix)


def test_filled_template_parses_and_identity_is_checked(ideas):
    idea = ideas["d01.algebra.linear-transformations"]
    text = filled_document(idea, 4)
    proposal = checked_proposal(text, idea, 4)
    assert proposal.header.environment is Environment.SHELLSIM
    with pytest.raises(ValueError, match="id"):
        checked_proposal(text, idea, 5)


def test_null_slot_becomes_a_parseable_null_proposal(ideas):
    idea = ideas["d01.algebra.linear-transformations"]
    planned = PlannedSlot.model_validate(slot(7, "reasoning", "judge", "null"))
    outcome = null_slot_proposal(idea, planned)
    assert outcome.completions == ()
    proposal = outcome.proposal
    assert proposal.header.null_reason == "no grader"
    assert proposal.header.verification is Verification.JUDGE
    assert parse(render(proposal)) == proposal


def run_with_client(fake_glm, call):
    async def go():
        endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, backoff=ExponentialBackoff(initial=0.001, maximum=0.001)) as client:
            return await call(client)

    return asyncio.run(go())


def author(fake_glm, idea) -> SlotOutcome:
    two_slots = slot_plan_type(2).model_validate(plan([slot(1, "shellsim", "code"), slot(2, "reasoning", "code")], []))
    return run_with_client(
        fake_glm, lambda client: author_proposal(client, LLMPolicy(), idea, two_slots, two_slots.slots[0])
    )


def test_unparseable_document_is_repaired_once_after_the_prefilled_front_matter_start(fake_glm, ideas):
    idea = ideas["d01.algebra.linear-transformations"]
    valid = filled_document(idea, 1)
    broken = valid.replace("verification: code\n", "")
    fake_glm.stream(content=broken)
    fake_glm.stream(content=continuation(idea, 1, valid))

    outcome = author(fake_glm, idea)

    assert isinstance(outcome, SlotProposal)
    assert outcome.repair_error is not None and "verification" in outcome.repair_error
    assert [c.content for c in outcome.completions] == [broken, valid]
    repair_request = fake_glm.requests[1]["messages"]
    assert repair_request[:2] == list(outcome.request)
    assert repair_request[2] == {"role": "assistant", "content": broken}
    assert "missing key(s) ['verification']" in repair_request[3]["content"]
    assert repair_request[4] == {
        "role": "assistant",
        "content": proposal_prefix("d01.algebra.linear-transformations/1", idea),
    }
    assert "enable_thinking" not in fake_glm.requests[0]["chat_template_kwargs"]
    assert fake_glm.requests[1]["continue_final_message"] is True


def test_document_still_invalid_after_repair_is_a_slot_failure_with_both_completions(fake_glm, ideas):
    idea = ideas["d01.algebra.linear-transformations"]
    fake_glm.stream(content="not a proposal")
    fake_glm.stream(content=" null\nenvironment: shellsim\n---\nstill no body")

    outcome = author(fake_glm, idea)

    assert isinstance(outcome, SlotFailure)
    assert "invalid after repair" in outcome.error
    prefix = proposal_prefix("d01.algebra.linear-transformations/1", idea)
    assert [c.content for c in outcome.completions] == [
        "not a proposal",
        prefix + " null\nenvironment: shellsim\n---\nstill no body",
    ]


def test_propose_returns_one_outcome_per_slot_in_order_without_calling_the_model_for_null_slots(fake_glm, ideas):
    idea = ideas["d01.algebra.linear-transformations"]
    portfolio = plan([slot(1, "reasoning", "judge", "null"), slot(2, "shellsim", "code")], [])
    fake_glm.stream(tool_calls=(("plan_slots", json.dumps(portfolio)),))
    fake_glm.stream(content=filled_document(idea, 2))

    batch: ProposalBatch = run_with_client(
        fake_glm, lambda client: CapabilitySource(client, LLMPolicy()).propose(idea, 2)
    )

    assert len(fake_glm.requests) == 2
    assert [s.slot for s in batch.slots] == [1, 2]
    assert [p.header.null_reason for p in batch.proposals] == ["no grader", None]
    assert batch.slots[0].completions == () and len(batch.slots[1].completions) == 1
    assert len(batch.planning) == 1 and batch.failures == ()
