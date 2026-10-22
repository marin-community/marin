# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import json
import re
from pathlib import Path

from pydantic import ValidationError
from taskforge.llm.client import FinishReason, ModelRequestRejected
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError, response_format, schema_instruction
from taskforge.stage10_proposal.model import (
    DOCUMENT_REPAIR_PROMPT,
    Environment,
    ResearchNeed,
    SolverMachine,
    Verification,
    document_regex,
    document_request_fields,
    solver_machine_violation,
)
from taskforge.stage10_proposal.source import ProposalBatch, SlotFailure, SlotOutcome, SlotProposal

from experiments.post_training.capability_driven_envs.catalog import (
    CapabilityIdea,
    capability_idea_record,
    load_capability_ideas,
)
from experiments.post_training.capability_driven_envs.proposal_source import (
    RESEARCH_DECISION_CRITERION,
    CapabilitySource,
    SlotStatus,
    author_proposal,
    checked_proposal,
    document_template,
    plan_prompt,
    proposal_prompt,
    slot_plan_type,
    source_ref,
)
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel

CAPABILITY_ID = "d01.algebra.linear-transformations"
CAPABILITY = {
    "id": CAPABILITY_ID,
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
EDGE = {"dependent_id": CAPABILITY_ID, "prerequisite_id": "d01.algebra.exact-linear-systems"}
CATALOG_VERSION = "v3"
ALL_ENVIRONMENTS = frozenset(Environment)
NO_CONTAINER = frozenset({Environment.REASONING, Environment.SHELLSIM})


@pytest.fixture
def ideas(tmp_path: Path) -> dict[str, CapabilityIdea]:
    catalog = {
        "catalog_version": CATALOG_VERSION,
        "curricula": [
            {
                "curriculum": {
                    "subject_id": "D01",
                    "subject_name": "Mathematics",
                    "sections": [{"id": "d01.algebra", "kind": "group"}, CAPABILITY],
                }
            }
        ],
        "learning_progression": {
            "catalog_version": CATALOG_VERSION,
            "edges": [EDGE, {**EDGE, "dependent_id": "other"}],
        },
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    return load_capability_ideas(path)


def test_catalog_loads_capabilities_with_their_incoming_edges(ideas):
    assert list(ideas) == [CAPABILITY_ID]
    idea = ideas[CAPABILITY_ID]
    assert idea.subject_id == "D01"
    assert idea.prerequisite_edges == (EDGE,)
    progression = {"catalog_version": CATALOG_VERSION, "edges": [EDGE]}
    assert capability_idea_record(idea)["record"] == {**CAPABILITY, "learning_progression": progression}


def slot(number: int, environment: str, verification: str, status: str = "propose") -> dict:
    """A planned slot; an imageless slot declares ShellSim, a container slot no solver machine."""
    return {
        "slot": number,
        "title": f"t{number}",
        "workflow": "w",
        "environment": environment,
        "solver_machine": None if environment == "container" else "shellsim",
        "verification": verification,
        "distinctive_challenge": "c",
        "research_need": "not_needed",
        "research_reason": "synthetic fixtures",
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
    plan_type = slot_plan_type(2, ALL_ENVIRONMENTS)
    ok = plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "container", "llm_as_judge", "null")], []))
    assert [s.status for s in ok.slots] == [SlotStatus.PROPOSE, SlotStatus.NULL]

    with pytest.raises(ValidationError):
        plan_type.model_validate(plan([slot(1, "reasoning", "code")], []))
    excluded = [{"environment": "container", "verification": "llm_as_judge", "reason": "x"}]
    with pytest.raises(ValidationError, match="excluded_combinations"):
        plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "container", "llm_as_judge")], excluded))


def test_the_slot_plan_does_not_offer_or_accept_the_reserved_composite_verification():
    plan_type = slot_plan_type(1, ALL_ENVIRONMENTS)
    schema = json.dumps(response_format(plan_type))

    assert '"decision_model"' in schema and '"composite"' not in schema
    with pytest.raises(ValidationError, match="verification"):
        plan_type.model_validate(plan([slot(1, "reasoning", "composite")], []))
    excluded = [{"environment": "reasoning", "verification": "composite", "reason": "x"}]
    with pytest.raises(ValidationError, match="verification"):
        plan_type.model_validate(plan([slot(1, "reasoning", "code")], excluded))


@pytest.mark.parametrize(
    ("environment", "machine"), [("container", "cpython"), ("reasoning", None)], ids=["container", "imageless"]
)
def test_slot_plan_declares_a_solver_machine_exactly_for_an_imageless_slot(environment, machine):
    slots = [{**slot(1, environment, "code"), "solver_machine": machine}]
    violation = solver_machine_violation(Environment(environment), None if machine is None else SolverMachine(machine))
    assert violation is not None

    with pytest.raises(ValidationError, match=re.escape(f"slot 1: {violation}")):
        slot_plan_type(1, ALL_ENVIRONMENTS).model_validate(plan(slots, []))


def test_slot_plan_caps_how_many_proposed_slots_share_a_pairing():
    plan_type = slot_plan_type(4, ALL_ENVIRONMENTS)
    crowded = [slot(n, "reasoning", "code") for n in (1, 2, 3)] + [slot(4, "container", "llm_as_judge")]
    with pytest.raises(ValidationError, match="reasoning x code"):
        plan_type.model_validate(plan(crowded, []))
    crowded[2] = slot(3, "reasoning", "code", "null")
    assert plan_type.model_validate(plan(crowded, [])).slots[2].status is SlotStatus.NULL


def test_decision_model_slots_share_an_environment_without_the_pairing_cap():
    plan_type = slot_plan_type(2, ALL_ENVIRONMENTS)
    both = plan_type.model_validate(
        plan([slot(1, "reasoning", "decision_model"), slot(2, "reasoning", "decision_model")], [])
    )
    assert [s.verification for s in both.slots] == [Verification.DECISION_MODEL, Verification.DECISION_MODEL]

    with pytest.raises(ValidationError, match="reasoning x code"):
        plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "reasoning", "code")], []))


def test_plan_with_a_slot_in_an_unavailable_environment_fails_validation():
    plan_type = slot_plan_type(2, NO_CONTAINER)

    with pytest.raises(ValidationError, match="slot 2 is proposed in container"):
        plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "container", "code")], []))
    nulled = plan_type.model_validate(plan([slot(1, "reasoning", "code"), slot(2, "container", "code", "null")], []))
    assert nulled.slots[1].status is SlotStatus.NULL


def test_the_plan_instruction_states_the_slot_count_and_the_status_values():
    instruction = schema_instruction(slot_plan_type(3, NO_CONTAINER))

    assert "Exactly 3 slots, numbered 1 through 3 in order." in instruction
    assert "status (one of 'propose', 'null', required)" in instruction


def test_plan_and_proposal_prompts_share_the_research_criterion_and_the_catalog_record(ideas):
    idea = ideas[CAPABILITY_ID]
    portfolio = slot_plan_type(1, NO_CONTAINER).model_validate(plan([slot(1, "reasoning", "code")], []))

    planning = plan_prompt(idea, 1, NO_CONTAINER)
    authoring = proposal_prompt(idea, portfolio, portfolio.slots[0], f"{CAPABILITY_ID}/1")

    assert RESEARCH_DECISION_CRITERION in planning
    assert " ".join(RESEARCH_DECISION_CRITERION.split()) in " ".join(authoring.split())
    for prompt in (planning, authoring):
        record = prompt.split("CAPABILITY RECORD:\n")[1].split("\n\nPORTFOLIO:")[0]
        assert json.loads(record) == capability_idea_record(idea)


def filled_document(idea, number: int, research_need: ResearchNeed = ResearchNeed.NEEDED) -> str:
    """Fill every placeholder of the prompt's template; a needed document cites its research item."""
    document = (
        document_template(f"{CAPABILITY_ID}/{number}", idea)
        .replace("<reasoning | shellsim | container>", "shellsim")
        .replace("<none (reasoning only) | shellsim | cpython | packages, or null for container>", "cpython")
        .replace("<decision_model | answer_in_box | code | llm_as_judge>", "decision_model")
        .replace("<needed | not_needed>", str(research_need))
        .replace("<web | github>", "github")
    )
    if research_need is ResearchNeed.NOT_NEEDED:
        return re.sub(r"research:\n  - .*\n", "research: []\n", document)
    return document.replace("## Research plan\n<...>", "## Research plan\nFind the repository (research[0]).")


@pytest.mark.parametrize("research_need", list(ResearchNeed))
def test_the_filled_template_is_a_document_the_regex_accepts_and_the_identity_is_checked(ideas, research_need):
    idea = ideas[CAPABILITY_ID]
    text = filled_document(idea, 4, research_need)

    assert re.fullmatch(document_regex(f"{CAPABILITY_ID}/4", source_ref(idea)), text)
    proposal = checked_proposal(text, idea, 4)
    assert proposal.header.environment is Environment.SHELLSIM
    assert proposal.header.solver_machine is SolverMachine.CPYTHON
    assert proposal.header.research_need is research_need
    with pytest.raises(ValueError, match="id"):
        checked_proposal(text, idea, 5)


def author(model: ScriptedModel, idea) -> SlotOutcome:
    two_slots = slot_plan_type(2, ALL_ENVIRONMENTS).model_validate(
        plan([slot(1, "shellsim", "code"), slot(2, "reasoning", "code")], [])
    )
    return asyncio.run(author_proposal(model, LLMPolicy(), idea, two_slots, two_slots.slots[0]))


def test_author_request_carries_the_document_regex_with_continuations_off(ideas):
    idea = ideas[CAPABILITY_ID]
    model = ScriptedModel()
    model.reply(filled_document(idea, 1))

    outcome = author(model, idea)

    assert isinstance(outcome, SlotProposal) and outcome.repair_error is None
    (request,) = model.requests
    assert request.fields == document_request_fields(f"{CAPABILITY_ID}/1", source_ref(idea))
    assert request.policy.max_continuations == 0


def test_length_cut_document_is_a_slot_failure_without_continuation(ideas):
    idea = ideas[CAPABILITY_ID]
    model = ScriptedModel()
    model.reply(filled_document(idea, 1), FinishReason.LENGTH)

    outcome = author(model, idea)

    assert isinstance(outcome, SlotFailure)
    assert outcome.error == "cut off by the output limit"
    assert len(model.requests) == 1


def test_consistency_error_gets_one_regex_repair(ideas):
    idea = ideas[CAPABILITY_ID]
    valid = filled_document(idea, 1)
    uncited = valid.replace("Find the repository (research[0]).", "Find the repository.")
    model = ScriptedModel()
    model.reply(uncited)
    model.reply(valid)

    outcome = author(model, idea)

    assert isinstance(outcome, SlotProposal)
    assert outcome.repair_error == "body: research[0] is never cited as research[0]"
    assert [c.content for c in outcome.completions] == [uncited, valid]
    first, repair = model.requests
    assert repair.fields == first.fields
    assert repair.messages == [
        *outcome.request,
        {"role": "assistant", "content": uncited},
        {"role": "user", "content": DOCUMENT_REPAIR_PROMPT.format(error=outcome.repair_error)},
    ]


def test_document_still_invalid_after_repair_is_a_slot_failure_with_both_completions(ideas):
    idea = ideas[CAPABILITY_ID]
    uncited = filled_document(idea, 1).replace("(research[0])", "")
    model = ScriptedModel()
    model.reply(uncited)
    model.reply(uncited)

    outcome = author(model, idea)

    assert isinstance(outcome, SlotFailure)
    assert outcome.error == "invalid after repair: body: research[0] is never cited as research[0]"
    assert [c.content for c in outcome.completions] == [uncited, uncited]


def test_a_rejected_repair_request_is_a_slot_failure_that_keeps_the_first_completion(ideas):
    idea = ideas[CAPABILITY_ID]
    uncited = filled_document(idea, 1).replace("(research[0])", "")
    rejected = ModelRequestRejected(400, "the request is malformed", ())
    model = ScriptedModel()
    model.reply(uncited)
    model.fail(rejected)

    outcome = author(model, idea)

    assert isinstance(outcome, SlotFailure)
    assert str(rejected) in outcome.error
    assert [c.content for c in outcome.completions] == [uncited]


def test_a_plan_slot_the_host_cannot_run_is_repaired_before_any_document_is_written(ideas):
    idea = ideas[CAPABILITY_ID]
    model = ScriptedModel()
    model.reply(json.dumps(plan([slot(1, "container", "code")], [])))
    model.reply(json.dumps(plan([slot(1, "shellsim", "code")], [])))
    model.reply(filled_document(idea, 1))

    batch: ProposalBatch = asyncio.run(CapabilitySource(model, LLMPolicy(), NO_CONTAINER).propose(idea, 1))

    repair_prompt = model.requests[1].messages[-2]["content"]
    assert isinstance(repair_prompt, str) and "slot 1 is proposed in container" in repair_prompt
    assert len(batch.planning) == 2
    assert [p.header.environment for p in batch.proposals] == [Environment.SHELLSIM]


def test_a_plan_still_invalid_after_its_repair_raises_before_any_document_is_written(ideas):
    model = ScriptedModel()
    unrunnable = json.dumps(plan([slot(1, "container", "code")], []))
    model.reply(unrunnable)
    model.reply(unrunnable)

    with pytest.raises(StructuredOutputError, match="slot 1 is proposed in container"):
        asyncio.run(CapabilitySource(model, LLMPolicy(), NO_CONTAINER).propose(ideas[CAPABILITY_ID], 1))
    assert len(model.requests) == 2


def test_propose_returns_one_outcome_per_slot_in_order_without_calling_the_model_for_null_slots(ideas):
    idea = ideas[CAPABILITY_ID]
    model = ScriptedModel()
    model.reply(json.dumps(plan([slot(1, "reasoning", "llm_as_judge", "null"), slot(2, "shellsim", "code")], [])))
    model.reply(filled_document(idea, 2))

    batch: ProposalBatch = asyncio.run(CapabilitySource(model, LLMPolicy(), ALL_ENVIRONMENTS).propose(idea, 2))

    assert len(model.requests) == 2
    assert model.requests[0].fields == {"response_format": response_format(slot_plan_type(2, ALL_ENVIRONMENTS))}
    assert batch.planning_request[-1] == model.requests[0].messages[-1]
    assert [s.slot for s in batch.slots] == [1, 2]
    assert [p.header.null_reason for p in batch.proposals] == ["no grader", None]
    assert batch.slots[0].completions == () and len(batch.slots[1].completions) == 1
    assert len(batch.planning) == 1 and batch.failures == ()
