# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment, SolverMachine
from taskforge.stage10_proposal.source import SlotProposal

from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalConfig,
    OpenEndedProposalSource,
    checked_proposal,
    document_template,
    planning_record,
)
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_open_ended import bundle

CONFIG = OpenEndedProposalConfig(Environment.SHELLSIM, SolverMachine.SHELLSIM)


def slot(number: int, *, status: str = "propose", question: str | None = None) -> dict[str, object]:
    return {
        "slot": number,
        "title": f"task {number}",
        "scientific_question": question or f"scientific question {number}",
        "analysis_obligations": ["compute a donor-aware comparison"],
        "distinctive_challenge": f"challenge {number}",
        "status": status,
        "reason": "the bundle cannot support another distinct analysis" if status == "null" else None,
    }


def plan(*slots: dict[str, object]) -> str:
    return json.dumps({"coverage_rationale": "the slots cover distinct supported analyses", "slots": slots})


def filled_document(number: int) -> str:
    return document_template(bundle(), number, CONFIG).replace(
        "<...>",
        "Use the supplied counts for a donor-aware analysis. Keep the frozen rubric private and validate positive, "
        "partial, severe-negative, verbosity, and evaluator-injection controls.",
    )


def test_planning_view_contains_solver_inputs_but_no_private_evidence():
    record = planning_record(bundle())
    rendered = json.dumps(record)

    assert "workspace/data/counts.csv" in rendered
    assert "group,success,total" in rendered
    assert "reference/expected.json" not in rendered
    assert "A=80%" not in rendered
    assert "author_evidence" not in record


def test_source_plans_slots_and_writes_only_the_non_null_proposals():
    model = ScriptedModel()
    model.reply(plan(slot(1, status="null"), slot(2)))
    model.reply(filled_document(2))

    batch = asyncio.run(OpenEndedProposalSource(model, LLMPolicy(), CONFIG).propose(bundle(), 2))

    assert len(model.requests) == 2
    assert [outcome.slot for outcome in batch.slots] == [1, 2]
    assert all(isinstance(outcome, SlotProposal) for outcome in batch.slots)
    first, second = batch.proposals
    assert first.header.null_reason == "the bundle cannot support another distinct analysis"
    assert second.header.null_reason is None
    assert second.header.environment is Environment.SHELLSIM
    assert second.header.solver_machine is SolverMachine.SHELLSIM
    assert second.header.resources == ("workspace/data/counts.csv",)
    assert second.header.source.hash == bundle().content_digest


def test_writer_cannot_change_campaign_owned_proposal_fields():
    model = ScriptedModel()
    changed = filled_document(1).replace("verification: llm_as_judge", "verification: decision_model")
    changed = changed.replace("workspace/data/counts.csv", "workspace/data/wrong.csv")
    model.reply(plan(slot(1)))
    model.reply(changed)

    batch = asyncio.run(OpenEndedProposalSource(model, LLMPolicy(), CONFIG).propose(bundle(), 1))

    outcome = batch.slots[0]
    assert isinstance(outcome, SlotProposal)
    assert outcome.repair_error is None
    assert len(outcome.completions) == 1
    assert outcome.proposal.header.verification.value == "llm_as_judge"
    assert outcome.proposal.header.resources == ("workspace/data/counts.csv",)
    assert checked_proposal(changed, bundle(), 1, CONFIG) == outcome.proposal
