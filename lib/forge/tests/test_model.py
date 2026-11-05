# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from forge.model import serialization as wire
from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, Message, Sampling, ToolCall, ToolSchema, Usage
from forge.model.proposal import (
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
)
from forge.model.review import (
    Accept,
    AdversaryResult,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    ControlResult,
    Incomplete,
    Modify,
    Reject,
    RejectKind,
    Retry,
    TrialResult,
    ValidationResult,
)
from forge.model.source import PlannedSlot, ProposalBatch, SlotFailure, SlotProposal
from forge.model.workpiece import Conversation, Submission, Terminal, Workpiece
from forge.policy import Policy, Prompt, load_policy
from forge.policy.defaults import DEFAULTS_DIR


@pytest.fixture
def workpiece() -> Workpiece:
    task = TaskSpec(
        id="sum",
        context=ConversationInput(events=(TextMessage(role="user", content="1 + 1?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="serialization fixture"),
        source=Source(dataset="forge", revision="v1", row="1", importer_revision="v1"),
    )
    proposal = TaskProposal(
        ProposalHeader(
            id="sum",
            source=SourceRef(SourceKind.CAPABILITY, "arithmetic", "abc"),
            environment=Environment.REASONING,
            solver_machine=SolverMachine.NONE,
            verification=Verification.CODE,
            composite=(),
            grounding=Grounding.UNVERIFIED,
            research_need=ResearchNeed.NOT_NEEDED,
            research_reason="synthetic",
            research=(),
            build=(),
            resources=(),
            null_reason="fixture",
        ),
        "fixture\n",
    )
    controls = (Control(Candidate("2"), RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
    evidence = ValidationResult(
        "task-digest",
        (TrialResult(0, GradeResult(Outcome.GRADED, 1.0)), TrialResult(1, None, Cause.MODEL_TIMEOUT)),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1.0), True),),
        (AdversaryResult("shortcut", True, ("No defect found",), (GradeResult(Outcome.GRADED, 0),)),),
    )
    session = Conversation(
        (Message("user", "make a task"), Message("assistant", "", (ToolCall("1", "submit", "{}"),), reasoning="done")),
        (ToolSchema("submit", "Submit the task", {"type": "object", "properties": {"task": {"type": "string"}}}),),
    )
    return Workpiece(
        "w1",
        proposal,
        session=session,
        submission=Submission(task, controls, "authored"),
        evidence=(evidence,),
        decisions=(Modify(("harder",)), Incomplete(("retry later",))),
        terminal=Terminal.HELD,
    )


def test_saved_workpiece_round_trip_keeps_conversation_evidence_and_decisions(workpiece: Workpiece) -> None:
    restored = wire.WORKPIECE.validate_json(wire.WORKPIECE.dump_json(workpiece))
    assert restored == workpiece
    assert restored.evidence[0].solver[1].cause is Cause.MODEL_TIMEOUT
    assert isinstance(restored.decisions[0], Modify)
    assert isinstance(restored.decisions[1], Incomplete)
    assert restored.session is not None
    assert restored.session.messages[-1].reasoning == "done"


@pytest.mark.parametrize(
    "decision",
    [
        Accept(BandOutcome.IN_BAND),
        Reject(RejectKind.TASK, ("ambiguous",)),
        Modify(("revise",)),
        Retry(Cause.NO_GRADE, 2),
        Incomplete(("machine gone",)),
    ],
)
def test_decision_round_trip_preserves_variant(decision: wire.Decision) -> None:
    restored = wire.DECISION.validate_json(wire.DECISION.dump_json(decision))
    assert restored == decision
    assert type(restored) is type(decision)


def test_workpiece_replace_does_not_change_previous_round(workpiece: Workpiece) -> None:
    revised = workpiece.with_(
        round=1, session=None, decisions=(*workpiece.decisions, Accept(BandOutcome.IN_BAND)), terminal=Terminal.ACCEPTED
    )
    assert workpiece.round == 0 and workpiece.session is not None and workpiece.terminal is Terminal.HELD
    assert len(workpiece.decisions) == 2
    assert revised.round == 1 and revised.session is None and revised.terminal is Terminal.ACCEPTED
    assert revised.submission == workpiece.submission


def test_submission_digest_tracks_task_and_controls_but_excludes_notes(workpiece: Workpiece) -> None:
    submission = workpiece.submission
    assert submission is not None
    assert replace(submission, notes="new note").digest == submission.digest
    assert replace(submission, controls=()).digest != submission.digest
    assert replace(submission, task=submission.task.model_copy(update={"id": "different"})).digest != submission.digest


@pytest.mark.parametrize(
    ("rate", "placement"),
    [
        (0.1, BandOutcome.TOO_HARD),
        (0.2, BandOutcome.IN_BAND),
        (0.5, BandOutcome.IN_BAND),
        (0.8, BandOutcome.IN_BAND),
        (0.9, BandOutcome.TOO_EASY),
    ],
)
def test_band_placement_includes_both_boundaries(rate: float, placement: BandOutcome) -> None:
    band = BandRules(0.2, 0.8, frozenset({BandOutcome.IN_BAND}))
    assert band.place(rate) is placement
    assert wire.BAND_RULES.validate_json(wire.BAND_RULES.dump_json(band)).place(rate) is placement


def test_expected_zero_accepts_submission_failure_but_reference_does_not() -> None:
    failure = GradeResult(Outcome.SUBMISSION_FAILURE, None)
    assert RewardBand(Outcome.GRADED, reward_max=0).met_by(failure)
    assert not RewardBand(Outcome.GRADED, reward_min=1).met_by(failure)


def test_policy_overlay_and_serialization_preserve_digest_and_original() -> None:
    policy = Policy({"author.system": Prompt("Hello {name}"), "author.turns": 2, "author.tags": ("a",)})
    updated = policy.overlaid({"author.system": "Welcome {name}", "author.tags": ["b"]})
    restored = wire.POLICY.validate_json(wire.POLICY.dump_json(updated))
    assert restored == updated
    assert restored.prompt("author.system").render(name="Ada") == "Welcome Ada"
    assert restored.stage_digest("author") != policy.stage_digest("author")
    assert policy.prompt("author.system").render(name="Ada") == "Hello Ada"
    assert restored.setting("author.tags", tuple) == ("b",)


def test_bundled_policy_survives_serialization() -> None:
    policy = load_policy(DEFAULTS_DIR)
    restored = wire.POLICY.validate_json(wire.POLICY.dump_json(policy))
    assert restored == policy
    assert restored.stage_digest("proposer") == policy.stage_digest("proposer")


def test_individual_model_files_round_trip(workpiece: Workpiece) -> None:
    """Each standalone file representation also works outside a workpiece envelope."""
    assert workpiece.session is not None and workpiece.submission is not None
    proposal = workpiece.proposal
    completion = Completion("done", usage=Usage(10, 2, 1, 8))
    slot = SlotProposal(1, proposal, workpiece.session.messages, (completion,), None)
    failure = SlotFailure(2, "invalid proposal", (), (completion,))
    batch = ProposalBatch(workpiece.session.messages, (completion,), (slot, failure))
    assert wire.PROPOSAL_BATCH.validate_json(wire.PROPOSAL_BATCH.dump_json(batch)) == batch
    assert wire.SLOT_PROPOSAL.validate_json(wire.SLOT_PROPOSAL.dump_json(slot)) == slot
    assert wire.SLOT_FAILURE.validate_json(wire.SLOT_FAILURE.dump_json(failure)) == failure
    planned = PlannedSlot(1, None)
    assert wire.PLANNED_SLOT.validate_json(wire.PLANNED_SLOT.dump_json(planned)) == planned
    sampling = Sampling(temperature=0.5, top_p=0.9, max_tokens=100)
    assert wire.SAMPLING.validate_json(wire.SAMPLING.dump_json(sampling)) == sampling
    assert wire.COMPLETION.validate_json(wire.COMPLETION.dump_json(completion)) == completion
    budget = Budget(3, 2)
    assert wire.BUDGET.validate_json(wire.BUDGET.dump_json(budget)) == budget
    assert wire.SUBMISSION.validate_json(wire.SUBMISSION.dump_json(workpiece.submission)) == workpiece.submission
    assert wire.CONVERSATION.validate_json(wire.CONVERSATION.dump_json(workpiece.session)) == workpiece.session
    assert wire.TASK_PROPOSAL.validate_json(wire.TASK_PROPOSAL.dump_json(proposal)) == proposal
    evidence = workpiece.evidence[0]
    assert wire.VALIDATION_RESULT.validate_json(wire.VALIDATION_RESULT.dump_json(evidence)) == evidence


def test_tool_schema_snapshot_survives_caller_mutation() -> None:
    properties = {"path": {"type": "string"}}
    schema = ToolSchema("read", "Read a file", {"required": ("path",), "properties": properties})
    saved = wire.TOOL_SCHEMA.dump_json(schema)
    properties["path"]["type"] = "number"
    assert wire.TOOL_SCHEMA.dump_json(schema) == saved
    restored = wire.TOOL_SCHEMA.validate_json(saved)
    assert restored == schema
