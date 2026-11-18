# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derived contract tags and grader types come from the final TaskSpec, not declarations."""

import pytest
from verifyit.spec import RUBRIC_CHECKLIST, ExactSpec, JudgeSpec, PytestSpec, ScriptSpec, UnsolvableSpec

from taskcompendium.derived_contract import (
    DerivedContract,
    derive_contract,
    shape_contradictions,
    summarize_contracts,
    summary_contradictions,
    task_contracts,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.grader_type import GraderType, derive_grader_type
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    CommandSemantics,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    ToolResult,
)

IMAGE = "private/grader@sha256:" + "a" * 64


def grader_environment() -> EnvironmentRequirements:
    return EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS)


def task(grader, **updates) -> TaskSpec:
    specification = TaskSpec(
        id="derived-contract-example",
        context=ConversationInput(events=(TextMessage(role="user", content="Answer in your reply."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )
    return specification.model_copy(update=updates)


JUDGE = verifyit_package(
    JudgeSpec(rubric=RUBRIC_CHECKLIST, criteria=("be complete",), question="Is the answer complete?")
).grader
EXACT = verifyit_package(ExactSpec(expected=("done",))).grader
TESTS = verifyit_package(PytestSpec(), environment=grader_environment()).grader
SCRIPT = verifyit_package(ScriptSpec(path="/tests/check.sh"), environment=grader_environment()).grader


def test_reply_only_judged_source_derives_nonagentic_single_turn_llm_judge():
    contract = derive_contract(task(JUDGE))
    assert contract == DerivedContract(
        agentic=False,
        single_turn=True,
        reply_only=True,
        workspace_graded=False,
        grader_type=GraderType.LLM_JUDGE,
    )


@pytest.mark.parametrize(
    ("events", "single_turn"),
    [
        (
            (TextMessage(role="system", content="Follow instructions."), TextMessage(role="user", content="Answer.")),
            True,
        ),
        (
            (TextMessage(role="developer", content="Use a short reply."), TextMessage(role="user", content="Answer.")),
            True,
        ),
        (
            (TextMessage(role="user", content="Answer."), TextMessage(role="assistant", content="The answer is")),
            True,
        ),
        (
            (
                TextMessage(role="user", content="Look this up."),
                AssistantToolCalls(calls=(ConversationToolCall(call_id="lookup-1", name="lookup", arguments={}),)),
                ToolResult(call_id="lookup-1", content="Found it."),
            ),
            False,
        ),
        (
            (
                TextMessage(role="user", content="First question."),
                TextMessage(role="assistant", content="First answer."),
                TextMessage(role="user", content="Second question."),
            ),
            False,
        ),
        ((TextMessage(role="system", content="Wait for a question."),), False),
    ],
    ids=("system", "developer", "prefill", "tool-history", "second-user-turn", "no-user-turn"),
)
def test_single_turn_counts_user_messages_without_tool_history(events, single_turn):
    specification = task(JUDGE, context=ConversationInput(events=events))
    assert derive_contract(specification).single_turn is single_turn


def test_workspace_answer_delivery_is_not_reply_only():
    specification = TaskSpec(
        id="derived-contract-workspace",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the project."),)),
        environment_requirements=grader_environment(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=TESTS,
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )
    assert derive_contract(specification).reply_only is False


def test_declared_contradiction_fails():
    contract = derive_contract(task(JUDGE))
    assert shape_contradictions({"agentic", "multi-turn"}, contract) == ("agentic", "multi-turn")
    assert shape_contradictions(set(), contract) == ()
    summary = task_contracts((task(JUDGE),))
    assert summary_contradictions({"agentic", "multi-turn", "reply-only"}, summary) == (
        "agentic",
        "multi-turn",
    )
    assert summary_contradictions({"reply-only"}, summary) == ()


def test_agentic_workspace_contract():
    specification = TaskSpec(
        id="derived-contract-workspace",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the project."),)),
        environment_requirements=grader_environment(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=TESTS,
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )
    contract = derive_contract(specification)
    assert contract.agentic
    assert contract.workspace_graded
    assert contract.grader_type is GraderType.UNIT_TESTS


def test_mixed_source_yields_mixed_with_counts():
    summary = task_contracts((task(JUDGE), task(EXACT)))
    assert summary.grader_type is GraderType.MIXED
    assert summary.mixed
    assert summary.grader_type_counts == {"exact-match": 1, "llm-judge": 1}
    assert summary.agentic_count == 0
    assert summary.single_turn_count == 2
    assert summary.reply_only_count == 2


def test_single_type_summary_is_not_mixed():
    summary = task_contracts((task(JUDGE), task(JUDGE)))
    assert summary.grader_type is GraderType.LLM_JUDGE
    assert not summary.mixed
    assert summary.judge_rubric_counts == {"checklist": 2}


def test_script_and_nograder_types():
    assert derive_grader_type(SCRIPT) is GraderType.SCRIPT
    assert derive_grader_type(NoGrader(reason="preference labels have no runnable grader")) is GraderType.NONE
    contract = derive_contract(task(NoGrader(reason="preference labels have no runnable grader")))
    assert contract.grader_type is GraderType.NONE
    assert not contract.workspace_graded


def test_reward_model_contract_recognition():
    grader = NoGrader(reason="upstream trainer scores reward model outputs", contract={"reward_model": {}})
    assert derive_grader_type(grader) is GraderType.REWARD_MODEL


def test_unclassifiable_grader_raises():
    unsolvable = verifyit_package(UnsolvableSpec()).grader
    with pytest.raises(ValueError, match="no GraderType mapping"):
        derive_grader_type(unsolvable)


def test_empty_population_summary_raises():
    with pytest.raises(ValueError, match="at least one task"):
        _ = summarize_contracts(()).grader_type
