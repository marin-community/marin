# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pivot tasks grade the final assistant message in process, and preparation checks the expert's own turn."""

import json

import pytest

from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
    GradingAttempt,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.pipeline.models import CheckStatus
from taskcompendium.pipeline.verification import verify_task

BASH = FunctionDefinition(name="bash", parameters={"type": "object", "properties": {"command": {"type": "string"}}})
GREP = {"type": "function_call", "name": "bash", "arguments": json.dumps({"command": "grep -rn foo src"})}
TERMINUS = json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}]})
SOURCE = Source(dataset="fixture", revision="1", row="0", importer_revision="1")


def _tool_task(expected_action: dict) -> TaskSpec:
    return TaskSpec(
        id="pivot",
        context=ConversationInput(events=(TextMessage(role="user", content="Find foo."),)),
        environment_requirements=EnvironmentRequirements(),
        final_tools=(BASH,),
        answer_type=AnswerType.NATIVE_ACTION,
        answer_format=FinalAction(),
        grader=VerifyitGrader(mode="pivot-tool-call", parameters={"expected_action": json.dumps(expected_action)}),
        source=SOURCE,
    )


def _grade(task: TaskSpec, reply: TextMessage | AssistantToolCalls):
    return grade_answer(task, GradingAttempt(ConversationTrace(events=(*task.context.events, reply))))


def _call(command: str) -> AssistantToolCalls:
    return AssistantToolCalls(calls=(ConversationToolCall(call_id="c", name="bash", arguments={"command": command}),))


@pytest.mark.parametrize(
    "reply,reward",
    [(_call("grep foo"), 1.0), (_call("ls"), 0.0), (TextMessage(role="assistant", content="Done."), 0.0)],
)
def test_tool_call_pivot_grades_the_final_message(reply, reward):
    result = _grade(_tool_task(GREP), reply)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


def test_preparation_checks_the_expert_action_and_an_empty_reply():
    checks = {check.check: check.status for check in verify_task(_tool_task(GREP))}
    assert checks == {"empty": CheckStatus.PASS, "reference": CheckStatus.PASS}


def test_a_message_pivot_that_an_empty_reply_passes_is_defective():
    checks = {check.check: check.status for check in verify_task(_tool_task({"type": "message"}))}
    assert checks == {"empty": CheckStatus.DEFECT, "reference": CheckStatus.PASS}


def test_terminus_pivot_grades_reply_text():
    task = TaskSpec(
        id="terminal",
        context=ConversationInput(events=(TextMessage(role="user", content="List files."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=VerifyitGrader(mode="terminus", parameters={"expected_answer": TERMINUS}),
        source=SOURCE,
    )
    assert {check.check: check.status for check in verify_task(task)} == {
        "empty": CheckStatus.PASS,
        "reference": CheckStatus.PASS,
    }
    assert _grade(task, TextMessage(role="assistant", content=f"<think>plan</think>{TERMINUS}")).reward == 1.0
