# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Symbolic grading across serialized contracts and submission conventions."""

import json
import subprocess
import sys

import pytest
from tasktrove_verify.spec import MathType

from taskcompendium.grading import Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import AnswerCall, GradingAttempt, JsonAnswer, PlainText
from taskcompendium.verifier_registry import grade_answer, resolve_verifier
from taskcompendium.verifiers.mathematical import mathematical_answer


def _task(expected, math_type):
    return TaskSpec(
        id="hand-authored-math",
        context=ConversationInput(events=(TextMessage(role="user", content="Simplify the expression."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=mathematical_answer(expected, math_type),
        source=Source(dataset="hand-authored", revision="1", row="math", importer_revision="1"),
    )


def _attempt(task, convention, candidate):
    if isinstance(convention, AnswerCall):
        response = AssistantToolCalls(
            calls=(ConversationToolCall(call_id="answer", name="submit_answer", arguments={"answer": candidate}),)
        )
    else:
        content = json.dumps({"answer": candidate}) if isinstance(convention, JsonAnswer) else candidate
        response = TextMessage(role="assistant", content=content)
    return GradingAttempt(ConversationTrace(events=(*task.context.events, response)), {}, object())


@pytest.mark.parametrize("convention", [PlainText(id="plain"), JsonAnswer(id="json"), AnswerCall(id="call")])
@pytest.mark.parametrize(
    "expected, math_type, candidate, reward",
    [
        ("1/2", MathType.SCALAR, "0.5", 1.0),
        (r"2\sqrt{3}", MathType.SCALAR, r"\sqrt{12}", 1.0),
        ("1/2", MathType.SCALAR, "2", 0.0),
        ("1/2", MathType.SCALAR, "???", 0.0),
        ("1/2", MathType.SCALAR, r"\boxed{1/2}\n\boxed{", 0.0),
        ("x^2+1", MathType.SCALAR, "1+x^2", 1.0),
        ("[1/2, x+1]", MathType.LIST, "0.5, 1+x", 1.0),
        ("[1/2, x+1]", MathType.LIST, "1+x, 0.5", 0.0),
        (r"\{1,2\}", MathType.SET, r"\{2,1\}", 1.0),
        (r"(2,\infty)", MathType.INTERVAL, "x > 2", 1.0),
        ("y=2x+1", MathType.EQUATION, "y=2x+2", 0.0),
    ],
)
async def test_mathematical_answers_share_scoring_across_conventions(expected, math_type, candidate, reward, convention):
    task = TaskSpec.model_validate_json(_task(expected, math_type).model_dump_json())
    result = await grade_answer(task, convention, _attempt(task, convention, candidate))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


def test_invalid_reference_fails_construction_and_loading():
    with pytest.raises(ValueError, match="Invalid mathematical verifier contract"):
        mathematical_answer("???", MathType.SCALAR)
    invalid = VerifierSpec(
        kind=VerifierKind.MATHEMATICAL_ANSWER,
        parameters_json=json.dumps({"expected": "???", "math_type": "scalar"}),
    )
    with pytest.raises(ValueError, match="Invalid mathematical verifier contract"):
        resolve_verifier(invalid)


def test_mathematical_verifier_grades_in_fresh_process(tmp_path):
    task = _task(r"\sqrt{2}", MathType.SCALAR)
    path = tmp_path / "specification.json"
    path.write_text(task.model_dump_json())
    script = """
import asyncio
import sys
from pathlib import Path
from taskcompendium.models import TaskSpec, ConversationTrace, TextMessage
from taskcompendium.submission import GradingAttempt, PlainText
from taskcompendium.verifier_registry import grade_answer

task = TaskSpec.model_validate_json(Path(sys.argv[1]).read_text())
async def run():
    for answer in [r"2/\\sqrt{2}", "2"]:
        trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
        result = await grade_answer(task, PlainText(id="plain"), GradingAttempt(trace, {}, object()))
        print(result.status, result.reward)
asyncio.run(run())
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ["graded 1.0", "graded 0.0"]
