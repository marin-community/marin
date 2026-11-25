# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric grades with the starter notebook's prompt and the judge's last Output line."""

import hashlib

import pytest
from verifyit.grade import GradingInfraError, Status
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate

from taskcompendium.importers.evalchemy.judge_prompts import FACTS_PARAMETRIC_GRADER_TEMPLATE, facts_parametric_spec

# SHA-256 of the starter notebook's GRADER_TEMPLATE as pinned in NVIDIA-NeMo/Gym#3588.
STARTER_TEMPLATE_SHA256 = "9d6a61f9ce3b875b5f97d25305abe6c073911f97e98297780483dd84fd9b548c"
QUESTION = "yair yint aung birthday"
GOLD = "August 18, 1993"
NO_VERDICT = {"accuracy": 0.0, "incorrect": 0.0, "not_attempted": 0.0, "unknown": 0.0, "judge_failed": 0.0}


@pytest.fixture
def grade(judge_server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    connection = JudgeConnection(
        f"http://127.0.0.1:{judge_server.server_port}/v1", "fixture-key", default_model="fixture-judge"
    )

    def run(reply: str, candidate: str = "1990"):
        judge_server.reply = reply
        return grade_judge_candidate(facts_parametric_spec(QUESTION, GOLD), candidate, connection=connection)

    return run


def test_grader_template_is_the_starter_prompt_byte_for_byte():
    assert hashlib.sha256(FACTS_PARAMETRIC_GRADER_TEMPLATE.encode()).hexdigest() == STARTER_TEMPLATE_SHA256


def test_judge_sees_question_gold_answer_and_prediction(grade, judge_server):
    verdict = grade("Matches the gold answer.\n```\nOutput: CORRECT\n```", "18 August 1993.")
    [request] = judge_server.requests
    [message] = request["messages"]
    assert message["role"] == "user"
    assert message["content"].endswith(
        f"```\nQuestion: {QUESTION}\nGold answer: {GOLD}\nPredicted answer: 18 August 1993.\n```"
    )
    assert (verdict.status, verdict.reward) == (Status.SCORED, 1.0)
    assert verdict.detail["metrics"] == {**NO_VERDICT, "accuracy": 1.0}


@pytest.mark.parametrize(
    "reply,metrics",
    [
        ("The answer is not a MISTAKE; it matches.\nOutput: [CORRECT]", {"accuracy": 1.0}),
        ("It is CORRECT in spirit but contradicts the year.\n**Output:** MISTAKE", {"incorrect": 1.0}),
        ("output: incorrect", {"incorrect": 1.0}),
        # UNKNOWN is attempted but not correct, as in the paper's attempted accuracy.
        ("Output: [UNKNOWN]", {"incorrect": 1.0, "unknown": 1.0}),
        ("The model refuses.\nOutput: NOT ATTEMPTED", {"not_attempted": 1.0}),
        # An echo of the prompt's response format is not a verdict; the last real line wins.
        ("Format: Output: [CORRECT/MISTAKE/UNKNOWN/NOT_ATTEMPTED]\nOutput: NOT_ATTEMPTED", {"not_attempted": 1.0}),
        ("Output: CORRECT\nOn reflection:\nOutput: MISTAKE", {"incorrect": 1.0}),
    ],
)
def test_last_output_line_decides_the_verdict(grade, reply, metrics):
    verdict = grade(reply)
    assert verdict.status is Status.SCORED
    assert verdict.reward == metrics.get("accuracy", 0.0)
    assert verdict.detail["metrics"] == {**NO_VERDICT, **metrics}


def test_empty_answer_is_judged(grade, judge_server):
    assert grade("Nothing was answered.\nOutput: NOT_ATTEMPTED", "").detail["metrics"]["not_attempted"] == 1.0
    assert len(judge_server.requests) == 1


def test_reply_without_output_line_is_a_grading_failure(grade):
    with pytest.raises(GradingInfraError):
        grade("I think this is CORRECT.")
