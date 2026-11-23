# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric scoring uses the starter notebook's prompt and the judge's last Output verdict."""

import hashlib

import pytest
from verifyit.grade import GradingInfraError
from verifyit.modes.grade_judge import JudgeReply

from taskcompendium.importers.evalchemy.scorers.facts_parametric import (
    GRADER_TEMPLATE,
    grade_facts_parametric,
    output_verdict,
)

# SHA-256 of the starter notebook's GRADER_TEMPLATE as pinned in NVIDIA-NeMo/Gym#3588.
STARTER_TEMPLATE_SHA256 = "9d6a61f9ce3b875b5f97d25305abe6c073911f97e98297780483dd84fd9b548c"
QUESTION = "yair yint aung birthday"
GOLD = "August 18, 1993"


class ScriptedJudge:
    def __init__(self, reply: str):
        self.reply = reply
        self.conversations: list[list[dict[str, str]]] = []

    def __call__(self, messages):
        self.conversations.append(messages)
        return JudgeReply(self.reply, {"model": "fake/judge"})


def test_grader_template_is_the_starter_prompt_byte_for_byte():
    assert hashlib.sha256(GRADER_TEMPLATE.encode()).hexdigest() == STARTER_TEMPLATE_SHA256


def test_judge_sees_question_gold_answer_and_prediction():
    judge = ScriptedJudge("Matches the gold answer.\n```\nOutput: CORRECT\n```")
    reward = grade_facts_parametric(QUESTION, GOLD, "18 August 1993.", judge)
    [[message]] = judge.conversations
    assert message["role"] == "user"
    assert message["content"].endswith(
        f"```\nQuestion: {QUESTION}\nGold answer: {GOLD}\nPredicted answer: 18 August 1993.\n```"
    )
    assert reward.reward == 1.0
    assert reward.detail["metrics"] == {
        "accuracy": 1.0,
        "incorrect": 0.0,
        "not_attempted": 0.0,
        "unknown": 0.0,
        "judge_failed": 0.0,
    }


@pytest.mark.parametrize(
    "reply,metrics",
    [
        ("Output: MISTAKE", {"accuracy": 0.0, "incorrect": 1.0, "not_attempted": 0.0, "unknown": 0.0}),
        # UNKNOWN is attempted but not correct, as in the paper's attempted accuracy.
        ("Output: [UNKNOWN]", {"accuracy": 0.0, "incorrect": 1.0, "not_attempted": 0.0, "unknown": 1.0}),
        ("Output: NOT_ATTEMPTED", {"accuracy": 0.0, "incorrect": 0.0, "not_attempted": 1.0, "unknown": 0.0}),
    ],
)
def test_non_correct_verdicts_map_to_classification_metrics(reply, metrics):
    reward = grade_facts_parametric(QUESTION, GOLD, "1990", ScriptedJudge(reply))
    assert reward.reward == 0.0
    assert reward.detail["metrics"] == {**metrics, "judge_failed": 0.0}


@pytest.mark.parametrize(
    "reply,verdict",
    [
        ("The answer is not a MISTAKE; it matches.\nOutput: [CORRECT]", "correct"),
        ("It is CORRECT in spirit but contradicts the year.\n**Output:** MISTAKE", "mistake"),
        ("Output: INCORRECT", "mistake"),
        ("The model refuses.\nOutput: NOT ATTEMPTED", "not_attempted"),
        # An echo of the prompt's response format is not a verdict; the last real line wins.
        ("Format: Output: [CORRECT/MISTAKE/UNKNOWN/NOT_ATTEMPTED]\nOutput: UNKNOWN", "unknown"),
        ("Output: CORRECT\nOn reflection:\nOutput: MISTAKE", "mistake"),
        ("I think this is CORRECT.", None),
    ],
)
def test_output_line_parse(reply, verdict):
    assert output_verdict(reply) == verdict


def test_reply_without_output_line_is_a_grading_failure():
    with pytest.raises(GradingInfraError):
        grade_facts_parametric(QUESTION, GOLD, "1993", ScriptedJudge("Looks right to me."))
