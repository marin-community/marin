# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Grounding v2 scoring follows the starter notebook's flow and verdict parsing."""

import json

import pytest
from verifyit.modes.grade_judge import JudgeReply

from taskcompendium.importers.evalchemy.scorers.facts_grounding import (
    grade_facts_grounding,
    parse_ufg_rev21_verdict,
)

FULL_PROMPT = "Answer only from the document.\n\nWhat colour is the sky?\n\nThe sky is blue."
REQUEST = "What colour is the sky?"
DOCUMENT = "The sky is blue."
RESPONSE = "The document says the sky is blue, not grey."
BASELINE = "Blue."


def sentence(text: str, label: str) -> str:
    return json.dumps({"sentence": text, "label": label, "rationale": "Checked.", "excerpt": None})


def rating(verdict: str) -> str:
    return f'Analysis: fine.\n```json\n{{\n    "Instruction Following": "{verdict}"\n}}\n```'


class ScriptedJudge:
    """Replays judge replies in order and records each conversation it was asked."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.conversations: list[list[dict[str, str]]] = []

    def __call__(self, messages):
        self.conversations.append(messages)
        return JudgeReply(self.replies.pop(0), {"model": "fake/judge"})


def test_eligible_grounded_response_scores_one_after_three_judge_steps():
    judge = ScriptedJudge(BASELINE, rating("No Issues"), sentence(RESPONSE, "supported"))
    reward = grade_facts_grounding(FULL_PROMPT, REQUEST, DOCUMENT, RESPONSE, judge)
    assert reward.reward == 1.0
    assert reward.detail["metrics"] == {"factuality": 1.0, "eligible": 1.0}
    baseline_request, eligibility_request, grounding_request = judge.conversations
    assert baseline_request == [{"role": "user", "content": FULL_PROMPT}]
    eligibility = eligibility_request[-1]["content"]
    assert f"<|begin_of_query|>\n{REQUEST}\n<|end_of_query|>" in eligibility
    assert f"<|begin_of_test_response|>\n{RESPONSE}\n<|end_of_test_response|>" in eligibility
    assert f"<|begin_of_baseline_response|>\n{BASELINE}\n<|end_of_baseline_response|>" in eligibility
    assert DOCUMENT not in eligibility
    assert grounding_request[-1]["content"].endswith(
        f"**User Query:**\n{REQUEST}\n\n**Context:**\n{DOCUMENT}\n\n**Response:**\n{RESPONSE}"
    )


def test_major_issue_makes_the_response_ineligible_without_a_grounding_step():
    judge = ScriptedJudge(BASELINE, rating("Major Issue(s)"))
    reward = grade_facts_grounding(FULL_PROMPT, REQUEST, DOCUMENT, RESPONSE, judge)
    assert reward.reward == 0.0
    assert reward.detail["metrics"] == {"factuality": 0.0, "eligible": 0.0}
    assert len(judge.conversations) == 2


@pytest.mark.parametrize("eligibility", [rating("Minor Issue(s)"), "I cannot decide."])
def test_minor_or_unparseable_rating_is_eligible(eligibility):
    judge = ScriptedJudge(BASELINE, eligibility, sentence(RESPONSE, "supported"))
    reward = grade_facts_grounding(FULL_PROMPT, REQUEST, DOCUMENT, RESPONSE, judge)
    assert reward.detail["metrics"] == {"factuality": 1.0, "eligible": 1.0}


def test_one_unsupported_sentence_fails_an_eligible_response():
    grounding = "\n".join(
        [sentence("The sky is blue.", "supported"), sentence("It is green at night.", "not_supported")]
    )
    judge = ScriptedJudge(BASELINE, rating("No Issues"), grounding)
    reward = grade_facts_grounding(FULL_PROMPT, REQUEST, DOCUMENT, RESPONSE, judge)
    assert reward.detail["metrics"] == {"factuality": 0.0, "eligible": 1.0}
    assert [item["sentence"] for item in reward.detail["unsupported_sentences"]] == ["It is green at night."]


@pytest.mark.parametrize(
    "answer,grounded",
    [
        # Fenced chunks are joined; introductions and no-claim sentences pass.
        (f"Here:\n```json\n{sentence('Intro.', 'no_rad')}\n{sentence('A.', 'supported')}\n```", True),
        # Raw quotes inside values are stripped before parsing.
        ('{"sentence": "He said "hi".", "label": "supported", "rationale": "ok", "excerpt": "said "hi""}', True),
        # A sentence without a label is not counted against the response.
        ('{"sentence": "A."}', True),
        # An unparseable line that mentions not_supported still fails the response.
        ('{"sentence": broken "not_supported"', False),
        # Windows line endings between verdict lines still separate them.
        (sentence("A.", "supported") + "\r\n" + sentence("B.", "supported"), True),
        (sentence("A.", "supported") + "\r\n" + sentence("B.", "not_supported"), False),
        ("No verdicts at all.", False),
    ],
)
def test_grounding_parser_matches_the_notebook_rules(answer, grounded):
    assert parse_ufg_rev21_verdict(answer)[0] is grounded
