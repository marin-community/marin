# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.lexam import lexam_mcq_prompt
from taskcompendium.importers.evalchemy.scorers.lexam import grade_lexam_mcq

# LEXam's MCQ_PROMPT["letters"] and choice formatting from litellm_eval.py, rendered for the synthetic row below.
LEXAM_RENDERED = (
    "You are an expert in Privatrecht and address legal issues in a structured, exam-style manner.\n"
    "You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.\n"
    "Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, "
    "consider legal frameworks beyond Swiss law as well.\n\nPlease reason through the question step by step, "
    "using a chain-of-thought approach:\n- Clarify the facts: Briefly restate or highlight the key facts in the "
    "question to anchor your reasoning.\n- Issue Identification: What legal issue(s) arise from the facts?\n"
    "- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, "
    "case law, doctrine)?\n- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing "
    "any ambiguities, exceptions, or competing interpretations.\n- Eliminate Incorrect Answers: Briefly explain "
    "why each incorrect answer is wrong or less convincing.\n- Conclusion: Clearly state the correct answer choice "
    "(e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.\n\nFormat your final "
    "answer as follows:\n Correct Answer: ###C### \n\nQuestion:\n Which statement is correct?\nConsider {braces}."
    "\nA. Art. 1 ZGB applies\nB. None\nC. Both\nD. Neither\n\nAnswer:"
)


def test_prompt_matches_lexam_rendering():
    prompt = lexam_mcq_prompt(
        "Privatrecht",
        "Which statement is correct?\nConsider {braces}.",
        ["Art. 1 ZGB applies", "None", "Both", "Neither"],
    )
    assert prompt == LEXAM_RENDERED


@pytest.mark.parametrize(
    "response,reward,extracted",
    [
        ("Conclusion: C fits best.\n Correct Answer: ###C### ", 1, "C"),
        ("Correct Answer: ###A###", 0, "A"),
        # The last marker decides, as in the reference's matches[-1].
        ("Not ###A###. Correct Answer: ###C###", 1, "C"),
        ("Correct Answer: ###C###, or perhaps ###B###", 0, "B"),
        # Reasoning before the end-of-thinking marker is ignored.
        ("I think ###C###.</think>Correct Answer: ###B###", 0, "B"),
        ("Correct Answer: ###c###", 0, None),
        ("Correct Answer: C", 0, None),
        ("Correct Answer: ### C ###", 0, None),
        ("", 0, None),
    ],
)
def test_lexam_extraction(response, reward, extracted):
    result = grade_lexam_mcq("C", response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted
