# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.mmlu_redux import zeroeval_mcqa_prompt
from taskcompendium.importers.evalchemy.scorers.zeroeval import grade_zeroeval_mcqa

# ZeroEval's MCQA template rendered by its apply_mc_template for the synthetic row below.
ZEROEVAL_RENDERED = (
    "\n## Question: \n\nWhich number is prime?\nPick one.\n\n## Choices:\n\n- (A) 4\n- (B) 6\n- (C) 7\n"
    "- (D) {choices}\n\n\n## Instruction \n\nPlease answer this question by first reasoning and then selecting "
    "the correct choice. \nPresent your reasoning and solution in the following json format. \nPlease show your "
    'choice in the `answer` field with only the choice letter, e.g.,`"answer": "C"`.\n\n```json\n{\n    '
    '"reasoning": "___",\n    "answer": "___"\n}\n```\n'
)


def test_prompt_matches_zeroeval_rendering():
    assert zeroeval_mcqa_prompt("Which number is prime?\nPick one.", ["4", "6", "7", "{choices}"]) == ZEROEVAL_RENDERED


@pytest.mark.parametrize(
    "response,reward,extracted",
    [
        ('```json\n{\n    "reasoning": "7 has no divisors.",\n    "answer": "C"\n}\n```', 1, "C"),
        ('{"reasoning": "4 is even.", "answer": "A"}', 0, "A"),
        # ZeroEval accepts the gold letter followed by a parenthesis anywhere in the answer.
        ('{"reasoning": "r", "answer": "(C) 7"}', 1, "(C) 7"),
        ('{"reasoning": "r", "answer": "C. 7"}', 0, "C. 7"),
        ('{"reasoning": "r", "answer": "c"}', 0, "c"),
        ('{"reasoning": "r", "answer": "B or C"}', 0, "B or C"),
        # The first complete JSON object decides, even when a later one disagrees.
        ('{"reasoning": "r", "answer": "C"} Wait, {"reasoning": "r", "answer": "A"}', 1, "C"),
        # Braces before the JSON fail to parse, so the quoted-value regex recovers the answer.
        ('Since \\frac{1}{2} is not prime: {"reasoning": "r", "answer": "C"}', 1, "C"),
        # An unquoted answer runs to the next comma or whitespace, so a closing brace stays attached.
        ('{"reasoning": "unterminated, "answer": C\n}', 1, "C"),
        ('{"reasoning": "unterminated, "answer": C}', 0, "C}"),
        ('{"reasoning": "r", "answer": 3}', 0, "3"),
        ("The answer is C.", 0, None),
        ('{"reasoning": "r", "answer": ""}', 0, None),
        ("", 0, None),
    ],
)
def test_zeroeval_extraction(response, reward, extracted):
    result = grade_zeroeval_mcqa("C", response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted
