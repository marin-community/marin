# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.scorers.longbench_v2 import grade_longbench_v2

_UPSTREAM_DEFECT = pytest.mark.xfail(strict=True, reason="upstream regex reads the first letter of the next word")


# Labeled from the requested answer sentence alone; the gold letter is B. Format decisions keep
# upstream pred.py's reading.
@pytest.mark.parametrize(
    "response,reward",
    [
        pytest.param("The correct answer is (B)", 1.0, id="requested-form"),
        pytest.param("The correct answer is B.", 1.0, id="bare-letter"),
        pytest.param("**The correct answer is (B)**", 1.0, id="markdown-emphasis"),
        pytest.param("Reasoning first.\n\nThe correct answer is (B) Rachael.", 1.0, id="letter-then-choice-text"),
        pytest.param("<think>The correct answer is (A)?</think>\n\nThe correct answer is (B)", 1.0, id="inline-think"),
        pytest.param("The correct answer is (C)", 0.0, id="wrong-letter"),
        pytest.param("I believe it is B", 0.0, id="no-answer-sentence"),
        pytest.param("", 0.0, id="empty"),
        pytest.param("The correct answer is (insert answer here)", 0.0, id="restated-instruction"),
        pytest.param("The correct answer is (A). On reflection, The correct answer is (B)", 0.0, id="first-statement"),
        pytest.param("Therefore, the correct answer is (B).", 0.0, id="lowercase-sentence"),
        pytest.param("The correct answer is: B", 0.0, id="colon"),
        pytest.param("The correct answer is Rachael", 0.0, id="choice-text-only"),
        pytest.param("The correct answer is Choice B", 1.0, id="next-word-choice", marks=_UPSTREAM_DEFECT),
        pytest.param("The correct answer is Because of X.", 0.0, id="next-word-false-pass", marks=_UPSTREAM_DEFECT),
    ],
)
def test_longbench_v2_grades_the_requested_answer_sentence(response, reward):
    result = grade_longbench_v2("B", response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["answer_extraction_errors"] == ([] if result.detail["model_answer"] else ["failed_to_parse"])
