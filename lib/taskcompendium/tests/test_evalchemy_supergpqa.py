# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.scorers import supergpqa
from taskcompendium.importers.evalchemy.scorers.supergpqa import grade_supergpqa
from taskcompendium.importers.evalchemy.supergpqa import supergpqa_prompt

# The authors' zero-shot template filled by infer/data_loader.py for the synthetic row below.
SUPERGPQA_RENDERED = (
    "Answer the following multiple choice question. There is only one correct answer. The last line of your "
    "response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, "
    "F, G, H, I, or J.\n\nWhich number is prime?\nPick one.\nA) 4\nB) 6\nC) 7\nD) {x}\n"
)
OPTIONS = ["4", "6", "7", "9"]


def test_prompt_matches_reference_rendering():
    assert supergpqa_prompt("Which number is prime?\nPick one.", ["4", "6", "7", "{x}"]) == SUPERGPQA_RENDERED


# Expected letters are the reference's eval/eval.py zero-shot extraction on the same responses.
@pytest.mark.parametrize(
    "response,reward,extracted",
    [
        ("Seven has no divisors.\nAnswer: C", 1, "C"),
        ("The answer is (C).", 1, "C"),
        ("I think B\nAnswer: $\\boxed{C}$", 1, "C"),
        ("\\boxed{C}", 1, "C"),
        ("Answer: A", 0, "A"),
        # A letter-pattern match in lowercase is kept as is and scored wrong.
        ("answer: c", 0, "c"),
        ("a total of 7", 0, "a"),
        # Without a letter, quoted option text maps back to its letter.
        ("So the answer is 7.", 1, "C"),
        ("The correct option is 7", 1, "C"),
        ("no idea", 0, None),
        ("", 0, None),
    ],
)
def test_supergpqa_extraction(response, reward, extracted):
    result = grade_supergpqa("C", OPTIONS, response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted


@pytest.mark.parametrize("run", ["*" * 40, "(" * 40, "\\" * 40])
def test_backtracking_search_times_out_as_a_miss(monkeypatch, run):
    monkeypatch.setattr(supergpqa, "SEARCH_TIMEOUT", 0.05)
    result = grade_supergpqa("C", OPTIONS, run)
    assert result.status == Status.SCORED
    assert result.detail["extracted"] is None
    assert grade_supergpqa("C", OPTIONS, run + "\nAnswer: C").reward == 1


def test_reasoning_trace_is_ignored():
    result = grade_supergpqa("C", OPTIONS, "Maybe the answer is A.</think>\nAnswer: C")
    assert result.reward == 1
