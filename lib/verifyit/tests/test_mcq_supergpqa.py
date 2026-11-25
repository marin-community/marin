# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import signal

import pytest
from verifyit import supergpqa_extraction
from verifyit.grade import Status
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_mcq
from verifyit.spec import McqExtraction, McqSpec

OPTIONS = ("4", "6", "7", "9")
SPEC = McqSpec("C", options=4, extraction=McqExtraction.SUPERGPQA_REFERENCE, option_texts=OPTIONS)


# Expected letters are the SuperGPQA runner's eval/eval.py zero-shot extraction on the same responses.
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
        # Without a letter, option text maps back to its letter.
        ("So the answer is 7.", 1, "C"),
        ("The correct option is 7", 1, "C"),
        ("no idea", 0, None),
        ("", 0, None),
    ],
)
def test_supergpqa_reference_extraction(response, reward, extracted):
    result = grade_mcq.grade_mcq_candidate(SPEC, response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted


def test_supergpqa_reference_grades_the_answer_file(tmp_path):
    (tmp_path / "answer.txt").write_text("Maybe the answer is A.</think>\nAnswer: C")
    assert dispatch(SPEC, tmp_path, tmp_path).reward == 1


@pytest.mark.parametrize("run", ["*" * 40, "(" * 40, "\\" * 40])
def test_backtracking_search_times_out_as_a_miss(monkeypatch, run):
    monkeypatch.setattr(supergpqa_extraction, "SUPERGPQA_SEARCH_TIMEOUT", 0.05)
    result = grade_mcq.grade_mcq_candidate(SPEC, run)
    assert result.status == Status.SCORED
    assert result.detail["extracted"] is None
    assert grade_mcq.grade_mcq_candidate(SPEC, run + "\nAnswer: C").reward == 1


def test_search_timeout_restores_the_callers_alarm():
    def caller_handler(_signum, _frame):
        raise AssertionError("caller alarm fired early")

    previous = signal.signal(signal.SIGALRM, caller_handler)
    signal.setitimer(signal.ITIMER_REAL, 30.0)
    try:
        assert grade_mcq.grade_mcq_candidate(SPEC, "Answer: C").reward == 1
        remaining, _interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining <= 30.0
        assert signal.getsignal(signal.SIGALRM) is caller_handler
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
