# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from verifyit.grade import Status
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_exact
from verifyit.modes.grade_puzzle import puzzle_spec
from verifyit.spec import ExactSpec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "spec, text, reward",
    [
        (ExactSpec(expected=("Paris",)), "  Paris \n", 1.0),
        (ExactSpec(expected=("Paris",)), "paris\n", 1.0),
        (ExactSpec(expected=("Paris",), ignore_case=False), "paris\n", 0.0),
        (ExactSpec(expected=("Paris",)), "Lyon\n", 0.0),
        (ExactSpec(expected=("hello world",)), "hello    world\n", 1.0),
        (ExactSpec(expected=("hello world",), ignore_whitespace=False), "hello    world\n", 0.0),
        # A trailing newline never decides the reward, even with whitespace significant.
        (ExactSpec(expected=("hello world",), ignore_whitespace=False), "hello world\n", 1.0),
        # A boxed answer is compared on its own, so surrounding prose does not spoil it.
        (ExactSpec(expected=("Paris",)), "The capital is \\boxed{Paris} of course.\n", 1.0),
        (ExactSpec(expected=("Paris",)), "The capital is \\boxed{Lyon} of course.\n", 0.0),
        (ExactSpec(expected=("Paris",)), "The capital is Paris of course.\n", 0.0),
    ],
)
def test_exact_single_expected_compares_the_whole_candidate(tmp_path, spec, text, reward):
    _answer(tmp_path, text)
    assert grade_exact.grade(spec, tmp_path, tmp_path).reward == reward


@pytest.mark.parametrize(
    "spec, text, reward",
    [
        (ExactSpec(expected=("red", "green", "blue")), "red, green, blue\n", 1.0),
        (ExactSpec(expected=("red", "green", "blue")), "red\ngreen\nblue\n", 1.0),
        (ExactSpec(expected=("red", "green", "blue")), "blue, green, red\n", 0.0),
        (ExactSpec(expected=("red", "green", "blue"), ordered=False), "blue, green, red\n", 1.0),
        (ExactSpec(expected=("red", "green", "blue")), "red, green\n", 0.0),
        (ExactSpec(expected=("red", "green", "blue")), "red, green, blue, yellow\n", 0.0),
        # A multiset counts repeats: two "a" are not the same answer as one.
        (ExactSpec(expected=("a", "a", "b"), ordered=False), "a, b, a\n", 1.0),
        (ExactSpec(expected=("a", "a", "b"), ordered=False), "a, b, b\n", 0.0),
        (ExactSpec(expected=("red", "green")), "\\boxed{red, green}\n", 1.0),
    ],
)
def test_exact_several_expected_compares_the_candidate_as_a_list(tmp_path, spec, text, reward):
    _answer(tmp_path, text)
    assert grade_exact.grade(spec, tmp_path, tmp_path).reward == reward


def test_exact_empty_output_scores_zero_with_no_output(tmp_path):
    _answer(tmp_path, "\n \n")
    result = grade_exact.grade(ExactSpec(expected=("Paris",)), tmp_path, tmp_path)
    assert (result.status, result.reward, result.detail["reason"]) == (Status.SCORED, 0.0, "no_output")


def test_exact_reward_detail_carries_the_extracted_candidate(tmp_path):
    _answer(tmp_path, "The capital is \\boxed{Lyon}.\n")
    detail = grade_exact.grade(ExactSpec(expected=("Paris",)), tmp_path, tmp_path).detail
    assert detail == {"extracted": "Lyon", "expected": ["Paris"]}


def test_exact_without_an_expected_string_is_an_invalid_task(tmp_path):
    _answer(tmp_path, "Paris\n")
    assert dispatch(ExactSpec(expected=()), tmp_path, tmp_path).status == Status.INVALID_TASK


def test_puzzle_ordered_list_accepts_line_breaks_but_rejects_reordering(tmp_path):
    spec = puzzle_spec("Defect, Salt, chair, donate", "ordered_list")
    _answer(tmp_path, "Defect\nSalt\nchair\ndonate")
    assert dispatch(spec, tmp_path, tmp_path).reward == 1.0
    _answer(tmp_path, "chair, Defect, donate, Salt")
    assert dispatch(spec, tmp_path, tmp_path).reward == 0.0
