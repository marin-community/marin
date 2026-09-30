# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from pathlib import Path

import pytest

pytest.importorskip("math_verify", reason="math mode needs the `answer` extra")

from tasktrove_verify.grade import InvalidTask, Status
from tasktrove_verify.grade import grade as dispatch
from tasktrove_verify.modes import grade_math
from tasktrove_verify.spec import MathSpec, MathType


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "expected, math_type, text, reward",
    [
        ("0.5", MathType.SCALAR, "So the area is one half.\n\\boxed{\\frac{1}{2}}\n", 1.0),
        ("0.5", MathType.SCALAR, "\\boxed{1/2}", 1.0),
        ("0.5", MathType.SCALAR, "\\boxed{0.5}", 1.0),
        ("0.5", MathType.SCALAR, "\\boxed{2}", 0.0),
        ("2\\sqrt{3}", MathType.SCALAR, "\\boxed{2\\sqrt 3}", 1.0),
        ("2\\sqrt{3}", MathType.SCALAR, "\\boxed{3\\sqrt{2}}", 0.0),
        ("x^2+1", MathType.SCALAR, "\\boxed{1 + x^2}", 1.0),
        # No boxed expression: the last line is the answer.
        ("42", MathType.SCALAR, "First I add the parts.\nThe answer is 42\n", 1.0),
        ("42", MathType.SCALAR, "First I add the parts.\nThe answer is 41\n", 0.0),
        # The last box wins, as a model that revises itself boxes twice.
        ("42", MathType.SCALAR, "\\boxed{7}\nthat was wrong, actually \\boxed{42}\n", 1.0),
        # A malformed final box cannot expose an earlier answer to the parser.
        ("42", MathType.SCALAR, "\\boxed{42}\nthat was wrong, actually \\boxed{\n", 0.0),
        ("42", MathType.SCALAR, "\\boxed{42}\nthat was wrong, actually \\boxed{}\n", 0.0),
        ("42", MathType.SCALAR, "$\\boxed{42}$", 1.0),
        # An unreadable candidate is a scored wrong answer.
        ("42", MathType.SCALAR, "\\boxed{???}", 0.0),
        ("42", MathType.SCALAR, "I have no idea how to do this problem.\n", 0.0),
        ("\\{1,2,3\\}", MathType.SET, "\\boxed{\\{3,2,1\\}}", 1.0),
        ("\\{1,2,3\\}", MathType.SET, "\\boxed{\\{1,2\\}}", 0.0),
        ("(1,3]", MathType.INTERVAL, "\\boxed{(1,3]}", 1.0),
        ("(1,3]", MathType.INTERVAL, "\\boxed{[1,3]}", 0.0),
        # The set/relation comparison the interval and set types enable.
        ("(2,\\infty)", MathType.INTERVAL, "\\boxed{x > 2}", 1.0),
        ("(2,\\infty)", MathType.INTERVAL, "\\boxed{x < 2}", 0.0),
        ("y = 2x + 1", MathType.EQUATION, "\\boxed{y=2x+1}", 1.0),
        ("y = 2x + 1", MathType.EQUATION, "\\boxed{y=2x+2}", 0.0),
        ("(1, 2)", MathType.TUPLE, "\\boxed{(1,2)}", 1.0),
        ("(1, 2)", MathType.TUPLE, "\\boxed{(2,1)}", 0.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{[1,2,3]}", 1.0),
        # A list is ordered, and the brackets around it are optional.
        ("[1, 2, 3]", MathType.LIST, "\\boxed{3, 2, 1}", 0.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{1, 2, 3}", 1.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{\\left[1, 2, 3\\right]}", 1.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{[1, 2]}", 0.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{[1, 2, 3, 4]}", 0.0),
        # Member comparison uses expression equality.
        ("[1/2, x+1]", MathType.LIST, "\\boxed{[0.5, 1+x]}", 1.0),
        # A comma inside a member does not split it.
        ("[(1,2), 3]", MathType.LIST, "\\boxed{[(1,2), 3]}", 1.0),
        ("[(1,2), 3]", MathType.LIST, "\\boxed{[(2,1), 3]}", 0.0),
    ],
)
def test_math_answers_are_compared_symbolically(tmp_path, expected, math_type, text, reward):
    _answer(tmp_path, text)
    spec = MathSpec(expected=expected, math_type=math_type)
    candidate_result = grade_math.grade_math_candidate(spec, text)
    assert candidate_result.reward == reward
    assert grade_math.grade(spec, tmp_path, tmp_path) == candidate_result


def test_math_empty_output_scores_zero_with_no_output(tmp_path):
    _answer(tmp_path, "\n  \n")
    result = grade_math.grade(MathSpec(expected="42"), tmp_path, tmp_path)
    assert (result.reward, result.detail["reason"]) == (0.0, "no_output")


def test_math_reward_detail_carries_the_extracted_expression(tmp_path):
    _answer(tmp_path, "after simplifying, \\boxed{\\frac{3}{4}}\n")
    detail = grade_math.grade(MathSpec(expected="0.75"), tmp_path, tmp_path).detail
    assert detail["extracted"] == "\\frac{3}{4}"


@pytest.mark.parametrize("expected", ["", "   "])
def test_math_unparsable_expected_is_an_invalid_task(tmp_path, expected):
    _answer(tmp_path, "\\boxed{42}")
    assert dispatch(MathSpec(expected=expected), tmp_path, tmp_path).status == Status.INVALID_TASK


def test_math_grades_from_a_worker_thread(tmp_path):
    """Pipelines grade in worker threads, where math-verify's signal-based timeout cannot be armed."""
    _answer(tmp_path, "\\boxed{\\frac{1}{2}}")
    results: list = []
    worker = threading.Thread(
        target=lambda: results.append(grade_math.grade(MathSpec(expected="0.5"), tmp_path, tmp_path))
    )
    worker.start()
    worker.join()
    assert results[0].status == Status.SCORED and results[0].reward == 1.0


@pytest.mark.parametrize("candidate", [None, "42", "???"])
def test_invalid_reference_is_not_a_wrong_candidate(candidate):
    with pytest.raises(InvalidTask):
        grade_math.grade_math_candidate(MathSpec(expected="???"), candidate)


@pytest.mark.parametrize(
    "expected, math_type, numeric",
    [
        ("42", MathType.SCALAR, True),
        ("-0.5", MathType.SCALAR, True),
        ("1/2", MathType.SCALAR, True),
        (r"2\sqrt{3}", MathType.SCALAR, True),
        (r"\pi", MathType.SCALAR, True),
        ("(1,2)", MathType.SCALAR, False),
        ("1,2", MathType.SCALAR, False),
        ("[1,2]", MathType.SCALAR, False),
        (r"\{1,2\}", MathType.SCALAR, False),
        ("1 or 2", MathType.SCALAR, False),
        ("1 or ???", MathType.SCALAR, False),
        ("42 ???", MathType.SCALAR, False),
        ("42,", MathType.SCALAR, False),
        ("2 +/- 1", MathType.SCALAR, False),
        ("x+1", MathType.SCALAR, False),
        ("x=2", MathType.SCALAR, False),
        (r"\sqrt{-1}", MathType.SCALAR, False),
        (r"\infty", MathType.SCALAR, False),
        ("1/0", MathType.SCALAR, False),
        ("???", MathType.SCALAR, False),
        ("", MathType.SCALAR, False),
        ("42", MathType.LIST, False),
        ("42", MathType.SET, False),
        ("42", MathType.INTERVAL, False),
        ("42", MathType.TUPLE, False),
        ("42", MathType.EQUATION, False),
    ],
)
def test_finite_real_scalar_classification(expected, math_type, numeric):
    assert grade_math.is_finite_real_scalar(MathSpec(expected=expected, math_type=math_type)) is numeric
