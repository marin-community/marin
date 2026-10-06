# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from verifyit.candidate import grade_text_candidate
from verifyit.grade import InvalidTask, Status, negative_candidate
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_math
from verifyit.numeric import NumericCandidateError
from verifyit.spec import NumericSpec, parse_spec, render_spec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "expected,text,reward",
    [
        ("42", "42", 1),
        ("42", "The answer is 42.\n", 1),
        ("42", "My calculation gives 42.", 1),
        ("42", "43", 0),
        ("-1234.5", "Answer: -1,234.5", 1),
        ("6.02e23", "6.02e23", 1),
        ("0.5", ".5", 1),
        ("12", "First I guessed 7, then 9.\nThe answer is 12", 1),
        ("42", r"\boxed{42}" + "\nchecked against 999 samples", 1),
        ("999", r"\boxed{42}" + "\nchecked against 999 samples", 0),
        ("1/2", "1/2", 1),
        ("1/2", "The result is 1 / 2.", 1),
        ("1/2", "0.5", 1),
        ("2", "1/2", 0),
        ("1000", "1,000", 1),
        ("3/10", "0.3", 1),
        ("3/10", "Approximately 3e-1 units", 1),
        ("9007199254740993", "9007199254740993", 1),
        ("9007199254740993", "9007199254740992", 0),
        (str(10**400), str(10**400), 1),
        (str(10**400), str(10**400 + 1), 0),
    ],
)
def test_numeric_file_and_candidate_paths_grade_the_same_exact_value(tmp_path, expected, text, reward):
    spec = parse_spec(render_spec(NumericSpec(expected, tolerance_abs=0.0, tolerance_rel=0.0)))
    _answer(tmp_path, text)
    direct = grade_text_candidate(spec, text)
    file_result = grade_math.grade(spec, tmp_path, tmp_path)
    assert (direct.status, direct.reward) == (Status.SCORED, reward)
    assert file_result == direct
    json.dumps(direct.detail, allow_nan=False)


@pytest.mark.parametrize(
    "spec,text,reward",
    [
        (NumericSpec("3.14159", tolerance_abs=0.0, tolerance_rel=0.0), "3.1416", 0),
        (NumericSpec("3.14159", tolerance_abs=1e-3, tolerance_rel=0.0), "3.1416", 1),
        (NumericSpec("3.14159", tolerance_abs=1e-3, tolerance_rel=0.0), "3.2", 0),
        (NumericSpec("1e6", tolerance_abs=0.0, tolerance_rel=1e-5), "1000001", 1),
        (NumericSpec("1e6", tolerance_abs=0.0, tolerance_rel=1e-5), "1000100", 0),
        (NumericSpec("0.3", tolerance_abs=0.01, tolerance_rel=0.0), "0.31", 1),
        (NumericSpec("0.3", tolerance_abs=0.01, tolerance_rel=0.0), "0.31000000000000001", 0),
        (NumericSpec("1/3", tolerance_abs=0.0003333333333333334, tolerance_rel=0.0), "0.333", 1),
        (NumericSpec("1/3", tolerance_abs=0.0003333333333333333, tolerance_rel=0.0), "0.333", 0),
        (NumericSpec("1/3", tolerance_abs=0.0, tolerance_rel=0.0), "0.333", 0),
        (NumericSpec("1e3000", tolerance_abs=0.0, tolerance_rel=1.0), "0", 1),
    ],
)
def test_numeric_explicit_tolerances_bound_exact_differences(tmp_path, spec, text, reward):
    _answer(tmp_path, text)
    assert grade_math.grade(spec, tmp_path, tmp_path).reward == reward
    assert grade_text_candidate(spec, text).reward == reward
    json.dumps(grade_text_candidate(spec, text).detail, allow_nan=False)


@pytest.mark.parametrize(
    "text",
    [
        "not a number",
        "",
        "  \n",
        "12 or 13",
        "2 + 2",
        "1,00",
        "Result: 1efoo",
        "Result: value42",
        "1/0",
        "nan",
        "inf",
        "0.2+0.1",
        "1e999999999",
        r"\boxed{42}" + "\n" + r"\boxed{",
        r"\boxed{42}" + "\n" + r"\boxed{}",
    ],
)
def test_malformed_numeric_submission_cannot_expose_a_partial_value(tmp_path, text):
    spec = NumericSpec("42", tolerance_abs=0.0, tolerance_rel=0.0)
    _answer(tmp_path, text)
    result = grade_math.grade(spec, tmp_path, tmp_path)
    assert (result.status, result.reward) == (Status.SCORED, 0)
    with pytest.raises(NumericCandidateError):
        grade_text_candidate(spec, text)


def test_numeric_negative_candidate_exceeds_the_configured_tolerance(tmp_path):
    spec = NumericSpec("42", tolerance_abs=2.0, tolerance_rel=0.0)
    candidate = negative_candidate(spec)
    assert candidate is not None
    _answer(tmp_path, candidate)
    assert grade_math.grade(spec, tmp_path, tmp_path).reward == 0


@pytest.mark.parametrize(
    "expected,absolute,relative", [("nan", 0.0, 0.0), ("1/0", 0.0, 0.0), ("42", -1.0, 0.0), ("42", 0.0, float("inf"))]
)
def test_invalid_private_numeric_contract_is_checked_before_malformed_candidate(tmp_path, expected, absolute, relative):
    spec = NumericSpec(expected, tolerance_abs=absolute, tolerance_rel=relative)
    _answer(tmp_path, "not a number")
    assert dispatch(spec, tmp_path, tmp_path).status == Status.INVALID_TASK
    with pytest.raises(InvalidTask):
        grade_text_candidate(spec, "not a number")


@pytest.mark.parametrize(
    "expected,candidate,tolerance,reward",
    [
        (0.3, 0.2 + 0.1, 0.0, 0),
        (0.3, 0.2 + 0.1, 1e-15, 1),
        (1.0, 1.01, 0.01, 0),
        (0.01, 0.02, 0.01, 1),
        (1e308, -1e308, 0.01, 0),
    ],
)
def test_float_numeric_grader_preserves_binary_subtraction(expected, candidate, tolerance, reward):
    result = grade_math.grade_numeric_candidate_float(expected, candidate, tolerance_abs=tolerance)
    assert (result.status, result.reward) == (Status.SCORED, reward)
    assert result.detail == {"extracted": candidate, "expected": expected}


def test_regression_multioutput_fraction_matches_exact_reference():
    # Per-output normalized squared errors are 1 and 1/2; mean R² is 1/4.
    reward = grade_math.grade_regression_candidate([[[1, 10], [3, 14]]], [[[2, 10], [2, 12]]], variance_floor=0)
    assert (reward.status, reward.reward) == (Status.SCORED, 0.25)
    assert reward.detail == {"nmse": 0.75, "nmae": 0.75, "r2": 0.25}


def test_regression_negative_fit_is_zero_with_raw_metric():
    reward = grade_math.grade_regression_candidate([[[0], [2]]], [[[10], [10]]], variance_floor=0)
    assert (reward.status, reward.reward, reward.detail["r2"]) == (Status.SCORED, 0, -81)


def test_regression_variance_floor_is_explicit_and_constant_reference_is_valid():
    truth = [[0], [0.00001]]
    assert grade_math.grade_regression_candidate([truth], [truth], variance_floor=0).reward == 1
    assert grade_math.grade_regression_candidate([truth], [truth], variance_floor=1e-9).reward == 0
    result = grade_math.grade_regression_candidate([[[1], [1]]], [[[1], [1]]], variance_floor=0)
    assert (result.status, result.reward) == (Status.SCORED, 0)


@pytest.mark.parametrize("bad", [[], [[]], [[1], [2, 3]], [[float("nan")]], [[float("inf")]], [[True]]])
def test_regression_validates_reference_before_candidate(bad):
    reference = grade_math.grade_regression_candidate([bad], None, variance_floor=0)
    assert (reference.status, reference.reward) == (Status.INVALID_TASK, 0)
    candidate = grade_math.grade_regression_candidate([[[1], [2]]], [bad], variance_floor=0)
    assert (candidate.status, candidate.reward) == (Status.SCORED, 0)


def test_regression_constant_output_keeps_multioutput_reward_at_floor():
    # Source NMSE uses a large sentinel for the constant column, so even a
    # perfect second output cannot turn the aggregate into positive credit.
    truth = [[1, 0], [1, 2]]
    reward = grade_math.grade_regression_candidate([truth], [truth], variance_floor=1e-9)
    assert (reward.status, reward.reward) == (Status.SCORED, 0)


def test_regression_cannot_hide_missing_group_rows_with_extra_rows_elsewhere():
    truth = [[[0], [1]], [[2], [3]]]
    # Flattened predictions are exactly correct, but came from the wrong groups.
    shifted = [[[0]], [[1], [2], [3]]]
    result = grade_math.grade_regression_candidate(truth, shifted, variance_floor=0)
    assert (result.status, result.reward) == (Status.SCORED, 0)
    assert grade_math.grade_regression_candidate(truth, truth, variance_floor=0).reward == 1
