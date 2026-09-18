# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from tasktrove_verify.grade import Status, negative_candidate
from tasktrove_verify.grade import grade as dispatch
from tasktrove_verify.modes import grade_math
from tasktrove_verify.spec import NumericSpec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "expected, text, reward",
    [
        (42.0, "42", 1.0),
        (42.0, "The answer is 42.\n", 1.0),
        (42.0, "43", 0.0),
        (-1234.5, "Total: -1,234.5\n", 1.0),
        (6.02e23, "6.02e23 molecules\n", 1.0),
        (0.5, ".5", 1.0),
        # The result the model states last is the one graded.
        (12.0, "First I guessed 7, then 9, but the answer is 12\n", 1.0),
        # A boxed result wins over numbers written after it.
        (42.0, "\\boxed{42}\nchecked against 999 samples\n", 1.0),
        (999.0, "\\boxed{42}\nchecked against 999 samples\n", 0.0),
        # A malformed final box cannot expose an earlier number to the parser.
        (42.0, "\\boxed{42}\nthat was wrong, actually \\boxed{\n", 0.0),
        (42.0, "\\boxed{42}\nthat was wrong, actually \\boxed{}\n", 0.0),
    ],
)
def test_numeric_reads_the_final_number(tmp_path, expected, text, reward):
    _answer(tmp_path, text)
    assert grade_math.grade(NumericSpec(expected=expected), tmp_path, tmp_path).reward == reward


@pytest.mark.parametrize(
    "spec, text, reward",
    [
        (NumericSpec(expected=3.14159), "3.1416", 0.0),
        (NumericSpec(expected=3.14159, tolerance_abs=1e-3), "3.1416", 1.0),
        (NumericSpec(expected=3.14159, tolerance_abs=1e-3), "3.2", 0.0),
        (NumericSpec(expected=1e6, tolerance_rel=1e-5), "1000001", 1.0),
        (NumericSpec(expected=1e6, tolerance_rel=1e-5), "1000100", 0.0),
    ],
)
def test_numeric_tolerances_bound_the_match(tmp_path, spec, text, reward):
    _answer(tmp_path, text)
    assert grade_math.grade(spec, tmp_path, tmp_path).reward == reward


@pytest.mark.parametrize(
    "text, reason",
    [("I could not work it out.\n", "no_number"), ("", "no_output"), ("  \n", "no_output")],
)
def test_numeric_output_without_a_number_scores_zero(tmp_path, text, reason):
    _answer(tmp_path, text)
    result = grade_math.grade(NumericSpec(expected=42.0), tmp_path, tmp_path)
    assert (result.status, result.reward, result.detail["reason"]) == (Status.SCORED, 0.0, reason)


def test_numeric_reward_detail_carries_the_extracted_value(tmp_path):
    _answer(tmp_path, "after rounding, 17.5\n")
    detail = grade_math.grade(NumericSpec(expected=42.0), tmp_path, tmp_path).detail
    assert detail["extracted"] == 17.5


@pytest.mark.parametrize(
    "spec",
    [
        NumericSpec(expected=float("nan")),
        NumericSpec(expected=float("inf")),
        NumericSpec(expected=42.0, tolerance_abs=-1.0),
        NumericSpec(expected=42.0, tolerance_rel=-1.0),
        NumericSpec(expected=42.0, tolerance_abs=float("inf")),
        NumericSpec(expected=1e308, tolerance_rel=1e308),
    ],
)
def test_numeric_invalid_contract_is_an_invalid_task(tmp_path, spec):
    _answer(tmp_path, "42")
    assert dispatch(spec, tmp_path, tmp_path).status == Status.INVALID_TASK


def test_numeric_negative_candidate_exceeds_the_configured_tolerance(tmp_path):
    spec = NumericSpec(expected=42.0, tolerance_abs=2.0)
    candidate = negative_candidate(spec)
    assert candidate is not None
    _answer(tmp_path, candidate)
    assert grade_math.grade(spec, tmp_path, tmp_path).reward == 0.0
