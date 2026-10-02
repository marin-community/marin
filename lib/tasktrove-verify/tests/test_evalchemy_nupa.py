# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
from tasktrove_verify.adapters.evalchemy_nupa import grade_nupa_answer
from tasktrove_verify.grade import Status


def full_answer(value, answer_format):
    patterns = {
        "Integer": r"\d+",
        "Float": r"\d+\.\d+",
        "Fraction": r"\d+/\d+",
        "ScientificNotation": r"\d+\.\d+[eE][+-]?\d+",
    }
    return value if re.fullmatch(patterns[answer_format], value) else None


def digit_parts(value, answer_format):
    separators = {"Integer": (), "Float": (".",), "Fraction": ("/",), "ScientificNotation": (".", "e")}
    parts = [value]
    for separator in separators[answer_format]:
        parts = [piece for part in parts for piece in part.lower().split(separator, 1)]
    if len(parts) != len(separators[answer_format]) + 1:
        return tuple("" for _ in range(len(separators[answer_format]) + 1))
    return tuple("".join(char for char in part if char.isdigit()) for part in parts)


def grade(candidate, expected, answer_format):
    return grade_nupa_answer(
        candidate,
        expected,
        answer_format,
        extract_answer=lambda value, fmt: full_answer(value, fmt) if isinstance(value, str) else None,
        full_answer=full_answer,
        digit_parts=digit_parts,
    )


@pytest.mark.parametrize(
    "candidate,expected,answer_format,digit_match,length_difference",
    [
        ("23", "123", "Integer", 2 / 3, 1),
        ("2.340", "12.345", "Float", 3 / 5, 1),
        ("12/3", "12/43", "Fraction", 3 / 4, 1),
        ("1.23e5", "1.23e45", "ScientificNotation", 4 / 5, 1),
    ],
)
def test_component_alignment_preserves_partial_digit_credit(
    candidate, expected, answer_format, digit_match, length_difference
):
    verdict = grade(candidate, expected, answer_format)
    assert verdict.status is Status.SCORED
    assert verdict.reward == 0
    assert verdict.detail["metrics"] == {
        "exact_match": 0.0,
        "digit_match": digit_match,
        "dlength": float(length_difference),
        "format_valid": 1.0,
        "no_answer": 0.0,
    }


def test_representation_equality_does_not_coerce_numeric_equivalents():
    assert grade("1.00", "1.0", "Float").reward == 0
    assert grade("1.0", "1.0", "Float").reward == 1


def test_source_component_metric_omits_scientific_exponent_sign():
    verdict = grade("1.23e-5", "1.23e5", "ScientificNotation")
    assert verdict.reward == 1
    assert verdict.detail["metrics"]["digit_match"] == 1


@pytest.mark.parametrize("candidate", [None, "missing", {"answer": "123"}])
def test_missing_answer_keeps_reference_length_and_zero_credit(candidate):
    verdict = grade(candidate, "123", "Integer")
    assert verdict.reward == 0
    assert verdict.detail["metrics"] == {
        "exact_match": 0.0,
        "digit_match": 0.0,
        "dlength": 3.0,
        "format_valid": 0.0,
        "no_answer": 1.0,
    }


@pytest.mark.parametrize("expected", ["", "abc123", "nan", "inf", None])
def test_malformed_reference_cannot_be_reduced_to_matching_digits(expected):
    verdict = grade("123", expected, "Integer")
    assert verdict.status is Status.INVALID_TASK
    assert verdict.reward == 0
