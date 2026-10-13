# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Typed source extraction stays separate from actual primitive answer grading."""

import pytest
from verifyit.adapters.harness_mmmu import grade_mmmu_mcq, grade_mmmu_open
from verifyit.grade import InvalidTask


def test_choice_alternatives_and_missing_extraction():
    assert grade_mmmu_mcq(["apple", "pear"], ["A", "B"], "B").reward == 1.0
    assert grade_mmmu_mcq(["apple", "pear"], "A", None).reward == 0.0
    assert grade_mmmu_mcq(["apple"], "A", "A").reward == 1.0


@pytest.mark.parametrize("gold", ["Z", "", "AB", [], [None], " A "])
def test_invalid_gold_cannot_pass_missing_choice_path(gold):
    with pytest.raises(InvalidTask):
        grade_mmmu_mcq(["apple", "pear"], gold, None)


def test_prepared_open_floats_use_zero_tolerance_and_text_uses_substring():
    assert grade_mmmu_open([1.23], [1.23]).reward == 1.0
    assert grade_mmmu_open([1.23], [1.24]).reward == 0.0
    assert grade_mmmu_open([0.0], [-0.0]).reward == 1.0
    assert grade_mmmu_open([0.2 + 0.1], [0.3]).reward == 0.0
    assert grade_mmmu_open(["cat"], ["caterpillar"]).reward == 1.0
    assert grade_mmmu_open([" cat", "cat "], ["catfish"]).reward == 0.0


def test_one_character_source_padding_is_not_stripped():
    assert grade_mmmu_open([" a", "a "], ["road"]).reward == 0.0
    assert grade_mmmu_open([" a", "a "], [" a", "a "]).reward == 1.0


def test_any_nonfinite_candidate_penalizes_whole_response():
    assert grade_mmmu_open([1.0], [float("inf"), 1.0]).reward == 0.0
    assert grade_mmmu_open([1.0], [float("nan")]).reward == 0.0


@pytest.mark.parametrize("references", [[""], ["  "], [float("inf")], [float("nan")], [], [True]])
def test_nonfinite_or_vacuous_reference_is_invalid(references):
    with pytest.raises(InvalidTask):
        grade_mmmu_open(references, [])


def test_invalid_source_parser_result_is_infrastructure_failure():
    with pytest.raises(RuntimeError, match="parser returned"):
        grade_mmmu_open(["cat"], [None])
