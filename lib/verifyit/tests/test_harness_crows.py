# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Source preference metrics preserve strict ties and unbounded likelihood differences."""

from types import SimpleNamespace

import pytest
from verifyit.adapters.harness_crows import crows_mean, crows_metrics, validate_crows_output, validate_crows_task
from verifyit.grade import InvalidTask


@pytest.mark.parametrize(
    "responses,expected",
    [
        ([(-1.0, True), (-2.0, False)], {"pct_stereotype": 1.0, "likelihood_diff": 1.0}),
        ([(-2.0, False), (-1.0, True)], {"pct_stereotype": 0.0, "likelihood_diff": 1.0}),
        ([(-2.0, False), (-2.0, True)], {"pct_stereotype": 0.0, "likelihood_diff": 0.0}),
        ([(-1e308, False), (-1.0, True)], {"pct_stereotype": 0.0, "likelihood_diff": 1e308}),
    ],
)
def test_source_preference_and_difference_domains(responses, expected):
    assert crows_metrics(["stereotype", "alternative"], responses) == expected


@pytest.mark.parametrize(
    "choices,responses",
    [
        (["", "alternative"], [(-1.0, True), (-2.0, False)]),
        (["stereotype"], [(-1.0, True), (-2.0, False)]),
        (["stereotype", "alternative"], [(-1.0, True)]),
        (["stereotype", "alternative"], [(float("nan"), True), (-2.0, False)]),
        (["stereotype", "alternative"], [(float("-inf"), True), (-2.0, False)]),
        (["stereotype", "alternative"], [(1e308, True), (-1e308, False)]),
        (["stereotype", "alternative"], [(True, True), (-2.0, False)]),
        (["stereotype", "alternative"], [(-1.0, 1), (-2.0, False)]),
        (["stereotype", "alternative"], [(10**1000, True), (-2.0, False)]),
    ],
)
def test_bad_source_reference_or_likelihood_is_unscored(choices, responses):
    with pytest.raises(InvalidTask):
        crows_metrics(choices, responses)


def test_no_custom_scorer_does_not_require_other_task_fields():
    assert validate_crows_task(SimpleNamespace(config=SimpleNamespace(process_results=None))) is False


def test_source_mean_overflow_aborts_the_batch():
    with pytest.raises(InvalidTask, match="nonfinite"):
        crows_mean([1e308, 1e308])
    assert crows_mean([1.0, 0.0, 999.0]) == 1000.0 / 3


def test_nonfinite_stderr_is_not_exported():
    with pytest.raises(InvalidTask, match="nonfinite"):
        validate_crows_output(float("nan"))
