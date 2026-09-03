# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.coverage_ratio import ProblemAttempts, pass_at_k, tail_sft_coverage_ratio


def test_pass_at_k_applies_the_issue_defined_independent_draw_transform():
    assert pass_at_k(0.0, 16) == 0.0
    assert pass_at_k(1.0, 16) == 1.0
    assert pass_at_k(0.5, 2) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("successes", "attempts"),
    [(-1, 16), (17, 16), (0, 0)],
)
def test_problem_attempts_rejects_invalid_counts(successes, attempts):
    with pytest.raises(ValueError):
        ProblemAttempts(successes=successes, attempts=attempts)


def test_coverage_ratio_reports_loss_gain_ratio_and_mean_change():
    base = {
        "loss": ProblemAttempts(2, 4),
        "gain": ProblemAttempts(2, 4),
        "never": ProblemAttempts(0, 4),
        "always": ProblemAttempts(4, 4),
    }
    before = {
        "loss": ProblemAttempts(3, 4),
        "gain": ProblemAttempts(1, 4),
        "never": ProblemAttempts(0, 4),
        "always": ProblemAttempts(4, 4),
    }
    after = {
        "loss": ProblemAttempts(1, 4),
        "gain": ProblemAttempts(2, 4),
        "never": ProblemAttempts(0, 4),
        "always": ProblemAttempts(4, 4),
    }

    result = tail_sft_coverage_ratio(base, before, after, k=2, expected_attempts=4)

    assert result.reachable_problem_ids == ("gain", "loss")
    assert result.loss == pytest.approx(0.5)
    assert result.gain == pytest.approx(0.3125)
    assert result.ratio == pytest.approx(1.6)
    assert result.mean_pass_at_k_change == pytest.approx(-0.09375)


def test_coverage_ratio_uses_open_reachable_bounds():
    base = {
        "lower": ProblemAttempts(1, 4),
        "middle": ProblemAttempts(2, 4),
        "upper": ProblemAttempts(3, 4),
    }

    result = tail_sft_coverage_ratio(
        base,
        base,
        base,
        k=1,
        expected_attempts=4,
        reachable_lower=0.25,
        reachable_upper=0.75,
    )

    assert result.reachable_problem_ids == ("middle",)


def test_coverage_ratio_rejects_a_mixed_attempt_budget():
    base = {"p": ProblemAttempts(8, 16)}
    before = {"p": ProblemAttempts(7, 15)}
    after = {"p": ProblemAttempts(8, 16)}

    with pytest.raises(ValueError, match="before problems do not have exactly 16 attempts"):
        tail_sft_coverage_ratio(base, before, after)


def test_coverage_ratio_rejects_missing_reachable_problems():
    base = {"p": ProblemAttempts(1, 16)}

    with pytest.raises(ValueError, match=r"before=\['p'\]"):
        tail_sft_coverage_ratio(base, {}, base)


def test_coverage_ratio_marks_zero_gain_as_undefined():
    base = {"p": ProblemAttempts(1, 16)}
    after = {"p": ProblemAttempts(0, 16)}

    result = tail_sft_coverage_ratio(base, base, after)

    assert result.loss > 0.0
    assert result.gain == 0.0
    assert result.ratio is None


def test_coverage_ratio_reports_an_empty_reachable_set():
    base = {"never": ProblemAttempts(0, 16), "always": ProblemAttempts(16, 16)}

    result = tail_sft_coverage_ratio(base, base, base)

    assert result.reachable_problem_ids == ()
    assert result.loss == 0.0
    assert result.gain == 0.0
    assert result.ratio is None
    assert result.mean_pass_at_k_change is None
