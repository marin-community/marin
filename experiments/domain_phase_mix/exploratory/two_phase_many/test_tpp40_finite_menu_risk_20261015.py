# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Decision-risk identities independent of any saved scientific outputs."""

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_finite_menu_decisions_20261015 as runner,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_finite_menu_risk_20261015 as risk


@pytest.mark.parametrize("distance", [0.0, 0.02, 2.0, 12.0])
def test_two_actions_reduce_to_gaussian_testing(distance):
    result = risk.finite_menu_risk(np.array([1.0, 1.25]), np.array([1.25, 1.0]), distance)
    expected = 0.25 * ndtr(-distance / 2)
    assert result.bayes_regret == pytest.approx(expected, rel=2e-14, abs=1e-30)
    assert result.overlap_bound == pytest.approx(expected, rel=2e-14, abs=1e-30)


def test_unequal_regrets_use_the_decision_threshold_not_test_threshold():
    first, second, distance = 0.5, 0.125, 1.7
    result = risk.finite_menu_risk(np.array([0.0, first]), np.array([second, 0.0]), distance)
    cutoff = math.log(first / second)
    expected = 0.5 * (first * ndtr(-cutoff / distance - distance / 2) + second * ndtr(cutoff / distance - distance / 2))
    assert result.bayes_regret == pytest.approx(expected, rel=2e-14)
    assert result.bayes_regret > result.overlap_bound


def test_compromise_action_changes_bayes_risk_and_agrees_with_direct_quadrature():
    first = np.array([0.0, 1.0, 0.15])
    second = np.array([1.0, 0.0, 0.15])
    distance = 1.4
    result = risk.finite_menu_risk(first, second, distance)

    def integrand(z):
        density_zero = norm.pdf(z)
        density_one = norm.pdf(z - distance)
        return 0.5 * float(np.min(density_zero * first + density_one * second))

    cuts = [math.log(float(value)) / distance + distance / 2 for value in (Fraction(3, 17), Fraction(17, 3))]
    expected = sum(quad(integrand, lo, hi, epsabs=1e-13)[0] for lo, hi in zip([-12, *cuts], [*cuts, 14], strict=True))
    assert len(result.intervals) == 3
    assert result.bayes_regret == pytest.approx(expected, abs=2e-12)
    assert result.overlap_bound <= result.bayes_regret <= 0.5 * result.compromise_sum_regret


def test_shared_optimum_and_duplicate_actions_are_not_decision_ambiguity():
    result = risk.finite_menu_risk(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]), 0.3)
    assert result.shared_optimum
    assert result.optimal_actions_zero == (0, 1)
    assert result.bayes_regret == 0
    assert result.log_bayes_regret == -math.inf
    assert len(result.intervals) == 1


def test_rational_envelope_matches_every_exact_probe_with_parallel_and_dominated_lines():
    intercepts = list(map(Fraction, [0, 0, 1, 2, 3, 5]))
    slopes = list(map(Fraction, [6, 6, 3, 4, 1, 0]))
    intervals = risk.lower_envelope(intercepts, slopes)
    for interval in intervals:
        probes = [interval.lower + 1] if interval.upper is None else [(interval.lower + interval.upper) / 2]
        for ratio in probes:
            values = [a + ratio * b for a, b in zip(intercepts, slopes, strict=True)]
            assert values[interval.action] == min(values)
    assert intervals[0].action == 0
    assert all(interval.action not in (1, 3) for interval in intervals)


def test_zero_distance_and_extreme_tail_remain_distinct():
    first, second = np.array([0.0, 1.0, 0.2]), np.array([1.0, 0.0, 0.2])
    result = risk.finite_menu_risk(first, second, 0.0)
    assert result.bayes_regret == pytest.approx(0.2)
    tail = risk.finite_menu_risk(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 100)
    assert tail.bayes_regret == 0
    assert math.isfinite(tail.log_bayes_regret)
    assert tail.log_bayes_regret < -1000


def test_near_zero_distance_preserves_a_near_unit_decision_threshold():
    first, second, distance = float(np.nextafter(0.5, 1.0)), 0.5, 1e-16
    result = risk.finite_menu_risk(np.array([0.0, first]), np.array([second, 0.0]), distance)
    cutoff = math.log1p((first - second) / second)
    expected_zero = float(log_ndtr(cutoff / distance + distance / 2))
    expected_one = float(log_ndtr(cutoff / distance - distance / 2))
    assert result.interval_log_probabilities[0] == pytest.approx((expected_zero, expected_one), rel=2e-14)
    assert math.exp(result.interval_log_probabilities[0][0]) > 0.98
    assert result.bayes_regret == pytest.approx(0.25, abs=1e-15)


def test_positive_compromise_interval_below_resolution_does_not_become_zero_mass():
    compromise = float(np.nextafter(0.5, 0.0))
    with pytest.raises(ArithmeticError):
        risk.finite_menu_risk(np.array([0.0, 1.0, compromise]), np.array([1.0, 0.0, compromise]), 1000.0)


@pytest.mark.parametrize("eligible", [True, False])
def test_menu_eligibility_preserves_unresolved_function_and_original_status(eligible):
    head = {
        "pair_id": "invented-pair",
        "source": "invented",
        "world": "native_null",
        "noise": 0.003,
        "outer_fold": 0,
        "candidate": "native",
        "distance": 1.0,
        "bayes_error": float(ndtr(-0.5)),
        "analytic_pair_eligible": eligible,
        "selected_fit_unresolved": True,
        "original_numerical_status": '{"failed_starts":1}',
        "within_class_exclusion": "" if eligible else "shared-shape model misspecification",
    }
    endpoints = pd.DataFrame(
        {
            "role": ["asymmetric"] * 3,
            "source_index": [0, 1, 2],
            "row_id": ["a", "b", "c"],
            "truth": [0.0, 1.0, float(np.nextafter(0.5, 0.0))],
            "predicted": [1.0, 0.0, float(np.nextafter(0.5, 0.0))],
        }
    )
    row, _ = runner.diagnose_menu(head, endpoints, "held_asymmetric")
    assert row["status"] == "finite"
    assert row["finite_pair_bayes_regret"] > 0
    assert (row["within_class_menu_regret_bound"] is not None) == eligible
    assert all(row[key] == value for key, value in head.items())

    failed_head = head | {"distance": 1000.0, "bayes_error": 0.0}
    failure, intervals = runner.diagnose_menu(failed_head, endpoints, "held_asymmetric")
    assert failure["status"] == "numerical_failure"
    assert all(failure[key] == value for key, value in failed_head.items())
    assert failure["prerequisite_heads_path"].endswith("/gaussian_pair_diagnostic/heads.csv")
    assert intervals == []
