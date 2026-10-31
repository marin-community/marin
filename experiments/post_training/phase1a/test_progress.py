# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.phase1a.progress import (
    ProgressPoint,
    confirmed_crossings,
    interference_at_crossing,
    metrics_at_step,
    seed_bootstrap_interval,
)


def test_crossings_reject_overshoot_and_retain_first_and_last_confirmed_matches():
    # The first crossing overshoots; later crossings have the required confirmation.
    points = [
        ProgressPoint(step, progress, {"drift": drift})
        for step, progress, drift in [
            (0, 0, 0),
            (2, 0.6, 2),
            (4, 0.8, 4),
            (6, 0.4, 6),
            (8, 0.6, 8),
            (10, 0.54, 10),
            (30, 0.3, 30),
            (50, 0.7, 50),
            (100, 0.5, 100),
        ]
    ]
    hits = confirmed_crossings(points, 0.5)
    assert [(hit.lower_step, hit.upper_step, hit.confirmation_step) for hit in hits] == [(6, 8, 10), (30, 50, 100)]
    assert hits[0].metrics["drift"] == pytest.approx(7)
    assert hits[-1].metrics["drift"] == pytest.approx(40)


@pytest.mark.parametrize("confirmation,expected", [(0.45, 1), (0.55, 1), (0.449, 0), (0.551, 0)])
def test_crossing_confirmation_uses_absolute_five_point_distance(confirmation, expected):
    points = [ProgressPoint(0, 0.4, {}), ProgressPoint(2, 0.5, {}), ProgressPoint(4, confirmation, {})]
    assert len(confirmed_crossings(points, 0.5)) == expected


def test_terminal_crossing_is_unconfirmed_and_unknown_baseline_metrics_stay_missing():
    points = [
        ProgressPoint(0, 0, {"drift": 0}),
        ProgressPoint(2, 0.6, {"drift": 12, "task_fidelity": 3}),
        ProgressPoint(4, 0.51, {"drift": 14, "task_fidelity": 2}),
    ]
    assert confirmed_crossings(points[:2], 0.5) == []
    (hit,) = confirmed_crossings(points, 0.5)
    assert hit.metrics == {"drift": pytest.approx(10)}


def test_interference_matches_the_single_task_budget_in_both_kl_forms():
    single = [
        ProgressPoint(0, 0, {"forward": 1, "reverse": 2}),
        ProgressPoint(10, 0.6, {"forward": 0.4, "reverse": 0.8}),
        ProgressPoint(30, 0.52, {"forward": 0.48, "reverse": 0.96}),
    ]
    joint = [
        ProgressPoint(0, 0, {"forward": 1, "reverse": 2}),
        ProgressPoint(10, 0.3, {"forward": 0.7, "reverse": 1.4}),
        ProgressPoint(30, 0.8, {"forward": 0.2, "reverse": 0.4}),
    ]
    (crossing,) = confirmed_crossings(single, 0.5)
    result = interference_at_crossing(crossing, joint, single, {"forward": 1, "reverse": 2})
    assert result["forward"].difference == pytest.approx(0.25)
    assert result["reverse"].difference == pytest.approx(0.5)
    assert result["forward"].gain_fraction == pytest.approx(0.5)
    assert result["reverse"].gain_fraction == pytest.approx(0.5)


def test_step_interpolation_does_not_invent_baseline_metrics_or_extrapolate():
    points = [ProgressPoint(0, 0, {}), ProgressPoint(10, 0.6, {"flips": 0.2})]
    assert metrics_at_step(points, 5) == {}
    assert metrics_at_step(points, 10) == {"flips": 0.2}
    with pytest.raises(ValueError, match="outside"):
        metrics_at_step(points, 11)


def test_seed_bootstrap_interval_enumerates_seed_draws_and_preserves_paired_contrasts():
    interval = seed_bootstrap_interval([0, 2])
    assert interval.mean == 1
    assert interval.lower == pytest.approx(0.075)
    assert interval.upper == pytest.approx(1.925)
    sft, opd = [1, 10, 100], [0.5, 9.5, 99.5]
    paired = seed_bootstrap_interval([right - left for left, right in zip(sft, opd, strict=True)])
    assert paired.mean == paired.lower == paired.upper == -0.5
