# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.phase1a.progress import ProgressPoint, confirmed_crossings


def test_crossings_reject_overshoot_and_retain_first_and_last_confirmed_matches():
    # The first crossing overshoots; later crossings have the required confirmation.
    points = [
        ProgressPoint(step, progress, {"drift": drift})
        for step, progress, drift in [
            (0, 0, 0), (2, 0.6, 2), (4, 0.8, 4), (6, 0.4, 6),
            (8, 0.6, 8), (10, 0.54, 10), (30, 0.3, 30), (50, 0.7, 50), (100, 0.5, 100),
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
    hit, = confirmed_crossings(points, 0.5)
    assert hit.metrics == {"drift": pytest.approx(10)}
