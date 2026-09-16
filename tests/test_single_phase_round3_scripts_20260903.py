# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioural checks for the round-3 development scripts of the single-phase Observatory benchmark."""

import itertools

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round3_grid_rules_20260903 as rules,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round3_proposals_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round3_union_loso_20260903 as union_loso,
)


def test_selection_row_scores_the_predicted_argmin_against_the_measured_best():
    loss = np.array([1.00, 0.98, 0.99, 1.02, 0.97, 1.01])
    guess = np.array([0.90, 0.95, 0.92, 0.99, 0.96, 1.00])
    row = selection.selection_row(loss, guess, tolerance=0.001)
    assert row["regret_at_1"] == pytest.approx(1.00 - 0.97)
    assert row["top5_regret"] == pytest.approx(0.0)
    assert row["frontier_predicted_rank"] == 4.0
    assert row["selected_rank"] == 4.0
    assert row["basin_hit"] == 0.0
    assert row["random_regret_at_1"] == pytest.approx(loss.mean() - loss.min())


def test_dynamic_programme_matches_brute_force_with_box_and_cap(monkeypatch):
    monkeypatch.setattr(proposals, "BLOCKS", 6)
    rng = np.random.default_rng(0)
    curves = tuple(
        proposals.ComponentCurve(
            weight=0.5,
            intercept=1.0,
            benefit=rng.uniform(0.0, 0.2, 3),
            harm=rng.uniform(0.0, 0.05, 3),
            shape={"rate": 0.5, "power": 0.7, "threshold": 1.0},
        )
        for _ in range(2)
    )
    inventory = np.array([3.0, 12.0, 30.0])
    lower = np.array([0, 1, 0])
    upper = np.array([4, 6, 2])
    counts = proposals.solve(curves, inventory, lower, upper)
    assert counts.sum() == 6 and np.all(counts >= lower) and np.all(counts <= upper)
    best = None
    for candidate in itertools.product(*(range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper, strict=True))):
        if sum(candidate) != 6:
            continue
        value = float(proposals.predict(curves, (inventory * np.array(candidate) / 6.0)[None, :])[0])
        if best is None or value < best[0]:
            best = (value, candidate)
    assert best is not None
    assert float(proposals.predict(curves, (inventory * counts / 6.0)[None, :])[0]) == pytest.approx(best[0])


def test_rule_prediction_uses_the_inner_cv_argmin_inside_the_mask():
    grid = np.zeros((1, 2, 2, 1, 3))  # component, shape, ridge, link, bank rows
    grid[0, 0, 0, 0] = [1.0, 2.0, 3.0]
    grid[0, 1, 1, 0] = [7.0, 8.0, 9.0]
    cv = np.array([[[[0.5], [0.4]], [[0.3], [0.2]]]])  # component, shape, ridge, link; best overall is shape 1, ridge 1
    weights = np.array([1.0])
    full = np.ones((2, 2, 1), dtype=bool)
    assert np.allclose(rules.rule_prediction(grid, cv, weights, full), [7.0, 8.0, 9.0])
    restricted = full.copy()
    restricted[1] = False  # exclude shape 1: best inside the mask is shape 0, ridge 1 (never predicted), so zeros
    assert np.allclose(rules.rule_prediction(grid, cv, weights, restricted), [0.0, 0.0, 0.0])


def test_loso_splits_hold_out_every_membership_and_pool_by_primary_source():
    memberships = (
        frozenset({"panel"}),
        frozenset({"A"}),
        frozenset({"A", "B"}),
        frozenset({"B"}),
        frozenset({union_loso.DOSE_SOURCE}),
    )
    union = union_loso.Union(
        target="uncheatable",
        features=None,
        outcomes=np.zeros((5, 1)),
        aggregate=np.zeros(5),
        memberships=memberships,
        primary=np.array(["panel", "A", "A", "B", union_loso.DOSE_SOURCE]),
        coordinate_id=np.array([f"c{index}" for index in range(5)]),
        distance=np.zeros(5),
        trainable=np.array([True, True, True, False, True]),
    )
    splits = {split.held_out: split for split in union_loso.regime_splits(union, "loso")}
    assert set(splits) == {"A", "B"}
    assert set(splits["B"].test) == {2, 3}  # the A;B coordinate is held out with B too
    assert 2 not in splits["B"].train and 3 not in splits["B"].train
    assert set(splits["B"].pooled) == {3}  # but its pooled prediction comes from the split that held out A
    assert set(splits["A"].pooled) == {1, 2}
    assert 3 not in splits["A"].train  # test-only rows never train
