# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioural checks for the round-6 training-set designs: pruning, coverage selection, budget and eligibility."""

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round6_training_sets_20260904 as training,
)


def test_prune_redundant_removes_the_near_duplicate_first():
    weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.98, 0.02, 0.0]])
    kept = training.prune_redundant(weights, np.arange(4), 1)
    assert 3 not in kept or 0 not in kept  # one of the duplicate pair leaves
    assert len(kept) == 3 and set(kept) <= {0, 1, 2, 3}


def test_coverage_greedy_prefers_the_farthest_pool_row_and_never_repeats():
    weights = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5]])
    chosen = training.coverage_greedy(weights, np.array([0]), np.array([1, 2, 3]), 2)
    assert chosen.tolist() == [2, 3]
    assert len(training.coverage_greedy(weights, np.array([0]), np.array([1, 2, 3]), 10)) == 3


def test_total_variation_is_half_the_l1_distance():
    a = np.array([[0.5, 0.5]])
    b = np.array([[1.0, 0.0], [0.5, 0.5]])
    np.testing.assert_allclose(training.total_variation(a, b), [[0.5, 0.0]])


def _synthetic_union(panel_rows: int, dose_rows: int):
    from experiments.domain_phase_mix.exploratory.two_phase_many import (
        single_phase_observatory_models_20260902 as models,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many import (
        single_phase_round3_union_loso_20260903 as loso,
    )

    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(3), size=panel_rows + dose_rows)
    features = models.Features(
        exposures=weights * 2.0,
        weights=weights,
        inventory=np.full(3, 2.0),
        early_fraction=np.zeros(len(weights)),
        families=models.no_families(3),
        label="synthetic",
    )
    memberships = tuple([frozenset({"panel"})] * panel_rows + [frozenset({loso.DOSE_SOURCE})] * dose_rows)
    return loso.Union(
        target="table9",
        features=features,
        outcomes=np.zeros((len(weights), 1)),
        aggregate=np.zeros(len(weights)),
        memberships=memberships,
        primary=np.array(["panel"] * panel_rows + [loso.DOSE_SOURCE] * dose_rows),
        coordinate_id=np.array([f"row{i}" for i in range(len(weights))]),
        distance=np.zeros(len(weights)),
        trainable=np.ones(len(weights), dtype=bool),
    )


def test_designs_stay_within_budget_and_never_use_rows_outside_panel_and_pool():
    union = _synthetic_union(panel_rows=8, dose_rows=6)
    pool = np.arange(8, 12)  # four of the six dose rows; the other two stand in for held-out evaluation rows
    designs = training.build_designs(union, pool)
    allowed = set(range(8)) | set(pool.tolist())
    names = [design.name for design in designs]
    assert len(names) == len(set(names))  # capped swap sizes are deduplicated
    for design in designs:
        assert set(design.train.tolist()) <= allowed, design.name
        assert len(set(design.train.tolist())) == len(design.train), design.name
        if not design.name.endswith("_over_budget"):
            assert len(design.train) <= training.BUDGET
    over_budget = [design for design in designs if design.name.endswith("_over_budget")]
    assert len(over_budget) == 1 and set(over_budget[0].train.tolist()) == allowed
