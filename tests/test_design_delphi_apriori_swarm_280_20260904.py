# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioural checks for the a priori 280-row swarm design: budget, simplex rows, reserved rows, subsample semantics."""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_apriori_swarm_280_20260904 as design,
)


def test_shares_from_levels_scale_with_unique_tokens():
    inventory = np.array([1.0, 4.0])  # epochs at full share; the second pool is four times smaller
    shares = design.shares_from_levels(np.array([1.0, 1.0]), inventory)
    np.testing.assert_allclose(shares, [0.8, 0.2])
    np.testing.assert_allclose(shares * inventory, [0.8, 0.8])  # equal levels give equal epochs


def test_farthest_point_picks_the_most_distant_rows_first():
    points = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5]])
    chosen = design.farthest_point(points, 2, np.array([1.0, 0.0]))
    assert chosen.tolist() == [2, 3]


@pytest.fixture(scope="module")
def built():
    panel = harness.load_panel(design.PANEL)
    rng = np.random.default_rng(0)
    dose = rng.dirichlet(np.ones(len(panel.buckets)), size=60)
    return panel, design.build_design(panel, dose)


def test_design_meets_budget_with_simplex_rows_and_reserved_baselines(built):
    panel, frame = built
    assert len(frame) == design.BUDGET_ROWS
    weights = np.vstack(frame["weights"].to_numpy())
    assert weights.min() >= 0 and np.allclose(weights.sum(1), 1.0)
    kinds = set(frame["kind"])
    assert {"baseline_proportional", "baseline_unimax", "baseline_uniform"} <= kinds
    assert sum(k.startswith("proportional_repeat_") for k in kinds) == design.PROPORTIONAL_REPEATS
    assert sum(k.startswith("pctrl_del_") for k in kinds) == len(panel.buckets)
    assert frame["run_name"].is_unique


def test_subsampled_rows_keep_their_anchor_weights_and_only_shrink_the_target_pool(built):
    panel, frame = built
    buckets = list(panel.buckets)
    subsampled = frame[frame["block"].eq("subsampled_pool")]
    assert len(subsampled) == 40
    proportional = frame.loc[frame["kind"].eq("baseline_proportional"), "weights"].iloc[0]
    for _, row in subsampled[subsampled["anchor"].eq("A_proportional")].iterrows():
        np.testing.assert_allclose(row["weights"], proportional)
        fractions = row["pool_fractions"]
        assert set(fractions.values()) <= {0.5, 0.25}
        if row["target_bucket"] == "cc_high_group":
            assert all(b.startswith("dolma3_cc/") and b.endswith("_high") for b in fractions)
        else:
            assert list(fractions) == [row["target_bucket"]]
    assert all(bucket in buckets for fractions in subsampled["pool_fractions"] for bucket in fractions)
