# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioural checks for the frozen 280-row swarm design: budget, simplex rows, reserved rows, conditions, seeds."""

import numpy as np
import pandas as pd
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
    weights = rng.dirichlet(np.ones(len(panel.buckets)), size=60)
    dose = pd.DataFrame(weights, columns=[f"weight::{b}" for b in panel.buckets])
    dose["coordinate_id"] = [f"coord{i}" for i in range(60)]
    dose["source_run_names"] = [f"run{i}" for i in range(60)]
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


def test_run_conditions_are_unique_and_repeats_differ_only_by_seed_block(built):
    panel, frame = built
    buckets = list(panel.buckets)
    fractions = [np.array([f.get(b, 1.0) for b in buckets]) for f in frame["pool_fractions"]]
    keys = [
        design.condition_key(w, f, int(block))
        for w, f, block in zip(frame["weights"], fractions, frame["seed_block"], strict=True)
    ]
    assert not pd.Series(keys).duplicated().any()
    mixtures = [design.condition_key(w, f, 0) for w, f in zip(frame["weights"], fractions, strict=True)]
    same_mixture = pd.Series(mixtures).duplicated(keep="first").to_numpy()
    assert frame.loc[same_mixture, "seed_block"].gt(0).all()  # every repeated mixture is a fresh seed block


def test_reused_rows_keep_their_provenance(built):
    _panel, frame = built
    reused = frame[frame["source"].ne("new")]
    assert reused["source_run_names"].ne("").all()
    dose = frame[frame["block"].eq("reused_dose_ladder")]
    assert dose["source_coordinate_id"].str.startswith("coord").all()


def test_subsampled_rows_keep_anchor_weights_and_pair_seeds_within_blocks(built):
    panel, frame = built
    buckets = list(panel.buckets)
    subsampled = frame[frame["block"].eq("subsampled_pool")]
    assert len(subsampled) == 40
    pilot = subsampled[subsampled["wave"].eq("pilot")]
    assert len(pilot) == 32 and set(pilot["seed_block"]) == {0, 1}
    proportional = frame.loc[frame["kind"].eq("baseline_proportional"), "weights"].iloc[0]
    for _, row in pilot[pilot["anchor"].eq("A_proportional")].iterrows():
        np.testing.assert_allclose(row["weights"], proportional)
        assert row["data_seed"] == design.BASE_DATA_SEED + row["seed_block"]
        fractions = row["pool_fractions"]
        assert set(fractions.values()) <= set(design.POOL_FRACTIONS)
        if row["target_bucket"] == design.CC_HIGH_GROUP:
            assert all(b.startswith("dolma3_cc/") and b.endswith("_high") for b in fractions)
        else:
            assert list(fractions) == [row["target_bucket"]]
    anchors = frame[frame["block"].eq("anchor")]
    assert set(anchors.loc[anchors["seed_block"].eq(1), "kind"]) == {"anchor_B_small_pools_forward_repeat"}
    assert all(bucket in buckets for fractions in subsampled["pool_fractions"] for bucket in fractions)


def test_dose_rows_below_the_reuse_count_are_rejected():
    panel = harness.load_panel(design.PANEL)
    dose = pd.DataFrame(
        np.full((5, len(panel.buckets)), 1 / len(panel.buckets)), columns=[f"weight::{b}" for b in panel.buckets]
    )
    dose["coordinate_id"] = [f"c{i}" for i in range(5)]
    dose["source_run_names"] = ""
    with pytest.raises(ValueError, match="carry both targets"):
        design.build_design(panel, dose)
