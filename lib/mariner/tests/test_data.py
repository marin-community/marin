# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pandas as pd
import pytest
from mariner.data import (
    CALIBRATION_RUN,
    MAIN_SWARM,
    REPEAT,
    anchors_from_runs,
    heldout_bank,
    objective_from_tasks,
    read_table,
    release_anchors,
    release_heldout_bank,
    release_partition,
    release_runs,
    release_swarm,
    release_tasks,
    swarm_from_runs,
)


def test_release_swarm_keeps_fit_rows_and_divides_exposure_by_pool_fraction(release_snapshot):
    swarm = release_swarm(release_snapshot, MAIN_SWARM, "obe_easy")
    # The 13 fit rows in table order, without the interleaved repeats; fit_01 is the proportional baseline.
    assert swarm.runs == tuple(f"fit_{row:02d}" for row in range(13))
    np.testing.assert_array_equal(np.flatnonzero(swarm.calibration), [1])
    assert list(swarm.outcomes.columns) == ["t0", "t1"]
    np.testing.assert_array_equal(swarm.outcomes["t0"], 1 + np.arange(13) / 100)
    # fit_05: weights (0.5, 0.25, 0.25) at 2, 4 and 8 epochs per unit weight, from half of b2's pool.
    np.testing.assert_array_equal(swarm.exposures[5], [1.0, 1.0, 4.0])


def test_release_anchors_add_the_june_run_only_for_main_swarm_uncheatable(release_snapshot):
    obe_easy = release_anchors(release_snapshot, MAIN_SWARM, "obe_easy")
    assert set(obe_easy) == {"t0", "t1"}
    # The three repeats, t0 = 1.0, 1.1 and 1.2; the June run (t0 5.0) does not count.
    assert (obe_easy["t0"].proportional, obe_easy["t0"].repeat_sd) == pytest.approx((1.1, 0.1))
    uncheatable = release_anchors(release_snapshot, MAIN_SWARM, "uncheatable")
    # The repeats and the June run, u0 = 2.0, 2.2, 2.4 and 2.6; not the ladder's copy of a repeat, its MARINER run
    # or its proportional run at 2e19.
    assert (uncheatable["u0"].proportional, uncheatable["u0"].repeat_sd) == pytest.approx((2.3, math.sqrt(0.2 / 3)))


def test_heldout_bank_averages_runs_at_one_exposure(release_snapshot):
    bank = release_heldout_bank(release_snapshot, "obe_easy")
    # h0 and h2 share e0; h1 alone is e1, from half of b2's pool; h3 is held out for uncheatable only.
    assert bank.exposure_ids == ("e0", "e1")
    np.testing.assert_array_equal(bank.runs, [2, 1])
    np.testing.assert_allclose(bank.measured[["t0", "t1"]].to_numpy(), [[1.1, 0.9], [1.4, 1.2]], rtol=1e-12)
    # The table records no bpb__obe_easy, so the objective is the task means at weights 1/2 each.
    np.testing.assert_allclose(bank.measured_objective, [1.0, 1.3], rtol=1e-12)
    np.testing.assert_array_equal(bank.weights, [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    np.testing.assert_array_equal(bank.pool_fractions, [[1.0, 1.0, 1.0], [1.0, 1.0, 0.5]])


def test_heldout_bank_takes_the_recorded_objective_where_task_values_are_missing(release_snapshot):
    bank = release_heldout_bank(release_snapshot, "uncheatable")
    assert bank.exposure_ids == ("e0", "e1", "e2")
    # h3, alone at e2, recorded bpb__uncheatable but no u0.
    assert np.isnan(bank.measured["u0"].iloc[2])
    np.testing.assert_allclose(bank.measured_objective, [2.1, 2.4, 2.6], rtol=1e-12)


@pytest.mark.parametrize("change", [{"w__b0": 0.25, "w__b1": 0.25}, {"pf__b2": 0.5}], ids=["weights", "pool_fractions"])
def test_heldout_bank_rejects_an_exposure_whose_runs_differ(release_snapshot, change):
    runs = read_table(release_snapshot, "heldout")
    # h2 shares e0 with h0, whose mixture is (0.2, 0.3, 0.5) from whole pools.
    runs.loc[runs["observation_id"] == "h2", list(change)] = list(change.values())
    objective = objective_from_tasks(release_tasks(release_snapshot), "obe_easy")
    with pytest.raises(ValueError, match="e0"):
        heldout_bank(runs, release_partition(release_snapshot).bucket_ids, objective)


@pytest.mark.parametrize("blank", [math.nan, "", " "], ids=["missing", "empty", "whitespace"])
def test_heldout_bank_rejects_a_run_without_an_exposure_id(release_snapshot, blank):
    runs = read_table(release_snapshot, "heldout")
    runs.loc[runs["observation_id"] == "h3", "exposure_id"] = blank
    objective = objective_from_tasks(release_tasks(release_snapshot), "obe_easy")
    with pytest.raises(ValueError, match="exposure_id"):
        heldout_bank(runs, release_partition(release_snapshot).bucket_ids, objective)


def first_row_set(runs: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    """``runs`` with ``column`` set to ``value`` in its first row, the fit row fit_00."""
    changed = runs.copy()
    changed.loc[0, column] = value
    return changed


def with_missing_fit_inclusion(runs: pd.DataFrame) -> pd.DataFrame:
    """``runs`` with ``fit_inclusion`` as pandas' nullable boolean, missing in its first row, the fit row fit_00."""
    inclusion = pd.array(runs["fit_inclusion"], dtype="boolean")
    inclusion[0] = pd.NA
    return runs.assign(fit_inclusion=inclusion)


DEFECTS = {
    "no_calibration": lambda runs: runs.replace({"run_label": {CALIBRATION_RUN: "baseline_other"}}),
    "missing_outcome": lambda runs: first_row_set(runs, "bpb__t0", math.nan),
    "unnormalized": lambda runs: first_row_set(runs, "w__b0", runs.loc[0, "w__b0"] + 0.01),
    "missing_pool_fraction": lambda runs: runs.drop(columns="pf__b2"),
    "zero_pool_fraction": lambda runs: first_row_set(runs, "pf__b0", 0.0),
    "integer_fit_inclusion": lambda runs: runs.assign(fit_inclusion=runs["fit_inclusion"].astype(int)),
    "missing_fit_inclusion": with_missing_fit_inclusion,
    "duplicate_observation_id": lambda runs: runs.replace({"observation_id": {"fit_03": "fit_02"}}),
    "blank_observation_id": lambda runs: first_row_set(runs, "observation_id", math.nan),
}


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("no_calibration", "baseline_proportional"),
        ("missing_outcome", "bpb__t0"),
        ("unnormalized", "sum to one"),
        ("missing_pool_fraction", "pf__b2"),
        ("zero_pool_fraction", "pf__b0"),
        ("integer_fit_inclusion", "fit_inclusion"),
        ("missing_fit_inclusion", "fit_inclusion"),
        ("duplicate_observation_id", "fit_02"),
        ("blank_observation_id", "observation_id"),
    ],
)
def test_swarm_from_runs_rejects_invalid_run_tables(release_snapshot, defect, message):
    runs = DEFECTS[defect](release_runs(release_snapshot, MAIN_SWARM))
    with pytest.raises(ValueError, match=message):
        swarm_from_runs(runs, release_partition(release_snapshot), ("t0", "t1"))


def test_anchors_from_runs_require_two_repeats(release_snapshot):
    runs = release_runs(release_snapshot, MAIN_SWARM)
    swarm = swarm_from_runs(runs, release_partition(release_snapshot), ("t0", "t1"))
    with pytest.raises(ValueError, match="two"):
        anchors_from_runs(swarm, runs[runs["design_kind"] == REPEAT].head(1), ("t0", "t1"))


def test_anchors_from_runs_reject_a_run_of_another_mixture(release_snapshot):
    runs = release_runs(release_snapshot, MAIN_SWARM)
    # fit_03, a sampled run, labeled as a proportional repeat: its outcomes would move the anchors.
    runs.loc[runs["observation_id"] == "fit_03", "design_kind"] = REPEAT
    swarm = swarm_from_runs(runs, release_partition(release_snapshot), ("t0", "t1"))
    with pytest.raises(ValueError, match="fit_03"):
        anchors_from_runs(swarm, runs[runs["design_kind"] == REPEAT], ("t0", "t1"))
