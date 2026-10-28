# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from mariner.data import REPEAT, anchors_from_runs, partition_from_table, swarm_from_runs
from mariner.design import proportional, removals, sobol_candidates, stratified, swarm_table, unimax
from scipy.stats import qmc

BUCKETS = ("b0", "b1", "b2", "b3")
POOL_TOKENS = np.array([4000.0, 2000.0, 1000.0, 3000.0])
TARGET_BUDGET = 8000.0
TASKS = ("t0", "t1")


def total_variation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(left - right).sum(axis=-1)


def test_sobol_candidates_lie_on_the_simplex_with_small_weights_zeroed():
    candidates = sobol_candidates(64, 10, seed=0, min_weight=0.02)
    assert candidates.shape == (64, 10)
    np.testing.assert_allclose(candidates.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert (candidates >= 0).all() and (candidates == 0).any()
    assert candidates[candidates > 0].min() >= 0.02


def test_sobol_candidates_fall_back_to_the_uniform_mixture_when_every_weight_is_small():
    # No weight of 4 buckets is 0.9 or more in most rows: they become the uniform mixture, the rest a single bucket.
    candidates = sobol_candidates(3, 4, seed=0, min_weight=0.9, min_distance=0.0)
    uniform = np.all(candidates == 0.25, axis=1)
    # One uniform row at most: the four-decimal deduplication skips every later row that repeats it.
    assert uniform.sum() == 1
    assert np.all(np.sort(candidates[~uniform], axis=1)[:, -1] == 1.0)


@pytest.mark.parametrize(
    ("n_buckets", "n_existing", "n", "min_distance"),
    [(10, 8, 32, 0.05), (3, 4, 16, 0.15)],  # the second packs the triangle so tightly that the distance's scale matters
)
def test_sobol_candidates_keep_min_distance_from_existing_and_each_other(n_buckets, n_existing, n, min_distance):
    existing = sobol_candidates(n_existing, n_buckets, seed=1)
    candidates = sobol_candidates(n, n_buckets, seed=1, existing=existing, min_distance=min_distance)
    assert candidates.shape == (n, n_buckets)
    within = total_variation(candidates[:, None], candidates[None])
    within[np.diag_indices(n)] = np.inf
    assert within.min() >= min_distance
    # The same seed draws the existing mixtures again first, so the filter had to skip them.
    assert total_variation(candidates[:, None], existing[None]).min() >= min_distance


def test_sobol_candidates_are_the_scrambled_sobol_points_in_order():
    # With nothing to reject, candidate i is the simplex point of the i-th point of one 128-point draw, though the
    # candidates are drawn in blocks that double in size (8, 8, 16, 32, 64).
    points = np.clip(qmc.Sobol(d=6, scramble=True, seed=3).random_base2(7), 1e-9, 1 - 1e-9)
    raw = -np.log(points)
    expected = raw / raw.sum(axis=1, keepdims=True)
    candidates = sobol_candidates(100, 6, seed=3, min_weight=0.0, min_distance=0.0)
    np.testing.assert_allclose(candidates, expected[:100], rtol=0, atol=1e-15)


def test_sobol_candidates_reproduce_their_seed():
    # The first two rows that the research code's sobol_weight_configs draws at seed 0 for one phase of four buckets:
    # a seed must give the same swarm under any SciPy release.
    expected = [
        [0.07172839107386524, 0.03151509625886846, 0.449495114903172, 0.4472613977640943],
        [0.3956863496478225, 0.3468286384671171, 0.2193128301673509, 0.03817218171770959],
    ]
    np.testing.assert_allclose(sobol_candidates(2, 4, seed=0), expected, rtol=0, atol=1e-15)


def test_sobol_candidates_never_repeat_a_mixture_at_four_decimals():
    # Two buckets have 10,001 mixtures at four decimals, so the first 2,000 points of their stream repeat one another.
    candidates = sobol_candidates(2000, 2, seed=0, min_weight=0.0, min_distance=0.0)
    assert len(np.unique(np.round(candidates, 4), axis=0)) == 2000


def test_sobol_candidates_raise_when_too_few_mixtures_are_available():
    # One bucket has a single mixture, so a second candidate always repeats the first.
    np.testing.assert_array_equal(sobol_candidates(1, 1, seed=0), [[1.0]])
    with pytest.raises(ValueError):
        sobol_candidates(2, 1, seed=0)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 0, 2, 1]])
@pytest.mark.parametrize(
    ("max_epochs", "expected"),
    [(1.0, [0.01, 0.1, 0.445, 0.445]), (2.0, [0.02, 0.2, 0.39, 0.39])],
)
def test_unimax_caps_small_pools_and_splits_the_rest_evenly(max_epochs, expected, order):
    pool_tokens = np.array([10.0, 100.0, 1000.0, 1000.0])
    weights = unimax(pool_tokens[order], budget=1000.0, max_epochs=max_epochs)
    np.testing.assert_allclose(weights, np.array(expected)[order], rtol=0, atol=1e-15)


def test_unimax_rejects_a_budget_beyond_the_pools_at_max_epochs():
    pool_tokens = np.array([10.0, 20.0])
    # The pools hold 30 tokens at one epoch, so a budget of 100 tokens would repeat each bucket 3.33 times.
    with pytest.raises(ValueError, match="budget"):
        unimax(pool_tokens, budget=100.0, max_epochs=1.0)
    # A budget of exactly the pools takes each bucket once.
    np.testing.assert_allclose(unimax(pool_tokens, budget=30.0, max_epochs=1.0), [1 / 3, 2 / 3], rtol=0, atol=1e-15)


def test_proportional_and_stratified_baselines():
    np.testing.assert_allclose(proportional(np.array([10.0, 30.0, 60.0])), [0.1, 0.3, 0.6], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(stratified(4), np.full(4, 0.25))


def test_removals_zero_one_bucket_per_row_and_renormalize_the_rest():
    weights = removals(np.array([0.5, 0.25, 0.125, 0.125]))
    expected = [[0, 1 / 2, 1 / 4, 1 / 4], [2 / 3, 0, 1 / 6, 1 / 6], [4 / 7, 2 / 7, 0, 1 / 7], [4 / 7, 2 / 7, 1 / 7, 0]]
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)


def test_swarm_table_lists_baselines_candidates_removals_then_repeats():
    proportional_weights = np.array([0.5, 0.25, 0.25])
    unimax_weights = np.array([0.2, 0.3, 0.5])
    removal_base = np.array([0.4, 0.4, 0.2])
    candidates = np.array([[0.1, 0.1, 0.8], [0.6, 0.2, 0.2]])
    bucket_ids = ("a", "b/c", "d")
    table = swarm_table(
        bucket_ids,
        candidates,
        {"unimax": unimax_weights, "proportional": proportional_weights},
        removal_base,
        repeats=2,
        data_seed_start=500,
    )

    assert list(table.columns) == [
        "run_label",
        "design_kind",
        "fit_inclusion",
        "data_seed",
        "trainer_seed",
        *[f"w__{bucket}" for bucket in bucket_ids],
        *[f"pf__{bucket}" for bucket in bucket_ids],
    ]
    assert list(table["run_label"]) == [
        "baseline_unimax",
        "baseline_proportional",
        "run_00000",
        "run_00001",
        "deletion_a",
        "deletion_b/c",
        "deletion_d",
        "proportional_repeat_00",
        "proportional_repeat_01",
    ]
    assert list(table["design_kind"]) == ["baseline"] * 2 + ["sampled"] * 2 + ["deletion"] * 3 + ["repeat"] * 2
    assert list(table["fit_inclusion"]) == [True] * 7 + [False] * 2
    np.testing.assert_array_equal(table["data_seed"], 500 + np.arange(9))
    np.testing.assert_array_equal(table["trainer_seed"], np.zeros(9))
    weights = table[[f"w__{bucket}" for bucket in bucket_ids]].to_numpy()
    # The removals start from removal_base, and the repeats train the proportional baseline wherever it sits.
    expected = np.vstack(
        [
            unimax_weights,
            proportional_weights,
            candidates,
            removals(removal_base),
            proportional_weights,
            proportional_weights,
        ]
    )
    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_array_equal(table[[f"pf__{bucket}" for bucket in bucket_ids]].to_numpy(), np.ones((9, 3)))


def test_swarm_table_may_omit_candidates_and_removals():
    table = swarm_table(
        BUCKETS,
        sobol_candidates(0, len(BUCKETS), seed=0),
        {"proportional": proportional(POOL_TOKENS)},
        removal_base=None,
        repeats=2,
        data_seed_start=0,
    )
    assert list(table["run_label"]) == ["baseline_proportional", "proportional_repeat_00", "proportional_repeat_01"]


def test_swarm_table_loads_as_a_swarm_after_outcomes_join():
    proportional_weights = proportional(POOL_TOKENS)
    table = swarm_table(
        BUCKETS,
        sobol_candidates(5, len(BUCKETS), seed=0),
        {
            "proportional": proportional_weights,
            "unimax": unimax(POOL_TOKENS, budget=5000.0, max_epochs=1.0),
            "stratified": stratified(len(BUCKETS)),
        },
        removal_base=proportional_weights,
        repeats=3,
        data_seed_start=7000,
    )
    # Training the 15 runs gives them observation ids and outcomes.
    rows = np.arange(len(table))
    trained = table.assign(
        observation_id=[f"obs_{row:02d}" for row in rows], bpb__t0=1.0 + 0.01 * rows, bpb__t1=2.0 + 0.02 * rows
    )
    partition = partition_from_table(
        pd.DataFrame({"bucket_id": BUCKETS, "pool_tokens": POOL_TOKENS}), target_budget=TARGET_BUDGET
    )

    swarm = swarm_from_runs(trained, partition, TASKS)
    # The 3 baselines, 5 candidates and 4 removals fit; the proportional baseline calibrates; the repeats do not fit.
    assert swarm.runs == tuple(f"obs_{row:02d}" for row in range(12))
    np.testing.assert_array_equal(np.flatnonzero(swarm.calibration), [0])
    np.testing.assert_array_equal(swarm.weights, table[[f"w__{bucket}" for bucket in BUCKETS]].iloc[:12])
    np.testing.assert_array_equal(swarm.pool_fractions, np.ones((12, len(BUCKETS))))
    np.testing.assert_allclose(swarm.outcomes["t1"], 2.0 + 0.02 * rows[:12], rtol=0, atol=1e-12)

    # The repeats are rows 12 to 14: t0 1.12, 1.13 and 1.14; t1 2.24, 2.26 and 2.28.
    anchors = anchors_from_runs(swarm, trained[trained["design_kind"] == REPEAT], TASKS)
    assert (anchors["t0"].proportional, anchors["t0"].repeat_sd) == pytest.approx((1.13, 0.01))
    assert (anchors["t1"].proportional, anchors["t1"].repeat_sd) == pytest.approx((2.26, 0.02))
