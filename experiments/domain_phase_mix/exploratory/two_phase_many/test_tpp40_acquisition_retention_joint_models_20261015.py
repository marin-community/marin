# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented two-bucket laws only; no frozen geometry or scientific labels."""

import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_acquisition_retention_joint_models_20261015 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_causal_acquisition_retention_20261015 as law
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as native
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_joint_models_20261015 as original_union

SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
CONFIG = models.FeatureConfig((10.0, 2.0), "equal_group", (0.5, 0.5), 1e-10, 512)
ALPHA = 0.8


def toy_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261015)
    total = np.exp(rng.uniform(-2, 3, (36, 2)))
    fractions = rng.uniform(0.05, 0.95, total.shape)
    return total * fractions, total * (1 - fractions), np.arange(len(total)) % 3


def cache_replay(path: str, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    return models.design_and_offset(
        early, late, ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, CONFIG, models.BenefitCache(Path(path))
    )[0]


def fitted_replay(fit: models.JointFit, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    return models.JointFit.from_json(json.loads(json.dumps(fit.to_json(), allow_nan=False))).predict(early, late)


def test_direct_epoch_design_matches_pure_policy_with_native_harm_and_tied_null() -> None:
    weights = np.array([[0.2, 0.8], [0.8, 0.2]])
    fractions = np.array([ALPHA, 1 - ALPHA])
    epochs = fractions[:, None] * weights * CONFIG.scales
    settings = law.Law(SHAPE["rate"], SHAPE["power"], 1.0, ("first", "second"), np.array(CONFIG.masses))
    expected = law.evaluate_policy(weights, fractions, np.array(CONFIG.scales), settings, law.Quadrature(1e-10, 512))
    matrix, offset = models.design_and_offset(epochs[:1], epochs[1:], ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, CONFIG)
    np.testing.assert_allclose(matrix[0, :2], -expected.benefit, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(matrix[:, 2:], native.mariner.harm(epochs.sum(axis=0)[None], SHAPE["threshold"]))
    np.testing.assert_array_equal(offset, [0.0])
    # Physical epochs remain admissible when an optimizer's query misses its token simplex.
    off_simplex, _ = models.design_and_offset(
        epochs[:1] * 1.3, epochs[1:] * 0.4, ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, CONFIG
    )
    assert np.isfinite(off_simplex).all()
    for total in (np.zeros((1, 2)), np.array([[1e-20, 3e-20]]), np.array([[1.0, 4.0]])):
        early_tied, late_tied = ALPHA * total, (1 - ALPHA) * total
        tied, _ = models.design_and_offset(early_tied, late_tied, ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, CONFIG)
        np.testing.assert_array_equal(tied, native.mariner.design_matrix(early_tied + late_tied, SHAPE))
    null, _ = models.design_and_offset(epochs[:1], epochs[1:], ALPHA, SHAPE, models.MODEL, {"mu": 0.0}, CONFIG)
    np.testing.assert_array_equal(null, native.mariner.design_matrix(epochs.sum(axis=0)[None], SHAPE))
    assert not models.native_tied(np.array([0.0, 1e-30]), np.array([1e-30, 0.0]), ALPHA)
    assert not models.native_tied(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.nextafter(1.0, 0.0))
    total = np.array([1.0, 4.0])
    tiny_contrast = np.array([1e-9, -1e-9])
    feature = models.policy_feature(
        ALPHA * total + tiny_contrast,
        (1 - ALPHA) * total - tiny_contrast,
        ALPHA,
        SHAPE["rate"],
        SHAPE["power"],
        1.0,
        CONFIG,
    )
    assert not feature["exact_native"] and feature["evaluations"] > 0
    assert (
        np.max(abs(np.array(feature["benefit"]) - native.mariner.benefit(total, SHAPE["rate"], SHAPE["power"]))) > 1e-13
    )


def test_cache_reuses_policy_rows_across_thresholds_reorders_and_separates_clocks(tmp_path: Path) -> None:
    early, late, _ = toy_geometry()
    early, late = early[:4], late[:4]
    path = tmp_path / "features.sqlite"
    cache = models.BenefitCache(path)
    first, _ = models.design_and_offset(early, late, ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, CONFIG, cache)
    changed, _ = models.design_and_offset(
        early[[3, 0]],
        late[[3, 0]],
        ALPHA,
        {**SHAPE, "threshold": 5.0},
        models.MODEL,
        {"mu": 1.0},
        CONFIG,
        models.BenefitCache(path),
    )
    np.testing.assert_array_equal(changed[:, :2], first[[3, 0], :2])
    assert not np.array_equal(changed[:, 2:], first[[3, 0], 2:])
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM features").fetchone()[0] == 4
    second_clock = replace(CONFIG, clock="global_pool", masses=(0.2, 0.8))
    changed_clock, _ = models.design_and_offset(
        early, late, ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, second_clock, cache
    )
    assert np.max(abs(changed_clock - first)) > 1e-4
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM contexts").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM features").fetchone()[0] == 8
        connection.execute("UPDATE features SET response='{}'")
    with pytest.raises(ValueError, match="corruption"):
        models.design_and_offset(early, late, ALPHA, SHAPE, models.MODEL, {"mu": 1.0}, CONFIG, cache)


def test_failed_numerical_entry_is_durable_and_a_new_budget_is_a_separate_context(tmp_path: Path) -> None:
    cache = models.BenefitCache(tmp_path / "features.sqlite")
    config = replace(CONFIG, scales=(100.0, 0.02), relative_tolerance=1e-12, max_panels=1)
    shape = {"rate": 0.25, "power": 0.3, "threshold": 3.0}
    early = np.array([[0.8 * 0.001 * 100, 0.8 * 0.999 * 0.02]])
    late = np.array([[0.2 * 0.9 * 100, 0.2 * 0.1 * 0.02]])
    for _ in range(2):
        with pytest.raises(ArithmeticError, match="Cached numerical failure"):
            models.design_and_offset(early, late, ALPHA, shape, models.MODEL, {"mu": 20.0}, config, cache)
    with sqlite3.connect(cache.path) as connection:
        assert connection.execute("SELECT count(*) FROM features").fetchone()[0] == 1
        saved = json.loads(connection.execute("SELECT response FROM features").fetchone()[0])
        assert saved["status"] == "numerical_failure"
    result, _ = models.design_and_offset(
        early, late, ALPHA, shape, models.MODEL, {"mu": 20.0}, replace(config, max_panels=512), cache
    )
    assert np.isfinite(result).all()
    with sqlite3.connect(cache.path) as connection:
        assert connection.execute("SELECT count(*) FROM contexts").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM features").fetchone()[0] == 2


def test_cache_concurrent_spawn_writers_commit_identical_replay(tmp_path: Path) -> None:
    early, late, _ = toy_geometry()
    early, late = early[:3], late[:3]
    path = tmp_path / "features.sqlite"
    models.BenefitCache(path)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
        tasks = [executor.submit(cache_replay, str(path), early, late) for _ in range(2)]
        results = [task.result(timeout=30) for task in tasks]
    np.testing.assert_array_equal(results[0], results[1])
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM features").fetchone()[0] == 3


def test_j2_native_parity_fresh_control_union_and_canonical_spawn(tmp_path: Path) -> None:
    early, late, labels = toy_geometry()
    matrix = native.mariner.design_matrix(early + late, SHAPE)
    observed = native.mariner.Head(0.5, np.array([0.5, 0.8, 0.03, 0.02]), 0.6).predict(matrix)
    native_grid = models.TestGrid(({},), (0.0,), (1.0, 6.0))
    before = native.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    cache = models.BenefitCache(tmp_path / "features.sqlite")
    aggregate = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        "aggregate_mariner",
        config=CONFIG,
        strengths=(0.0, 1.0),
        cache=cache,
        shapes=(SHAPE,),
        test_grid=native_grid,
    )
    np.testing.assert_array_equal(aggregate.predict(early, late), before.predict(early, late))
    poisoned = models.JointFit(
        replace(aggregate.selected, inner_rmse=-100.0, head=native.mariner.Head(10.0, np.ones(4), -9.0)), CONFIG
    )
    fit = models.fit_joint(
        early,
        late,
        observed,
        labels,
        ALPHA,
        models.MODEL,
        config=CONFIG,
        strengths=(0.0, 1.0),
        cache=cache,
        controls=(poisoned,),
        shapes=(SHAPE,),
        test_grid=models.TestGrid(({"mu": 0.0}, {"mu": 1.0}), (0.0,), (1.0, 6.0)),
    )
    retained = fit.search_space["retained_control_candidates"][0]
    expected, floors = np.empty(len(observed)), []
    for train, held in native.temporal.inner_splits(labels):
        head = native.fit_fixed_head(
            matrix[train], np.zeros(len(train)), observed[train], aggregate.ridge, aggregate.kappa, None
        )
        floors.append(head.floor)
        expected[held] = head.predict(matrix[held])
    assert np.ptp(floors) > 1e-4
    np.testing.assert_allclose(retained["inner_predictions"], expected, atol=1e-13, rtol=1e-13)
    assert retained["fresh_inner_rmse"] == aggregate.inner_rmse
    assert fit.inner_rmse <= aggregate.inner_rmse
    assert fit.search_space["complete_aggregate_source"]["parameters"] == {}
    assert retained["parameters"] == {"mu": 0.0}
    assert original_union.parameter_grid("aggregate_mariner") == ({"kappa": 0.0, "nu": 0.0},)
    after = native.fit_joint(
        early, late, observed, labels, ALPHA, "aggregate_mariner", shapes=(SHAPE,), test_grid=native_grid
    )
    np.testing.assert_array_equal(before.predict(early, late), after.predict(early, late))
    nonnull = models.JointFit(replace(fit.selected, parameters={"mu": 1.0}), CONFIG)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        replay = executor.submit(fitted_replay, nonnull, early[:3], late[:3]).result(timeout=30)
    np.testing.assert_array_equal(replay, nonnull.predict(early[:3], late[:3]))
    stale = nonnull.to_json()
    stale["feature_runtime"] = {**stale["feature_runtime"], "numpy": "different-runtime"}
    with pytest.raises(ValueError, match="runtime identity"):
        models.JointFit.from_json(stale)
    isolated_payload = nonnull.to_json()
    isolated_payload["feature_runtime"]["numpy"] = "caller-edited-copy"
    isolated_payload["feature_source_hashes"][next(iter(isolated_payload["feature_source_hashes"]))] = (
        "caller-edited-copy"
    )
    fresh = models.JointFit.from_json(nonnull.to_json())
    np.testing.assert_array_equal(fresh.predict(early[:3], late[:3]), replay)


def test_known_floor_retains_complete_native_candidate_as_temporal_mu_zero(tmp_path: Path) -> None:
    early, late, labels = toy_geometry()
    matrix = native.mariner.design_matrix(early + late, SHAPE)
    observed = native.mariner.Head(0.5, np.array([0.5, 0.8, 0.03, 0.02]), 0.6).predict(matrix)
    cache = models.BenefitCache(tmp_path / "features.sqlite")
    aggregate = models.fit_known_floor(
        early,
        late,
        observed,
        labels,
        ALPHA,
        "aggregate_mariner",
        floor=0.6,
        config=CONFIG,
        strengths=(0.0, 1.0),
        cache=cache,
        shapes=(SHAPE,),
        test_grid=models.TestGrid(({},), (0.0,), (1.0,)),
    )
    fit = models.fit_known_floor(
        early,
        late,
        observed,
        labels,
        ALPHA,
        models.MODEL,
        floor=0.6,
        config=CONFIG,
        strengths=(0.0, 1.0),
        cache=cache,
        shapes=(SHAPE,),
        matched_null_fit=aggregate,
        test_grid=models.TestGrid(({"mu": 0.0}, {"mu": 1.0}), (0.0,), (1.0,)),
    )
    assert fit.parameters == {"mu": 0.0} and aggregate.parameters == {}
    assert fit.search_space["retained_matched_null"]["rescored_inner_rmse"] == aggregate.inner_rmse
    np.testing.assert_array_equal(fit.predict(early, late), aggregate.predict(early, late))
    np.testing.assert_allclose(fit.predict(early, late), observed, rtol=0, atol=2e-14)
