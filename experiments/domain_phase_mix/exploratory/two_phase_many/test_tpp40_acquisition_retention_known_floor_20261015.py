# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "pytest"]
# ///
"""Invented-data checks of the complete known-floor search and input barrier."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_known_floor_20261015 as runner,
)


@pytest.mark.timeout(120)
def test_full_native_shape_and_temporal_grid_recover_a_different_toy_shape(tmp_path: Path):
    # This test traverses all168shapes and7strengths with actual numerical features.
    rng = np.random.default_rng(640)
    weights = rng.uniform(0.04, 0.96, (2, 30))
    scales, alpha = np.array([1.0, 5.0]), 0.75
    early = alpha * scales * np.c_[1 - weights[0], weights[0]]
    late = (1 - alpha) * scales * np.c_[1 - weights[1], weights[1]]
    config = runner.models.FeatureConfig(tuple(scales), "equal_group", (0.5, 0.5), 1e-9, 256)
    cache = runner.models.BenefitCache(tmp_path / "features.sqlite")
    shape = {"rate": 0.5, "power": 1.0, "threshold": 2.0}
    matrix, offset = runner.models.design_and_offset(
        early, late, alpha, shape, runner.models.MODEL, {"mu": 3.0}, config, cache
    )
    truth = runner.models.mariner.Head(0.1, np.array([0.8, 1.7, 0.2, 0.3]), 0.6)
    observed = truth.predict(matrix)
    labels = np.arange(len(observed)) % 3
    aggregate = runner.fit_family(early, late, observed, labels, alpha, config, "aggregate_mariner", None, cache)
    fitted = runner.fit_family(early, late, observed, labels, alpha, config, "equal_group", aggregate, cache)
    assert fitted.shape == shape
    assert fitted.parameters == {"mu": 3.0}
    assert fitted.known_floor == fitted.head.floor == 0.6
    assert fitted.kappa is None and fitted.ridge == 0.0
    assert fitted.inner_rmse < 1e-10
    assert fitted.inner_rmse < aggregate.inner_rmse
    np.testing.assert_allclose(fitted.predict(early, late, cache), observed, atol=2e-11, rtol=0)
    assert fitted.search_space["retained_matched_null"]["rescored_inner_rmse"] == aggregate.inner_rmse
    assert len(fitted.selection_profile) == 168 * 7 * 5 + 1
    assert not fitted.search_space["test_grid_override"]
    assert fitted.search_space["gammas"] == []
    replay = runner.models.JointFit.from_json(json.loads(json.dumps(fitted.to_json(), allow_nan=False)))
    np.testing.assert_array_equal(replay.predict(early, late, cache), fitted.predict(early, late, cache))
    inner = runner.selected_inner(fitted, early, late, observed, labels, cache)
    assert abs(inner["selected_score_replay_difference"]) < 1e-14
    altered = observed.copy()
    altered[labels == 0] += 0.1
    poisoned = runner.selected_inner(fitted, early, late, altered, labels, cache)
    assert inner["inner_heads"][0] == poisoned["inner_heads"][0]
    assert inner["inner_heads"][1] != poisoned["inner_heads"][1]
    assert not np.any(offset)


def test_training_noise_and_held_query_poison_are_separated():
    clean = np.array([0.9, 1.0, 0.95, 1.03, 0.83, 0.88])
    noise = np.array([0.001, -0.002, 0.003, -0.004, 0.005, -0.006])
    train = np.array([0, 2, 4])
    observed = runner.training_values(clean, noise, train)
    changed_clean, changed_noise = clean.copy(), noise.copy()
    changed_clean[[1, 3, 5]] = [-20, 30, 40]
    changed_noise[[1, 3, 5]] = [80, 90, 100]
    np.testing.assert_array_equal(runner.training_values(changed_clean, changed_noise, train), observed)
    np.testing.assert_array_equal(clean, [0.9, 1.0, 0.95, 1.03, 0.83, 0.88])
    np.testing.assert_array_equal(observed, [0.901, 0.953, 0.835])


def test_draft_cannot_freeze_or_read_saved_synthetic_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"status": "draft", "scope": runner.SCOPE}))
    for operation in (runner.freeze, lambda: runner.fit_all(1), runner.summarize):
        with pytest.raises(ValueError, match="Register this separate"):
            operation()
    assert not (tmp_path / "features.sqlite").exists()
    assert not (tmp_path / "cells").exists()
    assert not (tmp_path / "execution_identity.json").exists()
