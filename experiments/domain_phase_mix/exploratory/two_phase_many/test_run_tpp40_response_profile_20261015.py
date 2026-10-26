# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved-input and scoring contracts for the fixed-law response diagnostic."""

import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as runner


def test_immutable_cache_replays_each_clock_and_cannot_generate_missing_rows(tmp_path):
    path = tmp_path / "features.sqlite"
    data = {"alpha": 0.8, "scales": np.array([1.0, 4.0])}
    frozen = {
        "source_hashes": {"law.py": "frozen-toy-source"},
        "environment": {name: "toy" for name in ("python", "numpy", "scipy", "threadpoolctl", "sqlite")},
    }
    early, late = np.array([[0.2, 0.4]]), np.array([[0.1, 0.3]])
    request = {"early": early[0].tolist(), "late": late[0].tolist()}
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE contexts(key TEXT, identity TEXT)")
        connection.execute(
            "CREATE TABLE features(context TEXT, policy TEXT, request TEXT, response TEXT, response_sha256 TEXT)"
        )
        for index, clock in enumerate(("equal_group", "global_pool")):
            context = {
                "alpha": 0.8,
                "rate": 0.25,
                "power": 0.7,
                "mu": 3.0,
                "sources": frozen["source_hashes"],
                "runtime": frozen["environment"],
                "config": {
                    "clock": clock,
                    "scales": [1.0, 4.0],
                    "masses": [0.5, 0.5] if index == 0 else [0.8, 0.2],
                    "groups": ["first", "second"],
                    "relative_tolerance": 1e-11,
                    "max_panels": 512,
                },
            }
            key = runner.canonical_hash(context)
            response = json.dumps(
                {"status": "complete", "relative_error_estimate": 1e-13, "benefit": [0.2 + 0.1 * index, 0.4]}
            )
            connection.execute("INSERT INTO contexts VALUES (?,?)", (key, json.dumps(context)))
            connection.execute(
                "INSERT INTO features VALUES (?,?,?,?,?)",
                (
                    key,
                    runner.canonical_hash(request),
                    json.dumps(request),
                    response,
                    hashlib.sha256(response.encode()).hexdigest(),
                ),
            )
        connection.commit()
    before = path.read_bytes()
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as connection:
        cache = runner.FrozenFeatures(connection, data, frozen)
        equal = cache.matrix(early, late, {"clock": "equal_group", "mu": 3.0})
        pool = cache.matrix(early, late, {"clock": "global_pool", "mu": 3.0})
        np.testing.assert_allclose(equal[0, :2], [-0.2, -0.4], rtol=0, atol=0)
        np.testing.assert_allclose(pool[0, :2], [-0.3, -0.4], rtol=0, atol=1e-16)
        np.testing.assert_array_equal(equal[:, 2:], pool[:, 2:])
        with pytest.raises(ValueError, match="no new integration"):
            cache.matrix(early * 2, late, {"clock": "equal_group", "mu": 3.0})
    assert path.read_bytes() == before


def test_training_only_noise_and_targets_are_invariant_to_held_label_poisoning():
    clean, noise = np.arange(8, dtype=float), np.linspace(-0.003, 0.003, 8)
    train = np.array([0, 3, 5])
    expected = runner.training_values(clean, noise, train)
    clean[[1, 2, 4, 6, 7]], noise[[1, 2, 4, 6, 7]] = 1e20, -1e30
    np.testing.assert_array_equal(runner.training_values(clean, noise, train), expected)


def test_fixed_generating_kinetic_profile_ignores_saved_selected_profile():
    null = {"mu": 0.0, "head": {"floor": 0.6}, "query_predictions": [1.3, 1.4]}
    signal = {"mu": 3.0, "head": {"floor": 0.6}, "query_predictions": [1.2, 1.5]}
    saved = {"selected_mu": 0.0, "selected_index": 0, "profiles": [null, signal]}
    assert runner.fixed_profile(saved, 3.0)["query_predictions"] == [1.2, 1.5]
    saved["selected_index"], saved["selected_mu"] = 1, 3.0
    assert runner.fixed_profile(saved, 0.0)["query_predictions"] == [1.3, 1.4]


def test_matched_gain_scoring_retains_bias_cancellation_and_empty_actual_ties():
    query = pd.DataFrame({"role": ["asymmetric", "asymmetric", "matched_tied", "matched_tied"]})
    truth = np.array([1.0, 1.2, 1.1, 1.5])
    result = runner.score(truth + 0.2, truth, query)
    assert result["gain_rmse"] < 3e-16
    assert result["asymmetric_rmse"] == pytest.approx(0.2)
    assert result["matched_tied_bias"] == pytest.approx(0.2)
    assert result["matched_tied_max_abs_error"] == pytest.approx(0.2)
    assert result["actual_tied_count"] == 0
    assert result["actual_tied_rmse"] is None


def test_successful_selected_head_does_not_hide_other_start_failures():
    fit = {
        "selected": {
            "start_index": 1,
            "near_constant_phi": False,
            "nonregular_learned_floor": False,
            "training_exponential_underflow_count": 0,
            "first_order_tolerance_met": True,
        },
        "selected_from_successful_solve": True,
        "solves": [
            {"start_index": 0, "success": False, "termination": "evaluation_cap"},
            {"start_index": 1, "success": True, "termination": "gradient"},
            {"start_index": 2, "success": False, "termination": "numerical_failure"},
        ],
        "trials": [{"status": "finite"}, {"status": "numerical_failure"}],
    }
    result = runner.fit_diagnostics(fit)
    assert result["selected_from_successful_solve"]
    assert result["failed_start_count"] == result["nonwinning_failed_start_count"] == 2
    assert result["evaluation_cap_count"] == result["numerical_failure_evaluation_count"] == 1
