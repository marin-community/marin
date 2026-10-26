# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Cache isolation and paired-score behavior for the unchanged-J2 diagnostic."""

import sqlite3
from contextlib import closing

import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_unchanged_j2_diagnostic_20261015 as runner


def test_closed_cache_fork_can_extend_without_mutating_source(tmp_path):
    source, destination = tmp_path / "sealed.sqlite", tmp_path / "diagnostic/features.sqlite"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE features(key TEXT PRIMARY KEY, value REAL)")
        connection.execute("INSERT INTO features VALUES ('existing', 1.25)")
    original = source.read_bytes()
    expected = runner.benchmark.sha256(source)
    origin = runner.copy_closed_cache(source, destination, expected)
    assert destination.read_bytes() == original
    assert runner.copy_closed_cache(source, destination, expected) == origin
    with closing(sqlite3.connect(destination)) as connection, connection:
        connection.execute("INSERT INTO features VALUES ('query', 2.5)")
    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 2
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="Unreceipted cache"):
        runner.copy_closed_cache(source, destination, expected)


def test_paired_calibration_delta_and_zero_effect_baseline_are_distinct():
    supplied = {
        "query_rmse": 0.001,
        "query_max_abs_error": 0.002,
        "asymmetric_rmse": 0.001,
        "matched_tied_rmse": 0.0015,
        "actual_tied_rmse": 0.002,
        "gain_rmse": 0.0004,
        "frontier_mean_optimism": -0.001,
        "regret_at_1": 0.0,
        "regret_at_3": 0.0,
        "regret_at_5": 0.0,
    }
    current = supplied | {"actual_tied_rmse": 0.005, "gain_rmse": 0.003, "true_gain_rms": 0.002}
    result = runner.comparison_readout(current, supplied)
    assert result["delta_vs_known_floor"]["actual_tied_rmse"] == pytest.approx(0.003)
    assert result["delta_vs_known_floor"]["gain_rmse"] == pytest.approx(0.0026)
    assert result["gain_rmse_divided_by_zero_gain_baseline"] == pytest.approx(1.5)
    null = runner.comparison_readout(current | {"true_gain_rms": 0.0}, supplied)
    assert null["gain_rmse_divided_by_zero_gain_baseline"] is None
    assert null["delta_vs_known_floor"] == result["delta_vs_known_floor"]


def test_interrupted_cache_copy_is_preserved_before_retry(tmp_path):
    source, destination = tmp_path / "sealed.sqlite", tmp_path / "features.sqlite"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE features(key TEXT PRIMARY KEY)")
    partial = destination.with_name(destination.name + ".copying")
    original_partial = b"interrupted copy evidence"
    partial.write_bytes(original_partial)
    with pytest.raises(ValueError, match="interrupted cache copy"):
        runner.copy_closed_cache(source, destination, runner.benchmark.sha256(source))
    assert partial.read_bytes() == original_partial
    assert not destination.exists()
