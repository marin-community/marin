# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented-data response selection, local CV, and durable checkpoint behavior."""

import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_whole_response_clean_20261015 as model


def toy_data():
    t = np.linspace(0.05, 3, 42)
    matrix = np.column_stack((-t, t**2))
    observed = 0.6 + 1.4 * np.exp(matrix @ np.array([0.8, 0.06]))
    return matrix, observed


def test_conditional_checkpoint_recovers_replays_and_rejects_changed_training(tmp_path):
    matrix, observed = toy_data()
    path = tmp_path / "conditional.json.gz"
    fitted = model.conditional_fit(path, matrix, observed, {"toy": "clean"}, "invented-identity")
    np.testing.assert_allclose(model.solver.prior.predict(matrix, fitted["selected"]), observed, atol=1e-8, rtol=0)
    original_bytes, modified = path.read_bytes(), path.stat().st_mtime_ns
    resumed = model.conditional_fit(path, matrix, observed, {"toy": "clean"}, "invented-identity")
    assert resumed == fitted
    assert path.read_bytes() == original_bytes and path.stat().st_mtime_ns == modified
    persisted = json.loads(gzip.decompress(original_bytes))
    assert len(persisted["fit"]["solves"]) == 5
    assert persisted["fit"]["trials"]
    with pytest.raises(ValueError):
        model.conditional_fit(path, matrix, observed + 0.01, {"toy": "clean"}, "invented-identity")
    assert path.read_bytes() == original_bytes


def test_inner_held_values_cannot_change_their_fitted_prediction(tmp_path):
    matrix, observed = toy_data()
    labels = np.arange(len(observed)) % 3
    spec = {"shape_index": 0, "strength_index": 0, "mu": 0.0}
    baseline = model.profile_candidate(matrix, observed, labels, tmp_path / "baseline", spec, "toy")
    altered = observed.copy()
    altered[labels == 1] += 0.25
    poisoned = model.profile_candidate(matrix, altered, labels, tmp_path / "poisoned", spec, "toy")
    assert baseline["eligible_for_selection"] and poisoned["eligible_for_selection"]
    np.testing.assert_array_equal(
        np.asarray(baseline["inner_predictions"])[labels == 1],
        np.asarray(poisoned["inner_predictions"])[labels == 1],
    )
    assert baseline["inner_mse"] != poisoned["inner_mse"]
    held = baseline["conditional_records"][1]
    assert not set(held["train_indices"]) & set(held["held_indices"])


def test_null_tie_reuses_complete_native_prediction_and_failure_status(tmp_path):
    matrix, observed = toy_data()
    fit = model.conditional_fit(tmp_path / "native.json.gz", matrix, observed, {"toy": True}, "toy")
    native_spec = {"mu": 0.0, "strength_index": 0, "shape_index": 4}
    native_profile = {"spec": native_spec, "eligible_for_selection": True, "inner_mse": 0.1}
    other = {
        "spec": {"mu": 3.0, "strength_index": 5, "shape_index": 0},
        "eligible_for_selection": True,
        "inner_mse": 0.1,
    }
    selected = model.choose([other, native_profile])
    native = {
        "selected_candidate": native_spec,
        "selected": fit["selected"],
        "full_unresolved": True,
        "failed_source_status": "retained",
        "candidate": "aggregate_mariner",
    }
    reused = model.reuse_native_full(native, selected, "equal_group")
    np.testing.assert_array_equal(
        model.solver.prior.predict(matrix, reused["selected"]),
        model.solver.prior.predict(matrix, native["selected"]),
    )
    assert reused["full_unresolved"] and reused["failed_source_status"] == "retained"
    assert reused["native_full_fit_reused"] and native["candidate"] == "aggregate_mariner"


def test_interrupted_write_and_resource_stop_preserve_outputs(tmp_path, monkeypatch):
    destination = tmp_path / "conditional.json.gz"
    interrupted = destination.with_name(destination.name + ".writing")
    interrupted.write_bytes(b"unfinished evidence")
    with pytest.raises(ValueError):
        model.atomic_bytes(destination, b"new data")
    assert interrupted.read_bytes() == b"unfinished evidence" and not destination.exists()
    matrix, observed = toy_data()
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(model.shutil, "disk_usage", lambda path: type(usage)(usage.total, usage.used, 1024))
    with pytest.raises(ValueError):
        model.conditional_fit(destination, matrix, observed, {"toy": True}, "toy")
    assert not list(tmp_path.glob("resource_stop_*.json"))
    with pytest.raises(RuntimeError):
        model.conditional_fit(tmp_path / "fresh.json.gz", matrix, observed, {"toy": True}, "toy")
    assert not (tmp_path / "fresh.json.gz").exists()
    receipt = json.loads(next(tmp_path.glob("resource_stop_*.json")).read_text())
    assert receipt["scientific_failure"] is False and receipt["free_bytes"] == 1024
    assert interrupted.read_bytes() == b"unfinished evidence"


def test_degenerate_profile_remains_eligible_but_unresolved(tmp_path):
    matrix = np.zeros((18, 2))
    observed = np.ones(18)
    profile = model.profile_candidate(
        matrix, observed, np.arange(18) % 3, tmp_path, {"shape_index": 0, "strength_index": 0}, "toy"
    )
    assert profile["eligible_for_selection"] and profile["inner_mse"] == 0
    assert profile["any_unresolved_inner"]
    diagnostics = model.profile_diagnostics([profile])
    assert diagnostics["unresolved_candidates"] == 1
    assert diagnostics["failed_inner_starts"] == 15
    for path in tmp_path.glob("inner*.json.gz"):
        record = json.loads(gzip.decompress(path.read_bytes()))
        assert record["fit"]["unresolved"] and len(record["fit"]["trials"]) == 5


def test_invented_full_query_metrics_feed_historical_gates_without_hiding_missing_native():
    query = pd.DataFrame(
        {
            "role": ["asymmetric"] * 3 + ["matched_tied"] * 3 + ["actual_tied"] * 2,
            "row_id": [f"toy-{i}" for i in range(8)],
        }
    )
    truth = np.array([1.0, 1.1, 1.2, 1.0, 1.1, 1.2, 1.05, 1.15])
    predicted = truth.copy()
    predicted[1] = 0.95
    score = model.prior.score(predicted, truth, query) | model.stage.menu_scores(predicted, truth, query)
    gain = predicted[3:6] - predicted[:3]
    score.update({"maximum_positive_predicted_gain": float(max(0, np.max(gain))), "selected_upper_boundary": False})
    assert score["regret_at_1"] == pytest.approx(0.1)
    assert score["regret_at_3"] == 0 and score["regret_at_5"] == 0
    assert score["selected_row_id"] == "toy-1"
    assert score["frontier_indices"] == [1, 0]
    assert score["frontier_mean_optimism"] == pytest.approx(0.075)
    assert score["gain_rmse"] == pytest.approx(0.15 / np.sqrt(3))
    world = {"name": "invented", "kind": "native", "clock": None}
    checks = model.performance_checks(world, "aggregate_mariner", score, score)
    assert checks is not None
    failed = {item["name"] for item in checks if not item["passed"]}
    assert {"asymmetric_rmse", "regret_at_1", "exact_null_absolute_gain", "null_gain_rmse"} <= failed
    assert not any(item["name"].startswith("unresolved_") for item in checks)
    assert model.performance_checks(world, "equal_group", score, None) is None
