# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Noisy-context lineage, checkpoint reuse and extreme held-score preservation."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_whole_response_fold3_noise_20261015 as model,
)


def query_frame():
    return pd.DataFrame(
        {
            "role": ["asymmetric"] * 3 + ["matched_tied"] * 3 + ["actual_tied"] * 3,
            "row_id": [f"row{i}" for i in range(9)],
        }
    )


def test_ordinary_metrics_and_historical_checks_are_unchanged():
    query = query_frame()
    truth = np.linspace(0.8, 1.2, len(query))
    predicted = truth + np.linspace(-0.0001, 0.0001, len(query))
    metrics, affected = model.finite_metrics(predicted, truth, query)
    assert metrics == model.prior.score(predicted, truth, query) | model.stage.menu_scores(predicted, truth, query)
    assert affected == []
    score = metrics | {
        "maximum_positive_predicted_gain": 0.0,
        "selected_upper_boundary": False,
        "maximum_absolute_predicted_gain": 0.0,
        "gain_rmse": 0.0,
    }
    world = next(w for w in model.backend.WORLDS if w["kind"] == "native")
    checks = model.backend.performance_checks(world, "aggregate_mariner", score, score)
    assert checks is not None and all(c["passed"] for c in checks)
    assert model.backend.performance_checks(world, "equal_group", score, None) is None


def test_large_finite_predictions_keep_representable_rmse_without_clipping(tmp_path):
    query = query_frame()
    predicted = np.full(len(query), 1e200)
    truth = np.ones(len(query))
    metrics, affected, failure = model.save_query_summary(tmp_path, predicted, truth, query, {"toy": True})
    assert failure is None
    assert metrics["query_rmse"] == pytest.approx(1e200, rel=1e-15)
    assert metrics["gain_rmse"] == 0
    assert metrics["query_rmse"] > 1e199 and "query_rmse" in affected
    assert np.array_equal(pd.read_csv(tmp_path / "predictions.csv").predicted, predicted)
    json.dumps(metrics, allow_nan=False)


def test_unrepresentable_metrics_preserve_predictions_and_explicit_failure(tmp_path):
    query = query_frame()
    predicted = np.full(len(query), 1.6e308)
    truth = np.full(len(query), -1.6e308)
    _, _, failure = model.save_query_summary(tmp_path, predicted, truth, query, {"toy": True})
    assert failure is not None and failure["finite_predictions_saved"]
    assert "query_rmse" in failure["invalid_metrics"]
    assert failure["metrics"]["query_rmse"] is None
    assert model.prior.read(tmp_path / "query_failure.json") == failure
    assert model.prior.digest(tmp_path / "predictions.csv") == failure["predictions_sha256"]
    assert np.isfinite(pd.read_csv(tmp_path / "predictions.csv").predicted).all()


def test_training_uses_exact_noisy_fold3_receipt_and_row_order(tmp_path, monkeypatch):
    monkeypatch.setattr(model.prior, "OUTPUT", tmp_path)
    world = {"name": "invented", "kind": "native"}
    noise = model.NOISES[1]
    train = np.array([9, 2, 15])
    folder = tmp_path / "cells/outer3" / noise / "invented/initialization"
    folder.mkdir(parents=True)
    (tmp_path / "execution_identity.json").write_text('{"toy_identity": true}')
    (folder / "starts.json").write_text(json.dumps({"train_indices": train.tolist(), "observed": [1.1, 0.9, 1.3]}))
    receipt = {
        "identity": {"toy_identity": True},
        "spec": {"world": world, "outer_fold": 3, "noise": noise, "stage": "five_gamma_starts"},
        "files": {"starts.json": model.prior.digest(folder / "starts.json")},
    }
    (folder / "complete.json").write_text(json.dumps(receipt))
    np.testing.assert_array_equal(model.training_only(world, noise, train), [1.1, 0.9, 1.3])
    with pytest.raises(ValueError, match="row order"):
        model.training_only(world, noise, train[::-1])
    receipt["spec"]["outer_fold"] = 0
    (folder / "complete.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="world/fold/noise"):
        model.training_only(world, noise, train)


def test_noisy_conditional_fit_is_fresh_and_resumable(tmp_path):
    x = np.linspace(-0.6, 0.7, 15).reshape(-1, 1)
    y = 0.6 + np.exp(-0.2 + 0.3 * x[:, 0])
    path = tmp_path / "noisy.json.gz"
    spec = {"world": "invented", "noise": "20261016", "outer_fold": 3, "inner_fold": 0}
    fitted = model.template.conditional_fit(path, x, y, spec, "toy-identity")
    record = json.loads(gzip.decompress(path.read_bytes()))
    assert "adopted_from" not in record and record["header"]["spec"] == spec
    original = path.read_bytes()
    assert model.template.conditional_fit(path, x, y, spec, "toy-identity") == fitted
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        model.template.conditional_fit(path, x, y + 0.001, spec, "toy-identity")
    assert len(fitted["solves"]) == 5 and fitted["selected"] is not None
    np.testing.assert_allclose(model.solver.prior.predict(x, fitted["selected"]), y, rtol=0, atol=1e-8)


def test_held_truth_is_unopened_when_any_full_fit_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "OUTPUT", tmp_path)
    (tmp_path / "execution_identity.json").write_text("{}")
    monkeypatch.setattr(model, "execution_identity", lambda: {})
    monkeypatch.setattr(model.backend, "completed_fit", lambda *args: False)
    with pytest.raises(ValueError, match="All thirty"):
        model.score_all()
    assert not list(tmp_path.glob("cells/**/scores.json"))
