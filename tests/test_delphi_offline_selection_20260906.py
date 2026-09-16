# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import delphi_selection_models_20260906 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_selection_20260906 as scoring


def test_weighted_ridge_matches_sklearn_predictions():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 6))
    response = rng.normal(size=(40, 3))
    query = rng.normal(size=(9, 6))
    precision = rng.uniform(0.3, 2.0, size=40)
    mean = np.average(matrix, axis=0, weights=precision)
    scale = np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=precision))
    expected = Ridge(alpha=0.1).fit((matrix - mean) / scale, response, sample_weight=precision)
    actual = models.ridge_prediction(matrix, response, query, 0.1, precision, np.ones(3) / 3)
    np.testing.assert_allclose(actual.values, expected.predict((query - mean) / scale), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("spec", models.SPECS, ids=lambda spec: spec.name)
def test_outer_test_labels_cannot_change_tuning_or_predictions(tmp_path, spec):
    rng = np.random.default_rng(9)
    weights = rng.dirichlet(np.ones(5), size=42)
    outcomes = 2 + weights @ rng.normal(size=(5, 4)) + rng.normal(scale=0.03, size=(42, 4))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    panel = {
        "weights": weights,
        "exposures": weights * np.arange(1, 6),
        "table9_outcomes": outcomes,
        "table9_aggregation_weights": np.ones(4) / 4,
    }
    benchmark.harness.atomic_save(inputs / "panel.npz", panel)
    query = rng.dirichlet(np.ones(5), size=12)
    benchmark.harness.atomic_save(
        inputs / "table9_bank_features.npz", {"weights": query, "exposures": query * np.arange(1, 6)}
    )
    rows = np.arange(42)
    pd.DataFrame(
        {
            "repeat": 0,
            "fold": 0,
            "row": rows,
            "role": np.where(rows < 33, "train", "test"),
            "inner_fold": np.where(rows < 33, rows % 3, -1),
        }
    ).to_csv(inputs / "splits.csv", index=False)
    models.fit_alternative(tmp_path, spec, "table9", 0, 0, "first")
    path = tmp_path / "alternative_shards" / spec.name / "table9" / "r0_f0.npz"
    first = benchmark.read_npz(path)
    panel["table9_outcomes"][33:] += rng.normal(scale=1000, size=(9, 4))
    benchmark.harness.atomic_save(inputs / "panel.npz", panel)
    models.fit_alternative(tmp_path, spec, "table9", 0, 0, "changed_test_labels")
    second = benchmark.read_npz(path)
    assert str(first["selected_json"]) == str(second["selected_json"])
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    np.testing.assert_array_equal(first["bank_prediction"], second["bank_prediction"])


def test_source_membership_chains_stay_together():
    sources = pd.Series(["a", "a;b", "b;c", "c", "d", "e;f", "f"])
    blocks, memberships = scoring.source_blocks(sources)
    assert len(set(blocks[:4])) == 1
    assert len(set(blocks[5:])) == 1
    assert len(set(blocks)) == 3
    for block in np.unique(blocks):
        train = set.union(*(memberships[i] for i in np.flatnonzero(blocks != block)))
        test = set.union(*(memberships[i] for i in np.flatnonzero(blocks == block)))
        assert not train & test


def test_selection_regret_uses_predicted_shortlists():
    measured = np.arange(1.0, 13.0)
    result = benchmark.selection_metrics(measured, -measured)
    assert result["regret_at_1"] == 11
    assert result["best_of_5_regret"] == 7
    assert result["best_of_10_regret"] == 2
    assert result["selected_rank"] == 12
    assert result["optimism"] == 24
