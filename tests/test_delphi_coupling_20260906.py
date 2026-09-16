# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import fit_delphi_coupling_20260906 as coupling


@pytest.mark.parametrize("basis", list(coupling.Basis))
@pytest.mark.parametrize("link", list(coupling.Link))
def test_response_jacobian_matches_finite_differences(basis, link):
    rng = np.random.default_rng(8)
    weights = rng.dirichlet(np.ones(5), size=30)
    query = rng.dirichlet(np.ones(5), size=7)
    design = coupling.design_matrix(weights, query, np.arange(1, 6), basis)
    parameters = rng.uniform(0.01, 0.04, design.projection.shape[1] + 1)
    _, actual, _ = coupling.response_jacobian(parameters, design.query, design.projection, link)
    step = 1e-6
    numerical = []
    for direction in np.eye(len(parameters)):
        above = coupling.response_jacobian(parameters + step * direction, design.query, design.projection, link)[0]
        below = coupling.response_jacobian(parameters - step * direction, design.query, design.projection, link)[0]
        numerical.append((above - below) / (2 * step))
    np.testing.assert_allclose(actual, np.column_stack(numerical), atol=1e-7, rtol=1e-5)


def test_removing_interactions_preserves_single_bucket_responses():
    parameters = np.array([0.3, 0.2, -0.1, 0.4])
    matrix = np.zeros((9, 3, 1))
    matrix[:, 1, 0] = np.linspace(-2, 2, len(matrix))
    additive, additive_jacobian, _ = coupling.response_jacobian(parameters, matrix, np.eye(3), coupling.Link.ADDITIVE)
    coupled, coupled_jacobian, _ = coupling.response_jacobian(parameters, matrix, np.eye(3), coupling.Link.COUPLED)
    np.testing.assert_allclose(additive, coupled, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(additive_jacobian, coupled_jacobian, rtol=1e-13, atol=1e-13)


def test_share_gauge_removes_null_direction_for_both_links():
    rng = np.random.default_rng(22)
    weights = rng.dirichlet(np.ones(5), size=40)
    design = coupling.design_matrix(weights, weights[:3], np.ones(5), coupling.Basis.SHARES)
    np.testing.assert_allclose(design.projection.T @ design.scale.ravel(), 0, atol=1e-14)
    np.testing.assert_allclose(design.projection.T @ design.projection, np.eye(4), atol=1e-14)
    matrix = design.training.reshape(len(weights), -1) @ design.projection
    assert np.linalg.matrix_rank(matrix) == 4


def test_additive_extrapolation_is_not_silently_clipped():
    parameters = np.r_[0.0, -np.ones(4)]
    matrix = np.ones((1, 4, 1))
    actual, _, clipped = coupling.response_jacobian(parameters, matrix, np.eye(4), coupling.Link.ADDITIVE)
    np.testing.assert_allclose(actual, 1 + 4 * np.expm1(-1), atol=1e-14)
    assert actual[0] < 0
    assert clipped == 0


@pytest.mark.parametrize("spec", coupling.SPECS, ids=lambda spec: spec.name)
def test_outer_labels_cannot_change_coupling_fit(tmp_path, spec):
    rng = np.random.default_rng(19)
    weights = rng.dirichlet(np.ones(4), size=36)
    inventory = np.arange(1, 5)
    outcomes = np.exp(0.2 * weights @ rng.normal(size=(4, 2)))
    panel = {
        "weights": weights,
        "inventory": inventory,
        "table9_outcomes": outcomes,
        "table9_aggregation_weights": np.ones(2) / 2,
    }
    (tmp_path / "inputs").mkdir()
    benchmark.harness.atomic_save(tmp_path / "inputs" / "panel.npz", panel)
    benchmark.harness.atomic_save(tmp_path / "inputs" / "table9_bank_features.npz", {"weights": weights[:5]})
    rows = np.arange(len(weights))
    pd.DataFrame(
        {
            "repeat": 0,
            "fold": 0,
            "row": rows,
            "role": np.where(rows < 27, "train", "test"),
            "inner_fold": np.where(rows < 27, rows % 3, -1),
        }
    ).to_csv(tmp_path / "inputs" / "splits.csv", index=False)
    coupling.fit_specification(tmp_path, spec, "table9", 0, 0, "original")
    path = tmp_path / "shards" / spec.name / "table9" / "r0_f0.npz"
    original = benchmark.read_npz(path)
    panel["table9_outcomes"][27:] += 1000
    benchmark.harness.atomic_save(tmp_path / "inputs" / "panel.npz", panel)
    coupling.fit_specification(tmp_path, spec, "table9", 0, 0, "poisoned")
    poisoned = benchmark.read_npz(path)
    assert str(original["selected_json"]) == str(poisoned["selected_json"])
    np.testing.assert_array_equal(original["prediction"], poisoned["prediction"])
    np.testing.assert_array_equal(original["bank_prediction"], poisoned["bank_prediction"])
    np.testing.assert_array_equal(original["removed_bank_prediction"], poisoned["removed_bank_prediction"])
