# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy-only numerical and artifact-boundary checks for the geometry runner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_geometry_20261015 as geometry,
)


def physical_integrand(
    u: float,
    start: np.ndarray,
    increment: np.ndarray,
    total: np.ndarray,
    receiver: int,
    source: int,
    rate: float,
    power: float,
) -> float:
    exposure = start + increment * u ** (1 / power)
    h = (rate * exposure) ** power
    if start[source] == 0:
        dh = (rate * increment[source]) ** power
    else:
        dh = rate**power * exposure[source] ** (power - 1) * increment[source] * u ** (1 / power - 1)
    return float(
        np.exp(h[receiver] - (rate * total[receiver]) ** power) * (-np.expm1(-h[receiver])) * np.exp(-h[source]) * dh
    )


def physical_reference(early: np.ndarray, late: np.ndarray, rate: float, power: float) -> np.ndarray:
    """Independent physical-time quadrature, not the production benefit coordinate."""
    result = np.zeros(2)
    total = early + late
    for start, increment in ((np.zeros(2), early), (early, late)):
        for receiver, source in ((0, 1), (1, 0)):
            if increment[source] == 0:
                continue
            value, error = quad(
                physical_integrand,
                0,
                1,
                args=(start, increment, total, receiver, source, rate, power),
                epsabs=2e-13,
                epsrel=2e-12,
                limit=300,
            )
            assert error < 2e-10
            result[receiver] += value
    return result


@pytest.mark.parametrize("rate,power", [(0.05, 0.3), (0.5, 0.7), (4.0, 0.5), (1.0, 1.0)])
def test_pair_quadrature_matches_independent_physical_path(rate: float, power: float) -> None:
    early = np.array([[0.2, 1.56], [2.0, 0.0], [0.0, 0.4], [0.3, 0.02]])
    late = np.array([[0.0, 0.44], [0.2, 1.3], [3.0, 0.0], [1.4, 2.0]])
    result = geometry.pair_integrals(early, late, rate, power)
    expected = np.array([physical_reference(e, l, rate, power) for e, l in zip(early, late, strict=True)])
    assert np.all(result.converged)
    np.testing.assert_allclose(result.value, expected, rtol=3e-9, atol=5e-11)


def test_zero_kinetic_counterexample_and_saturated_receiver() -> None:
    early = np.array([[0.2, 1.56], [40.0, 0.0]])
    late = np.array([[0.0, 0.44], [0.0, 1.0]])
    result = geometry.zero_kinetic_geometry(early, late, 0.8, 1.0, 1.0)
    expected_first = np.exp(-np.array([0.2, 2.0])) * (
        np.array([0.06278096299040536, 0.3798026935683092]) - np.array([0.0793509866395033, 0.24464153275374875])
    )
    np.testing.assert_allclose(result["derivative"][0], expected_first, rtol=1e-10, atol=2e-11)
    x, y = 40.0, 1.0
    tied = y * np.expm1(x - y) / (x - y) - (-np.expm1(-y))
    path = np.expm1(x) * (-np.expm1(-y))
    expected_saturated = np.exp(-x) * (tied - path)
    assert result["derivative"][1, 0] < -0.6
    assert abs(result["derivative"][1, 0] - expected_saturated) < 3e-10


def test_tied_zero_dose_and_clock_weights_preserve_native_identity() -> None:
    total = np.array([[0.0, 0.0], [0.0, 2.0], [0.03, 40.0], [2.0, 1.0]])
    result = geometry.zero_kinetic_geometry(0.8 * total, 0.2 * total, 0.8, 0.25, 0.3)
    assert np.array_equal(result["derivative"], np.zeros_like(total))
    assert np.max(np.abs(result["raw_derivative"])) < 1e-10
    assert np.array_equal(result["path"][:2], np.zeros((2, 2)))
    weights = geometry.clock_weights(np.array([1.0, 4.0]))
    np.testing.assert_allclose(weights["global_pool"], [0.8, 0.2], rtol=0, atol=1e-15)
    # One common kinetic rescaling cannot match both receiver clocks.
    assert weights["global_pool"][0] / weights["equal_group"][0] != weights["global_pool"][1] / weights["equal_group"][1]


def test_shape_nuisance_columns_match_native_finite_differences() -> None:
    exposure = np.array([[0.0, 0.03], [0.1, 1.4], [3.0, 30.0]])
    shape = {"rate": 0.5, "power": 0.7, "threshold": 2.0}
    matrix = geometry.nuisance_design(exposure, shape, "columnwise_shape_known_floor")
    step = 1e-5
    for key, columns in (("rate", slice(5, 7)), ("power", slice(7, 9)), ("threshold", slice(9, 11))):
        plus, minus = dict(shape), dict(shape)
        plus[key] = shape[key] * np.exp(step) if key == "rate" else shape[key] + step
        minus[key] = shape[key] * np.exp(-step) if key == "rate" else shape[key] - step
        native = (geometry.mariner.design_matrix(exposure, plus) - geometry.mariner.design_matrix(exposure, minus)) / (
            2 * step
        )
        expected = native[:, :2] if key != "threshold" else native[:, 2:]
        np.testing.assert_allclose(matrix[:, columns], expected, rtol=1e-8, atol=2e-10)


def test_nuisance_projection_distinguishes_absorbed_and_surviving_directions() -> None:
    x = np.linspace(-1, 1, 11)
    design = np.column_stack([np.ones_like(x), x, 1000 * x])
    directions = np.column_stack([3 + 2 * x, x * x])
    metrics, residual = geometry.residual_projection(design, directions, np.zeros_like(directions))
    assert metrics["nuisance_numerical_rank"] == 2
    np.testing.assert_allclose(residual[:, 0], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(residual[:, 1], x * x - np.mean(x * x), rtol=0, atol=1e-14)


def toy_artifacts(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    p0 = np.linspace(0.05, 0.9, 10)
    p0[0] = 0
    p1 = np.linspace(0.95, 0.1, 10)
    early_weights = np.column_stack([1 - p0, p0])
    late_weights = np.column_stack([1 - p1, p1])
    early_weights = np.vstack([early_weights, [0.5, 0.5]])
    late_weights = np.vstack([late_weights, [0.5, 0.5]])
    c0, c1 = np.array([0.8, 3.2]), np.array([0.2, 0.8])
    np.savez_compressed(
        folder / "policies.npz",
        phase0=early_weights,
        phase1=late_weights,
        phase0_epochs=early_weights * c0,
        phase1_epochs=late_weights * c1,
        c0=c0,
        c1=c1,
    )
    (folder / "metadata.json").write_text(
        json.dumps({"name": geometry.PANEL, "alpha": 0.8, "buckets": ["nemotron_full", "starcoder"]})
    )
    rows = pd.DataFrame(
        {
            "row_id": [f"row_{i}" for i in range(11)],
            "policy_id": [f"p_{i}" for i in range(11)],
            "group_id": [f"g_{i}" for i in range(11)],
            "source": "toy",
            "region": "toy",
            "tied": [False] * 10 + [True],
        }
    )
    rows.to_csv(folder / "rows.csv", index=False)
    folds = [
        {
            "panel": geometry.PANEL,
            "repeat": 0,
            "outer_fold": fold,
            "row_id": f"row_{i}",
            "role": "test" if i // 2 == fold else "train",
            "inner_fold": -1,
        }
        for fold in range(5)
        for i in range(10)
    ]
    folds.append(
        {
            "panel": geometry.PANEL,
            "repeat": 0,
            "outer_fold": 0,
            "row_id": "row_10",
            "role": "tied_test",
            "inner_fold": -1,
        }
    )
    pd.DataFrame(folds).to_csv(folder / "folds.csv", index=False)
    # A nonexistent/invalid outcome table must be irrelevant to this reader.
    (folder / "outcomes.csv").write_bytes(b"not a valid outcome table\xff")


def test_geometry_reader_uses_only_covariates_and_original_training_rows(tmp_path: Path) -> None:
    toy_artifacts(tmp_path)
    data = geometry.read_geometry(tmp_path)
    assert len(data.subsets["full_asymmetric"]) == 10
    assert list(data.rows.iloc[data.subsets["outer_0_train"]].row_id) == [f"row_{i}" for i in range(2, 10)]
    assert all(10 not in subset for subset in data.subsets.values())
    with np.load(tmp_path / "policies.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.savez_compressed(tmp_path / "policies.npz", **arrays, final_targets=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="outcomes are forbidden"):
        geometry.read_geometry(tmp_path)


def test_geometry_cell_replay_checks_durable_outputs(tmp_path: Path) -> None:
    toy_artifacts(tmp_path / "inputs")
    data = geometry.read_geometry(tmp_path / "inputs")
    output = tmp_path / "cell"
    identity = {"toy_only": True}
    geometry.analyze_shape(data, 0.25, 0.7, output, identity)
    digest = geometry.sha256(output / "complete.json")
    geometry.analyze_shape(data, 0.25, 0.7, output, identity)
    assert geometry.sha256(output / "complete.json") == digest
    rows = pd.read_csv(output / "row_diagnostics.csv")
    assert set(rows.clock) == set(geometry.CLOCKS)
    assert rows.loc[rows.row_id == "row_0", "zero_early_code"].all()
    assert len(json.loads((output / "projections.json").read_text())) == 2 * 6 * 6 * 2
    with pytest.raises(ValueError, match="another shape"):
        geometry.analyze_shape(data, 0.5, 0.7, output, identity)
    (output / "projections.json").write_text("[]\n")
    with pytest.raises(ValueError, match="evidence changed"):
        geometry.analyze_shape(data, 0.25, 0.7, output, identity)
