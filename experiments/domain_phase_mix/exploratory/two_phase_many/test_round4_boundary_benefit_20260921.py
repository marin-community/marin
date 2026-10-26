# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import round4_boundary_benefit_20260921 as model
from test_nextgen_two_phase_round3_20260921 import panel as sampled_panel

SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}


def policies(first_code: np.ndarray, last_code: np.ndarray) -> model.benchmark.data.BenchPanel:
    return model.benchmark.data.BenchPanel(
        "synthetic_boundary_fraction",
        ("web", "code"),
        np.column_stack([1 - first_code, first_code]),
        np.column_stack([1 - last_code, last_code]),
        0.8,
        np.array([0.8, 21.0]),
        np.array([0.2, 5.25]),
        {},
        pd.DataFrame({"row_id": np.arange(len(first_code)).astype(str), "tied": first_code == last_code}),
        {},
        {},
    )


@pytest.mark.parametrize("q,kappa", [(0.0, 0.5), (0.5, 2.0), (1.0, 2.0)])
@pytest.mark.parametrize("floor", [-3.0, 0.7])
def test_tied_policies_map_exactly_to_native_response(q: float, kappa: float, floor: float) -> None:
    panel = sampled_panel()
    native = model.mariner.design_matrix(panel.epochs, SHAPE)
    actual = model.design_matrix(panel, SHAPE, q, kappa)
    head = model.mariner.Head(0.5, np.array([6, 1, 0.03, 0.06]), floor)
    np.testing.assert_array_equal(actual[panel.tied], native[panel.tied])
    np.testing.assert_array_equal(head.predict(actual[panel.tied]), head.predict(native[panel.tied]))


def test_zero_strength_is_exact_native_on_asymmetric_policies() -> None:
    panel = sampled_panel()
    np.testing.assert_array_equal(
        model.design_matrix(panel, SHAPE, 0.37, 0.0), model.mariner.design_matrix(panel.epochs, SHAPE)
    )


def test_factored_correction_matches_independent_difference_of_squares() -> None:
    panel = sampled_panel()
    total = model.mariner.benefit(panel.epochs, SHAPE["rate"], SHAPE["power"])
    prefix = model.mariner.benefit(panel.phase0_epochs, SHAPE["rate"], SHAPE["power"])
    tied = model.mariner.benefit(panel.alpha * panel.epochs, SHAPE["rate"], SHAPE["power"])
    for q in (0.0, 0.5, 1.0):
        expected = model.mariner.design_matrix(panel.epochs, SHAPE)
        expected[:, :2] += 2 * ((prefix - q * total) ** 2 - (tied - q * total) ** 2)
        np.testing.assert_allclose(model.design_matrix(panel, SHAPE, q, 2.0), expected, rtol=0, atol=5e-16)


def test_single_bucket_target_and_curvature_at_fixed_total_exposure() -> None:
    total_exposure, q, kappa = 4.0, 0.5, 2.0
    final_benefit = -np.expm1(-((SHAPE["rate"] * total_exposure) ** SHAPE["power"]))
    target = q * final_benefit
    step = 0.01
    acquired = target + np.array([-step, 0.0, step])
    prefix_exposure = (-np.log1p(-acquired)) ** (1 / SHAPE["power"]) / SHAPE["rate"]
    first_code = prefix_exposure / total_exposure
    panel = dataclasses.replace(
        policies(first_code, 1 - first_code), alpha=0.5, c0=np.array([1.0, 4.0]), c1=np.array([1.0, 4.0])
    )
    np.testing.assert_allclose(panel.epochs[:, 1], total_exposure, rtol=0, atol=1e-15)
    column = model.design_matrix(panel, SHAPE, q, kappa)[:, 1]
    assert column[1] < min(column[0], column[2])
    np.testing.assert_allclose(column[[0, 2]] - column[1], kappa * step**2, rtol=0, atol=5e-16)
    assert abs((column[2] - 2 * column[1] + column[0]) / step**2 - 2 * kappa) < 1e-11


def test_actual_simplex_can_have_an_interior_two_phase_minimum() -> None:
    first, last = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201), indexing="ij")
    panel = policies(first.ravel(), last.ravel())
    head = model.mariner.Head(0.5, np.array([6, 1, 0.03, 0.06]), 0.5)
    values = head.predict(model.design_matrix(panel, SHAPE, 0.5, 0.5)).reshape(first.shape)
    position = np.unravel_index(np.argmin(values), values.shape)
    assert 0 < position[0] < len(values) - 1
    assert 0 < position[1] < len(values) - 1
    boundary = np.r_[values[0], values[-1], values[:, 0], values[:, -1]]
    assert values[position] < boundary.min() - 1e-5
    assert values[position] < np.diag(values).min() - 1e-3
