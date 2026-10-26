# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy matching-law truth, qualification and execution-boundary behavior."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mean_coordinate_stage2_20261015 as runner


@pytest.mark.parametrize("family", runner.FAMILIES)
def test_gate_applicability_absolute_null_gain_and_physical_boundary(family: str) -> None:
    null, positive = (world for world in runner.WORLDS if world["matching_family"] == family)
    baseline = {"clean_actual_tied_rmse": 0.02}
    scores = {"max_abs_predicted_gain": 2e-8, "clean_gain_rmse": 0.0009, "clean_actual_tied_rmse": 0.022}
    result = runner.qualification_checks(null, family, {"lambda": 0.5}, scores, baseline)
    assert not result["null_gain_pass"]
    assert result["gain_recovery_pass"] and result["tied_calibration_pass"]
    boundary = runner.qualification_checks(positive, family, {"lambda": 1.0}, scores, baseline)
    assert boundary["null_gain_pass"] and not boundary["kinetic_boundary_pass"]
    missed = {**scores, "clean_gain_rmse": 0.002, "clean_actual_tied_rmse": 0.024}
    failed = runner.qualification_checks(positive, family, {"lambda": 0.5}, missed, baseline)
    assert not failed["gain_recovery_pass"] and not failed["tied_calibration_pass"]
    aggregate = runner.qualification_checks(positive, "aggregate_mariner", {}, missed, baseline)
    assert not aggregate["correct_family"]
    assert aggregate["gain_recovery_pass"] and aggregate["tied_calibration_pass"]
    # These are observable diagnostics, never retroactive noiseless acceptance gates.
    raw_failure = {**scores, "regret_vs_truth_numerical_incumbent": 10.0, "selected_solver_success": False}
    assert runner.qualification_checks(null, family, {"lambda": 0.5}, raw_failure, baseline) == result


@pytest.mark.parametrize("world", runner.WORLDS)
def test_family_input_truth_and_raw_predictor_match_independent_equations(world: dict) -> None:
    rng = np.random.default_rng(802)
    total = rng.uniform(0.1, 12.0, (280, 39))
    fraction = rng.uniform(0.0, 1.0, total.shape)
    alpha, beta = 0.8, 0.2
    fraction[238:] = alpha
    early, late = total * fraction, total * (1 - fraction)
    arrays = {
        "phase0_epochs": early,
        "phase1_epochs": late,
        "alpha": np.asarray(alpha),
        "tied": np.arange(280) >= 238,
        "actual_targets": np.full((280, 7), np.nan),
    }
    folds = pd.DataFrame(
        {"row_index": np.arange(238), "role": ["train"] * 172 + ["test"] * 66, "inner_fold": np.arange(238) % 3}
    )
    coefficients = np.r_[np.linspace(0.01, 0.03, 39), np.linspace(0.001, 0.003, 39)]
    head = runner.models.mariner.Head(35.0, coefficients, 0.6)
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    family, strength = world["matching_family"], world["parameters"]["lambda"]
    cumulative = early + late
    native = -np.expm1(-((0.25 * cumulative) ** 0.7))
    late_hazard = (0.25 * late / beta) ** 0.7
    if family == "hazard_coordinate_mean":
        benefit = -np.expm1(-((1 - strength) * (0.25 * cumulative) ** 0.7 + strength * late_hazard))
    else:
        benefit = (1 - strength) * native - strength * np.expm1(-late_hazard)
    harm = np.logaddexp(0.0, np.log1p(cumulative) - 3.0) ** 2
    expected = 0.6 + np.exp(35.0 - benefit @ coefficients[:39] + harm @ coefficients[39:])
    module = runner.world_runner(world)
    values = module.oracle.world_inputs(arrays, folds, head, strength)
    specification = {
        "world": world,
        "parameters": world["parameters"],
        "shape": shape,
        "head": {"intercept": 35.0, "coefficients": coefficients.tolist(), "floor": 0.6},
    }
    predicted = runner.SyntheticTruth(specification, alpha).predict(early, late)
    np.testing.assert_allclose(values["clean"], expected, rtol=8e-15, atol=0.0)
    np.testing.assert_array_equal(predicted, values["clean"])
    assert np.min(predicted) > np.exp(30)
    np.testing.assert_array_equal(values["train"], np.arange(172))
    np.testing.assert_array_equal(values["held"], np.arange(172, 238))
    np.testing.assert_array_equal(values["tied"], np.arange(238, 280))
    np.testing.assert_allclose(values["clean"][238:], values["clean_tied"][238:], rtol=8e-15, atol=0.0)


def test_draft_cannot_freeze_or_create_scientific_contexts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    draft = {
        "candidate_ids": runner.CANDIDATES,
        "stage": 2,
        "status": "draft",
        "execution_contract": runner.CONTRACT,
    }
    (tmp_path / "registration.json").write_text(json.dumps(draft))
    for command in (runner.freeze, runner.prepare):
        with pytest.raises(ValueError):
            command()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["registration.json"]
