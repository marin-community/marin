# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy saved-artifact behavior; no scientific outcome or result reads."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_conditional_phase_bounds_20261015 as runner


def toy_head(intercept: float, floor: float) -> dict[str, Any]:
    return {
        "head": {"intercept": intercept, "floor": floor, "coefficients": [0.0, 0.0, 0.0, 0.0]},
        "shape": {"rate": 0.25, "power": 0.7, "threshold": 3.0},
        "parameters": {"lambda": 0.5},
        "model": "late_refresh_hazard",
        "alpha": 0.8,
    }


def toy_policy(prediction: float, phase0: list[float], phase1: list[float]) -> dict[str, Any]:
    return {
        "best": {
            "phase0": phase0,
            "phase1": phase1,
            "predicted": prediction,
            "source": "toy_nonconverged",
            "tied": False,
            "feasible": True,
            "success": False,
            "simplex_residual": max(abs(sum(phase0) - 1), abs(sum(phase1) - 1)),
        },
        "phase_blind_search": False,
        "predicted_gain_vs_static": -0.002,
        "predicted_gain_same_aggregate": -0.001,
    }


def test_truth_and_fitted_gap_use_their_original_exponential_conventions() -> None:
    dimensions = runner.Geometry(0.8, np.array([2.0, 2.0]), np.array([0.5, 0.5]))
    payload = toy_head(35.0, 0.6)
    truth = 0.6 + np.exp(35.0)
    native = 0.6 + np.exp(30.0)
    results = []
    for kind, value in (("truth", truth), ("fitted", native)):
        spec = {"policy_field": "best", "source_head_kind": kind, "evaluator_kind": kind}
        result = runner.evaluate_case(
            spec, dimensions, toy_policy(value, [0.5, 0.5], [0.25, 0.75]), payload, payload, value
        )
        assert result["frozen_prediction_parity_pass"]
        assert result["matching_objective_bound_available"]
        assert result["matching_objective_lower_bound"] == pytest.approx(value, rel=2e-15)
        assert result["frozen_matching_remaining_improvement_upper_estimate"] == 0.0
        assert result["saved_policy"]["success"] is False
        assert result["saved_predicted_gain_vs_static"] == -0.002
        results.append(result)
    assert results[0]["matching_objective_lower_bound"] > 100 * results[1]["matching_objective_lower_bound"]
    assert not results[0]["diagnostic"]["weighted"]["unclipped_support_is_native_lower_bound"]


def test_truth_cross_evaluation_keeps_source_and_requested_parity_separate() -> None:
    dimensions = runner.Geometry(0.8, np.array([2.0, 2.0]), np.array([0.5, 0.5]))
    source, truth = toy_head(-1.0, 0.4), toy_head(-2.0, 0.6)
    source_value, truth_value = 0.4 + np.exp(-1.0), 0.6 + np.exp(-2.0)
    spec = {"policy_field": "best", "source_head_kind": "fitted", "evaluator_kind": "truth"}
    result = runner.evaluate_case(
        spec, dimensions, toy_policy(source_value, [0.5, 0.5], [0.2, 0.8]), source, truth, truth_value
    )
    assert result["frozen_prediction_parity_pass"]
    assert result["source_prediction_replay"] == source_value
    assert result["evaluator_prediction_replay"] == truth_value
    mismatched = runner.evaluate_case(
        spec, dimensions, toy_policy(source_value, [0.5, 0.5], [0.2, 0.8]), source, truth, truth_value + 0.01
    )
    assert not mismatched["frozen_prediction_parity_pass"]
    assert not mismatched["bound_parity_and_feasibility_checks_pass"]
    assert mismatched["source_prediction_error"] == 0.0
    assert mismatched["evaluator_prediction_error"] == pytest.approx(-0.01, abs=2e-16)


@pytest.mark.parametrize("phase0", [[0.5, 0.5000001], [-1e-12, 1.000000000001]])
def test_old_optimizer_feasibility_does_not_force_a_diagnostic_bound(phase0: list[float]) -> None:
    dimensions = runner.Geometry(0.8, np.array([2.0, 2.0]), np.array([0.5, 0.5]))
    payload = toy_head(-1.0, 0.6)
    value = 0.6 + np.exp(-1.0)
    spec = {"policy_field": "best", "source_head_kind": "fitted", "evaluator_kind": "fitted"}
    result = runner.evaluate_case(spec, dimensions, toy_policy(value, phase0, [0.5, 0.5]), payload, payload, value)
    assert result["saved_policy"]["feasible"]
    assert not result["diagnostic"]["feasible_at_tolerance"]
    assert not result["matching_objective_bound_available"]
    assert "matching_objective_lower_bound" not in result


def test_entire_prior_seal_is_checked_including_an_unused_file(tmp_path: Path) -> None:
    selected, unused = tmp_path / "head.json", tmp_path / "unused.csv"
    selected.write_text('{"head": 1}\n')
    unused.write_text("unused,1\n")
    entries = {path.name: {"sha256": runner.sha256(path), "bytes": path.stat().st_size} for path in (selected, unused)}
    seal = tmp_path / "result_manifest.json"
    seal.write_text(json.dumps({"files": entries}))
    expected = runner.sha256(seal)
    assert runner.verify_seal(tmp_path, expected) == entries
    unused.write_text("unused,2\n")
    with pytest.raises(ValueError, match="Sealed prior artifact changed"):
        runner.verify_seal(tmp_path, expected)


def test_geometry_loader_ignores_non_covariate_archive_members(tmp_path: Path) -> None:
    archive = tmp_path / "geometry.npz"
    np.savez(
        archive,
        c0=np.array([8.0, 0.8]),
        c1=np.array([2.0, 0.2]),
        alpha=0.8,
        final_targets=np.array([{"never_unpickle_this_member": True}], dtype=object),
    )
    dimensions = runner.geometry(archive)
    np.testing.assert_array_equal(dimensions.scales, [10.0, 1.0])
    np.testing.assert_array_equal(dimensions.costs, [0.1, 1.0])
    # Opening an outcome member would fail with allow_pickle=False; the loader uses only covariates.
    with np.load(archive, allow_pickle=False) as arrays:
        with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
            _ = arrays["final_targets"]


def test_resume_checks_output_and_identity_without_overwriting_evidence(tmp_path: Path) -> None:
    spec, identity = {"case_id": "toy"}, {"source_hash": "frozen"}
    result = tmp_path / "result.json"
    runner.write_json(result, {"gap": -1e-14})
    runner.write_json(
        tmp_path / "complete.json", {"case": spec, "identity": identity, "result_sha256": runner.sha256(result)}
    )
    assert runner.cached_case(tmp_path, spec, identity)
    runner.write_json(result, {"gap": -1e-14})
    with pytest.raises(ValueError, match="Preserve existing evidence"):
        runner.write_json(result, {"gap": 0})
    with pytest.raises(ValueError, match="Existing diagnostic receipt"):
        runner.cached_case(tmp_path, spec, {"source_hash": "changed"})
    result.write_text('{"gap": 0}\n')
    with pytest.raises(ValueError, match="Existing diagnostic receipt"):
        runner.cached_case(tmp_path, spec, identity)
