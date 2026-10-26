# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "pytest", "threadpoolctl"]
# ///
"""Invented-array tests; no scientific prefix covariates or saved fits are evaluated."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_prefix_auxiliary_pairs_20261015 as runner,
)


def test_covariance_aware_projection_matches_nnls_and_block_inverse() -> None:
    logits = np.array([-1.2, -0.7, 0.1, 0.5, 0.9])
    truth = np.array([0.7, 0.9, 1.0, 1.2, 1.4])
    delta = np.array([0.002, -0.004, 0.006, -0.001, 0.003])
    rho, ratio = 0.5, 3.0
    result, arrays = runner.profiled_extension(logits, truth, delta, rho, ratio)
    matrix = np.column_stack((np.ones(5), np.exp(logits - logits.max())))
    coefficients, _ = nnls(matrix, truth + rho * ratio * delta)
    np.testing.assert_allclose(arrays["candidate_prefix"], matrix @ coefficients, atol=2e-15, rtol=2e-15)
    covariance = runner.SIGMA**2 * np.array([[ratio**2, rho * ratio], [rho * ratio, 1]])
    differences = np.column_stack((arrays["candidate_prefix"] - truth, delta))
    direct = np.einsum("ni,ij,nj->", differences, np.linalg.inv(covariance), differences)
    assert math.isclose(result["distance_joint"] ** 2, direct, rel_tol=2e-13)


def test_shifted_target_beats_naive_prefix_fit_under_correlation() -> None:
    logits = np.log(np.array([0.1, 0.3, 0.8, 1.0]))
    truth = 0.7 + 0.5 * np.exp(logits)
    delta = np.array([0.01, 0.025, 0.05, 0.08])
    result, arrays = runner.profiled_extension(logits, truth, delta, 0.9, 10.0)
    naive, _ = runner.profiled_extension(logits, truth, delta, 0.0, 10.0)
    naive_conditional = 0.9 * delta / runner.SIGMA / math.sqrt(1 - 0.9**2)
    assert result["distance_conditional"] < np.linalg.norm(naive_conditional)
    assert naive["distance_prefix_only"] < 1e-10
    assert np.max(abs(arrays["candidate_prefix"] - truth)) > 0.2


@pytest.mark.parametrize("target", [np.array([0.4, 0.9, 1.4]), np.array([3.0, 1.0, -2.0]), np.array([-2.0, -1.0, -3.0])])
def test_closed_profile_faces_match_independent_nnls(target: np.ndarray) -> None:
    logits = np.log(np.array([0.2, 0.6, 1.0]))
    result, arrays = runner.profiled_extension(logits, target, np.zeros(3), 0.0, 1.0)
    design = np.column_stack((np.ones(3), np.exp(logits)))
    coefficient, _ = nnls(design, target)
    np.testing.assert_allclose(arrays["candidate_prefix"], design @ coefficient, rtol=2e-14, atol=2e-14)
    assert result["scale_zero_closure"] == (result["normalized_scale"] == 0)
    assert (result["log_original_scale"] is None) == result["scale_zero_closure"]


def test_normalization_gauge_and_common_prefix_references() -> None:
    logits = np.array([-1.0, -0.5, -0.1, 0.0])
    truth = 0.8 + 1.2 * np.exp(logits)
    delta = np.array([0.001, -0.003, 0.002, 0.004])
    first, values = runner.profiled_extension(logits, truth, delta, 0.0, 1.0)
    shifted, other = runner.profiled_extension(logits + 80, truth, delta, 0.0, 1.0)
    np.testing.assert_allclose(values["candidate_prefix"], other["candidate_prefix"], atol=1e-14, rtol=0)
    assert math.isclose(first["log_original_scale"] - shifted["log_original_scale"], 80, abs_tol=1e-12)
    assert math.isclose(first["distance_joint"], first["distance_final"], rel_tol=1e-13)
    assert first["distance_common_prefix_reference"] == first["distance_unrestricted_prefix_nuisance_reference"]
    u = delta / runner.SIGMA
    v = 0.9 * u
    assert np.linalg.norm((v - 0.9 * u) / math.sqrt(1 - 0.9**2)) == 0


def test_constant_feature_keeps_unidentified_closed_representation() -> None:
    result, arrays = runner.profiled_extension(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4), 0.0, 1.0)
    assert result["constant"] and result["prefix_decomposition_unidentified"]
    assert result["scale_zero_closure"] and result["log_original_scale"] is None
    np.testing.assert_array_equal(arrays["candidate_prefix"], np.full(4, 2.5))


def test_log_bound_preserves_positive_underflow_without_overlap_division() -> None:
    value = runner.scalar_bound(2.0, -10000.0)
    assert value["pair_mse_lower_bound"] == 0 and value["bound_underflowed"]
    assert math.isclose(value["pair_mse_log_lower_bound"], math.log(2) - 10000)
    assert runner.scalar_bound(0.0, -10000)["pair_mse_log_lower_bound"] is None


def test_numerical_failure_keeps_full_original_status_and_pair() -> None:
    head = {
        "pair_id": "invented",
        "world": "native_null",
        "original_numerical_status": "failed_nonwinning_refinement",
        "analytic_pair_eligible": True,
        "distance": 0,
    }
    item = {"head": head, "shape": runner.SHAPE, "amplitudes": [1e308, 0, 0, 0]}
    train = runner.pd.DataFrame(
        {"candidate_mean": [1.0, 1.0], "truth": [1.0, 1.0], "source_index": [0, 1], "row_id": ["a", "b"]}
    )
    result = runner.cell_result(item, train, np.array([[0.0, 0.0], [1.0, 0.0]]), np.ones(2), 0.5, 3.0)
    assert result["status"] == "failed" and not result["analytic_joint_pair_eligible"]
    assert result["head"] == head
    assert "underflow" in result["failure"]["message"]
    with pytest.raises(ArithmeticError, match="Nonfinite"):
        runner.profiled_extension(np.array([0.0, np.nan]), np.ones(2), np.zeros(2), 0.0, 1.0)


def test_canonical_parameter_ancestry_does_not_collapse_endpoint_duplicates() -> None:
    # A single observed total exposure can constrain only one linear combination.
    total = np.array([[2.0, 2.0]])
    early = np.array([[0.2, 1.6]])
    first, second = np.array([1.0, 3.0, 0.1, 0.1]), np.array([2.0, 2.0, 0.1, 0.1])
    np.testing.assert_allclose(
        runner.native_logits(total, runner.SHAPE, first), runner.native_logits(total, runner.SHAPE, second), atol=1e-15
    )
    assert (
        abs(runner.native_logits(early, runner.SHAPE, first)[0] - runner.native_logits(early, runner.SHAPE, second)[0])
        > 0.1
    )
    assert runner.ancestry(runner.SHAPE, first, "equal_positive")["kinetics_or_final_level_only_difference"]
    assert not runner.ancestry(runner.SHAPE, second, "equal_positive")["kinetics_or_final_level_only_difference"]


def test_receipts_resume_exact_outputs_and_reject_tampering(tmp_path: Path) -> None:
    execution, spec = {"source": "fixed"}, {"pair": "invented", "rho": 0.5}
    value = {"status": "failed", "failure": "retained"}
    runner.immutable_write(tmp_path / "result.json", runner.json_bytes(value))
    receipt = {
        "execution_identity_sha256": hashlib.sha256(runner.json_bytes(execution)).hexdigest(),
        "spec": spec,
        "result_sha256": runner.sha(tmp_path / "result.json"),
    }
    runner.immutable_write(tmp_path / "complete.json", runner.json_bytes(receipt))
    assert runner.verified_cell(tmp_path, execution, spec) == value
    runner.immutable_write(tmp_path / "result.json", runner.json_bytes(value))
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed cell"):
        runner.verified_cell(tmp_path, execution, spec)
    incomplete = tmp_path / "interrupted"
    incomplete.mkdir()
    (incomplete / "result.json.writing").write_text("partial")
    with pytest.raises(ValueError, match="Unexplained incomplete"):
        runner.verified_cell(incomplete, execution, spec)


def test_historical_source_adoption_refuses_live_replacement(tmp_path: Path) -> None:
    helper = tmp_path / "historical_helper.py"
    helper.write_text("def profile(): return 1\n")
    historical = {str(helper): runner.sha(helper)}
    assert runner.inherited_sources(historical) == historical
    helper.write_text("def profile(): return 2\n")
    with pytest.raises(ValueError, match="Inherited source changed"):
        runner.inherited_sources(historical)
