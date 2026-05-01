# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import (
    OlmixLoglinearFit,
    solve_olmix_loglinear_schedule,
)


def _phase_matrix(
    phase_weights: dict[str, dict[str, float]],
    phase_names: list[str],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [[float(phase_weights[phase_name][domain_name]) for domain_name in domain_names] for phase_name in phase_names],
        dtype=float,
    )


def test_olmix_exact_solver_matches_natural_prior_for_pure_kl():
    pytest.importorskip("cvxpy")
    phase_names = ["phase_0", "phase_1"]
    domain_names = ["a", "b"]
    fit = OlmixLoglinearFit(log_c=0.0, coefficients=(0.0, 0.0, 0.0, 0.0), huber_loss=0.0)
    natural = np.asarray([0.7, 0.3], dtype=float)
    phase_fractions = np.asarray([0.8, 0.2], dtype=float)

    phase_weights, predicted_objective, regularized_objective = solve_olmix_loglinear_schedule(
        fit,
        natural_proportions=natural,
        phase_fractions=phase_fractions,
        phase_names=phase_names,
        domain_names=domain_names,
        lambda_kl=0.05,
        solver="cvxpy",
    )

    solved = _phase_matrix(phase_weights, phase_names, domain_names)
    expected = np.tile(natural, (2, 1))
    np.testing.assert_allclose(solved, expected, atol=2e-4)
    assert predicted_objective == pytest.approx(2.0, abs=1e-6)
    assert regularized_objective == pytest.approx(2.0, abs=1e-6)


def test_olmix_exact_solver_hits_phasewise_vertex_without_kl():
    pytest.importorskip("cvxpy")
    phase_names = ["phase_0", "phase_1"]
    domain_names = ["a", "b"]
    fit = OlmixLoglinearFit(log_c=0.0, coefficients=(-1.0, 0.2, 0.1, -2.0), huber_loss=0.0)
    natural = np.asarray([0.5, 0.5], dtype=float)
    phase_fractions = np.asarray([0.8, 0.2], dtype=float)

    phase_weights, predicted_objective, regularized_objective = solve_olmix_loglinear_schedule(
        fit,
        natural_proportions=natural,
        phase_fractions=phase_fractions,
        phase_names=phase_names,
        domain_names=domain_names,
        lambda_kl=0.0,
        solver="cvxpy",
    )

    solved = _phase_matrix(phase_weights, phase_names, domain_names)
    np.testing.assert_allclose(solved[0], np.asarray([1.0, 0.0]), atol=5e-6)
    np.testing.assert_allclose(solved[1], np.asarray([0.0, 1.0]), atol=5e-6)
    expected_prediction = 1.0 + np.exp(-3.0)
    assert predicted_objective == pytest.approx(expected_prediction, rel=1e-6)
    assert regularized_objective == pytest.approx(expected_prediction, rel=1e-6)
