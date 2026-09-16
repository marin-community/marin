# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_phase_functional_independence_20260730 as audit,
)


def test_normalized_control_is_invariant_to_positive_gradient_scale() -> None:
    gradient = np.asarray([[1.0, -1.0], [2.0, -2.0]])
    delta = np.asarray([[0.2, -0.2], [-0.1, 0.1]])
    observed = audit.normalized_gradient_control(gradient, delta, reference_norm=2.0)
    scaled = audit.normalized_gradient_control(7.0 * gradient, delta, reference_norm=14.0)
    np.testing.assert_allclose(observed, scaled)


def test_normalized_control_is_zero_at_stationary_gradient() -> None:
    gradient = np.zeros((2, 3))
    delta = np.asarray([[0.2, -0.1, -0.1], [-0.3, 0.1, 0.2]])
    observed = audit.normalized_gradient_control(gradient, delta, reference_norm=1.0)
    np.testing.assert_array_equal(observed, 0.0)


def test_basis_diagnostics_detect_one_dimensional_basis() -> None:
    x = np.linspace(-1.0, 1.0, 100)
    columns = np.column_stack([x, 2.0 * x, -3.0 * x])
    observed = audit.basis_diagnostics(columns, ("a", "b", "c"))
    assert observed["first_component_explained_variance"] > 1.0 - 1e-12
    assert observed["stable_rank"] < 1.0 + 1e-12
    assert observed["participation_rank"] < 1.0 + 1e-12


def test_basis_diagnostics_preserves_independent_columns() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    columns = np.column_stack([np.sin(angle), np.cos(angle), np.sin(2.0 * angle)])
    observed = audit.basis_diagnostics(columns, ("a", "b", "c"))
    np.testing.assert_allclose(observed["stable_rank"], 3.0, atol=1e-12)
    np.testing.assert_allclose(observed["participation_rank"], 3.0, atol=1e-12)
