# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    survey_phase_functional_design_20260730 as survey,
)


def test_phase_functionals_are_zero_when_tied() -> None:
    phase = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]])
    weights = np.stack([phase, phase], axis=1)
    exposure = np.asarray([0.5, 3.0, 12.0])
    for functional in survey.FUNCTIONALS:
        np.testing.assert_allclose(
            survey.phase_functional(weights, exposure, functional),
            0.0,
            atol=0.0,
        )


def test_quadratic_functional_is_order_sensitive_in_two_domains() -> None:
    weights = np.asarray([[[0.9, 0.1], [0.5, 0.5]]])
    exposure = np.asarray([0.5, 13.0])
    observed = survey.phase_functional(weights, exposure, "quadratic")
    reversed_weights = weights[:, ::-1, :]
    reversed_value = survey.phase_functional(reversed_weights, exposure, "quadratic")
    assert observed[0] != 0.0
    np.testing.assert_allclose(observed, -reversed_value)


def test_two_domain_geometric_alignment_null_is_one() -> None:
    gradient = np.asarray([[-2.0, 2.0], [0.5, -0.5]])
    mass = np.asarray([0.2, 0.8])
    full, _trimmed = survey.geometric_alignment_null(
        gradient,
        mass,
        draws=64,
        seed=0,
    )
    np.testing.assert_allclose(full, 1.0)


def test_residualized_norm_detects_duplicate_column() -> None:
    base = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 20),
            np.linspace(-1.0, 1.0, 20) ** 2,
        ]
    )
    duplicate = 3.0 * base[:, 0]
    assert survey.residual_norm_fraction(duplicate, base) < 1e-12
    assert survey.design_condition_number(np.column_stack([base, duplicate])) == float("inf")
