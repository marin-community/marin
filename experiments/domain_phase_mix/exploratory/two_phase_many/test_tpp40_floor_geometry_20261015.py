# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavior checks for local nuisance projections on invented responses."""

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_tpp40_floor_geometry_20261015 as geometry


def test_adding_floor_removes_a_constant_component_without_rescaling_dependence() -> None:
    x = np.linspace(-1, 1, 21)
    y = 2 + x
    known = geometry.residualized(y, x[:, None])
    unknown = geometry.residualized(y, np.column_stack((x, np.ones(len(x)))))
    scaled = geometry.residualized(y, np.column_stack((1e5 * x, 1e-6 * np.ones(len(x)))))
    np.testing.assert_allclose(known["residual_norm"], 2 * np.sqrt(len(x)), rtol=1e-14)
    assert unknown["residual_norm"] < 1e-13
    assert scaled["residual_norm"] < 1e-13
    assert unknown["nuisance_rank"] == scaled["nuisance_rank"] == 2


def test_disjoint_synthetic_phase_direction_survives_unknown_floor() -> None:
    x = np.linspace(-1, 1, 41)
    phase = x**2 - np.mean(x**2)
    result = geometry.residualized(phase, np.column_stack((np.ones(len(x)), x, 2 * x)))
    assert result["nuisance_rank"] == 2
    np.testing.assert_allclose(result["retained_norm_ratio"], 1, rtol=1e-14)
    np.testing.assert_allclose(result["local_noise_scaled_inverse_norm"], 0.003 / np.linalg.norm(phase), rtol=1e-14)


def test_coarse_derivative_disagreement_remains_unresolved() -> None:
    x = np.linspace(-1, 1, 11)
    values = {
        "amplitudes": np.column_stack((np.ones(len(x)), x)),
        "derivatives_0.001": np.column_stack((x**2, x**3, x**4, x**5)),
        "derivatives_0.0005": np.column_stack((x**2, x**3, x**4, x**5 + 0.01)),
    }
    report = geometry.summarize_case(values, np.arange(len(x)))
    assert not report["finite_difference_resolved"]
    assert report["finite_difference_errors"][3] > report["finite_difference_limits"][3]
