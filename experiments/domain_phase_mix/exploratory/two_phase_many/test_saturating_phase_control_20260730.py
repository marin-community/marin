# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    saturating_phase_control_20260730 as saturating,
)


def aggregate_model() -> replay_control.AggregateFitted:
    geometry = replay_control.Geometry(
        c0=np.asarray([8.0, 4.0]),
        c1=np.asarray([2.0, 1.0]),
        phase_0_fraction=0.8,
    )
    shape = replay_control.AggregateShape(
        benefit_exponent=0.5,
        benefit_offset=0.01,
        damage_exponent=2.0,
        damage_threshold=1.0,
    )
    return replay_control.AggregateFitted(
        shape=shape,
        ridge=0.1,
        intercept=0.9,
        coefficients=np.asarray([0.01, 0.02, 0.001, 0.002]),
        geometry=geometry,
    )


def weights(phase0: tuple[float, float], phase1: tuple[float, float]) -> np.ndarray:
    return np.asarray([[phase0, phase1]], dtype=float)


def test_phase_columns_vanish_when_tied() -> None:
    model = aggregate_model()
    tied = weights((0.4, 0.6), (0.4, 0.6))
    np.testing.assert_allclose(saturating.phase_design_matrix(tied, model, tau=0.5), 0.0, atol=1e-12)


def test_control_is_bounded_by_pointwise_authority() -> None:
    model = aggregate_model()
    policies = np.concatenate(
        [
            weights((0.8, 0.2), (0.1, 0.9)),
            weights((0.2, 0.8), (0.9, 0.1)),
        ]
    )
    control, authority, ratio = saturating.control_statistics(policies, model)
    assert np.all(np.abs(control) <= authority + 1e-12)
    assert np.all(ratio <= 1.0 + 1e-12)


def test_infinite_tau_is_exact_linear_ablation() -> None:
    model = aggregate_model()
    policy = weights((0.8, 0.2), (0.1, 0.9))
    control, authority, _ratio = saturating.control_statistics(policy, model)
    np.testing.assert_array_equal(
        saturating.saturating_control(control, authority, np.inf),
        control,
    )


def test_saturating_response_is_odd_and_bounded() -> None:
    control = np.asarray([-0.3, -0.1, 0.0, 0.1, 0.3])
    authority = np.ones_like(control) * 0.4
    response = saturating.saturating_control(control, authority, tau=0.5)
    np.testing.assert_allclose(response, -response[::-1], atol=1e-12)
    assert np.all(np.abs(response) <= 0.5 * authority + 1e-12)
