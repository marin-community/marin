# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import staged_retained_phase_control_20260730 as staged


def geometry() -> replay_control.Geometry:
    return replay_control.Geometry(
        c0=np.asarray([2.0, 4.0]),
        c1=np.asarray([0.5, 1.0]),
        phase_0_fraction=0.8,
    )


def aggregate_model() -> replay_control.AggregateFitted:
    return replay_control.AggregateFitted(
        shape=replay_control.AggregateShape(0.5, 0.1, 2.0, 0.0),
        ridge=0.01,
        intercept=1.0,
        coefficients=np.asarray([0.2, 0.4, 0.01, 0.02]),
        geometry=geometry(),
    )


def test_normalized_state_preserves_tied_policy() -> None:
    weights = np.asarray([[[0.3, 0.7], [0.3, 0.7]]])
    for shape in staged.shape_grid():
        state = staged.normalized_retained_state(weights, geometry(), shape)
        np.testing.assert_allclose(state, weights[:, 0, :])


def test_phase_design_is_zero_for_tied_policy() -> None:
    weights = np.asarray([[[0.3, 0.7], [0.3, 0.7]]])
    shape = staged.Shape(retention=10.0, late_multiplier=4.0)
    for family_resolved in (False, True):
        np.testing.assert_allclose(
            staged.phase_design_matrix(
                weights,
                aggregate_model(),
                shape,
                family_resolved=family_resolved,
            ),
            0.0,
        )


def test_joint_phase_columns_are_zero_for_tied_policy() -> None:
    weights = np.asarray([[[0.3, 0.7], [0.3, 0.7]]])
    aggregate = aggregate_model()
    aggregate_columns = replay_control.aggregate_design_matrix(
        weights,
        aggregate.geometry,
        aggregate.shape,
    ).shape[1]
    for use_ordering in (False, True):
        design = staged.joint_design_matrix(
            weights,
            aggregate,
            staged.Shape(retention=10.0, late_multiplier=4.0),
            use_ordering=use_ordering,
        )
        np.testing.assert_allclose(design[:, aggregate_columns:], 0.0)


def test_retention_makes_late_presence_change_early_value() -> None:
    weights = np.asarray(
        [
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.2, 0.8], [0.8, 0.2]],
        ]
    )
    shape = staged.Shape(retention=5.0, late_multiplier=2.0)
    residual = staged.retained_benefit_residual(weights, aggregate_model(), shape)
    assert residual[0] != residual[1]
