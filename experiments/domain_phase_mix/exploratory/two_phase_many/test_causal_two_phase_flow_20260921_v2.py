# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import causal_two_phase_flow_20260921_v2 as flow
from experiments.domain_phase_mix.exploratory.two_phase_many.test_causal_two_phase_flow_20260921 import (
    _policies,
    _reference_phase,
)


@pytest.mark.parametrize("mode", ["additive", "ratio"])
def test_foreign_rows_permutation_and_subsets_preserve_each_policy(mode) -> None:
    early, late, first, second, alpha = _policies()
    expected = flow.causal_phase_benefit(early, late, first, second, alpha, 4, 0.7, 1, 0, mode)
    singleton = np.concatenate(
        [
            flow.causal_phase_benefit(
                early[i : i + 1], late[i : i + 1], first[i : i + 1], second[i : i + 1], alpha, 4, 0.7, 1, 0, mode
            )
            for i in range(len(early))
        ]
    )
    order = np.asarray([5, 2, 0, 5])
    changed = flow.causal_phase_benefit(
        early[order], late[order], first[order], second[order], alpha, 4, 0.7, 1, 0, mode
    )
    np.testing.assert_array_equal(singleton, expected)
    np.testing.assert_array_equal(changed, expected[order])


@pytest.mark.parametrize("mode", ["additive", "ratio"])
def test_tied_columns_are_analytic_even_inside_asymmetric_rows(mode) -> None:
    early, late, first, second, alpha = _policies()
    other_columns = [0, 2, 3]
    second[:, other_columns] *= ((1 - first[:, 1]) / (1 - second[:, 1]))[:, None]
    second[:, 1] = first[:, 1]
    scales = np.asarray([0.003, 0.5, 8, 150])
    late = (1 - alpha) * second * scales
    expected = -np.expm1(-((4 * (early + late)) ** 0.7))
    result = flow.causal_phase_benefit(early, late, first, second, alpha, 4, 0.7, 64, 16, mode)
    np.testing.assert_array_equal(result[:, 1], expected[:, 1])
    tied_early, tied_late = alpha * first * scales, (1 - alpha) * first * scales
    tied = flow.causal_phase_benefit(tied_early, tied_late, first, first, alpha, 4, 0.7, 64, 16, mode)
    native_tied = -np.expm1(-((4 * (tied_early + tied_late)) ** 0.7))
    np.testing.assert_array_equal(tied, native_tied)


@pytest.mark.parametrize("rate,power,forgetting,consolidation", [(0.5, 0.3, 64, 16), (4.0, 0.7, 1, 0)])
def test_row_local_flow_matches_independent_physical_time_reference(rate, power, forgetting, consolidation) -> None:
    early, late, first, second, alpha = _policies()
    actual = flow.causal_flow_readout(early, late, first, second, alpha, rate, power, forgetting, consolidation)
    expected = np.empty_like(actual)
    for index in np.ndindex(actual.shape):
        state = _reference_phase(
            np.asarray([1.0, 0.0, 0.0]), 0, early[index], first[index], alpha, rate, power, forgetting, consolidation
        )
        state = _reference_phase(
            state, early[index], late[index], second[index], 1 - alpha, rate, power, forgetting, consolidation
        )
        expected[index] = state[1:].sum()
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
    assert actual.min() >= 0
    assert actual.max() <= 1 + 2e-10


def test_row_local_state_keeps_prefix_subdivision_invariant() -> None:
    early, _, first, _, alpha = _policies()
    initial = np.zeros((*early.shape, 3))
    initial[..., 0] = 1
    full = flow.advance_causal_state(initial, np.zeros_like(early), early, first, alpha, 0.5, 0.3, 16, 4)
    state = initial.copy()
    previous = np.zeros_like(early)
    for fraction in [0.013, 0.187, 0.6, 0.2]:
        state = flow.advance_causal_state(state, previous, fraction * early, first, fraction * alpha, 0.5, 0.3, 16, 4)
        previous += fraction * early
    np.testing.assert_allclose(state, full, rtol=3e-8, atol=3e-10)
    np.testing.assert_allclose(state.sum(axis=-1), 1, rtol=0, atol=3e-15)


@pytest.mark.parametrize("power", [0.3, 1.0])
def test_zero_forgetting_and_tiny_ratio_keep_exact_limits(power) -> None:
    first = np.asarray([[0.9, 0.1, 0.0]])
    second = np.asarray([[0.1, 0.9, 0.0]])
    limits = []
    for dose in [1e-100, 1e-200]:
        early, late = 0.8 * dose * first, 0.2 * dose * second
        expected = -np.expm1(-((early + late) ** power))
        no_forgetting = flow.causal_phase_benefit(early, late, first, second, 0.8, 1, power, 0, 16)
        np.testing.assert_array_equal(no_forgetting, expected)
        ratio = flow.causal_phase_benefit(early, late, first, second, 0.8, 1, power, 4, 1, "ratio")
        assert ratio[0, 2] == 0
        assert np.isfinite(ratio).all()
        limits.append(ratio / dose**power)
    np.testing.assert_allclose(limits[0], limits[1], rtol=3e-8, atol=3e-10)
