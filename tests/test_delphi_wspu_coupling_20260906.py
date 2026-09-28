# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_delphi_wspu_coupling_20260906 as probe


@pytest.mark.parametrize("kappa", [0, 0.25, 0.5, 1])
@pytest.mark.parametrize("delta", [-4, -2, 0, 0.6, 3])
def test_coupling_preserves_complete_single_bucket_response(kappa, delta):
    anchor = np.array([2.0])
    deltas = np.array([[[0.0, delta, 0.0]]])
    prediction, counts = probe.coupled_values(anchor, deltas, kappa)
    np.testing.assert_allclose(prediction, [[2 + delta]], atol=1e-14)
    assert int(counts[0, 0]) == int(1 + kappa * delta / 2 <= 0)


@pytest.mark.parametrize("deltas", [[0.4, -0.3, 0.1], [-2.3, 0.2, -0.4], [-4, -3, 0.6]])
def test_coupling_matches_three_bucket_polynomial_and_gradient(deltas):
    anchor = 2.3
    d = np.array(deltas)
    kappa = 0.75
    prediction, _ = probe.coupled_values(np.array([anchor]), d[None, None], kappa)
    expected = anchor + d.sum() + kappa * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]) / anchor
    expected += kappa**2 * d.prod() / anchor**2
    assert float(prediction[0, 0]) == pytest.approx(expected, abs=1e-13)
    for bucket in range(3):
        step = np.zeros(3)
        step[bucket] = 1e-6
        high, _ = probe.coupled_values(np.array([anchor]), (d + step)[None, None], kappa)
        low, _ = probe.coupled_values(np.array([anchor]), (d - step)[None, None], kappa)
        others = np.delete(d, bucket)
        expected_gradient = 1 + kappa * others.sum() / anchor + kappa**2 * others.prod() / anchor**2
        assert float((high - low)[0, 0] / 2e-6) == pytest.approx(expected_gradient, abs=2e-9)


def test_signed_and_zero_factors_are_retained_without_clipping():
    deltas = np.array([[[-4.0, 0.0]], [[-2.0, 0.5]], [[-4.0, -4.0]]])
    prediction, counts = probe.coupled_values(np.array([2.0]), deltas, 1)
    np.testing.assert_allclose(prediction[:, 0], [-2, 0, 2], atol=1e-14)
    np.testing.assert_array_equal(counts[:, 0], [1, 1, 2])


def test_small_coupling_strength_has_stable_additive_limit():
    anchor = np.array([1.7, 2.3])
    deltas = np.array([[[0.3, -0.2, 0.1], [0.05, -0.12, 0.03]]])
    additive, _ = probe.coupled_values(anchor, deltas, 0)
    tiny, _ = probe.coupled_values(anchor, deltas, 1e-12)
    np.testing.assert_allclose(tiny, additive, atol=1e-12, rtol=0)
