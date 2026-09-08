# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_coupling_validation_20260906 as validation,
)


@pytest.mark.parametrize("kappa", [0.25, 0.5, 1.0])
def test_fractional_strength_atomic_and_macro_predictions_match_two_bucket_expansion(kappa):
    heads = validation.prior.coupling.AdditiveHeads(
        inventory=np.array([2.0, 3.0]),
        intercept=np.array([2.0, 3.0]),
        coefficients=np.array([[0.3, 0.2, 0.1, 0.2], [0.4, 0.1, 0.3, 0.2]]),
        rate=np.array([1.0, 2.0]),
        power=np.array([1.0, 1.5]),
        threshold=np.array([0.5, 0.7]),
    )
    anchor_effects = heads.bucket_effects(np.array([[0.5, 0.5]]))[0]
    anchor = heads.intercept + anchor_effects.sum(axis=-1)
    base = validation.prior.Incumbent("toy", 1, heads, np.array([0.25, 0.75]), anchor_effects, anchor)
    weights = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    deltas = heads.bucket_effects(weights) - anchor_effects
    expected = heads.predict(weights) + kappa * deltas[:, :, 0] * deltas[:, :, 1] / anchor
    actual = validation.evaluate(base, kappa, weights)
    np.testing.assert_allclose(actual.atomic, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.prediction, expected @ base.aggregation, rtol=1e-14, atol=1e-14)
