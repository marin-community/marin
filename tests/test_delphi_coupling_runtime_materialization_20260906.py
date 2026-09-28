# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_coupling_runtime_20260906 as runtime,
)


@pytest.mark.parametrize("kappa", [0.25, 0.5, 1.0])
def test_runtime_quantization_matches_exhaustive_two_bucket_search_under_fractional_count_cap(kappa):
    inventory = np.array([3.3, 1.7])
    heads = runtime.prior.coupling.AdditiveHeads(
        inventory=inventory,
        intercept=np.array([2.0]),
        coefficients=np.array([[0.8, 0.3, 0.1, 0.2]]),
        rate=np.array([1.0]),
        power=np.array([1.0]),
        threshold=np.array([0.5]),
    )
    anchor_effects = heads.bucket_effects(np.array([[0.5, 0.5]]))[0]
    anchor = heads.intercept + anchor_effects.sum(axis=-1)
    model = runtime.prior.Incumbent("toy", 1, heads, np.ones(1), anchor_effects, anchor)
    cap = 2
    initial = np.array([cap / inventory[0], 1 - cap / inventory[0]])
    counts, diagnostics = runtime.runtime_policy(model, kappa, initial, inventory, cap)

    first = np.arange(runtime.BLOCK_SIZE + 1)
    candidates = np.stack([first, runtime.BLOCK_SIZE - first], axis=1)
    candidates = candidates[np.all(candidates / runtime.BLOCK_SIZE * inventory <= cap, axis=1)]
    values = runtime.continuous.evaluate(model, kappa, candidates / runtime.BLOCK_SIZE).prediction
    expected = candidates[np.argmin(values)]
    np.testing.assert_array_equal(counts, expected)
    assert diagnostics["runtime_prediction"] == pytest.approx(values.min(), abs=1e-13)
    assert diagnostics["max_materialized_epoch"] <= cap
