# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_mu0_two_phase_20260921 as diagnostic
from experiments.domain_phase_mix.exploratory.two_phase_many.two_phase_benchmark_data_20260921 import BenchPanel


@pytest.mark.parametrize("model", ["NG2-005", "NG2-006"])
def test_mu0_ablation_whole_estimator_returns_aggregate_prediction(model) -> None:
    rng = np.random.default_rng(20260921)
    first = rng.dirichlet(np.ones(2), 18)
    second = rng.dirichlet(np.ones(2), 18)
    panel = BenchPanel(
        name="synthetic",
        buckets=("nemotron_full", "starcoder"),
        phase0=first,
        phase1=second,
        alpha=0.8,
        c0=np.asarray([0.8, 8.0]),
        c1=np.asarray([0.2, 2.0]),
        targets={},
        rows=pd.DataFrame({"tied": np.zeros(18, dtype=bool)}),
        input_hashes={},
        metadata={},
    )
    surrogate = diagnostic.surrogate
    matrix = surrogate.mariner.design_matrix(panel.epochs, {"rate": 0.5, "power": 0.7, "threshold": 3.0})
    response = 0.5 + np.exp(-0.2 + matrix @ np.asarray([1.2, 0.6, 0.03, 0.06]))
    features = np.asarray(
        [surrogate.mariner.benefit(panel.epochs, rate, power) for rate, power in surrogate.SHAPE_PAIRS]
    )
    original = np.repeat(features[:, None], len(surrogate.FORGETTING), axis=1)
    original[:, 1:] = 999
    frozen_original = original.copy()
    ablated = diagnostic.mu0_features(original)
    training = np.arange(15)
    with threadpool_limits(limits=1):
        fit = surrogate.fit_model(panel, response, training, training % 3, model, ablated)
        aggregate = panel.aggregate
        tied = dataclasses.replace(panel, phase0=aggregate, phase1=aggregate)
        prediction = fit.predict(panel, np.arange(18))
        tied_prediction = fit.predict(tied, np.arange(18))
    assert fit.forgetting == 0
    np.testing.assert_allclose(prediction, tied_prediction, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(original, frozen_original)
