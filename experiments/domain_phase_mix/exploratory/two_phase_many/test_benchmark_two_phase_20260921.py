# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark


@pytest.fixture
def panel() -> benchmark.data.BenchPanel:
    first = np.linspace(0.05, 0.95, 14)
    last = np.roll(first, 3)
    last[-1] = first[-1]
    phase0 = np.column_stack([first, 1 - first])
    phase1 = np.column_stack([last, 1 - last])
    return benchmark.data.BenchPanel(
        name="synthetic",
        buckets=("a", "b"),
        phase0=phase0,
        phase1=phase1,
        alpha=0.8,
        c0=np.asarray([2.0, 3.0]),
        c1=np.asarray([0.5, 0.75]),
        targets={"loss": 2.0 - first**0.7 + 0.2 * last},
        rows=pd.DataFrame(
            {
                "row_id": [str(index) for index in range(14)],
                "policy_id": [str(index) for index in range(14)],
                "group_id": [str(index) for index in range(14)],
                "source": "fixture",
                "region": "interior",
                "tied": [False] * 13 + [True],
            }
        ),
        input_hashes={},
        metadata={"family_index": [0, 1], "family_names": ["a", "b"]},
    )


@pytest.mark.parametrize("model", ["effective_exposure_dsp", "aggregate_mariner_train_median"])
def test_baseline_fit_excludes_outer_and_tied_outcomes(panel: benchmark.data.BenchPanel, model: str) -> None:
    train, query, inner = np.arange(12), np.asarray([12, 13]), np.arange(12) % 3
    predicted, fitted = benchmark.fit_predict_baseline(panel, "loss", model, train, query, inner)
    altered_labels = panel.targets["loss"].copy()
    altered_labels[query] = [-1e12, 1e12]
    altered = dataclasses.replace(panel, targets={"loss": altered_labels})
    replay, replay_fit = benchmark.fit_predict_baseline(altered, "loss", model, train, query, inner)
    np.testing.assert_array_equal(predicted, replay)
    assert fitted == replay_fit
    with pytest.raises(ValueError, match="Tied outcomes reached"):
        benchmark.fit_predict_baseline(panel, "loss", model, np.arange(14), query, np.arange(14) % 3)


def test_dsp_wrapper_matches_native_estimator_on_identical_nested_splits(panel: benchmark.data.BenchPanel) -> None:
    train, query, inner = np.arange(12), np.asarray([12, 13]), np.arange(12) % 3
    predicted, fitted = benchmark.fit_predict_baseline(panel, "loss", "effective_exposure_dsp", train, query, inner)
    training = benchmark.native_panel(panel, train)
    testing = benchmark.native_panel(panel, query)
    model = benchmark.baseline_model(training, "effective_exposure_dsp")
    native = benchmark.legacy.fit_model(
        training,
        model,
        "loss",
        n_splits=3,
        split_fn=lambda panel, n_splits, seed: [(inner != fold, inner == fold) for fold in range(3)],
    )
    assert fitted["shape"] == native.shape
    assert fitted["ridge"] == native.l2
    np.testing.assert_allclose(predicted, native.predict(testing, model), rtol=0, atol=1e-12)


def test_exposure_adapter_preserves_realized_dose_and_rpl_refuses_discrete_adapter(
    panel: benchmark.data.BenchPanel,
) -> None:
    early = panel.phase0 * panel.c0 * np.linspace(0.8, 1.2, len(panel.rows))[:, None]
    late = panel.phase1 * panel.c1 * np.linspace(1.1, 0.9, len(panel.rows))[:, None]
    discrete = dataclasses.replace(
        panel,
        metadata={**panel.metadata, "realized_phase0_epochs": early, "realized_phase1_epochs": late},
    )
    query = np.asarray([1, 5, 12])
    adapted = benchmark.native_panel(discrete, query)
    np.testing.assert_allclose(adapted.epochs, (early + late)[query], rtol=0, atol=1e-15)
    design = benchmark.legacy_models.build_effective_exposure_dsp(adapted, {"power": 1.0, "late_multiplier": 2.0})
    np.testing.assert_allclose(design.matrix[:, :2], -(early + 2 * late)[query], rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="constant exposure multipliers"):
        benchmark.fit_predict_baseline(discrete, "loss", "retained_power_law", np.arange(12), query, np.arange(12) % 3)
