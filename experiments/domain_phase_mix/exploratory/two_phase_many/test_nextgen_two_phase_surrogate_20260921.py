# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import nextgen_two_phase_surrogate_20260921 as surrogate


@pytest.fixture
def panel() -> surrogate.benchmark.data.BenchPanel:
    rng = np.random.default_rng(20260921)
    first = rng.dirichlet(np.ones(2), 18)
    second = rng.dirichlet(np.ones(2), 18)
    second[-2:] = first[-2:]
    alpha = 0.8
    tied = np.all(first == second, axis=1)
    rows = pd.DataFrame(
        {
            "row_id": [f"row_{i}" for i in range(len(first))],
            "policy_id": [f"policy_{i}" for i in range(len(first))],
            "group_id": [f"group_{i}" for i in range(len(first))],
            "source": "synthetic",
            "region": "synthetic",
            "tied": tied,
        }
    )
    return surrogate.benchmark.data.BenchPanel(
        name="synthetic_two_phase",
        buckets=("text", "code"),
        phase0=first,
        phase1=second,
        alpha=alpha,
        c0=alpha * np.array([0.7, 8.0]),
        c1=(1 - alpha) * np.array([0.7, 8.0]),
        targets={},
        rows=rows,
        input_hashes={},
        metadata={},
    )


@pytest.mark.parametrize("model,consolidation", [("NG2-001", 0), ("NG2-002", 4), ("NG2-003", 4)])
def test_design_without_forgetting_equals_native_mariner(panel, model, consolidation) -> None:
    shape = {"rate": 0.3, "power": 0.7, "threshold": 1.0}
    query = np.arange(len(panel.rows))
    expected = surrogate.mariner.design_matrix(panel.epochs, shape)
    actual = surrogate.design(panel, query, shape, model, 0, consolidation)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)


@pytest.mark.parametrize("model,consolidation", [("NG2-001", 0), ("NG2-002", 4), ("NG2-003", 4)])
def test_design_on_tied_queries_equals_native_mariner(panel, model, consolidation) -> None:
    shape = {"rate": 0.3, "power": 0.7, "threshold": 1.0}
    query = np.flatnonzero(panel.tied)
    expected = surrogate.mariner.design_matrix(panel.epochs[query], shape)
    actual = surrogate.design(panel, query, shape, model, 16, consolidation)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)


@pytest.mark.parametrize("defect", ["discrete_epochs", "inconsistent_phase_fraction"])
def test_design_rejects_unsupported_physical_exposure_geometry(panel, defect) -> None:
    if defect == "discrete_epochs":
        actual_early = panel.phase0_epochs.copy()
        actual_early[0, 1] += 0.01
        unsupported = replace(panel, metadata={"realized_phase0_epochs": actual_early})
    else:
        unsupported = replace(panel, c1=1.01 * panel.c1)
    shape = {"rate": 0.3, "power": 0.7, "threshold": 1.0}
    with pytest.raises(ValueError, match="constant exposure geometry"):
        surrogate.design(unsupported, np.arange(3), shape, "NG2-002", 4, 1)


def test_fitting_rejects_tied_training_labels(panel) -> None:
    train = np.array([0, 1, 2, 3, 16])
    with pytest.raises(ValueError, match="Tied labels"):
        surrogate.fit_model(panel, np.linspace(1, 2, len(panel.rows)), train, np.arange(len(train)) % 2, "NG2-001")


def test_complete_native_fit_ignores_all_nontraining_outcomes(panel) -> None:
    # Two full searches use identical allowed labels while every held/tied label
    # changes, including nonfinite sentinels. The search grid is never patched.
    train = np.arange(12)
    inner = np.arange(len(train)) % 3
    shape = {"rate": 0.3, "power": 0.7, "threshold": 1.0}
    response = 0.7 + np.exp(
        -0.1 + surrogate.mariner.design_matrix(panel.epochs, shape) @ np.array([0.7, 0.3, 0.05, 0.04])
    )
    changed = response.copy()
    changed[12:] = [np.nan, np.inf, -np.inf, -1000, 1e9, 0]
    with threadpool_limits(limits=1):
        original = surrogate.fit_model(panel, response, train, inner, "NG2-001")
        perturbed = surrogate.fit_model(panel, changed, train, inner, "NG2-001")
    assert original.to_json() == perturbed.to_json()
    query = np.arange(12, len(panel.rows))
    np.testing.assert_array_equal(original.predict(panel, query), perturbed.predict(panel, query))
