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
import nextgen_two_phase_round3_20260921 as model


def panel() -> model.benchmark.data.BenchPanel:
    rng = np.random.default_rng(20260921)
    first = rng.dirichlet(np.ones(2), 48)
    second = rng.dirichlet(np.ones(2), 48)
    second[-8:] = first[-8:]
    return model.benchmark.data.BenchPanel(
        "synthetic",
        ("web", "code"),
        first,
        second,
        0.8,
        np.array([0.8, 21.0]),
        np.array([0.2, 5.25]),
        {},
        pd.DataFrame({"row_id": [str(i) for i in range(48)], "tied": np.r_[np.zeros(40, bool), np.ones(8, bool)]}),
        {},
        {},
    )


@pytest.mark.parametrize("name", list(model.MODELS))
def test_tied_response_equals_native_head_with_signed_phase_parameters(name) -> None:
    data = panel()
    indices = np.arange(40, 48)
    for shape in ({"rate": 0.5, "power": 0.7, "threshold": 3.0}, {"rate": 4.0, "power": 0.3, "threshold": 1.0}):
        for mu in (0.0, 64.0):
            aggregate, phase, _, _ = model.matrices(data, indices, shape, name, mu)
            np.testing.assert_array_equal(phase, np.zeros_like(phase))
            for floor in (-3.0, 0.7):
                beta = np.array([6, 1, 0.03, 0.06])
                head = model.log_head.Head(0.5, beta, np.full(phase.shape[1], -100.0), floor)
                expected = model.mariner.Head(0.5, beta, floor).predict(
                    model.mariner.design_matrix(data.epochs[indices], shape)
                )
                np.testing.assert_allclose(head.predict(aggregate, phase), expected, rtol=0, atol=1e-8)


def test_increment_is_difference_of_cumulative_late_acquisitions() -> None:
    data = panel()
    shape = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
    benefit = model.mariner.benefit
    actual_increment = benefit(data.epochs, 0.5, 0.7) - benefit(data.phase0_epochs, 0.5, 0.7)
    tied_increment = benefit(data.epochs, 0.5, 0.7) - benefit(data.alpha * data.epochs, 0.5, 0.7)
    np.testing.assert_allclose(model.phase_design(data, shape, "NG2-010"), actual_increment - tied_increment, atol=2e-16)


def test_complete_search_ignores_nontraining_labels_and_recovers_same_predictions() -> None:
    data = panel()
    shape = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
    aggregate, phase, _, _ = model.matrices(data, np.arange(48), shape, "NG2-009", 0)
    response = 0.5 + np.exp(0.5 + aggregate @ np.array([6, 1, 0.03, 0.06]) + phase[:, 0] * 0.4)
    train, inner = np.arange(36), np.arange(36) % 3
    poisoned = response.copy()
    poisoned[36:] = np.linspace(-1000, 1000, 12)
    with threadpool_limits(limits=1):
        clean = model.fit_model(data, response, train, inner, "NG2-009")
        dirty = model.fit_model(
            dataclasses.replace(data, targets={"poison": poisoned}), poisoned, train, inner, "NG2-009"
        )
    assert clean.to_json() == dirty.to_json()
    np.testing.assert_array_equal(clean.predict(data, np.arange(36, 48)), dirty.predict(data, np.arange(36, 48)))
