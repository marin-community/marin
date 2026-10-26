# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nextgen_two_phase_round4_20260921_v2 as model


def synthetic_panel() -> model.benchmark.data.BenchPanel:
    rng = np.random.default_rng(20260925)
    phase0, phase1 = rng.dirichlet(np.ones(2), 48), rng.dirichlet(np.ones(2), 48)
    phase1[-8:] = phase0[-8:]
    return model.benchmark.data.BenchPanel(
        "synthetic_round4",
        ("web", "code"),
        phase0,
        phase1,
        0.8,
        np.array([0.8, 21.0]),
        np.array([0.2, 5.25]),
        {},
        pd.DataFrame({"row_id": [str(i) for i in range(48)], "tied": np.r_[np.zeros(40, bool), np.ones(8, bool)]}),
        {},
        {},
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("name", ["NG2-011", "NG2-012", "NG2-013"])
def test_complete_search_excludes_nontraining_labels_and_roundtrips_predictions(name, tmp_path) -> None:
    panel = synthetic_panel()
    shape = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
    matrix = model.matrices(panel, np.arange(48), shape, 0.5, 0.5)
    response = 0.5 + np.exp(0.5 + matrix @ np.array([6.0, 1.0, 0.03, 0.06]))
    train, inner = np.arange(36), np.arange(36) % 3
    poisoned = response.copy()
    poisoned[36:] = np.tile([np.nan, np.inf, -np.inf, -1e12], 3)
    # Each invocation performs the entire registered shape/state/ridge search
    # and native floor search. No grids, solver calls, or responses are mocked.
    with threadpool_limits(limits=1):
        clean = model.fit_model(panel, response, train, inner, name)
        dirty = model.fit_model(dataclasses.replace(panel, targets={"poison": poisoned}), poisoned, train, inner, name)
    assert clean.to_json() == dirty.to_json()
    query = np.arange(36, 48)
    np.testing.assert_array_equal(clean.predict(panel, query), dirty.predict(panel, query))

    output = tmp_path / "fit.json"
    output.write_text(json.dumps(clean.to_json(), allow_nan=False))
    saved = json.loads(output.read_text())
    arguments = (
        float(saved["intercept"]),
        np.array(saved["aggregate_coefficients"]),
        np.array(saved["phase_coefficients"]),
        float(saved["floor"]),
    )
    if saved["loss"] == "mean":
        solver = saved["solver"]
        head = model.mean_head.Head(
            *arguments,
            float(solver["objective"]),
            int(solver["iterations"]),
            float(solver["certified_kkt_residual"]),
        )
    else:
        head = model.log_head.Head(*arguments)
    restored = model.Fit(
        saved["model"],
        saved["shape"],
        float(saved["q"]),
        float(saved["kappa"]),
        float(saved["ridge"]),
        float(saved["gamma"]),
        bool(saved["flat_profile"]),
        float(saved["inner_rmse"]),
        head,
    )
    np.testing.assert_array_equal(restored.predict(panel, query), clean.predict(panel, query))
