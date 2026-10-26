# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nextgen_two_phase_round6_20260921 as model
from test_nextgen_two_phase_round5_20260921 import synthetic_panel


@pytest.mark.timeout(900)
def test_complete_search_excludes_nontraining_labels_and_transports_selected_floor(tmp_path) -> None:
    panel = synthetic_panel()
    shape = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
    matrix = model.matrices(panel, np.arange(48), shape, 0.5, 0.5)
    response = 0.5 + np.exp(0.5 + matrix @ np.array([6.0, 1.0, 0.03, 0.06]))
    train, inner = np.arange(36), np.arange(36) % 3
    poisoned = response.copy()
    poisoned[36:] = np.tile([np.nan, np.inf, -np.inf, -1e12], 3)
    # Both invocations use every native shape/ridge and the complete scalar
    # search. The phase model shares this same training-row selection path.
    with threadpool_limits(limits=1):
        clean = model.fit_model(panel, response, train, inner, "NG2-016")
        dirty = model.fit_model(
            dataclasses.replace(panel, targets={"poison": poisoned}), poisoned, train, inner, "NG2-016"
        )
    assert clean.to_json() == dirty.to_json()
    query = np.arange(36, 48)
    np.testing.assert_array_equal(clean.predict(panel, query), dirty.predict(panel, query))

    saved = clean.to_json()
    matrix = model.matrices(panel, train, clean.shape, clean.q, clean.kappa)
    error = 0.0
    count = 0
    for fold in sorted(set(inner)):
        fitting, validation = np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)
        # Independent native QR-NNLS solves verify the reported CV risk at the
        # same absolute floor later used by the outer head.
        native = model.mariner.fit_head(
            matrix[fitting], response[train[fitting]], clean.ridge, model.mariner.FloorSpec(clean.head.floor, 0, 0)
        )
        prediction = native.predict(matrix[validation])
        error += float(np.sum((prediction - response[train[validation]]) ** 2))
        count += len(validation)
    np.testing.assert_allclose(clean.inner_rmse, np.sqrt(error / count), rtol=1e-9, atol=1e-11)

    path = tmp_path / "fit.json"
    path.write_text(json.dumps(saved, allow_nan=False))
    restored = json.loads(path.read_text())
    head = model.log_head.Head(
        float(restored["intercept"]),
        np.array(restored["aggregate_coefficients"]),
        np.array(restored["phase_coefficients"]),
        float(restored["floor"]),
    )
    fit = model.Fit(
        restored["model"],
        restored["shape"],
        float(restored["q"]),
        float(restored["kappa"]),
        float(restored["ridge"]),
        float(restored["inner_rmse"]),
        head,
        restored["floor_search"],
    )
    np.testing.assert_array_equal(fit.predict(panel, query), clean.predict(panel, query))
