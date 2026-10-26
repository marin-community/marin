# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_refit_robustness_20260917 as validation,
)


@pytest.mark.parametrize("model_id", validation.MODEL_IDS)
def test_numeric_head_roundtrip_preserves_off_support_predictions(tmp_path, model_id):
    panel = bench.load_panel("delphi_3e18_39bucket")
    features = dataclasses.replace(panel.features, exposures=panel.features.exposures * 2.0)
    spec = next(spec for spec in candidates.candidate_specs() if spec.model_id == model_id)
    model = candidates.build_model(spec, features)
    head = models.FittedHead(0.2, np.linspace(0, 0.025, 2 * features.buckets), 0.4, 77, kappa=2.0)
    fitted = models.Fitted(
        {"rate": 0.5, "power": 0.5, "threshold": 2.0}, 0.1, head, {"link": "kappa_floor", "kappa": 2.0}
    )
    path = tmp_path / "head.npz"
    bench.atomic_save(
        path,
        {
            **{f"head_{key}": value for key, value in dataclasses.asdict(head).items()},
            "shape_json": json.dumps(fitted.shape),
            "ridge": fitted.ridge,
            "diagnostics_json": json.dumps(fitted.diagnostics),
        },
    )
    restored = validation.load_fitted(path)
    actual = model.predict(restored, features, np.arange(12))
    expected = model.predict(fitted, features, np.arange(12))
    np.testing.assert_array_equal(actual, expected)


def test_rms_scaling_changes_regularization_but_not_unpenalized_predictions():
    x = np.linspace(0.1, 2.0, 20)
    design = models.Design(np.column_stack((x, x**2 / 100)), np.ones(2), ("linear", "quadratic"))
    y = 0.3 + 0.2 * x + 0.04 * x**2
    raw = models.HeadSpec()
    rms = dataclasses.replace(raw, scale_columns=True)
    unscaled = models.fit_head(design, y, 0.0, raw)
    scaled = models.fit_head(design, y, 0.0, rms)
    np.testing.assert_allclose(
        models.predict_head(unscaled, design.values, raw), models.predict_head(scaled, design.values, rms), atol=1e-10
    )
    penalized_raw = models.fit_head(design, y, 0.1, raw)
    penalized_rms = models.fit_head(design, y, 0.1, rms)
    assert (
        np.max(
            np.abs(
                models.predict_head(penalized_raw, design.values, raw)
                - models.predict_head(penalized_rms, design.values, rms)
            )
        )
        > 0.001
    )


def test_aggregate_retains_archived_observation_separate_from_component_sum():
    panel = bench.load_panel("delphi_3e18_39bucket")
    group = panel.group("uncheatable")
    frame = pd.DataFrame(
        [
            {
                "model": "hs17_mariner",
                "panel": panel.name,
                "target": "uncheatable",
                "repeat": 0,
                "fold": 0,
                "row_index": 0,
                "component": component,
                "observed": 3.0,
                "prediction": 2.25,
            }
            for component in group.components
        ]
    )
    exported = validation.independent_aggregate(frame).iloc[0]
    assert exported.observed == group.aggregate[0]
    assert exported.component_sum_observed == pytest.approx(3.0)
    assert exported.prediction == pytest.approx(2.25)
