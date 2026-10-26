# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib"]
# ///

"""Targeted precision sensitivity for the predeclared five-member bank average."""

import decimal
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    optimize_mariner_convex_round2_stable_20260917 as stable,
)

MODEL = "cv2_shifted_power"
PANEL = "delphi_3e18_39bucket"
TARGET = "table9"
ENSEMBLE = "round2_convex_equal_average"
D = decimal.Decimal.from_float


def decimal_prediction(fitted, epochs: np.ndarray) -> float:
    with decimal.localcontext() as context:
        context.prec = 70
        rate, power = D(float(fitted.shape["rate"])), D(float(fitted.shape["power"]))
        value = D(float(fitted.head.intercept))
        for i, epoch in enumerate(epochs):
            exposure = D(float(epoch))
            benefit = (power * (1 + rate * exposure).ln()).exp() - 1
            value -= D(float(fitted.head.coefficients[i])) * benefit
            value += D(float(fitted.head.coefficients[i + len(epochs)])) * exposure / 32
        value = min(max(value, decimal.Decimal(-30)), decimal.Decimal(30))
        return float(D(float(fitted.head.floor)) + value.exp())


def selection(frame: pd.DataFrame, prediction: str) -> dict:
    selected = frame.loc[frame[prediction].idxmin()]
    return {
        "coordinate": selected.coordinate_id,
        "prediction": float(selected[prediction]),
        "observed": float(selected.measured),
        "regret": float(selected.measured - frame.measured.min()),
    }


def main() -> None:
    m = stable.original
    root = m.banks.OUTPUT.parent
    output = root / "convex_round2_optimizer/bank_precision_sensitivity"
    ensemble_path = root / "convex_round2_summary/bank_predictions.csv"
    own_path = m.banks.OUTPUT / "geometry/bank_predictions.csv"
    panel = m.canonical.bench.load_panel(PANEL)
    bank, query = m.banks.prior.bank_query(panel, TARGET)
    paths = [
        m.banks.prior.fit_path(m.banks.OUTPUT / "geometry", MODEL, PANEL, TARGET, i)
        for i in range(len(panel.group(TARGET).components))
    ]
    manifest = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stable_adapter_sha256": hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest(),
        "ensemble_csv_sha256": m.banks.prior.sha256(ensemble_path),
        "geometry_csv_sha256": m.banks.prior.sha256(own_path),
        "fit_hashes": {str(p): m.banks.prior.sha256(p) for p in paths},
        "member_weight": 0.2,
        "method": (
            "Stable affine-simplex evaluation for 339 fixed-pool rows; 70-digit original formula "
            "for 16 altered-pool rows, all component heads; original numerical clipping retained"
        ),
        "purpose": "Sensitivity of already predeclared selection; no new member/algorithm selection or refitting",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Bank sensitivity identity changed")
    m.canonical.shared.write_json(path, manifest)
    fits = [joblib.load(p) for p in paths]
    initial = m.objective_from_fits(MODEL, panel, TARGET, fits)
    objective = stable.StableObjective(initial.spec, panel, TARGET, fits)
    normal = np.all(
        np.isclose(query.exposures, query.weights * panel.features.inventory, rtol=1e-12, atol=1e-12), axis=1
    )
    assert int(normal.sum()) == 339 and int((~normal).sum()) == 16
    values = np.zeros((query.rows, len(fits)))
    floors = np.asarray([f.head.floor for f in fits])
    values[normal] = floors + np.exp(np.clip(objective.logits(query.weights[normal]), -30, 30))
    for row in np.flatnonzero(~normal):
        values[row] = [decimal_prediction(fit, query.exposures[row]) for fit in fits]
    alternative = values @ panel.group(TARGET).aggregation_weights
    own = pd.read_csv(own_path)
    own = (
        own[own.model.eq(MODEL) & own.panel.eq(PANEL) & own.target.eq(TARGET)]
        .set_index("coordinate_id")
        .loc[bank.coordinate_id]
        .reset_index()
    )
    parent = pd.read_csv(ensemble_path)
    parent = (
        parent[parent.model.eq(ENSEMBLE) & parent.panel.eq(PANEL) & parent.target.eq(TARGET)]
        .set_index("coordinate_id")
        .loc[bank.coordinate_id]
        .reset_index()
    )
    np.testing.assert_array_equal(parent.coordinate_id, own.coordinate_id)
    frame = parent.copy()
    frame["original_member_prediction"] = own.prediction.to_numpy()
    frame["precision_member_prediction"] = alternative
    frame["precision_ensemble_prediction"] = parent.prediction.to_numpy() + (alternative - own.prediction.to_numpy()) / 5
    frame["fixed_pool"] = normal
    frame.to_csv(output / "predictions.csv", index=False)
    readouts = []
    for labels in ("current_weights_complete", "historical_aggregate"):
        eligible = (
            frame[frame.complete_components].copy()
            if labels == "current_weights_complete"
            else frame.assign(measured=frame.historical_aggregate)
        )
        for stratum, mask in m.banks.prior.source_masks(eligible):
            subset = eligible.iloc[np.flatnonzero(mask)].dropna(
                subset=["measured", "prediction", "precision_ensemble_prediction"]
            )
            if subset.empty:
                continue
            before, after = selection(subset, "prediction"), selection(subset, "precision_ensemble_prediction")
            readouts.append(
                {
                    "labels": labels,
                    "stratum": stratum,
                    "rows": len(subset),
                    "original": before,
                    "precision": after,
                    "same_coordinate": before["coordinate"] == after["coordinate"],
                }
            )
    result = {
        "readouts": readouts,
        "max_fixed_pool_member_difference": float(
            np.max(np.abs(alternative[normal] - own.prediction.to_numpy()[normal]))
        ),
        "max_altered_pool_member_difference": float(
            np.max(np.abs(alternative[~normal] - own.prediction.to_numpy()[~normal]))
        ),
        "max_ensemble_difference": float(np.max(np.abs(frame.precision_ensemble_prediction - frame.prediction))),
    }
    m.canonical.shared.write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
