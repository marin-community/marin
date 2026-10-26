# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Retrospective convex-envelope and convex-head ensemble feasibility diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/mariner_local_validation_20260917"
OUTPUT = ROOT / "convex_geometry_round2"
COORDINATES = HERE / "reference_outputs/single_phase_heldout_benchmark_expanded_20260913/heldout_coordinates.csv"
CONVEX_MODELS = ("cv17_smooth_d1", "cv17_huber_d8")
LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}


def convex_envelope_point(weights: np.ndarray, losses: np.ndarray, index: int) -> dict:
    """Find a minimum-loss exact convex combination excluding the query point."""
    mask = np.arange(len(weights)) != index
    endpoints = weights[mask]
    scale = np.maximum(np.max(np.abs(weights), axis=0), 1e-8)
    equality = np.vstack([np.ones(len(endpoints)), (endpoints / scale).T])
    query = np.r_[1.0, weights[index] / scale]
    result = linprog(losses[mask], A_eq=equality, b_eq=query, bounds=(0, None), options=LP_OPTIONS)
    if not result.success and result.status != 2:
        result = linprog(
            losses[mask],
            A_eq=equality,
            b_eq=query,
            bounds=(0, None),
            method="highs-ds",
            options={**LP_OPTIONS, "presolve": False},
        )
    if result.status == 2:
        return {"status": "extreme", "gap": 0.0}
    if not result.success:
        return {"status": "unresolved", "gap": 0.0, "message": result.message}
    coefficients = result.x
    reconstruction = coefficients @ endpoints
    residual = float(np.max(np.abs(reconstruction - weights[index])))
    scaled_residual = float(np.max(np.abs(equality @ coefficients - query)))
    assert residual < 2e-8 and scaled_residual < 2e-8
    assert coefficients.min() >= -1e-9 and abs(coefficients.sum() - 1) < 2e-8
    bound = float(coefficients @ losses[mask])
    dual_gap = float(bound - result.eqlin.marginals @ query)
    dual_violation = float(np.max(equality.T @ result.eqlin.marginals - losses[mask]))
    assert abs(dual_gap) < 1e-8 and dual_violation < 1e-8
    return {
        "status": "represented",
        "gap": float(losses[index] - bound),
        "bound": bound,
        "coordinate_residual": residual,
        "scaled_residual": scaled_residual,
        "objective_dual_gap": dual_gap,
        "dual_feasibility_violation": dual_violation,
        "support": np.flatnonzero(mask)[coefficients > 1e-9].tolist(),
        "coefficients": coefficients[coefficients > 1e-9].tolist(),
        "coefficient_l2_squared": float(coefficients @ coefficients),
    }


def geometry_scope(frame: pd.DataFrame, weight_columns: list[str], scope: str, target: str, noise: float) -> dict:
    """Audit all points, excluding comparisons across different repetition pools."""
    destination = OUTPUT / f"envelope_{target}_{scope}.json"
    weights = frame[weight_columns].to_numpy(float)
    losses = frame.measured.to_numpy(float)
    identity = hashlib.sha256(weights.tobytes() + losses.tobytes()).hexdigest()
    if destination.exists():
        saved = json.loads(destination.read_text())
        assert saved["input_identity"] == identity
        if saved.get("schema_version") == 2:
            return saved
    records = []
    for index, row in frame.reset_index(drop=True).iterrows():
        result = convex_envelope_point(weights, losses, index)
        result.update({"id": row.coordinate_id, "source": row.source, "observed": float(row.measured)})
        if result["status"] == "represented":
            result["diagnostic_noise_sd"] = noise * np.sqrt(1 + result["coefficient_l2_squared"])
            result["diagnostic_z"] = result["gap"] / result["diagnostic_noise_sd"]
            result["support_ids"] = frame.iloc[result["support"]].coordinate_id.tolist()
        records.append(result)
    maximum = max(records, key=lambda value: value["gap"])
    represented = [record for record in records if record["status"] == "represented"]
    payload = {
        "schema_version": 2,
        "input_identity": identity,
        "scope": scope,
        "target": target,
        "rows": len(frame),
        "represented": len(represented),
        "extreme": sum(record["status"] == "extreme" for record in records),
        "unresolved": sum(record["status"] == "unresolved" for record in records),
        "positive_gap_count": sum(record["gap"] > 1e-8 for record in represented),
        "max_gap": maximum["gap"],
        "uniform_label_adjustment_lower_bound": max(0, maximum["gap"]) / 2,
        "max_diagnostic_z": max((record["diagnostic_z"] for record in represented), default=0),
        "max_gap_id": maximum["id"],
        "max_coordinate_residual": max((record["coordinate_residual"] for record in represented), default=0),
        "max_absolute_dual_gap": max((abs(record["objective_dual_gap"]) for record in represented), default=0),
        "records": records,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print({key: value for key, value in payload.items() if key != "records"}, flush=True)
    return payload


def nearest_hull_distance(weights: np.ndarray, index: int) -> float:
    """Return total variation distance to the convex hull of all other points."""
    others = weights[np.arange(len(weights)) != index]
    count, dimension = others.shape
    constraints = np.block([[others.T, -np.eye(dimension)], [-others.T, -np.eye(dimension)]])
    result = linprog(
        np.r_[np.zeros(count), np.ones(dimension)],
        A_ub=constraints,
        b_ub=np.r_[weights[index], -weights[index]],
        A_eq=np.r_[np.ones(count), np.zeros(dimension)][None, :],
        b_eq=[1],
        bounds=(0, None),
        options=LP_OPTIONS,
    )
    if not result.success:
        raise RuntimeError(f"Hull distance LP failed: {result.message}")
    actual = float(np.abs(result.x[:count] @ others - weights[index]).sum())
    assert abs(actual - result.fun) < 1e-8
    return actual / 2


def ensemble_margin(predictions: np.ndarray, target: int, componentwise: bool) -> dict:
    """Maximize the desired policy's selection margin over convex-head blends."""
    base = predictions[0].sum(axis=1)
    difference = predictions[1] - predictions[0]
    if not componentwise:
        difference = difference.sum(axis=1, keepdims=True)
    mask = np.arange(len(base)) != target
    constraints = np.c_[difference[target] - difference[mask], np.ones(mask.sum())]
    bounds = [(0, 1)] * difference.shape[1] + [(None, None)]
    result = linprog(
        np.r_[np.zeros(difference.shape[1]), -1.0],
        A_ub=constraints,
        b_ub=base[mask] - base[target],
        bounds=bounds,
        options=LP_OPTIONS,
    )
    if not result.success:
        raise RuntimeError(f"Ensemble LP failed: {result.message}")
    blended = base + difference @ result.x[:-1]
    margin = float(np.min(blended[mask] - blended[target]))
    assert abs(margin - result.x[-1]) < 2e-8
    return {
        "maximum_margin": margin,
        "attainable": margin >= -1e-8,
        "strictly_attainable": margin > 1e-8,
        "coefficients": result.x[:-1].tolist(),
        "selected_index": int(np.argmin(blended)),
        "desired_predicted": float(blended[target]),
    }


def ensemble_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    weights_records = []
    for (panel_name, target), rows in predictions.groupby(["panel", "target"]):
        observed = rows[rows.model.eq("hs17_mariner")].set_index("coordinate_id").sort_index()
        ids = observed.index
        panel = bench.load_panel(panel_name)
        group = panel.group(target)
        arrays = []
        for model in CONVEX_MODELS:
            components = []
            for index, component_weight in enumerate(group.aggregation_weights):
                path = ROOT / "banks/convex/fits" / model / panel_name / target / f"component_{index:03d}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    components.append(
                        pd.Series(saved["prediction"], index=saved["coordinate_id"]).loc[ids].to_numpy()
                        * component_weight
                    )
            arrays.append(np.asarray(components).T)
            reference = rows[rows.model.eq(model)].set_index("coordinate_id").loc[ids, "prediction"]
            np.testing.assert_allclose(arrays[-1].sum(axis=1), reference.to_numpy(), rtol=0, atol=2e-12)
        values = np.asarray(arrays)
        desired = {
            "mariner_choice": int(np.argmin(observed.prediction)),
            "measured_best": int(np.argmin(observed.measured)),
        }
        for label, index in desired.items():
            for componentwise in (False, True):
                result = ensemble_margin(values, index, componentwise)
                coefficients = result.pop("coefficients")
                selected = result.pop("selected_index")
                record = {
                    "panel": panel_name,
                    "target": target,
                    "desired": label,
                    "ensemble": "per_component_blend" if componentwise else "global_blend",
                    "rows": len(ids),
                    "desired_id": ids[index],
                    "desired_measured": float(observed.measured.iloc[index]),
                    "desired_regret": float(observed.measured.iloc[index] - observed.measured.min()),
                    "actual_selected_id": ids[selected],
                    "actual_selected_regret": float(observed.measured.iloc[selected] - observed.measured.min()),
                    **result,
                }
                records.append(record)
                weights_records.append({**record, "coefficients": coefficients})
    (OUTPUT / "ensemble_coefficients.json").write_text(json.dumps(weights_records, indent=2) + "\n")
    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT / "ensemble_feasibility.csv", index=False)
    return frame


def validate_geometry() -> None:
    """Check envelope signs, extreme-point exclusion, and exact ensemble recovery."""
    points = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert convex_envelope_point(points, np.array([0.0, 1.0, 0.0]), 1)["gap"] == 1.0
    assert convex_envelope_point(points, np.array([1.0, 0.0, 1.0]), 1)["gap"] == -1.0
    assert convex_envelope_point(points, np.array([0.0, 1.0, 0.0]), 0)["status"] == "extreme"
    # Every global blend misses the middle policy; choosing the best head per component recovers it.
    predictions = np.array([[[0.0, 0.0], [1.0, 3.0], [4.0, 2.0]], [[2.0, 4.0], [3.0, 1.0], [0.0, 0.0]]])
    assert ensemble_margin(predictions, 1, False)["maximum_margin"] == -1.0
    assert ensemble_margin(predictions, 1, True)["maximum_margin"] == 2.0


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validate_geometry()
    bank = pd.read_csv(COORDINATES)
    predictions = pd.concat(
        [pd.read_csv(ROOT / folder / "bank_predictions.csv") for folder in ("banks", "banks/convex")]
    )
    predictions = predictions[predictions.complete_components].copy()
    panel = bench.load_panel("delphi_3e18_39bucket")
    weight_columns = [f"weight::{bucket}" for bucket in panel.buckets]
    pool_columns = [f"pool_fraction::{bucket}" for bucket in panel.buckets]
    coordinates = bank[bank.panel.eq(panel.name)].set_index("coordinate_id")
    baseline = predictions[predictions.model.eq("hs17_mariner") & predictions.panel.eq(panel.name)]
    noise = pd.read_csv(ROOT / "banks/dose_repeat_noise.csv").set_index("target").external_panel_repeat_sd
    summaries = []
    ignored = []
    choices = []
    for target in ("table9", "uncheatable"):
        group = panel.group(target)
        observed = baseline[baseline.target.eq(target)].set_index("coordinate_id")
        rows = coordinates.loc[observed.index].copy()
        rows["measured"] = observed.measured
        rows["source"] = "bank"
        fixed_pool = np.isclose(rows[pool_columns].fillna(1), 1, rtol=0, atol=1e-12).all(axis=1)
        ignored.extend(
            {"target": target, "coordinate_id": index} for index in rows.iloc[np.flatnonzero(~fixed_pool)].index
        )
        rows = rows[fixed_pool].reset_index()
        fit = pd.DataFrame(panel.features.weights, columns=weight_columns)
        fit["coordinate_id"] = [f"fit::{name}" for name in panel.runs]
        fit["measured"] = group.outcomes @ group.aggregation_weights
        fit["source"] = "fit"
        assert np.allclose(panel.features.exposures, panel.features.weights * panel.features.inventory)
        for scope, frame in (("fit", fit), ("bank_fixed_pool", rows), ("combined_fixed_pool", pd.concat([fit, rows]))):
            result = geometry_scope(frame, weight_columns, scope, target, float(noise.loc[target]))
            summaries.append({key: value for key, value in result.items() if key != "records"})
            if scope == "combined_fixed_pool":
                values = frame[weight_columns].to_numpy(float)
                ordered_ids = frame.coordinate_id.tolist()
                for model, model_rows in predictions[
                    predictions.panel.eq(panel.name) & predictions.target.eq(target)
                ].groupby("model"):
                    selected = model_rows.loc[model_rows.prediction.idxmin(), "coordinate_id"]
                    index = ordered_ids.index(selected)
                    record = result["records"][index]
                    choices.append(
                        {
                            "target": target,
                            "model": model,
                            "id": selected,
                            "status": record["status"],
                            "jensen_gap": record["gap"],
                            "nearest_other_hull_tv": nearest_hull_distance(values, index),
                        }
                    )
    pd.DataFrame(summaries).to_csv(OUTPUT / "geometry_summary.csv", index=False)
    pd.DataFrame(ignored).to_csv(OUTPUT / "excluded_variable_pools.csv", index=False)
    pd.DataFrame(choices).to_csv(OUTPUT / "selected_policy_geometry.csv", index=False)
    ensembles = ensemble_diagnostics(predictions)
    print(ensembles.drop(columns=["desired_id", "actual_selected_id"]).to_string(index=False))
    manifest = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (COORDINATES, ROOT / "banks/bank_predictions.csv", ROOT / "banks/convex/bank_predictions.csv")
        },
        "canonical_panel_inputs": panel.input_hashes,
        "lp_options": LP_OPTIONS,
        "tests": "Convex/concave midpoint sign, extreme vertex, global versus per-component blend passed.",
        "scope": "Archived observations only; no fitting, model tuning, or prospective performance inference.",
    }
    (OUTPUT / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
