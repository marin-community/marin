# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 80/20 WSD StarCoder stage: representability, controls, and joint versus independent fitting.

This is the one panel where a two-phase policy is known to beat the entire one-phase class, so it
decides whether the interference-evidence state can represent a real phase advantage at all. It also
carries the sharpest negative control in the project: repaired retained power law predicts about
0.029 BPB of phase gain on C4 English and Falcon RefinedWeb, where the sampled optimum is tied and the
observed gain is exactly zero.

Every metric is fitted, including the negative controls. Fitting them is the point -- the incumbent
fitted them too and still invented the gain.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    interference_evidence_model_20260806 as ile,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import starcoder_wsd80_panel_20260728 as wsd80

PROTOCOLS = ("random", "blocked")
# The two retention laws are closed for structural reasons that do not involve the acquisition curve,
# so they are kept at the exponential curve the family started from. Only the surviving law is given the
# full curvature grid, and the exponential is inside that grid as an exact special case.
LAWS = {
    ile.InterferenceLaw.ABSOLUTE: (math.inf,),
    ile.InterferenceLaw.SHARE_DROP: (math.inf,),
    ile.InterferenceLaw.RECENCY_EXPOSURE: ile.CURVATURE_GRID,
}


def geometry() -> ile.Geometry:
    c0, c1 = wsd80.epoch_multipliers()
    return ile.Geometry(
        c0=c0,
        c1=c1,
        phase_1_fraction=wsd80.REALIZED_PHASE_1_FRACTION,
        family_index=np.arange(len(wsd80.DOMAIN_NAMES)),
    )


def load_targets() -> tuple[wsd80.Panel, harness.MultiTarget]:
    panel = wsd80.load_surface()
    metrics = pd.read_csv(wsd80.SURFACE_DIR / "wsd80_all_bpb_metrics.csv").drop_duplicates("wandb_run_id")
    joined = panel.frame[["wandb_run_id"]].merge(metrics, on="wandb_run_id", how="left", validate="one_to_one")
    names = tuple(c for c in metrics.columns if c != "wandb_run_id" and joined[c].notna().all())
    values = joined[list(names)].to_numpy(dtype=float)
    return panel, harness.MultiTarget(
        names=names,
        values=values,
        observed=np.ones(values.shape, dtype=bool),
        family=tuple("wsd80" for _ in names),
        family_share=np.full(len(names), 1.0 / len(names)),
    )


def grid_weights(phase_0_share: np.ndarray, phase_1_share: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.column_stack([1.0 - phase_0_share, phase_0_share]),
            np.column_stack([1.0 - phase_1_share, phase_1_share]),
        ],
        axis=1,
    )


def interior_mask(panel: wsd80.Panel) -> np.ndarray:
    phase_0, phase_1 = panel.phase_0[:, 1], panel.phase_1[:, 1]
    margin = harness.BOUNDARY_MARGIN
    return (phase_0 > margin) & (phase_1 > margin) & (phase_0 < 1 - margin) & (phase_1 < 1 - margin)


def selection_row(panel: wsd80.Panel, target: np.ndarray, prediction: np.ndarray) -> dict:
    interior = np.flatnonzero(interior_mask(panel))
    ranked = interior[np.argsort(prediction[interior])]
    selected, best = int(ranked[0]), int(interior[np.argmin(target[interior])])
    top_five = ranked[: min(5, len(ranked))]
    return {
        "regret_at_1": float(target[selected] - target[best]),
        "regret_at_5": float(np.min(target[top_five]) - target[best]),
        "selected_phase_0": float(panel.phase_0[selected, 1]),
        "selected_phase_1": float(panel.phase_1[selected, 1]),
        "selected_distance": float(
            np.hypot(
                panel.phase_0[selected, 1] - panel.phase_0[best, 1],
                panel.phase_1[selected, 1] - panel.phase_1[best, 1],
            )
        ),
    }


def continuous_optimum(panel: wsd80.Panel, target: np.ndarray, model: ile.Model, grid: int) -> dict:
    axis = np.linspace(0.0, 1.0, grid)
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = phase_0.ravel(), phase_1.ravel()
    prediction = model.predict(grid_weights(flat_0, flat_1))
    margin = harness.BOUNDARY_MARGIN
    interior = (flat_0 > margin) & (flat_1 > margin) & (flat_0 < 1 - margin) & (flat_1 < 1 - margin)
    interior_rows = np.flatnonzero(interior)
    best_interior = int(interior_rows[np.argmin(prediction[interior_rows])])

    tied_axis = np.linspace(0.0, 1.0, grid * grid)
    tied_prediction = model.predict(grid_weights(tied_axis, tied_axis))
    observed_interior = np.flatnonzero(interior_mask(panel))
    observed_best = int(observed_interior[np.argmin(target[observed_interior])])
    return {
        "predicted_best_interior_phase_0": float(flat_0[best_interior]),
        "predicted_best_interior_phase_1": float(flat_1[best_interior]),
        "predicted_two_phase_gain": float(np.min(tied_prediction) - np.min(prediction)),
        "predicted_two_phase_gain_interior": float(np.min(tied_prediction) - prediction[best_interior]),
        "optimum_distance_interior": float(
            np.hypot(
                flat_0[best_interior] - panel.phase_0[observed_best, 1],
                flat_1[best_interior] - panel.phase_1[observed_best, 1],
            )
        ),
        "predicted_best_is_boundary": int(not interior[int(np.argmin(prediction))]),
        "max_phase_weight": float(max(flat_0[best_interior], flat_1[best_interior])),
        "phase_tv": float(abs(flat_1[best_interior] - flat_0[best_interior])),
    }


def run(output_dir: Path, grid: int, draws: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel, targets = load_targets()
    geo = geometry()
    indices = np.arange(len(panel.y))
    interior = interior_mask(panel)

    score_rows, optimum_rows, trace_rows, oof_store = [], [], [], {}

    for law, curvature_grid in LAWS.items():
        shapes = ile.shape_grid(law=law, curvature_grid=curvature_grid)
        blind_shapes = tuple(shape for shape in shapes if shape.interference == 0.0)
        for protocol in PROTOCOLS:
            outer = harness.wsd80_folds(
                protocol, panel.weights, indices, harness.WSD_OUTER_SPLITS, harness.WSD_OUTER_SEED
            )

            def inner_for(fold_id: int, train: np.ndarray, protocol=protocol):
                local = harness.wsd80_folds(
                    protocol, panel.weights, train, harness.WSD_INNER_SPLITS, harness.WSD_INNER_SEED_BASE + fold_id
                )
                return tuple((np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in local)

            # The phase-blind ablation is the same estimator restricted to the zero-interference slice of
            # the grid, so it is run as a separate pass over `blind_shapes` rather than as a third mode.
            passes = ((("joint", "independent"), shapes), (("phase_blind",), blind_shapes))
            predictions_by_mode: dict[str, np.ndarray] = {}
            models_by_mode: dict[str, list] = {}
            for modes, grid_shapes in passes:
                fitting_modes = tuple("independent" if m == "phase_blind" else m for m in modes)
                predicted, trace = harness.nested_predictions(
                    panel.weights, geo, targets, outer, inner_for, grid_shapes, ile.HEAD_RIDGE_GRID, fitting_modes
                )
                fitted, full_trace = harness.full_fit(
                    panel.weights, geo, targets, outer, grid_shapes, ile.HEAD_RIDGE_GRID, fitting_modes
                )
                rename = dict(zip(fitting_modes, modes, strict=True))
                for record in (*trace, *full_trace):
                    record["mode"] = rename[record["mode"]]
                    record["law"] = str(law)
                    record["protocol"] = protocol
                trace_rows.extend(trace)
                trace_rows.extend(full_trace)
                for fitting_mode, mode in rename.items():
                    predictions_by_mode[mode] = predicted[fitting_mode]
                    models_by_mode[mode] = fitted[fitting_mode]

            for mode, predictions in predictions_by_mode.items():
                models = models_by_mode[mode]
                oof_store[(law, protocol, mode)] = predictions
                for j, name in enumerate(targets.names):
                    target = targets.values[:, j]
                    prediction = predictions[:, j]
                    residual = prediction - target
                    row = {
                        "law": str(law),
                        "protocol": protocol,
                        "mode": mode,
                        "metric": name,
                        "rmse_all": float(np.sqrt(np.mean(residual**2))),
                        "rmse_interior": float(np.sqrt(np.mean(residual[interior] ** 2))),
                        "median_absolute_interior": float(np.median(np.abs(residual[interior]))),
                    }
                    row.update(selection_row(panel, target, prediction))
                    score_rows.append(row)

                    model = models[j]
                    if model is None:
                        continue
                    optimum = {"law": str(law), "protocol": protocol, "mode": mode, "metric": name}
                    optimum.update(continuous_optimum(panel, target, model, grid))
                    optimum["rho"] = model.shape.rho
                    optimum["interference"] = model.shape.interference
                    optimum["ridge"] = model.ridge
                    optimum_rows.append(optimum)

    frames = []
    for (law, protocol, mode), predicted in oof_store.items():
        block = pd.DataFrame(predicted, columns=list(targets.names))
        block.insert(0, "row", np.arange(len(predicted)))
        block.insert(0, "mode", mode)
        block.insert(0, "protocol", protocol)
        block.insert(0, "law", str(law))
        frames.append(block)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "wsd80_out_of_fold_predictions.csv", index=False)

    scores = pd.DataFrame(score_rows)
    optima = pd.DataFrame(optimum_rows)
    trace = pd.DataFrame(trace_rows)
    scores.to_csv(output_dir / "wsd80_scores.csv", index=False)
    optima.to_csv(output_dir / "wsd80_optima.csv", index=False)
    trace.to_csv(output_dir / "wsd80_selection_trace.csv", index=False)

    gate_metrics = (harness.PRIMARY_TARGET, *harness.POSITIVE_CONTROLS, *harness.NEGATIVE_CONTROLS)
    comparisons = []
    for law in LAWS:
        for protocol in PROTOCOLS:
            for name in gate_metrics:
                j = targets.index(name)
                joint = oof_store[(law, protocol, "joint")][:, j] - targets.values[:, j]
                independent = oof_store[(law, protocol, "independent")][:, j] - targets.values[:, j]
                blind = oof_store[(law, protocol, "phase_blind")][:, j] - targets.values[:, j]
                comparison = harness.paired_bootstrap_difference(
                    joint[interior], independent[interior], indices[interior], draws=draws
                )
                comparison.update(
                    {"law": str(law), "protocol": protocol, "metric": name, "contrast": "joint_minus_independent"}
                )
                comparisons.append(comparison)
                ablation = harness.paired_bootstrap_difference(
                    joint[interior], blind[interior], indices[interior], draws=draws
                )
                ablation.update(
                    {"law": str(law), "protocol": protocol, "metric": name, "contrast": "joint_minus_phase_blind"}
                )
                comparisons.append(ablation)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "wsd80_joint_vs_independent.csv", index=False)

    gates = []
    for law in LAWS:
        for protocol in PROTOCOLS:
            for mode in ("joint", "independent", "phase_blind"):
                cell = (optima.law == str(law)) & (optima.protocol == protocol) & (optima["mode"] == mode)
                selected = optima[cell].set_index("metric")
                picked = scores[
                    (scores.law == str(law)) & (scores.protocol == protocol) & (scores["mode"] == mode)
                ].set_index("metric")
                if harness.PRIMARY_TARGET not in selected.index:
                    continue
                primary = selected.loc[harness.PRIMARY_TARGET]
                gain_error = abs(float(primary["predicted_two_phase_gain"]) - harness.OBSERVED_WSD_GAIN)
                record = {
                    "law": str(law),
                    "protocol": protocol,
                    "mode": mode,
                    "primary_predicted_gain": float(primary["predicted_two_phase_gain"]),
                    "primary_gain_error": gain_error,
                    "primary_gain_error_passes": bool(gain_error <= harness.WSD_GAIN_ERROR_LIMIT),
                    "primary_optimum_distance": float(primary["optimum_distance_interior"]),
                    "primary_optimum_distance_passes": bool(
                        float(primary["optimum_distance_interior"]) <= harness.WSD_OPTIMUM_DISTANCE_LIMIT
                    ),
                    "primary_optimum_phase_0": float(primary["predicted_best_interior_phase_0"]),
                    "primary_optimum_phase_1": float(primary["predicted_best_interior_phase_1"]),
                    "primary_regret_at_1": float(picked.loc[harness.PRIMARY_TARGET, "regret_at_1"]),
                    "primary_regret_passes": bool(
                        float(picked.loc[harness.PRIMARY_TARGET, "regret_at_1"])
                        <= harness.RPL_PRIMARY_REGRET + harness.REGRET_SLACK
                    ),
                    "primary_interior_rmse": float(picked.loc[harness.PRIMARY_TARGET, "rmse_interior"]),
                }
                for control in harness.NEGATIVE_CONTROLS:
                    gain = float(selected.loc[control, "predicted_two_phase_gain"])
                    record[f"negative_gain::{control}"] = gain
                    record[f"negative_gain_passes::{control}"] = bool(gain <= harness.WSD_NEGATIVE_GAIN_LIMIT)
                for control in harness.POSITIVE_CONTROLS:
                    record[f"positive_gain::{control}"] = float(selected.loc[control, "predicted_two_phase_gain"])
                    record[f"positive_regret::{control}"] = float(picked.loc[control, "regret_at_1"])
                gates.append(record)
    gate_frame = pd.DataFrame(gates)
    gate_frame.to_csv(output_dir / "wsd80_gates.csv", index=False)

    harness.write_json(
        output_dir / "wsd80_summary.json",
        {
            "protocol": harness.protocol_hash({"stage": "wsd80", "grid": grid}),
            "n_rows": len(panel.y),
            "n_metrics": targets.n_targets,
            "gates": gate_frame.to_dict(orient="records"),
        },
    )

    pd.set_option("display.width", 240)
    print("=== selected transition, full fit ===")
    full = trace[trace.fold == "full"]
    print(
        full[full.target.isin(gate_metrics)][
            ["law", "protocol", "mode", "target", "rho", "interference", "ridge"]
        ].to_string(index=False)
    )
    print()
    print("=== gates ===")
    print(gate_frame.to_string(index=False))
    print()
    print("=== joint vs independent, interior OOF RMSE difference ===")
    print(comparison_frame.to_string(index=False))
