# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered round-two synthetic recovery, raw menus, and leakage audit.

Only frozen policy geometry and partitions are read. No measured response file
is opened. Oracle fits are diagnosis, never evidence for candidate promotion.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as previous_audit
import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round2_20260921 as surrogate
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
FLOOR = 0.5
INTERCEPT = -0.2
COEFFICIENTS = np.asarray([1.2, 0.6, 0.03, 0.06])
WORLDS = (
    "causal_scalar_mu4_noiseless",
    "causal_scalar_mu4_noise002",
    "phase_null_noise002",
    "static_impostor_noise002",
)
SEEDS = (20260921, 20260922, 20260923)
MENU_POINTS = 101


def raw_menu(panel: BenchPanel, points: int = MENU_POINTS) -> BenchPanel:
    """Construct the complete two-bucket simplex product, including its diagonal."""
    if panel.buckets != ("nemotron_full", "starcoder"):
        raise ValueError("The registered raw synthetic menu requires the StarCoder two-bucket geometry")
    coordinates = np.linspace(0, 1, points)
    early, late = np.meshgrid(coordinates, coordinates, indexing="ij")
    first = np.column_stack([1 - early.ravel(), early.ravel()])
    second = np.column_stack([1 - late.ravel(), late.ravel()])
    identifiers = [f"p{i:03d}_q{j:03d}" for i in range(points) for j in range(points)]
    rows = pd.DataFrame(
        {
            "row_id": identifiers,
            "policy_id": identifiers,
            "group_id": [
                f"aggregate_{value:.15g}" for value in panel.alpha * first[:, 1] + (1 - panel.alpha) * second[:, 1]
            ],
            "source": "registered_synthetic_geometry",
            "region": "full_simplex",
            "tied": early.ravel() == late.ravel(),
        }
    )
    metadata = {
        **panel.metadata,
        "realized_phase0_epochs": first * panel.c0,
        "realized_phase1_epochs": second * panel.c1,
    }
    return dataclasses.replace(
        panel, name=f"{panel.name}_raw_{points}", phase0=first, phase1=second, rows=rows, metadata=metadata
    )


def generating_design(panel: BenchPanel, world: str) -> np.ndarray:
    """Return the registered synthetic design, including static-impostor terms."""
    if world not in WORLDS:
        raise ValueError(f"Unknown registered world: {world}")
    forgetting = 4.0 if world.startswith("causal_scalar_mu4") else 0.0
    matrix = surrogate.design(panel, np.arange(len(panel.rows)), SHAPE, forgetting, "causal")
    if world == "static_impostor_noise002":
        contrast = panel.contrast[:, panel.buckets.index("starcoder")]
        matrix = np.column_stack([matrix, contrast, contrast**2])
    return matrix


def world_mean(panel: BenchPanel, world: str) -> np.ndarray:
    coefficients = np.concatenate([COEFFICIENTS, [0.04, 0.02]]) if world == "static_impostor_noise002" else COEFFICIENTS
    return FLOOR + np.exp(INTERCEPT + generating_design(panel, world) @ coefficients)


def menu_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Score actual true values at fitted picks within one jointly fitted menu."""
    truth = frame.truth.to_numpy(float)
    predicted = frame.prediction.to_numpy(float)
    truth_gain = frame.tied_truth.to_numpy(float) - truth
    predicted_gain = frame.tied_prediction.to_numpy(float) - predicted
    best = int(np.argmin(truth))
    selected = int(np.argmin(predicted))
    diagonal = np.flatnonzero(frame.tied.to_numpy(bool))
    if not len(diagonal):
        raise ValueError("The raw menu must include tied policies")
    best_tied = int(diagonal[np.argmin(truth[diagonal])])
    selected_tied = int(diagonal[np.argmin(predicted[diagonal])])
    return {
        "rows": len(frame),
        "tied_rows": len(diagonal),
        "endpoint_rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
        "tied_rmse": float(np.sqrt(np.mean((frame.tied_prediction - frame.tied_truth) ** 2))),
        "matched_gain_rmse": float(np.sqrt(np.mean((predicted_gain - truth_gain) ** 2))),
        "matched_gain_bias": float(np.mean(predicted_gain - truth_gain)),
        "true_matched_gain_rms": float(np.sqrt(np.mean(truth_gain**2))),
        "true_max_matched_gain": float(np.max(truth_gain)),
        "predicted_max_matched_gain": float(np.max(predicted_gain)),
        "predicted_max_abs_matched_gain": float(np.max(np.abs(predicted_gain))),
        "true_minimum": float(truth[best]),
        "predicted_minimum": float(predicted[selected]),
        "true_value_at_predicted_minimum": float(truth[selected]),
        "raw_policy_regret": float(truth[selected] - truth[best]),
        "selected_optimism": float(truth[selected] - predicted[selected]),
        "true_grid_optimum_gain_over_tied": float(truth[best_tied] - truth[best]),
        "predicted_grid_optimum_gain_over_tied": float(predicted[selected_tied] - predicted[selected]),
        "true_gain_of_selected_policies": float(truth[selected_tied] - truth[selected]),
        "selected_phase0_code": float(frame.iloc[selected].phase0_code),
        "selected_phase1_code": float(frame.iloc[selected].phase1_code),
        "true_best_phase0_code": float(frame.iloc[best].phase0_code),
        "true_best_phase1_code": float(frame.iloc[best].phase1_code),
        "interpretation": (
            "Fixed 101x101 feasible simplex menu; not a certified continuous optimum or false-positive rate"
        ),
    }


def prediction_frame(
    panel: BenchPanel, truth: np.ndarray, predicted: np.ndarray, tied_truth: np.ndarray, tied_predicted: np.ndarray
) -> pd.DataFrame:
    frame = panel.rows[["row_id", "policy_id", "group_id", "tied"]].copy()
    frame["phase0_code"], frame["phase1_code"] = panel.phase0[:, 1], panel.phase1[:, 1]
    frame["truth"], frame["prediction"] = truth, predicted
    frame["tied_truth"], frame["tied_prediction"] = tied_truth, tied_predicted
    return frame


def identity(prepared: Path) -> dict[str, Any]:
    names = ["rows.csv", "policies.npz", "metadata.json", "folds.csv"]
    relative = [f"starcoder_wsd80_dense/{name}" for name in names]
    manifest = json.loads((prepared / "manifest.json").read_text())
    benchmark.verify_hashes({name: manifest["outputs"][name] for name in relative}, prepared)
    sources = {
        **surrogate.source_hashes(),
        str(Path(__file__).resolve()): benchmark.sha256(Path(__file__)),
        str(Path(previous_audit.__file__).resolve()): benchmark.sha256(Path(previous_audit.__file__)),
    }
    return {
        "sources": sources,
        "geometry": {name: manifest["outputs"][name] for name in relative},
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "environment": benchmark.environment_identity(),
        "partition": {"repeat": 0, "outer_fold": 0},
        "menu_points": MENU_POINTS,
        "claim": "synthetic_development_diagnostic_not_promotion",
    }


def floor_compatibility(response: np.ndarray, train: np.ndarray, inner: np.ndarray) -> list[dict[str, Any]]:
    partitions = [("outer_train", train)] + [
        (f"inner_train_{fold}", train[inner != fold]) for fold in sorted(set(inner))
    ]
    rows = []
    for name, indices in partitions:
        values = response[indices]
        minimum, median = float(values.min()), float(np.median(values))
        rho = FLOOR / minimum
        gamma = (median - FLOOR) / (median - minimum)
        rows.append(
            {
                "partition": name,
                "required_rho": rho,
                "nearest_rho_distance": min(abs(rho - r) for r in surrogate.FLOORS),
                "required_native_gamma": gamma,
                "native_gamma_in_range": 1 <= gamma <= 6,
            }
        )
    return rows


def audit_fit(arguments: tuple[str, str, str, int, str]) -> dict[str, Any]:
    prepared_name, output_name, world, seed, model = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    folder = output / world / str(seed) / model
    config = {**identity(prepared), "world": world, "seed": seed, "model": model}
    if previous_audit.check_completed(folder, config):
        return {"world": world, "seed": seed, "model": model, "cached": True, "folder": str(folder)}
    panel = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    train, test, inner = previous_audit.frozen_partition(prepared, panel)
    tied = previous_audit.tied_queries(panel)
    menu = raw_menu(panel)
    tied_menu = previous_audit.tied_queries(menu)
    noise_sd = 0.0 if world.endswith("noiseless") else 0.002
    started = time.monotonic()
    with threadpool_limits(limits=1):
        truth = world_mean(panel, world)
        response = truth + np.random.default_rng(seed).normal(0, noise_sd, len(truth))
        values = surrogate.geometry_cache(panel, output / "geometry", surrogate.MODELS[model][0])
        fitted = surrogate.fit_model(panel, response, train, inner, model, values)
        frame = prediction_frame(
            panel,
            truth,
            fitted.predict(panel, np.arange(len(panel.rows))),
            world_mean(tied, world),
            fitted.predict(tied, np.arange(len(tied.rows))),
        )
        frame["role"] = "excluded"
        frame.loc[train, "role"], frame.loc[test, "role"] = "train", "test"
        frame["response"] = response
        raw_frame = prediction_frame(
            menu,
            world_mean(menu, world),
            fitted.predict(menu, np.arange(len(menu.rows))),
            world_mean(tied_menu, world),
            fitted.predict(tied_menu, np.arange(len(tied_menu.rows))),
        )
        known_matrix = generating_design(panel, world)
        oracle = surrogate.profile.fit_profile_head(
            known_matrix[train],
            response[train],
            0.0,
            FLOOR / float(response[train].min()),
            np.ones(known_matrix.shape[1]),
        )
        oracle_frame = prediction_frame(
            menu,
            raw_frame.truth.to_numpy(),
            oracle.predict(generating_design(menu, world)),
            raw_frame.tied_truth.to_numpy(),
            oracle.predict(generating_design(tied_menu, world)),
        )
    metrics = {
        "archival_held_out": previous_audit.recovery_metrics(frame[frame.role != "excluded"]),
        "raw_menu": menu_metrics(raw_frame),
        "known_floor_and_design_oracle": menu_metrics(oracle_frame),
        "floor_compatibility": floor_compatibility(response, train, inner),
        "elapsed_seconds": time.monotonic() - started,
    }
    benchmark.write_json(folder / "fit.json", fitted.to_json())
    benchmark.write_json(
        folder / "config.json",
        {
            **config,
            "noise_sd": noise_sd,
            "training_row_ids": panel.rows.iloc[train].row_id.tolist(),
            "inner_folds": inner.tolist(),
        },
    )
    benchmark.write_json(folder / "metrics.json", metrics)
    benchmark.write_json(
        folder / "oracle.json",
        {
            "floor": oracle.floor,
            "intercept": oracle.intercept,
            "coefficients": oracle.coefficients.tolist(),
            "claim": "Known floor and true generating design: diagnosis only, excluded from promotion",
        },
    )
    frame.to_csv(folder / "predictions.csv", index=False)
    raw_frame.to_csv(folder / "raw_menu.csv", index=False)
    oracle_frame.to_csv(folder / "oracle_raw_menu.csv", index=False)
    files = [
        "fit.json",
        "config.json",
        "metrics.json",
        "oracle.json",
        "predictions.csv",
        "raw_menu.csv",
        "oracle_raw_menu.csv",
    ]
    benchmark.write_json(
        folder / "complete.json",
        {"config": config, "outputs": {name: benchmark.sha256(folder / name) for name in files}},
    )
    return {
        "world": world,
        "seed": seed,
        "model": model,
        "cached": False,
        "folder": str(folder),
        "selected_mu": fitted.forgetting,
        "floor": fitted.head.floor,
        "gamma": fitted.gamma,
        "rho": fitted.rho,
        "elapsed_seconds": metrics["elapsed_seconds"],
        **metrics["raw_menu"],
    }


def leakage_audit(prepared: Path, output: Path) -> None:
    """Repeat the full NG2-006 fit after poisoning every nontraining response."""
    folder = output / "nontraining_invariance"
    config = {**identity(prepared), "world": WORLDS[0], "seed": SEEDS[0], "model": "NG2-006"}
    if previous_audit.check_completed(folder, config):
        return
    panel = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    train, _, inner = previous_audit.frozen_partition(prepared, panel)
    baseline_folder = output / WORLDS[0] / str(SEEDS[0]) / "NG2-006"
    baseline = json.loads((baseline_folder / "fit.json").read_text())
    baseline_predictions = pd.read_csv(baseline_folder / "predictions.csv", float_precision="round_trip")
    response = world_mean(panel, WORLDS[0])
    outside = np.setdiff1d(np.arange(len(response)), train)
    response[outside[::2]], response[outside[1::2]] = np.nan, 1e100
    with threadpool_limits(limits=1):
        values = surrogate.geometry_cache(panel, output / "geometry", "causal")
        fitted = surrogate.fit_model(panel, response, train, inner, "NG2-006", values)
        predicted = fitted.predict(panel, np.arange(len(panel.rows)))
    same_parameters = fitted.to_json() == baseline
    same_predictions = np.array_equal(predicted, baseline_predictions.prediction.to_numpy())
    if not same_parameters or not same_predictions:
        raise AssertionError("The full estimator depends on a nontraining response")
    benchmark.write_json(
        folder / "receipt.json",
        {
            "status": "pass",
            "same_fit_json": same_parameters,
            "bitwise_same_predictions": same_predictions,
            "poisoned_rows": len(outside),
            "training_rows": len(train),
            "model": "NG2-006",
            "config": config,
        },
    )
    benchmark.write_json(
        folder / "complete.json",
        {"config": config, "outputs": {"receipt.json": benchmark.sha256(folder / "receipt.json")}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=previous_audit.PREPARED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(surrogate.MODELS), default=list(surrogate.MODELS))
    parser.add_argument("--worlds", nargs="+", choices=WORLDS, default=list(WORLDS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--nontraining-invariance", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    panel = previous_audit.geometry_panel(args.prepared, "starcoder_wsd80_dense")
    with threadpool_limits(limits=1):
        for clock in sorted({surrogate.MODELS[model][0] for model in args.models}):
            surrogate.geometry_cache(panel, args.output / "geometry", clock)
    tasks = [
        (str(args.prepared), str(args.output), world, seed, model)
        for world in args.worlds
        for seed in ([SEEDS[0]] if world.endswith("noiseless") else args.seeds)
        for model in args.models
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(audit_fit, task) for task in tasks]):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    if args.nontraining_invariance:
        leakage_audit(args.prepared, args.output)
        print(json.dumps({"nontraining_invariance": "pass"}), flush=True)
    summary = []
    for complete in sorted(args.output.glob("*/*/*/complete.json")):
        folder = complete.parent
        fit = json.loads((folder / "fit.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        summary.append(
            {
                "world": folder.parent.parent.name,
                "seed": int(folder.parent.name),
                "model": folder.name,
                "forgetting": fit["forgetting"],
                "floor": fit["floor"],
                "rho": fit["rho"],
                "gamma": fit["gamma"],
                **metrics["raw_menu"],
            }
        )
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    benchmark.write_json(
        args.output / "run.json",
        {
            "identity": identity(args.prepared),
            "last_invocation": records,
            "completed_fits": len(summary),
            "measured_outcomes_read": False,
            "lm_jobs": 0,
        },
    )


if __name__ == "__main__":
    main()
