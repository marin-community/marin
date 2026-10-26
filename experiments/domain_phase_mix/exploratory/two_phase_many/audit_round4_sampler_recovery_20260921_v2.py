# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered round-four synthetic recovery with the version-two numerical solver.

All 72 fits use asymmetric labels and the unchanged grouped inner search.
Native-floor incompatibility is recorded without altering the registered truth.
Tied curves and the raw 101-by-101 menu supply evaluation queries only. This
diagnostic introduces no promotion thresholds or false-positive-rate claim.
Run with ``uv run --no-sync python`` from the repository's frozen environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as previous_audit
import audit_round3_sampler_recovery_20260921 as sampler_audit
import audit_two_phase_round2_20260921 as menu_audit
import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round4_20260921_v2 as surrogate
import numpy as np
import pandas as pd
import round4_boundary_benefit_20260921 as boundary
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_nextgen_20260921/round4"
SHAPE = sampler_audit.SHAPE
FLOOR = sampler_audit.FLOOR
INTERCEPT = sampler_audit.INTERCEPT
COEFFICIENTS = sampler_audit.COEFFICIENTS
WORLDS = ("null", "boundary", "misspecified_null")
SAMPLERS = sampler_audit.SAMPLERS
MODELS = tuple(surrogate.MODELS)
SEEDS = sampler_audit.SEEDS
NOISE_LEVELS = (0.0, 0.002)
EXPECTED_FITS = 72
REPLAY_TOLERANCE = 1e-12


def world_mean(panel: BenchPanel, world: str) -> np.ndarray:
    """Evaluate the registered world without any fitted coefficients or labels."""
    if world not in WORLDS:
        raise ValueError(f"Unknown synthetic world: {world}")
    if world == "boundary":
        matrix = boundary.design_matrix(panel, SHAPE, q=0.5, kappa=0.5)
    else:
        matrix = benchmark.mariner.design_matrix(panel.epochs, SHAPE)
    linear = INTERCEPT + matrix @ COEFFICIENTS
    if world == "misspecified_null":
        linear += 0.08 * np.sin(2 * np.pi * panel.aggregate[:, panel.buckets.index("starcoder")])
    return FLOOR + np.exp(linear)


def designated_in_family(model: str, world: str) -> bool:
    return world == "null" or (world == "boundary" and model in {"NG2-012", "NG2-013"})


def identity(prepared: Path) -> dict[str, Any]:
    """Bind the diagnostic to geometry, implementation, environment, and registration."""
    freeze_path = ROOT / "numerical_v2/source_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["environment"] != benchmark.environment_identity():
        raise ValueError("Use uv run --no-sync python in the frozen numerical environment before any fitting")
    benchmark.verify_hashes(freeze["sources"])
    benchmark.verify_hashes(freeze["driver_sources"])
    if any(freeze["sources"].get(name) != digest for name, digest in surrogate.source_hashes().items()):
        raise ValueError("The version-two runner source hashes differ from the numerical source freeze")
    registration = json.loads((ROOT / "registration.json").read_text())
    if set(MODELS) != {"NG2-011", "NG2-012", "NG2-013"}:
        raise ValueError("The registered round-four diagnostic requires exactly NG2-011/012/013")
    if {model["id"] for model in registration["models"]} != set(MODELS):
        raise ValueError("Runner and registration model identities differ")
    synthetic = registration["synthetic"]
    if (
        set(synthetic["worlds"]) != set(WORLDS)
        or synthetic["noise_sd"] != list(NOISE_LEVELS)
        or synthetic["seeds"] != list(SEEDS)
        or synthetic["noiseless_one_seed"] is not True
    ):
        raise ValueError("The registered synthetic worlds, seeds, or noise levels changed")
    names = [f"starcoder_wsd80_dense/{name}" for name in ("rows.csv", "policies.npz", "metadata.json")]
    manifest = json.loads((prepared / "manifest.json").read_text())
    hashes = {name: manifest["outputs"][name] for name in names}
    benchmark.verify_hashes(hashes, prepared)
    modules = (previous_audit, sampler_audit, menu_audit, boundary)
    paths = (Path(__file__), *(Path(module.__file__) for module in modules))
    return {
        "sources": {**surrogate.source_hashes(), **{str(p.resolve()): benchmark.sha256(p) for p in paths}},
        "geometry": hashes,
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "environment": benchmark.environment_identity(),
        "registration": benchmark.sha256(ROOT / "registration.json"),
        "source_freeze": benchmark.receipt(freeze_path),
        "worlds": list(WORLDS),
        "models": list(MODELS),
        "samplers": list(SAMPLERS),
        "noise_levels": list(NOISE_LEVELS),
        "seeds": list(SEEDS),
        "expected_fits": EXPECTED_FITS,
        "claim": "synthetic_development_diagnostic_not_promotion",
    }


def preparation(prepared: Path) -> dict[str, Any]:
    """Preflight every actual floor subset while preserving incompatible worlds."""
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panels = {sampler: sampler_audit.sampler_panel(base, sampler) for sampler in SAMPLERS}
    drift = float(np.max(np.abs(panels["balanced"][0].aggregate - panels["positive"][0].aggregate)))
    if drift > 3e-16:
        raise ValueError("Synthetic samplers no longer share one aggregate population")
    records = []
    for world in WORLDS:
        for sampler, (panel, inner) in panels.items():
            if panel.tied.any() or set(inner) != {0, 1, 2}:
                raise ValueError("Synthetic training must be asymmetric and use all three grouped inner folds")
            for fold in sorted(set(inner)):
                training = set(panel.rows.loc[inner != fold, "group_id"])
                held = set(panel.rows.loc[inner == fold, "group_id"])
                if training & held:
                    raise ValueError("An aggregate group crosses a synthetic inner split")
            for noise_sd in NOISE_LEVELS:
                for seed in SEEDS[:1] if noise_sd == 0 else SEEDS:
                    response = world_mean(panel, world) + np.random.default_rng(seed).normal(
                        0, noise_sd, len(panel.rows)
                    )
                    compatibility = sampler_audit.floor_compatibility(response, inner)
                    compatible = all(
                        row["native_gamma_in_range"] and row["all_observations_above_true_floor"]
                        for row in compatibility
                    )
                    records.append(
                        {
                            "world": world,
                            "sampler": sampler,
                            "noise_sd": noise_sd,
                            "seed": seed,
                            "native_floor_compatible": compatible,
                            "folds": compatibility,
                        }
                    )
    return {
        "maximum_sampler_aggregate_drift": drift,
        "sampler_rows": 200,
        "aggregate_groups": 20,
        "all_training_rows_asymmetric": True,
        "geometry_sha256": {
            sampler: hashlib.sha256(np.hstack([panel.phase0, panel.phase1]).tobytes()).hexdigest()
            for sampler, (panel, _) in panels.items()
        },
        "floor_compatibility": records,
        "incompatible_cells": [record for record in records if not record["native_floor_compatible"]],
        "incompatibility_action": "Preserve truth and run as out-of-family estimator diagnostic; no truth repair",
    }


def replay_predictions(panel: BenchPanel, fitted: dict[str, Any]) -> np.ndarray:
    """Replay serialized coefficients through the native head, without fitting."""
    if fitted["phase_coefficients"] or (fitted["model"] == "NG2-011" and fitted["kappa"] != 0):
        raise ValueError("The registered law has no independent phase coefficients or aggregate-arm boundary term")
    head = benchmark.mariner.Head(
        intercept=fitted["intercept"],
        coefficients=np.asarray(fitted["aggregate_coefficients"]),
        floor=fitted["floor"],
    )
    if fitted["model"] == "NG2-011":
        matrix = benchmark.mariner.design_matrix(panel.epochs, fitted["shape"])
    else:
        matrix = boundary.design_matrix(panel, fitted["shape"], fitted["q"], fitted["kappa"])
    return head.predict(matrix)


def replay_frame(panel: BenchPanel, frame: pd.DataFrame, fitted: dict[str, Any], world: str) -> float:
    """Check saved coordinates, registered truths, and independent head replay."""
    tied = previous_audit.tied_queries(panel)
    reconstructed = menu_audit.prediction_frame(
        panel,
        world_mean(panel, world),
        replay_predictions(panel, fitted),
        world_mean(tied, world),
        replay_predictions(tied, fitted),
    )
    errors = []
    for column in reconstructed:
        if column in {"prediction", "tied_prediction"}:
            error = float(np.max(np.abs(frame[column].to_numpy() - reconstructed[column].to_numpy())))
            if not np.isfinite(error) or error > REPLAY_TOLERANCE:
                raise ValueError(f"Serialized head replay differs in {column}: {error}")
            errors.append(error)
        elif not np.array_equal(frame[column].to_numpy(), reconstructed[column].to_numpy()):
            raise ValueError(f"Saved synthetic {column} differs from registered geometry or truth")
    return max(errors)


def audit_fit(arguments: tuple[str, str, str, str, int, float, str]) -> dict[str, Any]:
    prepared_name, output_name, world, sampler, seed, noise_sd, model = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    noise_name = "noiseless" if noise_sd == 0 else "noise002"
    folder = output / world / noise_name / str(seed) / model / sampler
    config = {
        **identity(prepared),
        "world": world,
        "sampler": sampler,
        "seed": seed,
        "noise_sd": noise_sd,
        "model": model,
    }
    if previous_audit.check_completed(folder, config):
        return {"cached": True, "folder": str(folder)}
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panel, inner = sampler_audit.sampler_panel(base, sampler)
    train = np.arange(len(panel.rows))
    if panel.tied.any():
        raise ValueError("Tied synthetic labels cannot enter fitting")
    truth = world_mean(panel, world)
    response = truth + np.random.default_rng(seed).normal(0, noise_sd, len(truth))
    compatibility = sampler_audit.floor_compatibility(response, inner)
    compatible = all(row["native_gamma_in_range"] and row["all_observations_above_true_floor"] for row in compatibility)
    tied = previous_audit.tied_queries(panel)
    menu = menu_audit.raw_menu(base)
    tied_menu = previous_audit.tied_queries(menu)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = surrogate.fit_model(panel, response, train, inner, model)
        train_frame = menu_audit.prediction_frame(
            panel, truth, fitted.predict(panel, train), world_mean(tied, world), fitted.predict(tied, train)
        )
        train_frame["response"], train_frame["inner_fold"] = response, inner
        raw_frame = menu_audit.prediction_frame(
            menu,
            world_mean(menu, world),
            fitted.predict(menu, np.arange(len(menu.rows))),
            world_mean(tied_menu, world),
            fitted.predict(tied_menu, np.arange(len(tied_menu.rows))),
        )
    saved_fit = fitted.to_json() | {
        "training_rows": panel.rows.row_id.tolist(),
        "inner_folds": inner.tolist(),
    }
    replay_error = max(
        replay_frame(panel, train_frame, saved_fit, world), replay_frame(menu, raw_frame, saved_fit, world)
    )
    metrics = {
        "raw_menu": menu_audit.menu_metrics(raw_frame),
        "uniform_101_tied_curve_rmse": float(
            np.sqrt(np.mean((raw_frame.loc[raw_frame.tied, "prediction"] - raw_frame.loc[raw_frame.tied, "truth"]) ** 2))
        ),
        "training_endpoint_rmse": float(np.sqrt(np.mean((train_frame.prediction - truth) ** 2))),
        "floor_compatibility": compatibility,
        "native_floor_compatible": compatible,
        "designated_in_family": designated_in_family(model, world),
        "designated_law_and_floor_compatible": designated_in_family(model, world) and compatible,
        "floor_compatibility_scope": (
            "True floor attainable separately in each actual subset; this does not certify joint gamma selection "
            "or recovery by the complete estimator"
        ),
        "maximum_serialized_prediction_replay_error": replay_error,
        "elapsed_seconds": time.monotonic() - started,
    }
    benchmark.write_json(folder / "fit.json", saved_fit)
    benchmark.write_json(folder / "config.json", config)
    benchmark.write_json(folder / "metrics.json", metrics)
    train_frame.to_csv(folder / "training_predictions.csv", index=False)
    raw_frame.to_csv(folder / "raw_menu.csv", index=False)
    files = ("fit.json", "config.json", "metrics.json", "training_predictions.csv", "raw_menu.csv")
    benchmark.write_json(
        folder / "complete.json",
        {"config": config, "outputs": {name: benchmark.sha256(folder / name) for name in files}},
    )
    return {"cached": False, "folder": str(folder), "model": model, "world": world, "sampler": sampler, **metrics}


def summarize(output: Path, prepared: Path, current_identity: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every saved fit and compare identical menus across the two samplers."""
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    menu = menu_audit.raw_menu(base)
    summaries = []
    frames: dict[tuple[str, float, int, str, str], pd.DataFrame] = {}
    for complete in sorted(output.glob("*/*/*/*/*/complete.json")):
        folder = complete.parent
        config = json.loads((folder / "config.json").read_text())
        if any(config.get(key) != value for key, value in current_identity.items()):
            raise ValueError(f"Cannot summarize fits from another source or data identity: {folder}")
        previous_audit.check_completed(folder, config)
        key = (config["world"], config["noise_sd"], int(config["seed"]), config["model"], config["sampler"])
        if key in frames:
            raise ValueError(f"Duplicate synthetic fitted cell: {key}")
        metrics = json.loads((folder / "metrics.json").read_text())
        fitted = json.loads((folder / "fit.json").read_text())
        if fitted["model"] != config["model"]:
            raise ValueError("Serialized model differs from its completed configuration")
        frame = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
        training = pd.read_csv(folder / "training_predictions.csv", float_precision="round_trip")
        panel, inner = sampler_audit.sampler_panel(base, config["sampler"])
        if fitted["training_rows"] != panel.rows.row_id.tolist() or fitted["inner_folds"] != inner.tolist():
            raise ValueError("Saved fit identities differ from the actual asymmetric training subset")
        response = world_mean(panel, config["world"]) + np.random.default_rng(config["seed"]).normal(
            0, config["noise_sd"], len(panel.rows)
        )
        if not np.array_equal(training.response.to_numpy(), response) or not np.array_equal(training.inner_fold, inner):
            raise ValueError("Saved fitting responses or actual grouped folds differ from the registered sampler")
        compatibility = sampler_audit.floor_compatibility(response, inner)
        compatible = all(
            row["native_gamma_in_range"] and row["all_observations_above_true_floor"] for row in compatibility
        )
        if (
            compatibility != metrics["floor_compatibility"]
            or compatible != metrics["native_floor_compatible"]
            or designated_in_family(config["model"], config["world"]) != metrics["designated_in_family"]
        ):
            raise ValueError("Saved floor or designated-family diagnostics differ from the registered truth")
        replay_error = max(
            replay_frame(panel, training, fitted, config["world"]), replay_frame(menu, frame, fitted, config["world"])
        )
        replayed_metrics = menu_audit.menu_metrics(frame)
        if replayed_metrics != metrics["raw_menu"]:
            raise ValueError("Saved raw-menu metrics differ from their prediction replay")
        training_rmse = float(np.sqrt(np.mean((training.prediction - training.truth) ** 2)))
        tied_rmse = float(np.sqrt(np.mean((frame.loc[frame.tied, "prediction"] - frame.loc[frame.tied, "truth"]) ** 2)))
        if training_rmse != metrics["training_endpoint_rmse"] or tied_rmse != metrics["uniform_101_tied_curve_rmse"]:
            raise ValueError("Saved training or tied-curve error differs from its prediction replay")
        frames[key] = frame
        summaries.append(
            {
                "world": config["world"],
                "noise_sd": config["noise_sd"],
                "seed": config["seed"],
                "model": config["model"],
                "sampler": config["sampler"],
                "designated_in_family": metrics["designated_in_family"],
                "native_floor_compatible": metrics["native_floor_compatible"],
                "designated_law_and_floor_compatible": metrics["designated_law_and_floor_compatible"],
                "uniform_101_tied_curve_rmse": metrics["uniform_101_tied_curve_rmse"],
                "floor": fitted["floor"],
                "gamma": fitted["gamma"],
                "ridge": fitted["ridge"],
                "maximum_serialized_prediction_replay_error": replay_error,
                "elapsed_seconds": metrics["elapsed_seconds"],
                **replayed_metrics,
            }
        )
    expected = {
        (world, noise_sd, seed, model, sampler)
        for world in WORLDS
        for noise_sd in NOISE_LEVELS
        for seed in (SEEDS[:1] if noise_sd == 0 else SEEDS)
        for model in MODELS
        for sampler in SAMPLERS
    }
    if set(frames) != expected or len(frames) != EXPECTED_FITS:
        raise ValueError(f"Synthetic summary requires every one of the {EXPECTED_FITS} registered fits")
    comparisons = []
    for key, balanced in sorted(frames.items()):
        if key[-1] != "balanced":
            continue
        positive = frames[(*key[:-1], "positive")]
        for column in ("row_id", "phase0_code", "phase1_code", "truth", "tied_truth"):
            if not np.array_equal(balanced[column].to_numpy(), positive[column].to_numpy()):
                raise ValueError("Sampler comparison has nonidentical raw geometry or truth")
        tied = balanced.tied.to_numpy(bool)
        drift = balanced.prediction.to_numpy()[tied] - positive.prediction.to_numpy()[tied]
        balanced_gain = balanced.tied_prediction.to_numpy() - balanced.prediction.to_numpy()
        positive_gain = positive.tied_prediction.to_numpy() - positive.prediction.to_numpy()
        comparisons.append(
            {
                "world": key[0],
                "noise_sd": key[1],
                "seed": key[2],
                "model": key[3],
                "designated_in_family": designated_in_family(key[3], key[0]),
                "tied_curve_sampler_drift_rmse": float(np.sqrt(np.mean(drift**2))),
                "tied_curve_sampler_drift_max_abs": float(np.max(np.abs(drift))),
                "matched_gain_sampler_drift_rmse": float(np.sqrt(np.mean((balanced_gain - positive_gain) ** 2))),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=previous_audit.PREPARED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    current_identity = identity(args.prepared)
    preflight = preparation(args.prepared)
    preparation_path = args.output / "preparation.json"
    if preparation_path.exists():
        saved = json.loads(preparation_path.read_text())
        if saved != {"identity": current_identity, **preflight}:
            raise ValueError("Existing synthetic preparation changed; use a new output directory")
    benchmark.write_json(preparation_path, {"identity": current_identity, **preflight})
    if args.prepare_only:
        print(json.dumps({"prepared": True, "incompatible_cells": len(preflight["incompatible_cells"])}))
        return
    records = []
    if not args.replay_only:
        tasks = [
            (str(args.prepared), str(args.output), world, sampler, seed, noise_sd, model)
            for world in WORLDS
            for sampler in SAMPLERS
            for noise_sd in NOISE_LEVELS
            for seed in (SEEDS[:1] if noise_sd == 0 else SEEDS)
            for model in MODELS
        ]
        if len(tasks) != EXPECTED_FITS:
            raise ValueError("The registered synthetic task count changed")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(audit_fit, task) for task in tasks]):
                record = future.result()
                records.append(record)
                print(json.dumps(record), flush=True)
    summary, comparison = summarize(args.output, args.prepared, current_identity)
    summary.to_csv(args.output / "summary.csv", index=False)
    comparison.to_csv(args.output / "sampler_comparison.csv", index=False)
    benchmark.write_json(
        args.output / ("replay.json" if args.replay_only else "run.json"),
        {
            "identity": current_identity,
            "completed_fits": len(summary),
            "last_invocation": records,
            "incompatible_cells": len(preflight["incompatible_cells"]),
            "replayed_predictions_and_raw_metrics": True,
            "maximum_serialized_prediction_replay_error": float(
                summary.maximum_serialized_prediction_replay_error.max()
            ),
            "measured_outcomes_read": False,
            "lm_jobs": 0,
        },
    )


if __name__ == "__main__":
    main()
