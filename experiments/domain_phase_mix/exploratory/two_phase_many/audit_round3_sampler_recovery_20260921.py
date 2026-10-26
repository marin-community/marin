# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered round-three sampler-shift recovery using synthetic outcomes only.

The two asymmetric training samplers have identical aggregate populations.
Every fit searches the complete registered estimator. Tied queries and the
raw 101-by-101 phase menu are evaluated after fitting and never supply labels
to the estimator. Recovery is a diagnostic, not a promotion decision.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as previous_audit
import audit_two_phase_round2_20260921 as menu_audit
import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round3_20260921 as surrogate
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

ROOT = Path(__file__).resolve().parent / "reference_outputs/two_phase_nextgen_20260921/round3"
SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
FLOOR = 0.5
INTERCEPT = 0.5
COEFFICIENTS = np.array([6.0, 1.0, 0.03, 0.06])
WORLDS = ("null", "linear", "increment")
SAMPLERS = ("balanced", "positive")
MODELS = ("NG2-007", "NG2-009", "NG2-010")
SEEDS = (20260921, 20260922, 20260923)
IN_FAMILY = {"NG2-007": "null", "NG2-009": "linear", "NG2-010": "increment"}


def sampler_panel(base: BenchPanel, sampler: str) -> tuple[BenchPanel, np.ndarray]:
    """Construct 200 feasible asymmetric policies with 20 shared aggregates."""
    if base.buckets != ("nemotron_full", "starcoder") or sampler not in SAMPLERS:
        raise ValueError("The registered samplers require StarCoder two-bucket geometry")
    aggregate = np.repeat(np.linspace(0.05, 0.95, 20), 10)
    positive = np.minimum(aggregate / (1 - base.alpha), (1 - aggregate) / base.alpha)
    negative = np.minimum((1 - aggregate) / (1 - base.alpha), aggregate / base.alpha)
    if sampler == "balanced":
        fractions = np.r_[-np.array([0.1, 0.3, 0.5, 0.7, 0.9]), [0.1, 0.3, 0.5, 0.7, 0.9]]
        contrast = np.tile(fractions, 20) * np.minimum(positive, negative)
    else:
        contrast = np.tile(np.linspace(0.05, 0.95, 10), 20) * positive
    early = aggregate - (1 - base.alpha) * contrast
    late = aggregate + base.alpha * contrast
    phase0, phase1 = np.column_stack([1 - early, early]), np.column_stack([1 - late, late])
    if np.any(phase0 < 0) or np.any(phase1 < 0) or np.any(phase0 > 1) or np.any(phase1 > 1):
        raise ValueError("Synthetic policies leave the simplex")
    if np.any(early == late):
        raise ValueError("Synthetic training must be asymmetric only")
    groups = np.repeat(np.arange(20), 10)
    identifiers = [f"{sampler}_a{group:02d}_d{index % 10:02d}" for index, group in enumerate(groups)]
    rows = pd.DataFrame(
        {
            "row_id": identifiers,
            "policy_id": identifiers,
            "group_id": [f"aggregate_{group:02d}" for group in groups],
            "source": "registered_synthetic_sampler",
            "region": sampler,
            "tied": False,
        }
    )
    panel = dataclasses.replace(
        base,
        name=f"starcoder_round3_{sampler}",
        phase0=phase0,
        phase1=phase1,
        rows=rows,
        targets={},
        metadata={
            **base.metadata,
            "realized_phase0_epochs": base.c0 * phase0,
            "realized_phase1_epochs": base.c1 * phase1,
        },
    )
    if not np.allclose(panel.aggregate[:, 1], aggregate, rtol=0, atol=2e-16):
        raise ValueError("Constructed aggregate population drifted")
    return panel, groups % 3


def world_mean(panel: BenchPanel, world: str) -> np.ndarray:
    """Evaluate the fixed registered law without any fitted quantities."""
    if world not in WORLDS:
        raise ValueError(f"Unknown synthetic world: {world}")
    linear = INTERCEPT + benchmark.mariner.design_matrix(panel.epochs, SHAPE) @ COEFFICIENTS
    if world == "linear":
        linear += 0.6 * panel.contrast[:, 1]
    elif world == "increment":
        benefit = benchmark.mariner.benefit
        departure = benefit(panel.alpha * panel.epochs, SHAPE["rate"], SHAPE["power"]) - benefit(
            panel.phase0_epochs, SHAPE["rate"], SHAPE["power"]
        )
        linear += departure @ np.array([0.8, -0.5])
    return FLOOR + np.exp(linear)


def floor_compatibility(response: np.ndarray, inner: np.ndarray) -> list[dict[str, Any]]:
    """Check that the true floor is attainable in every actual fitting subset."""
    partitions = [("outer_train", np.arange(len(response)))] + [
        (f"inner_train_{fold}", np.flatnonzero(inner != fold)) for fold in sorted(set(inner))
    ]
    records = []
    for name, indices in partitions:
        values = response[indices]
        median, minimum = float(np.median(values)), float(np.min(values))
        gamma = (median - FLOOR) / (median - minimum)
        records.append(
            {
                "partition": name,
                "rows": len(indices),
                "minimum": minimum,
                "median": median,
                "required_native_gamma": gamma,
                "native_gamma_in_range": 1 <= gamma <= 6,
                "all_observations_above_true_floor": bool(np.all(values > FLOOR)),
            }
        )
    return records


def identity(prepared: Path) -> dict[str, Any]:
    names = [f"starcoder_wsd80_dense/{name}" for name in ("rows.csv", "policies.npz", "metadata.json")]
    manifest = json.loads((prepared / "manifest.json").read_text())
    hashes = {name: manifest["outputs"][name] for name in names}
    benchmark.verify_hashes(hashes, prepared)
    paths = (Path(__file__), Path(previous_audit.__file__), Path(menu_audit.__file__))
    return {
        "sources": {**surrogate.source_hashes(), **{str(p.resolve()): benchmark.sha256(p) for p in paths}},
        "geometry": hashes,
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "environment": benchmark.environment_identity(),
        "registration": benchmark.sha256(ROOT / "registration.json"),
        "claim": "synthetic_development_diagnostic_not_promotion",
    }


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
    panel, inner = sampler_panel(base, sampler)
    train = np.arange(len(panel.rows))
    truth = world_mean(panel, world)
    response = truth + np.random.default_rng(seed).normal(0, noise_sd, len(truth))
    compatibility = floor_compatibility(response, inner)
    if not all(row["native_gamma_in_range"] and row["all_observations_above_true_floor"] for row in compatibility):
        raise ValueError(f"Registered synthetic true floor is outside a training-subset native family: {config}")
    menu = menu_audit.raw_menu(base)
    tied_menu = previous_audit.tied_queries(menu)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = surrogate.fit_model(panel, response, train, inner, model)
        train_frame = menu_audit.prediction_frame(
            panel,
            truth,
            fitted.predict(panel, train),
            world_mean(previous_audit.tied_queries(panel), world),
            fitted.predict(previous_audit.tied_queries(panel), train),
        )
        train_frame["response"], train_frame["inner_fold"] = response, inner
        raw_frame = menu_audit.prediction_frame(
            menu,
            world_mean(menu, world),
            fitted.predict(menu, np.arange(len(menu.rows))),
            world_mean(tied_menu, world),
            fitted.predict(tied_menu, np.arange(len(tied_menu.rows))),
        )
    metrics = {
        "raw_menu": menu_audit.menu_metrics(raw_frame),
        "uniform_101_tied_curve_rmse": float(
            np.sqrt(np.mean((raw_frame.loc[raw_frame.tied, "prediction"] - raw_frame.loc[raw_frame.tied, "truth"]) ** 2))
        ),
        "training_endpoint_rmse": float(np.sqrt(np.mean((train_frame.prediction - truth) ** 2))),
        "floor_compatibility": compatibility,
        "designated_in_family": IN_FAMILY[model] == world,
        "elapsed_seconds": time.monotonic() - started,
    }
    benchmark.write_json(folder / "fit.json", fitted.to_json())
    benchmark.write_json(folder / "config.json", config)
    benchmark.write_json(folder / "metrics.json", metrics)
    train_frame.to_csv(folder / "training_predictions.csv", index=False)
    raw_frame.to_csv(folder / "raw_menu.csv", index=False)
    files = ("fit.json", "config.json", "metrics.json", "training_predictions.csv", "raw_menu.csv")
    benchmark.write_json(
        folder / "complete.json",
        {"config": config, "outputs": {name: benchmark.sha256(folder / name) for name in files}},
    )
    return {
        "cached": False,
        "folder": str(folder),
        "model": model,
        "world": world,
        "sampler": sampler,
        "noise_sd": noise_sd,
        "seed": seed,
        **metrics,
    }


def summarize(output: Path, current_identity: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare identical raw menus and uniform tied curves across samplers."""
    summaries = []
    frames: dict[tuple[str, str, int, str, str], pd.DataFrame] = {}
    for complete in sorted(output.glob("*/*/*/*/*/complete.json")):
        folder = complete.parent
        config = json.loads((folder / "config.json").read_text())
        if any(config.get(key) != value for key, value in current_identity.items()):
            raise ValueError(f"Cannot summarize fits from another source or data identity: {folder}")
        previous_audit.check_completed(folder, config)
        metrics = json.loads((folder / "metrics.json").read_text())
        fit = json.loads((folder / "fit.json").read_text())
        noise_name = "noiseless" if config["noise_sd"] == 0 else "noise002"
        key = (config["world"], noise_name, int(config["seed"]), config["model"], config["sampler"])
        frames[key] = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
        summaries.append(
            {
                "world": config["world"],
                "noise_sd": config["noise_sd"],
                "seed": config["seed"],
                "model": config["model"],
                "sampler": config["sampler"],
                "designated_in_family": metrics["designated_in_family"],
                "uniform_101_tied_curve_rmse": metrics["uniform_101_tied_curve_rmse"],
                "floor": fit["floor"],
                "gamma": fit["gamma"],
                "ridge": fit["ridge"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                **metrics["raw_menu"],
            }
        )
    comparisons = []
    for key in sorted(frames):
        if key[-1] != "balanced":
            continue
        other_key = (*key[:-1], "positive")
        if other_key not in frames:
            continue
        balanced, positive = frames[key], frames[other_key]
        for column in ("row_id", "phase0_code", "phase1_code", "truth", "tied_truth"):
            if not np.array_equal(balanced[column].to_numpy(), positive[column].to_numpy()):
                raise ValueError("Sampler comparison has nonidentical raw menu geometry or truth")
        tied = balanced.tied.to_numpy(bool)
        drift = balanced.prediction.to_numpy()[tied] - positive.prediction.to_numpy()[tied]
        balanced_gain = balanced.tied_prediction.to_numpy() - balanced.prediction.to_numpy()
        positive_gain = positive.tied_prediction.to_numpy() - positive.prediction.to_numpy()
        comparisons.append(
            {
                "world": key[0],
                "noise": key[1],
                "seed": key[2],
                "model": key[3],
                "designated_in_family": IN_FAMILY[key[3]] == key[0],
                "tied_curve_sampler_drift_rmse": float(np.sqrt(np.mean(drift**2))),
                "tied_curve_sampler_drift_max_abs": float(np.max(np.abs(drift))),
                "matched_gain_sampler_drift_rmse": float(np.sqrt(np.mean((balanced_gain - positive_gain) ** 2))),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def preparation(prepared: Path) -> dict[str, Any]:
    """Verify outcome-free sampler identity and every generated floor interval."""
    base = previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panels = {sampler: sampler_panel(base, sampler) for sampler in SAMPLERS}
    drift = float(np.max(np.abs(panels["balanced"][0].aggregate - panels["positive"][0].aggregate)))
    if drift > 3e-16:
        raise ValueError("Synthetic samplers no longer share one aggregate population")
    records = []
    for world in WORLDS:
        for sampler, (panel, inner) in panels.items():
            for noise_sd in (0.0, 0.002):
                for seed in SEEDS[:1] if noise_sd == 0 else SEEDS:
                    response = world_mean(panel, world) + np.random.default_rng(seed).normal(
                        0, noise_sd, len(panel.rows)
                    )
                    compatibility = floor_compatibility(response, inner)
                    if not all(
                        r["native_gamma_in_range"] and r["all_observations_above_true_floor"] for r in compatibility
                    ):
                        raise ValueError("A registered synthetic response excludes its true floor")
                    records.append(
                        {"world": world, "sampler": sampler, "noise_sd": noise_sd, "seed": seed, "folds": compatibility}
                    )
    return {
        "maximum_sampler_aggregate_drift": drift,
        "sampler_rows": 200,
        "aggregate_groups": 20,
        "all_training_rows_asymmetric": all(not panel.tied.any() for panel, _ in panels.values()),
        "geometry_sha256": {
            sampler: hashlib.sha256(np.hstack([panel.phase0, panel.phase1]).tobytes()).hexdigest()
            for sampler, (panel, _) in panels.items()
        },
        "floor_compatibility": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=previous_audit.PREPARED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    current_identity = identity(args.prepared)
    prepared = preparation(args.prepared)
    benchmark.write_json(args.output / "preparation.json", {"identity": current_identity, **prepared})
    if args.prepare_only:
        return
    tasks = [
        (str(args.prepared), str(args.output), world, sampler, seed, noise_sd, model)
        for world in WORLDS
        for sampler in SAMPLERS
        for noise_sd in (0.0, 0.002)
        for seed in (SEEDS[:1] if noise_sd == 0 else SEEDS)
        for model in MODELS
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(audit_fit, task) for task in tasks]):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summary, comparison = summarize(args.output, current_identity)
    summary.to_csv(args.output / "summary.csv", index=False)
    comparison.to_csv(args.output / "sampler_comparison.csv", index=False)
    benchmark.write_json(
        args.output / "run.json",
        {
            "identity": current_identity,
            "completed_fits": len(summary),
            "last_invocation": records,
            "measured_outcomes_read": False,
            "lm_jobs": 0,
        },
    )


if __name__ == "__main__":
    main()
