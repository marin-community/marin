# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Adaptive synthetic oracle attribution; no deployable model or promotion claim.

Run with ``uv run --no-sync python`` in the frozen benchmark runtime. The two
oracle arms fix the true boundary law; the second also supplies the true floor.
All fitting still uses asymmetric synthetic endpoints and grouped inner folds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import audit_round4_sampler_recovery_20260921_v2 as synthetic
import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round4_20260921_v2 as runner
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

ROOT = synthetic.ROOT / "oracle_attribution"
ARMS = ("known_shape_state", "known_shape_state_floor")
MODELS = ("NG2-012", "NG2-013")
EXPECTED_FITS = 32


def true_floor_gamma(response: np.ndarray) -> float:
    """Map the known synthetic floor into the native subset-local floor rule."""
    median, minimum = float(np.median(response)), float(response.min())
    gamma = (median - synthetic.FLOOR) / (median - minimum)
    if not 1 <= gamma <= 6 or minimum <= synthetic.FLOOR:
        raise ValueError("Known synthetic floor is outside the native fitting subset's floor range")
    return gamma


def oracle_fit(
    panel: benchmark.data.BenchPanel, response: np.ndarray, inner: np.ndarray, model: str, arm: str
) -> tuple[runner.Fit, int]:
    """Fit the fixed true law with native ridge/floor selection or known floor."""
    if panel.tied.any() or arm not in ARMS or model not in MODELS:
        raise ValueError("Invalid oracle arm, model or tied training rows")
    shape, q, kappa = synthetic.SHAPE, 0.5, 0.5
    matrix = runner.matrices(panel, np.arange(len(response)), shape, q, kappa)
    folds = tuple((np.flatnonzero(inner != fold), np.flatnonzero(inner == fold)) for fold in sorted(set(inner)))
    conditional_heads = 0

    def head(x: np.ndarray, y: np.ndarray, ridge: float, gamma: float):
        nonlocal conditional_heads
        if arm == "known_shape_state_floor":
            gamma = true_floor_gamma(y)
        fitted = runner.conditional_head(x, y, ridge, gamma, model)
        conditional_heads += 1
        if arm == "known_shape_state_floor" and abs(fitted.floor - synthetic.FLOOR) > 1e-12:
            raise ValueError("Native mapping did not realize the fixed oracle floor")
        return fitted

    def cross_validation(ridge: float, gamma: float) -> float:
        squared_error, count = 0.0, 0
        for train, valid in folds:
            fitted = head(matrix[train], response[train], ridge, gamma)
            predicted = fitted.predict(matrix[valid], np.empty((len(valid), 0)))
            squared_error += float(np.sum((predicted - response[valid]) ** 2))
            count += len(valid)
        return math.sqrt(squared_error / count)

    score, ridge_index = min(
        (cross_validation(ridge, runner.mariner.KAPPA_PROVISIONAL), index)
        for index, ridge in enumerate(runner.mariner.RIDGE_GRID)
    )
    ridge = runner.mariner.RIDGE_GRID[ridge_index]
    flat = False
    if arm == "known_shape_state_floor":
        gamma = true_floor_gamma(response)
    else:

        def objective(log_gamma: float) -> float:
            return cross_validation(ridge, math.exp(log_gamma))

        low, high = runner.mariner.KAPPA_BOUNDS
        search = minimize_scalar(
            objective,
            bounds=(math.log(low), math.log(high)),
            method="bounded",
            options={
                "maxiter": runner.mariner.KAPPA_SEARCH_EVALUATIONS,
                "xatol": runner.mariner.KAPPA_SEARCH_XATOL,
            },
        )
        gamma, score = math.exp(search.x), float(search.fun)
        flat = math.log(gamma) >= (1 - runner.mariner.FLAT_PROFILE_FRACTION) * math.log(high)
        if flat:
            gamma = runner.mariner.FLAT_PROFILE_KAPPA
            score = objective(math.log(gamma))
    fitted = head(matrix, response, ridge, gamma)
    return runner.Fit(model, dict(shape), q, kappa, ridge, gamma, flat, score, fitted), conditional_heads


def identity(prepared: Path) -> dict[str, Any]:
    registration = ROOT / "registration.json"
    if not registration.exists():
        raise ValueError("The oracle attribution must be registered before fitting")
    registered = json.loads(registration.read_text())
    truth = registered["fixed_truth"]
    if (
        registered["world"] != "boundary"
        or registered["models"] != list(MODELS)
        or registered["noise_sd"] != list(synthetic.NOISE_LEVELS)
        or registered["seeds"] != list(synthetic.SEEDS)
        or registered["noiseless_one_seed"] is not True
        or set(registered["arms"]) != set(ARMS)
        or registered["fits"] != EXPECTED_FITS
        or truth != {"shape": synthetic.SHAPE, "q": 0.5, "kappa": 0.5, "floor": synthetic.FLOOR}
    ):
        raise ValueError("Oracle implementation differs from the registered truth or 32-context design")
    frozen = json.loads((synthetic.ROOT / "source_freeze.json").read_text())
    if frozen["environment"] != benchmark.environment_identity():
        raise ValueError("The oracle audit requires the frozen benchmark runtime")
    paths = (Path(__file__), Path(synthetic.__file__), registration)
    return {
        "sources": {**runner.source_hashes(), **{str(path.resolve()): benchmark.sha256(path) for path in paths}},
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "environment": benchmark.environment_identity(),
        "registration": benchmark.receipt(registration),
        "full_search_identity": synthetic.identity(prepared),
        "claim": "Adaptive oracle attribution using known synthetic truth; not deployable or promotion evidence",
    }


def audit_context(
    prepared: Path, context: dict[str, Any], current: dict[str, Any], replay_only: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    arm, model, sampler = context["arm"], context["model"], context["sampler"]
    seed, noise_sd = context["seed"], context["noise_sd"]
    noise_name = "noiseless" if noise_sd == 0 else "noise002"
    folder = ROOT / arm / noise_name / str(seed) / model / sampler
    config = {**current, **context}
    completed = synthetic.previous_audit.check_completed(folder, config)
    base = synthetic.previous_audit.geometry_panel(prepared, "starcoder_wsd80_dense")
    panel, inner = synthetic.sampler_audit.sampler_panel(base, sampler)
    truth = synthetic.world_mean(panel, "boundary")
    response = truth + np.random.default_rng(seed).normal(0, noise_sd, len(truth))
    compatibility = synthetic.sampler_audit.floor_compatibility(response, inner)
    if not all(row["native_gamma_in_range"] and row["all_observations_above_true_floor"] for row in compatibility):
        raise ValueError("An actual oracle fitting subset excludes the known floor")
    reference_folder = synthetic.ROOT / "synthetic_v2/boundary" / noise_name / str(seed) / model / sampler
    reference_config = json.loads((reference_folder / "config.json").read_text())
    synthetic.previous_audit.check_completed(reference_folder, reference_config)
    if any(reference_config.get(key) != value for key, value in current["full_search_identity"].items()):
        raise ValueError("Full-search reference has a different source, environment or registered data identity")
    if any(reference_config[key] != context[key] for key in ("model", "sampler", "seed", "noise_sd")):
        raise ValueError("Full-search reference context differs from this oracle context")
    reference_training = pd.read_csv(reference_folder / "training_predictions.csv", float_precision="round_trip")
    if not np.array_equal(reference_training.response.to_numpy(), response):
        raise ValueError("Oracle and full-search fits do not use identical synthetic responses")
    if reference_training.row_id.tolist() != panel.rows.row_id.tolist() or not np.array_equal(
        reference_training.inner_fold, inner
    ):
        raise ValueError("Oracle and full-search fits do not use identical training rows and grouped folds")
    menu = synthetic.menu_audit.raw_menu(base)
    tied_menu = synthetic.previous_audit.tied_queries(menu)
    if not completed:
        if replay_only:
            raise ValueError(f"Missing completed oracle fit: {folder}")
        started = time.monotonic()
        fitted, conditional_heads = oracle_fit(panel, response, inner, model, arm)
        frame = synthetic.menu_audit.prediction_frame(
            menu,
            synthetic.world_mean(menu, "boundary"),
            fitted.predict(menu, np.arange(len(menu.rows))),
            synthetic.world_mean(tied_menu, "boundary"),
            fitted.predict(tied_menu, np.arange(len(tied_menu.rows))),
        )
        saved_fit = fitted.to_json() | {"training_rows": panel.rows.row_id.tolist(), "inner_folds": inner.tolist()}
        replay_error = synthetic.replay_frame(menu, frame, saved_fit, "boundary")
        metrics = {
            "raw_menu": synthetic.menu_audit.menu_metrics(frame),
            "uniform_101_tied_curve_rmse": float(
                np.sqrt(np.mean((frame.loc[frame.tied, "prediction"] - frame.loc[frame.tied, "truth"]) ** 2))
            ),
            "training_endpoint_rmse": float(
                np.sqrt(np.mean((fitted.predict(panel, np.arange(len(response))) - truth) ** 2))
            ),
            "conditional_heads": conditional_heads,
            "floor_compatibility": compatibility,
            "serialized_prediction_replay_error": replay_error,
            "runtime_seconds": time.monotonic() - started,
            "full_search_reference": benchmark.receipt(reference_folder / "complete.json"),
        }
        benchmark.write_json(folder / "fit.json", saved_fit)
        benchmark.write_json(folder / "config.json", config)
        benchmark.write_json(folder / "metrics.json", metrics)
        frame.to_csv(folder / "raw_menu.csv", index=False)
        names = ("fit.json", "config.json", "metrics.json", "raw_menu.csv")
        benchmark.write_json(
            folder / "complete.json",
            {"config": config, "outputs": {name: benchmark.sha256(folder / name) for name in names}},
        )
    frame = pd.read_csv(folder / "raw_menu.csv", float_precision="round_trip")
    saved_fit = json.loads((folder / "fit.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    if saved_fit["training_rows"] != panel.rows.row_id.tolist() or saved_fit["inner_folds"] != inner.tolist():
        raise ValueError("Saved oracle training rows or inner folds changed")
    if saved_fit["shape"] != synthetic.SHAPE or saved_fit["q"] != 0.5 or saved_fit["kappa"] != 0.5:
        raise ValueError("Saved oracle law is not the true fixed boundary law")
    synthetic.replay_frame(menu, frame, saved_fit, "boundary")
    if synthetic.menu_audit.menu_metrics(frame) != metrics["raw_menu"]:
        raise ValueError("Saved oracle metrics disagree with the unchanged raw-menu scorer")
    reference_frame = pd.read_csv(reference_folder / "raw_menu.csv", float_precision="round_trip")
    for column in ("row_id", "phase0_code", "phase1_code", "truth", "tied_truth"):
        if not np.array_equal(frame[column].to_numpy(), reference_frame[column].to_numpy()):
            raise ValueError("Oracle and full-search predictions have different query menus or truths")
    reference_metrics = json.loads((reference_folder / "metrics.json").read_text())
    reference_fit = json.loads((reference_folder / "fit.json").read_text())
    row = {
        **context,
        **{name: saved_fit[name] for name in ("ridge", "gamma", "floor")},
        "conditional_heads": metrics["conditional_heads"],
        "uniform_101_tied_curve_rmse": metrics["uniform_101_tied_curve_rmse"],
        **metrics["raw_menu"],
        **{
            f"full_search_{name}": value
            for name, value in reference_metrics["raw_menu"].items()
            if isinstance(value, (float, int))
        },
        "full_search_uniform_101_tied_curve_rmse": reference_metrics["uniform_101_tied_curve_rmse"],
        "full_search_floor": reference_fit["floor"],
        "full_search_gamma": reference_fit["gamma"],
    }
    return row, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=benchmark.DEFAULT_OUTPUT / "v3/prepared")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    current = identity(args.prepared)
    source_freeze = ROOT / "source_freeze.json"
    if source_freeze.exists():
        if json.loads(source_freeze.read_text()) != current:
            raise ValueError("Oracle implementation drifted after source registration")
    else:
        benchmark.write_json(source_freeze, current)
    contexts = [
        {"arm": arm, "model": model, "sampler": sampler, "seed": seed, "noise_sd": noise}
        for arm in ARMS
        for model in MODELS
        for sampler in synthetic.SAMPLERS
        for noise in synthetic.NOISE_LEVELS
        for seed in (synthetic.SEEDS[:1] if noise == 0 else synthetic.SEEDS)
    ]
    if len(contexts) != EXPECTED_FITS:
        raise ValueError("The registered oracle attribution requires exactly 32 contexts")
    rows, frames = [], {}
    with threadpool_limits(limits=1):
        for context in contexts:
            row, frame = audit_context(args.prepared, context, current, args.replay_only)
            rows.append(row)
            frames[(context["arm"], context["model"], context["noise_sd"], context["seed"], context["sampler"])] = frame
    comparisons = []
    for key, balanced in frames.items():
        if key[-1] != "balanced":
            continue
        positive = frames[(*key[:-1], "positive")]
        tied = balanced.tied.to_numpy(bool)
        drift = balanced.prediction.to_numpy()[tied] - positive.prediction.to_numpy()[tied]
        gain_drift = (balanced.tied_prediction - balanced.prediction) - (positive.tied_prediction - positive.prediction)
        comparisons.append(
            {
                "arm": key[0],
                "model": key[1],
                "noise_sd": key[2],
                "seed": key[3],
                "tied_curve_sampler_drift_rmse": float(np.sqrt(np.mean(drift**2))),
                "tied_curve_sampler_drift_max_abs": float(np.max(np.abs(drift))),
                "matched_gain_sampler_drift_rmse": float(np.sqrt(np.mean(gain_drift**2))),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(ROOT / "sampler_comparison.csv", index=False)
    benchmark.write_json(
        ROOT / "run.json",
        {
            "identity": current,
            "complete_oracle_fits": len(rows),
            "conditional_heads": sum(row["conditional_heads"] for row in rows),
            "outputs": {name: benchmark.sha256(ROOT / name) for name in ("summary.csv", "sampler_comparison.csv")},
        },
    )
    print(
        json.dumps(
            {"completed_oracle_fits": len(rows), "conditional_heads": sum(row["conditional_heads"] for row in rows)}
        )
    )


if __name__ == "__main__":
    main()
