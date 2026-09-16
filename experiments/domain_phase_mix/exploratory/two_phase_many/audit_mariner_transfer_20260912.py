# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "cvxpy==1.7.5", "fsspec==2026.1.0",
#   "gcsfs==2026.1.0", "plotly==6.5.1", "scikit-learn==1.7.2", "tabulate==0.9.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Replay frozen incumbent predictions and audit local continuation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

import benchmark_crossed_mariner_20260912 as crossed
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[3]))
REFERENCE = BASE / "reference_outputs"
OUTPUT = REFERENCE / "two_phase_mariner_transfer_20260912"
BRANCH = REFERENCE / "fixed_checkpoint_branch_wspu_20260907"
CALIBRATION = REFERENCE / "delphi_phase_frontier_calibration_20260910"
STANDALONE = Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py")
STATES = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)


class RawSolution(TypedDict):
    start: str
    success: bool
    message: str
    iterations: int
    latent: float
    weights: list[float]
    constraint_violation: float


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Score one finite candidate menu without using labels to rank proposals."""
    observed = frame.target.to_numpy(float)
    predicted = frame.prediction.to_numpy(float)
    assert np.isfinite(observed).all() and np.isfinite(predicted).all()
    selected = int(np.argmin(predicted))
    ranking = np.argsort(predicted, kind="stable")
    error = predicted - observed
    differences = error[:, None] - error[None, :]
    pairs = np.triu_indices(len(frame), 1)
    return {
        "n": len(frame),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pair_rmse": float(np.sqrt(np.mean(differences[pairs] ** 2))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "regret": float(observed[selected] - observed.min()),
        "shortlist3_regret": float(observed[ranking[:3]].min() - observed.min()),
        "selected_optimism": float(observed[selected] - predicted[selected]),
        "selected_action": str(frame.action_id.iloc[selected]),
        "observed_best_action": str(frame.action_id.iloc[int(np.argmin(observed))]),
    }


def frozen_baselines() -> None:
    """Score canonical-fit incumbents on later crossed branches, with no refit."""
    destination = OUTPUT / "frozen_baselines"
    destination.mkdir(parents=True, exist_ok=True)
    module = load_module(STANDALONE, "transfer_audit_mariner")
    hpr_path = CALIBRATION / "fits/canonical/uncheatable_hpr.pkl"
    mariner_path = CALIBRATION / "fits/mariner_uncheatable.json"
    paths = [
        Path(__file__),
        STANDALONE,
        hpr_path,
        mariner_path,
        CALIBRATION / "fits/canonical/uncheatable_selection.json",
        CALIBRATION / "fits/canonical/uncheatable_predictions.csv",
        CALIBRATION / "inputs/coordinates.npz",
        BRANCH / "data/rows.csv",
        BRANCH / "data/arrays.npz",
    ]
    identity = {str(path): sha(path) for path in paths}
    with hpr_path.open("rb") as handle:
        hpr = pickle.load(handle)
    mariner = module.ObjectiveFit.from_json(json.loads(mariner_path.read_text()))
    old_predictions = pd.read_csv(CALIBRATION / "fits/canonical/uncheatable_predictions.csv")
    with np.load(CALIBRATION / "inputs/coordinates.npz") as bank:
        replay_hpr = hpr.predict(bank["weights"])
        replay_mariner = mariner.predict(bank["aggregate"])
    replay_errors = {
        "hpr": float(np.max(np.abs(replay_hpr - old_predictions.hpr.to_numpy()))),
        "mariner": float(np.max(np.abs(replay_mariner - old_predictions.mariner.to_numpy()))),
    }
    assert max(replay_errors.values()) < 1e-10, replay_errors
    frame = pd.read_csv(BRANCH / "data/rows.csv")
    with np.load(BRANCH / "data/arrays.npz") as source:
        arrays = {name: source[name] for name in source.files}
    buckets = tuple(str(name) for name in arrays["bucket_names"])
    assert buckets == tuple(hpr.dataset.domains) == mariner.buckets
    # Only crossed rows are complete in every component. Other historical rows
    # intentionally contain NaNs and are outside this replay.
    rows = frame.index[
        frame.panel.eq("crossed_local") & frame.state_id.isin(STATES) & (frame.fit_budget | frame.is_tied_control)
    ].to_numpy()
    assert len(rows) == 66
    weights = np.stack([arrays["phase0_weight"][rows], arrays["phase1_weight"][rows]], axis=1)
    exposures = arrays["phase0_epochs"][rows] + arrays["phase1_epochs"][rows]
    aggregate = (2400 / 3007) * weights[:, 0] + (607 / 3007) * weights[:, 1]
    assert np.allclose(aggregate.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    geometry_errors = {
        "prefix": float(np.max(np.abs(weights[:, 0] * hpr.dataset.c0 - arrays["phase0_epochs"][rows]))),
        "continuation": float(np.max(np.abs(weights[:, 1] * hpr.dataset.c1 - arrays["phase1_epochs"][rows]))),
    }
    # The archived fit inventories and branch materialization differ by one
    # uniform scale (about 0.5 ppm). Verify it and measure prediction sensitivity.
    active = aggregate > 1e-8
    ratios = (exposures / np.maximum(aggregate * mariner.inventory, 1e-99))[active]
    runtime_scale = float(np.median(ratios))
    assert np.max(np.abs(ratios - runtime_scale)) < 1e-9
    hpr_scale = float(
        np.median(
            (arrays["phase0_epochs"][rows] / np.maximum(weights[:, 0] * hpr.dataset.c0, 1e-99))[weights[:, 0] > 1e-8]
        )
    )
    assert np.allclose(arrays["phase0_epochs"][rows], weights[:, 0] * hpr.dataset.c0 * hpr_scale, rtol=0, atol=1e-9)
    assert np.allclose(arrays["phase1_epochs"][rows], weights[:, 1] * hpr.dataset.c1 * hpr_scale, rtol=0, atol=1e-9)
    reconstructed = arrays["component_bpb"][rows] @ arrays["component_weights"]
    assert np.max(np.abs(reconstructed - frame.loc[rows, "target"].to_numpy())) < 2e-7
    canonical_weights = hpr.dataset.weights
    minimum_policy_l1 = np.abs(weights[:, None] - canonical_weights[None]).sum(axis=(2, 3)).min(axis=1)
    assert minimum_policy_l1.min() > 1e-8
    phase = hpr.predict(weights)
    phase_tied = hpr.predict(np.repeat(aggregate[:, None, :], 2, axis=1))
    spine = mariner.predict(aggregate)
    physical_hpr = replace(
        hpr, dataset=replace(hpr.dataset, c0=hpr.dataset.c0 * hpr_scale, c1=hpr.dataset.c1 * hpr_scale)
    )
    physical_mariner = replace(mariner, inventory=mariner.inventory * runtime_scale)
    coordinate_sensitivity = {
        "hpr_inventory_scale": hpr_scale,
        "mariner_inventory_scale": runtime_scale,
        "max_hpr_prediction_difference": float(np.max(np.abs(phase - physical_hpr.predict(weights)))),
        "max_mariner_prediction_difference": float(np.max(np.abs(spine - physical_mariner.predict(aggregate)))),
        "reported_coordinates": "Frozen native inventory; physical materialization sensitivity reported separately.",
    }
    tables = []
    for name, prediction in (
        ("frozen_mariner", spine),
        ("frozen_hpr", phase),
        ("frozen_mariner_hpr", spine + phase - phase_tied),
    ):
        table = frame.loc[rows, ["row_id", "state_id", "action_id", "target", "is_tied_control"]].copy()
        table["model"] = name
        table["prediction"] = prediction
        table["canonical_policy_min_l1"] = minimum_policy_l1
        tables.append(table)
    predictions = pd.concat(tables, ignore_index=True)
    metrics = []
    for (model, state), cell in predictions.groupby(["model", "state_id"], sort=False):
        for tied_selectable in (False, True):
            menu = cell if tied_selectable else cell[~cell.is_tied_control]
            metrics.append(
                {"model": model, "state_id": state, "tied_selectable": tied_selectable, **selection_metrics(menu)}
            )
    prediction_path = destination / "predictions.csv"
    metric_path = destination / "metrics.csv"
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(metrics).to_csv(metric_path, index=False)
    write_json_atomic(
        destination / "manifest.json",
        {
            "inputs": identity,
            "replay_errors": replay_errors,
            "geometry_errors": geometry_errors,
            "coordinate_sensitivity": coordinate_sensitivity,
            "outputs": {path.name: sha(path) for path in (prediction_path, metric_path)},
            "scope": (
                "Frozen canonical-training incumbents. Different training data from new crossed fits; "
                "descriptive external reference, not matched fitting-budget comparison."
            ),
            "no_fit": True,
            "no_remote_access": True,
        },
    )
    print(
        pd.DataFrame(metrics)
        .groupby(["model", "tied_selectable"])[["rmse", "pair_rmse", "regret", "selected_optimism"]]
        .mean()
        .to_string()
    )


def latent_problem(
    fit: dict[str, Any], prefix: np.ndarray, late_rates: np.ndarray, source: ModuleType
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Bind a fixed prefix and fit to its latent objective and derivative."""
    shape = fit["shape"]
    assert shape["power"] >= 1.0, "this analytic boundary derivative requires power at least one"
    coefficients = np.asarray(fit["coefficients"])
    prefix_columns = source.design_matrix(prefix[None], shape)[0]

    def objective(weight: np.ndarray) -> float:
        columns = source.design_matrix((prefix + late_rates * weight)[None], shape)[0] - prefix_columns
        return float(fit["intercept"] + columns @ coefficients)

    def gradient(weight: np.ndarray) -> np.ndarray:
        exposure = prefix + late_rates * weight
        rate, power, threshold = shape["rate"], shape["power"], shape["threshold"]
        scaled = rate * exposure
        benefit_derivative = rate * power * scaled ** (power - 1) * np.exp(-(scaled**power))
        argument = np.log1p(exposure) - threshold
        harm_derivative = 2 * np.logaddexp(argument, 0) * expit(argument) / (1 + exposure)
        return late_rates * (-coefficients[:39] * benefit_derivative + coefficients[39:] * harm_derivative)

    return objective, gradient


def raw_continuations() -> None:
    """Audit unregularized continuation optima across omitted-prefix refits."""
    destination = OUTPUT / "raw"
    prepared_path = OUTPUT / "crossed/prepared.json"
    source_path = OUTPUT / "crossed/sources/mixture_selection.py"
    source = load_module(source_path, "raw_transfer_mariner")
    frame, arrays = crossed.load_data()
    model_paths = [OUTPUT / "crossed/fits" / state / "MTP-002.json" for state in STATES]
    identity = {
        str(path): sha(path)
        for path in [
            Path(__file__),
            Path(crossed.__file__),
            destination / "PROTOCOL.md",
            prepared_path,
            source_path,
            *model_paths,
        ]
    }
    all_results = []
    all_weights = []
    gradient_errors = []
    for model_path in model_paths:
        stored = json.loads(model_path.read_text())
        fit = stored["fit"]
        assert fit["variant"] == "cumulative_increment_log"
        for state in STATES:
            cell_path = destination / "cells" / model_path.parent.name / f"{state}.json"
            state_rows = frame.index[
                frame.state_id.eq(state) & frame.panel.isin(["crossed_broad", "crossed_local"])
            ].to_numpy()
            first = state_rows[0]
            prefix = arrays["phase0_epochs"][first]
            prefix_weight = arrays["phase0_weight"][first]
            total_rates = np.max(
                np.divide(
                    arrays["phase0_epochs"][state_rows],
                    arrays["phase0_weight"][state_rows],
                    out=np.zeros_like(arrays["phase0_epochs"][state_rows]),
                    where=arrays["phase0_weight"][state_rows] > 0,
                ),
                axis=0,
            ) / (2400 / 3007)
            # Sparse prefixes omit some rates; recover those from continuations.
            late_rates = np.max(
                np.divide(
                    arrays["phase1_epochs"][state_rows],
                    arrays["phase1_weight"][state_rows],
                    out=np.zeros_like(arrays["phase1_epochs"][state_rows]),
                    where=arrays["phase1_weight"][state_rows] > 0,
                ),
                axis=0,
            )
            missing = total_rates == 0
            total_rates[missing] = late_rates[missing] / (607 / 3007)
            assert np.all(late_rates > 0)
            assert np.allclose(
                arrays["phase1_epochs"][state_rows], arrays["phase1_weight"][state_rows] * late_rates, rtol=0, atol=1e-9
            )
            observed_weights = arrays["phase1_weight"][state_rows]
            objective, gradient = latent_problem(fit, prefix, late_rates, source)

            center = np.full(39, 1 / 39)
            finite = np.empty(39)
            for index in range(39):
                displacement = np.zeros(39)
                displacement[index] = 1e-6
                finite[index] = (objective(center + displacement) - objective(center - displacement)) / 2e-6
            gradient_error = float(np.max(np.abs(finite - gradient(center))))
            assert gradient_error < 1e-6, gradient_error
            gradient_errors.append(gradient_error)
            if cell_path.exists():
                result: dict[str, Any] = json.loads(cell_path.read_text())
                assert result["identity"] == identity
            else:
                candidate_prediction = crossed.predict_fit(fit, prefix, arrays["phase1_epochs"][state_rows], source_path)
                broad = (
                    frame.panel.iloc[state_rows].eq("crossed_broad").to_numpy()
                    & frame.fit_budget.iloc[state_rows].to_numpy()
                )
                local = (
                    frame.panel.iloc[state_rows].eq("crossed_local").to_numpy()
                    & frame.fit_budget.iloc[state_rows].to_numpy()
                )
                broad_best = np.flatnonzero(broad)[np.argmin(candidate_prediction[broad])]
                local_best = np.flatnonzero(local)[np.argmin(candidate_prediction[local])]
                natural = 1 / total_rates
                natural /= natural.sum()
                starts = [
                    ("tied", prefix_weight),
                    ("proportional", natural),
                    ("uniform", center),
                    ("predicted_broad", observed_weights[broad_best]),
                    ("predicted_local", observed_weights[local_best]),
                ]
                rng = np.random.default_rng(20260912)
                starts.extend((f"dirichlet_{index}", rng.dirichlet(np.ones(39))) for index in range(3))
                solutions: list[RawSolution] = []
                for label, start in starts:
                    solution = minimize(
                        objective,
                        start,
                        jac=gradient,
                        method="SLSQP",
                        bounds=[(0, 1)] * 39,
                        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(39)}],
                        options={"maxiter": 1000, "ftol": 1e-12},
                    )
                    violation = max(abs(solution.x.sum() - 1), max(0.0, -float(solution.x.min())))
                    solutions.append(
                        {
                            "start": label,
                            "success": bool(solution.success),
                            "message": str(solution.message),
                            "iterations": int(solution.nit),
                            "latent": float(solution.fun),
                            "weights": solution.x.tolist(),
                            "constraint_violation": violation,
                        }
                    )
                feasible = [solution for solution in solutions if solution["constraint_violation"] < 1e-8]
                assert feasible
                best = min(feasible, key=lambda solution: solution["latent"])
                weight = np.asarray(best["weights"])
                prediction = float(crossed.predict_fit(fit, prefix, (late_rates * weight)[None], source_path)[0])
                tied_prediction = float(
                    crossed.predict_fit(fit, prefix, (late_rates * prefix_weight)[None], source_path)[0]
                )
                nearest = np.abs(observed_weights - weight).sum(axis=1) / 2
                result = {
                    "identity": identity,
                    "fit_held_state": model_path.parent.name,
                    "prefix_state": state,
                    "own_held_prefix": model_path.parent.name == state,
                    "best": best,
                    "starts": solutions,
                    "prediction": prediction,
                    "floor": fit["floor"],
                    "distance_above_floor": prediction - fit["floor"],
                    "predicted_tied": tied_prediction,
                    "predicted_gain_vs_tied": prediction - tied_prediction,
                    "predicted_best_local": float(candidate_prediction[local_best]),
                    "predicted_gain_vs_local": prediction - float(candidate_prediction[local_best]),
                    "nearest_observed_tv": float(nearest.min()),
                    "nearest_observed_action": str(frame.action_id.iloc[state_rows[int(nearest.argmin())]]),
                    "maximum_continuation_epochs": float(np.max(late_rates * weight)),
                    "maximum_total_epochs": float(np.max(prefix + late_rates * weight)),
                    "multistart_latent_range": (
                        max(solution["latent"] for solution in feasible)
                        - min(solution["latent"] for solution in feasible)
                    ),
                    "successful_starts": sum(solution["success"] for solution in solutions),
                    "gradient_max_error": gradient_error,
                }
                write_json_atomic(cell_path, result)
            all_results.append(
                {key: value for key, value in result.items() if key not in {"identity", "best", "starts"}}
            )
            all_weights.append(
                {
                    "fit_held_state": model_path.parent.name,
                    "prefix_state": state,
                    **dict(zip(arrays["bucket_names"].tolist(), result["best"]["weights"], strict=True)),
                }
            )
    metrics = pd.DataFrame(all_results)
    weights = pd.DataFrame(all_weights)
    stability = []
    for state, group in weights.groupby("prefix_state", sort=False):
        w = group[list(arrays["bucket_names"])].to_numpy()
        distances = np.abs(w[:, None] - w[None]).sum(axis=2) / 2
        pairs = distances[np.triu_indices(len(w), 1)]
        stability.append(
            {"state_id": state, "max_refit_tv": float(pairs.max()), "median_refit_tv": float(np.median(pairs))}
        )
    metrics.to_csv(destination / "metrics.csv", index=False)
    weights.to_csv(destination / "weights.csv", index=False)
    pd.DataFrame(stability).to_csv(destination / "stability.csv", index=False)
    write_json_atomic(
        destination / "validation.json",
        {
            "cells": len(metrics),
            "max_gradient_error": max(gradient_errors),
            "min_successful_starts": int(metrics.successful_starts.min()),
            "max_multistart_latent_range": float(metrics.multistart_latent_range.max()),
            "identity": identity,
            "no_measured_optimum_outcomes": True,
        },
    )
    print(
        metrics[metrics.own_held_prefix][
            [
                "prefix_state",
                "prediction",
                "predicted_gain_vs_tied",
                "predicted_gain_vs_local",
                "nearest_observed_tv",
                "maximum_total_epochs",
            ]
        ].to_string(index=False)
    )
    print(pd.DataFrame(stability).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["frozen-baselines", "raw"])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.command == "raw":
            raw_continuations()
        else:
            frozen_baselines()


if __name__ == "__main__":
    main()
