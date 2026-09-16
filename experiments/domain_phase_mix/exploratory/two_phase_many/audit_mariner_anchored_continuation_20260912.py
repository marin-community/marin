# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "fsspec==2026.1.0", "threadpoolctl==3.6.0"]
# ///
"""Audit a frozen MARINER aggregate anchor plus shared continuation residual."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

import benchmark_crossed_mariner_20260912 as crossed
import numpy as np
import pandas as pd
from audit_mariner_transfer_20260912 import selection_metrics, sha
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
ROOT = BASE / "reference_outputs/two_phase_mariner_transfer_20260912"
OUTPUT = ROOT / "anchored"
AGGREGATE_PATH = BASE / "reference_outputs/delphi_phase_frontier_calibration_20260910/fits/mariner_uncheatable.json"
SOURCE_PATH = ROOT / "crossed/sources/mixture_selection.py"
ALPHA = 2400 / 3007
LATE = 1 - ALPHA
WIDTH = 39
STATES = crossed.STATES
SEED = 20260912


@dataclass(frozen=True)
class Aggregate:
    model: Any
    rate: np.ndarray
    power: np.ndarray
    threshold: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    floor: np.ndarray


@dataclass(frozen=True)
class Continuation:
    held_state: str
    saved: dict[str, Any]
    coefficients: np.ndarray
    train_prefix_weights: np.ndarray
    train_action_weights: np.ndarray


@dataclass(frozen=True)
class Frozen:
    rows: pd.DataFrame
    arrays: dict[str, np.ndarray]
    aggregate: Aggregate
    source: ModuleType
    physical_inventory: np.ndarray
    fits: tuple[Continuation, ...]
    identity: dict[str, str]


class AnchoredSolution(TypedDict):
    start: str
    start_weights: list[float]
    success: bool
    message: str
    iterations: int
    prediction: float
    weights: list[float]
    constraint_violation: float
    stationarity_residual: float


def load_frozen(output: Path) -> Frozen:
    rows, arrays = crossed.load_data()
    baseline_path = ROOT / "frozen_baselines/manifest.json"
    baseline = json.loads(baseline_path.read_text())
    for path, digest in baseline["inputs"].items():
        assert sha(Path(path)) == digest, f"Frozen baseline source changed: {path}"
    for path, digest in baseline["outputs"].items():
        assert sha(ROOT / "frozen_baselines" / path) == digest
    prepared_path = ROOT / "crossed/prepared.json"
    prepared = json.loads(prepared_path.read_text())
    assert all(sha(Path(path)) == digest for path, digest in prepared["hashes"].items())
    source = load_module(SOURCE_PATH, "anchored_mariner_source")
    model = source.ObjectiveFit.from_json(json.loads(AGGREGATE_PATH.read_text()))
    assert tuple(arrays["bucket_names"]) == model.buckets
    assert tuple(task.component.split("/")[-2] for task in model.tasks) == tuple(arrays["component_names"])
    # Retain A's fitted objective weights exactly; the prepared canonical weights
    # differ slightly and are recorded as a sensitivity check, never substituted.
    aggregate = Aggregate(model, *model._arrays())
    assert np.all(aggregate.power >= 1), "Boundary derivative requires power at least one"
    scale = baseline["coordinate_sensitivity"]["mariner_inventory_scale"]
    physical_inventory = model.inventory * scale
    selected = rows.panel.isin(("crossed_broad", "crossed_local")) & rows.state_id.isin(STATES)
    indices = np.flatnonzero(selected)
    assert np.allclose(
        arrays["phase0_epochs"][indices],
        arrays["phase0_weight"][indices] * ALPHA * physical_inventory,
        rtol=0,
        atol=1e-9,
    )
    assert np.allclose(
        arrays["phase1_epochs"][indices],
        arrays["phase1_weight"][indices] * LATE * physical_inventory,
        rtol=0,
        atol=1e-9,
    )
    fits = []
    paths = [
        Path(__file__),
        Path(crossed.__file__),
        Path(selection_metrics.__code__.co_filename),
        BASE / "fixed_checkpoint_wspu_models_20260907.py",
        BASE / "fit_two_phase_link_spines_20260907.py",
        AGGREGATE_PATH,
        SOURCE_PATH,
        prepared_path,
        baseline_path,
        ROOT / "frozen_baselines/predictions.csv",
        ROOT / "raw/weights.csv",
        ROOT / "raw/metrics.csv",
        output / "PROTOCOL.md",
        crossed.INPUT / "data/rows.csv",
        crossed.INPUT / "data/arrays.npz",
    ]
    for state in STATES:
        path = ROOT / "crossed/fits" / state / "MTP-002.json"
        saved = json.loads(path.read_text())
        assert saved["input_hashes"] == prepared["hashes"]
        assert saved["fit"]["variant"] == "cumulative_increment_log"
        assert saved["fit"]["shape"]["power"] >= 1
        training = np.asarray(saved["training_rows"], int)
        assert not rows.state_id.iloc[training].eq(state).any()
        fits.append(
            Continuation(
                state,
                saved,
                np.asarray(saved["fit"]["coefficients"]),
                arrays["phase0_weight"][training],
                arrays["phase1_weight"][training],
            )
        )
        paths.append(path)
    identity = {str(path.resolve()): sha(path) for path in paths}
    return Frozen(rows, arrays, aggregate, source, physical_inventory, tuple(fits), identity)


def columns_and_derivatives(
    exposure: np.ndarray,
    rate: np.ndarray | float,
    power: np.ndarray | float,
    threshold: np.ndarray | float,
    source: ModuleType,
) -> tuple[np.ndarray, np.ndarray]:
    columns = np.concatenate([-source.benefit(exposure, rate, power), source.harm(exposure, threshold)], axis=-1)
    scaled = rate * exposure
    benefit_derivative = -rate * power * scaled ** (power - 1) * np.exp(-(scaled**power))
    argument = np.log1p(exposure) - threshold
    harm_derivative = 2 * np.logaddexp(argument, 0) * expit(argument) / (1 + exposure)
    return columns, np.concatenate([benefit_derivative, harm_derivative], axis=-1)


def weighted_derivative(derivative: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    product = derivative * coefficients
    return product[..., :WIDTH] + product[..., WIDTH:]


def evaluate_policy(
    frozen: Frozen,
    fit: Continuation,
    prefix_weight: np.ndarray,
    action_weight: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    aggregate_weight = ALPHA * prefix_weight + LATE * action_weight
    anchor = frozen.aggregate
    columns, derivative = columns_and_derivatives(
        aggregate_weight[None] * anchor.model.inventory,
        anchor.rate[:, None],
        anchor.power[:, None],
        anchor.threshold[:, None],
        frozen.source,
    )
    anchor_latent = anchor.intercept + np.sum(columns * anchor.coefficients, axis=1)
    anchor_deficit = np.exp(np.clip(anchor_latent, -frozen.source.LOG_CLIP, frozen.source.LOG_CLIP))
    anchor_components = anchor.floor + anchor_deficit
    anchor_value = float(anchor_components @ anchor.model.weights)
    anchor_gradient = np.sum(
        anchor.model.weights[:, None]
        * anchor_deficit[:, None]
        * (np.abs(anchor_latent) < frozen.source.LOG_CLIP)[:, None]
        * weighted_derivative(derivative, anchor.coefficients)
        * anchor.model.inventory,
        axis=0,
    )
    saved = fit.saved["fit"]
    shape = saved["shape"]
    exposures = np.stack(
        [
            (ALPHA * prefix_weight + LATE * action_weight) * frozen.physical_inventory,
            ALPHA * prefix_weight * frozen.physical_inventory,
            aggregate_weight * frozen.physical_inventory,
            ALPHA * aggregate_weight * frozen.physical_inventory,
        ]
    )
    columns, derivative = columns_and_derivatives(
        exposures,
        shape["rate"],
        shape["power"],
        shape["threshold"],
        frozen.source,
    )
    policy_latent = float(saved["intercept"] + (columns[0] - columns[1]) @ fit.coefficients)
    tied_latent = float(saved["intercept"] + (columns[2] - columns[3]) @ fit.coefficients)
    policy_deficit, tied_deficit = np.exp(
        np.clip(
            [policy_latent, tied_latent],
            -frozen.source.LOG_CLIP,
            frozen.source.LOG_CLIP,
        )
    )
    policy_value, tied_value = saved["floor"] + policy_deficit, saved["floor"] + tied_deficit
    policy_gradient = (
        policy_deficit
        * (abs(policy_latent) < frozen.source.LOG_CLIP)
        * weighted_derivative(derivative[0], fit.coefficients)
        * LATE
        * frozen.physical_inventory
    )
    tied_gradient = (
        tied_deficit
        * (abs(tied_latent) < frozen.source.LOG_CLIP)
        * frozen.physical_inventory
        * (
            weighted_derivative(derivative[2], fit.coefficients)
            - ALPHA * weighted_derivative(derivative[3], fit.coefficients)
        )
    )
    correction = float(policy_value - tied_value)
    value = anchor_value + correction
    terms = {
        "prediction": value,
        "anchor_prediction": anchor_value,
        "w_policy_prediction": float(policy_value),
        "w_tied_aggregate_prediction": float(tied_value),
        "phase_correction_bpb": correction,
        "cancellation_ratio": (
            float((abs(policy_value) + abs(tied_value)) / abs(correction)) if abs(correction) > 1e-12 else None
        ),
        "w_tied_minus_anchor_bpb": float(tied_value - anchor_value),
        "w_policy_latent": policy_latent,
        "w_tied_latent": tied_latent,
        "guard_count": int(
            np.sum(np.abs(anchor_latent) >= frozen.source.LOG_CLIP)
            + (abs(policy_latent) >= frozen.source.LOG_CLIP)
            + (abs(tied_latent) >= frozen.source.LOG_CLIP)
        ),
        "aggregate_weight": aggregate_weight.tolist(),
        "anchor_components": anchor_components.tolist(),
    }
    gradient = LATE * anchor_gradient + policy_gradient - LATE * tied_gradient
    assert np.isfinite(value) and np.isfinite(gradient).all()
    return terms, gradient


def support(frozen: Frozen, fit: Continuation, aggregate: np.ndarray) -> dict[str, float]:
    prefix_distance = np.abs(fit.train_prefix_weights - aggregate).sum(axis=1) / 2
    action_distance = np.abs(fit.train_action_weights - aggregate).sum(axis=1) / 2
    return {
        "tied_query_nearest_training_prefix_tv": float(prefix_distance.min()),
        "tied_query_nearest_training_policy_mean_phase_tv": float(((prefix_distance + action_distance) / 2).min()),
    }


def finite_candidates(frozen: Frozen, output: Path) -> dict[str, float]:
    rows, arrays = frozen.rows, frozen.arrays
    selected = rows.panel.eq("crossed_local") & rows.state_id.isin(STATES) & (rows.fit_budget | rows.is_tied_control)
    indices = np.flatnonzero(selected)
    assert len(indices) == 66
    baselines = pd.read_csv(ROOT / "frozen_baselines/predictions.csv")
    wanted = baselines.model.isin(("frozen_mariner", "frozen_mariner_hpr"))
    tables = [baselines[wanted].copy()]
    composite, raw, component_predictions = [], [], []
    replay_errors, weight_sensitivity = [], []
    for fit in frozen.fits:
        state_indices = indices[rows.state_id.iloc[indices].eq(fit.held_state)]
        stored_predictions = dict(zip(fit.saved["prediction_rows"], fit.saved["predicted"], strict=True))
        for index in state_indices:
            prefix, action = arrays["phase0_weight"][index], arrays["phase1_weight"][index]
            terms, _ = evaluate_policy(frozen, fit, prefix, action)
            replay_errors.append(abs(terms["w_policy_prediction"] - stored_predictions[int(index)]))
            incumbent = baselines[baselines.row_id.eq(rows.row_id.iloc[index]) & baselines.model.eq("frozen_mariner")]
            assert len(incumbent) == 1
            replay_errors.append(abs(terms["anchor_prediction"] - float(incumbent.prediction.iloc[0])))
            weight_sensitivity.append(
                abs(np.asarray(terms["anchor_components"]) @ arrays["component_weights"] - terms["anchor_prediction"])
            )
            metadata = rows.iloc[index][["row_id", "state_id", "action_id", "target", "is_tied_control"]].to_dict()
            common = {"row": int(index), **metadata}
            composite.append(
                {
                    **common,
                    "model": "MTP-006",
                    **{
                        key: value
                        for key, value in terms.items()
                        if key not in ("aggregate_weight", "anchor_components")
                    },
                    **support(frozen, fit, np.asarray(terms["aggregate_weight"])),
                }
            )
            raw.append({**common, "model": "MTP-002", "prediction": terms["w_policy_prediction"]})
            for component, prediction in zip(arrays["component_names"], terms["anchor_components"], strict=True):
                component_predictions.append(
                    {
                        "row_id": rows.row_id.iloc[index],
                        "component": component,
                        "mariner_anchor_component_prediction": prediction,
                    }
                )
    assert max(replay_errors) < 1e-10
    tables.extend([pd.DataFrame(raw), pd.DataFrame(composite)])
    predictions = pd.concat(tables, ignore_index=True)
    predictions.to_csv(output / "finite_predictions.csv", index=False)
    pd.DataFrame(composite).to_csv(output / "finite_composite_terms.csv", index=False)
    component_table = pd.DataFrame(component_predictions)
    component_table.to_csv(output / "anchor_component_predictions.csv", index=False)
    metrics, components = [], []
    for (model, state), cell in predictions.groupby(["model", "state_id"], sort=False):
        tied = cell[cell.is_tied_control]
        assert len(tied) == 1
        tied_index = int(rows.index[rows.row_id.eq(tied.row_id.iloc[0])][0])
        for selectable in (False, True):
            menu = cell if selectable else cell[~cell.is_tied_control]
            score = selection_metrics(menu)
            pick = menu.iloc[int(np.argmin(menu.prediction.to_numpy()))]
            metrics.append(
                {
                    "model": model,
                    "state_id": state,
                    "tied_selectable": selectable,
                    **score,
                    "predicted_gain_vs_tied": float(tied.prediction.iloc[0] - pick.prediction),
                    "observed_gain_vs_tied": float(tied.target.iloc[0] - pick.target),
                }
            )
            picked_index = int(rows.index[rows.row_id.eq(pick.row_id)][0])
            for component_index, name in enumerate(arrays["component_names"]):
                anchor_prediction = component_table[
                    component_table.row_id.eq(pick.row_id) & component_table.component.eq(name)
                ]
                components.append(
                    {
                        "model": model,
                        "state_id": state,
                        "tied_selectable": selectable,
                        "selected_action": pick.action_id,
                        "component": name,
                        "selected_measured_bpb": arrays["component_bpb"][picked_index, component_index],
                        "tied_measured_bpb": arrays["component_bpb"][tied_index, component_index],
                        "measured_selected_minus_tied_bpb": (
                            arrays["component_bpb"][picked_index, component_index]
                            - arrays["component_bpb"][tied_index, component_index]
                        ),
                        "anchor_component_prediction": float(
                            anchor_prediction.mariner_anchor_component_prediction.iloc[0]
                        ),
                        "composite_component_prediction_available": False,
                    }
                )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "finite_metrics.csv", index=False)
    columns = [
        "rmse",
        "pair_rmse",
        "spearman",
        "regret",
        "shortlist3_regret",
        "selected_optimism",
        "predicted_gain_vs_tied",
        "observed_gain_vs_tied",
    ]
    metrics_frame.groupby(["model", "tied_selectable"], sort=False)[columns].mean().to_csv(output / "finite_summary.csv")
    pd.DataFrame(components).to_csv(output / "selected_components.csv", index=False)
    pd.DataFrame(
        {
            "component": arrays["component_names"],
            "anchor_frozen_weight": frozen.aggregate.model.weights,
            "prepared_canonical_weight": arrays["component_weights"],
        }
    ).to_csv(output / "component_weights.csv", index=False)
    return {
        "max_frozen_prediction_replay_error": max(replay_errors),
        "finite_candidates_per_model": 66,
        "maximum_component_weight_difference": float(
            np.max(np.abs(frozen.aggregate.model.weights - arrays["component_weights"]))
        ),
        "maximum_anchor_weight_sensitivity_bpb": max(weight_sensitivity),
    }


def structural_checks(frozen: Frozen, output: Path) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    prefixes = np.stack(
        [frozen.arrays["phase0_weight"][int(frozen.rows.index[frozen.rows.state_id.eq(state)][0])] for state in STATES]
    )
    policies = np.concatenate([prefixes, rng.dirichlet(np.ones(WIDTH), 20)])
    parity, gradient_records, replay = [], [], []
    for fit in frozen.fits:
        for index, policy in enumerate(policies):
            terms, _ = evaluate_policy(frozen, fit, policy, policy)
            error = abs(terms["prediction"] - float(frozen.aggregate.model.predict(policy)[0]))
            parity.append({"fit_held_state": fit.held_state, "policy": index, "error_bpb": error})
        prefix = prefixes[STATES.index(fit.held_state)]
        for index, action in enumerate((np.full(WIDTH, 1 / WIDTH), rng.dirichlet(np.ones(WIDTH)))):
            terms, analytic = evaluate_policy(frozen, fit, prefix, action)
            finite = np.empty(WIDTH)
            for bucket in range(WIDTH):
                step = np.zeros(WIDTH)
                step[bucket] = 1e-6
                high = evaluate_policy(frozen, fit, prefix, action + step)[0]["prediction"]
                low = evaluate_policy(frozen, fit, prefix, action - step)[0]["prediction"]
                finite[bucket] = (high - low) / 2e-6
            error = float(np.max(np.abs(finite - analytic)))
            gradient_records.append({"fit_held_state": fit.held_state, "policy": index, "max_error": error})
            predicted_w = crossed.predict_fit(
                fit.saved["fit"],
                ALPHA * frozen.physical_inventory * prefix,
                (LATE * frozen.physical_inventory * action)[None],
                SOURCE_PATH,
            )[0]
            replay.append(abs(terms["w_policy_prediction"] - predicted_w))
    parity_error = max(record["error_bpb"] for record in parity)
    gradient_error = max(record["max_error"] for record in gradient_records)
    assert parity_error < 1e-12 and gradient_error < 1e-6 and max(replay) < 1e-10
    pd.DataFrame(parity).to_csv(output / "tied_parity.csv", index=False)
    pd.DataFrame(gradient_records).to_csv(output / "gradient_checks.csv", index=False)
    return {
        "max_tied_parity_error": parity_error,
        "max_gradient_error": gradient_error,
        "max_random_w_replay_error": max(replay),
    }


def objective_and_gradient(
    weight: np.ndarray,
    frozen: Frozen,
    fit: Continuation,
    prefix: np.ndarray,
) -> tuple[float, np.ndarray]:
    terms, gradient = evaluate_policy(frozen, fit, prefix, weight)
    return terms["prediction"], gradient


def raw_continuations(frozen: Frozen, output: Path) -> dict[str, Any]:
    rows, arrays = frozen.rows, frozen.arrays
    raw_weights = pd.read_csv(ROOT / "raw/weights.csv")
    results, weights, raw_comparison = [], [], []
    for fit in frozen.fits:
        for state in STATES:
            path = output / "cells" / fit.held_state / f"{state}.json"
            indices = np.flatnonzero(rows.state_id.eq(state) & rows.panel.isin(("crossed_broad", "crossed_local")))
            prefix = arrays["phase0_weight"][indices[0]]
            observed = arrays["phase1_weight"][indices]
            result: dict[str, Any]
            if path.exists():
                result = json.loads(path.read_text())
                assert result["identity"] == frozen.identity, f"Cached cell identity changed: {path}"
            else:
                finite = [evaluate_policy(frozen, fit, prefix, weight)[0] for weight in observed]
                finite_values = np.asarray([record["prediction"] for record in finite])
                broad = (
                    rows.panel.iloc[indices].eq("crossed_broad").to_numpy() & rows.fit_budget.iloc[indices].to_numpy()
                )
                local = (
                    rows.panel.iloc[indices].eq("crossed_local").to_numpy() & rows.fit_budget.iloc[indices].to_numpy()
                )
                broad_best = np.flatnonzero(broad)[int(np.argmin(finite_values[broad]))]
                local_best = np.flatnonzero(local)[int(np.argmin(finite_values[local]))]
                natural = 1 / frozen.physical_inventory
                natural /= natural.sum()
                starts = [
                    ("tied", prefix),
                    ("proportional", natural),
                    ("uniform", np.full(WIDTH, 1 / WIDTH)),
                    ("predicted_broad", observed[broad_best]),
                    ("predicted_local", observed[local_best]),
                ]
                rng = np.random.default_rng(SEED)
                starts.extend((f"dirichlet_{index}", rng.dirichlet(np.ones(WIDTH))) for index in range(3))

                solutions: list[AnchoredSolution] = []
                for label, start in starts:
                    solution = minimize(
                        objective_and_gradient,
                        start,
                        args=(frozen, fit, prefix),
                        jac=True,
                        method="SLSQP",
                        bounds=[(0, 1)] * WIDTH,
                        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(WIDTH)}],
                        options={"maxiter": 1000, "ftol": 1e-12},
                    )
                    violation = max(abs(solution.x.sum() - 1), max(0.0, -float(solution.x.min())))
                    terms, gradient = evaluate_policy(frozen, fit, prefix, solution.x)
                    level = float(solution.x @ gradient)
                    active = solution.x > 1e-7
                    stationarity = max(
                        float(np.max(np.abs(gradient[active] - level))), float(np.max(np.maximum(level - gradient, 0)))
                    )
                    solutions.append(
                        {
                            "start": label,
                            "start_weights": start.tolist(),
                            "success": bool(solution.success),
                            "message": str(solution.message),
                            "iterations": int(solution.nit),
                            "prediction": terms["prediction"],
                            "weights": solution.x.tolist(),
                            "constraint_violation": violation,
                            "stationarity_residual": stationarity,
                        }
                    )
                feasible = [solution for solution in solutions if solution["constraint_violation"] < 1e-8]
                assert feasible
                best = min(feasible, key=lambda solution: solution["prediction"])
                weight = np.asarray(best["weights"])
                terms, _ = evaluate_policy(frozen, fit, prefix, weight)
                nearest = np.abs(observed - weight).sum(axis=1) / 2
                local_measured = rows.target.iloc[indices[local]].to_numpy(float)
                predicted_tied = evaluate_policy(frozen, fit, prefix, prefix)[0]["prediction"]
                result = {
                    "identity": frozen.identity,
                    "fit_held_state": fit.held_state,
                    "prefix_state": state,
                    "own_held_prefix": fit.held_state == state,
                    "starts": solutions,
                    "best": best,
                    **{
                        key: value
                        for key, value in terms.items()
                        if key not in ("aggregate_weight", "anchor_components")
                    },
                    **support(frozen, fit, np.asarray(terms["aggregate_weight"])),
                    "predicted_tied": predicted_tied,
                    "predicted_gain_vs_tied": predicted_tied - terms["prediction"],
                    "predicted_best_local": float(finite_values[local_best]),
                    "predicted_gain_vs_local": float(finite_values[local_best] - terms["prediction"]),
                    "measured_best_local": float(local_measured.min()),
                    "unmeasured_gap_below_best_local": float(local_measured.min() - terms["prediction"]),
                    "nearest_observed_tv": float(nearest.min()),
                    "nearest_broad_fit_tv": float(nearest[broad].min()),
                    "nearest_local_fit_tv": float(nearest[local].min()),
                    "nearest_observed_action": str(rows.action_id.iloc[indices[int(nearest.argmin())]]),
                    "maximum_bucket_weight": float(weight.max()),
                    "largest_bucket": str(arrays["bucket_names"][np.argmax(weight)]),
                    "maximum_continuation_epochs": float(np.max(LATE * frozen.physical_inventory * weight)),
                    "maximum_total_epochs": float(np.max(frozen.physical_inventory * (ALPHA * prefix + LATE * weight))),
                    "multistart_prediction_range": (
                        max(s["prediction"] for s in feasible) - min(s["prediction"] for s in feasible)
                    ),
                    "successful_starts": sum(s["success"] for s in solutions),
                    "best_success": best["success"],
                    "best_stationarity_residual": best["stationarity_residual"],
                }
                write_json_atomic(path, result)
                print(
                    json.dumps(
                        {
                            "fit": fit.held_state,
                            "prefix": state,
                            "prediction": terms["prediction"],
                            "largest_weight": weight.max(),
                        }
                    ),
                    flush=True,
                )
            results.append({key: value for key, value in result.items() if key not in ("identity", "best", "starts")})
            weights.append(
                {
                    "fit_held_state": fit.held_state,
                    "prefix_state": state,
                    **dict(zip(arrays["bucket_names"].tolist(), result["best"]["weights"], strict=True)),
                }
            )
            previous = raw_weights[raw_weights.fit_held_state.eq(fit.held_state) & raw_weights.prefix_state.eq(state)]
            assert len(previous) == 1
            old_weight = previous[list(arrays["bucket_names"])].to_numpy(float)[0]
            old_terms, _ = evaluate_policy(frozen, fit, prefix, old_weight)
            raw_comparison.append(
                {
                    "fit_held_state": fit.held_state,
                    "prefix_state": state,
                    "own_held_prefix": fit.held_state == state,
                    **{
                        key: value
                        for key, value in old_terms.items()
                        if key not in ("aggregate_weight", "anchor_components")
                    },
                    "anchored_optimum_prediction": result["prediction"],
                    "new_vs_old_optimum_tv": float(np.abs(old_weight - result["best"]["weights"]).sum() / 2),
                }
            )
    metrics, weights_frame = pd.DataFrame(results), pd.DataFrame(weights)
    metrics.to_csv(output / "raw_metrics.csv", index=False)
    weights_frame.to_csv(output / "raw_weights.csv", index=False)
    pd.DataFrame(raw_comparison).to_csv(output / "raw_mtp002_rescore.csv", index=False)
    stability, scores = [], []
    for state, group in weights_frame.groupby("prefix_state", sort=False):
        values = group[list(arrays["bucket_names"])].to_numpy(float)
        tv = np.abs(values[:, None] - values[None]).sum(axis=2) / 2
        pair_tv = tv[np.triu_indices(len(values), 1)]
        stability.append(
            {"prefix_state": state, "max_refit_tv": float(pair_tv.max()), "median_refit_tv": float(np.median(pair_tv))}
        )
        prefix = arrays["phase0_weight"][int(rows.index[rows.state_id.eq(state)][0])]
        for candidate, weight in zip(group.fit_held_state, values, strict=True):
            for fit in frozen.fits:
                terms, _ = evaluate_policy(frozen, fit, prefix, weight)
                scores.append(
                    {
                        "prefix_state": state,
                        "candidate_fit": candidate,
                        "scoring_fit": fit.held_state,
                        **{k: v for k, v in terms.items() if k not in ("aggregate_weight", "anchor_components")},
                    }
                )
    pd.DataFrame(stability).to_csv(output / "raw_stability.csv", index=False)
    pd.DataFrame(scores).to_csv(output / "refit_policy_scores.csv", index=False)
    return {
        "raw_cells": len(metrics),
        "raw_starts": len(metrics) * 8,
        "minimum_successful_starts": int(metrics.successful_starts.min()),
        "all_selected_optima_solver_success": bool(metrics.best_success.all()),
        "maximum_selected_stationarity_residual": float(metrics.best_stationarity_residual.max()),
        "maximum_multistart_prediction_range": float(metrics.multistart_prediction_range.max()),
        "maximum_guard_count": int(metrics.guard_count.max()),
        "raw_optima_measured": False,
    }


def report(output: Path, validation: dict[str, Any]) -> None:
    summary = pd.read_csv(output / "finite_summary.csv")
    finite = pd.read_csv(output / "finite_metrics.csv")
    raw = pd.read_csv(output / "raw_metrics.csv")
    primary = raw[raw.own_held_prefix]
    stability = pd.read_csv(output / "raw_stability.csv")
    lines = [
        "MTP-006 adds no fitted coefficient: it anchors the shared MARINER continuation response to a frozen "
        "MARINER aggregate prediction. Its phase correction is W(prefix, continuation) minus W(aggregate, aggregate). "
        "The results below follow an adaptive protocol prompted by the raw MTP-002 one-bucket optima.",
        "",
        "| Finite model | Tied selectable | RMSE | Pair RMSE | Spearman | Regret | Selected optimism |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.tied_selectable} | {row.rmse:.6f} | {row.pair_rmse:.6f} | "
            f"{row.spearman:.3f} | {row.regret:.6f} | {row.selected_optimism:.6f} |"
        )
    lines.append("")
    for row in summary[summary.model.eq("MTP-006")].itertuples():
        lines.append(
            f"For MTP-006 with tied selectable={row.tied_selectable}, mean predicted gain versus tied is "
            f"{row.predicted_gain_vs_tied:.6f} BPB and mean observed gain is {row.observed_gain_vs_tied:.6f}."
        )
    lines.append("")
    composite = finite[finite.model.eq("MTP-006") & ~finite.tied_selectable]
    lines.extend(
        [
            "",
            "All errors, regret, and optimism are bits per byte (BPB), with lower predicted and measured BPB preferred. "
            "Optimism is measured minus predicted BPB at the predicted-best measured candidate. "
            "Each mean weights six ordinary "
            "prefix configurations equally. Each primary composite uses the W fit that excludes its prefix family. "
            "Frozen A and the earlier A-plus-HPR composite use a different training bank and are external references.",
            "",
            f"The composite's six action-only choices are `{'; '.join(composite.selected_action)}`. "
            f"The worst observed regret is {composite.regret.max():.6f} BPB. "
            "`finite_metrics.csv` also reports tied-selectable choices, gains versus tied, and every per-state metric.",
            "",
            "| Raw held-prefix optimum | Prediction | A term | W phase correction | "
            "Largest weight | Nearest observed TV | Tied query prefix TV |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary.itertuples():
        lines.append(
            f"| {row.prefix_state} | {row.prediction:.6f} | {row.anchor_prediction:.6f} | "
            f"{row.phase_correction_bpb:+.6f} | {row.maximum_bucket_weight:.3%} | "
            f"{row.nearest_observed_tv:.3f} | {row.tied_query_nearest_training_prefix_tv:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Prefix | A | W actual | W tied aggregate | Composite |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in primary.itertuples():
        lines.append(
            f"| {row.prefix_state} | {row.anchor_prediction:.6f} | {row.w_policy_prediction:.6f} | "
            f"{row.w_tied_aggregate_prediction:.6f} | {row.prediction:.6f} |"
        )
    lines.extend(
        [
            "",
            f"The six primary optima have maximum bucket weights {primary.maximum_bucket_weight.min():.3%}-"
            f"{primary.maximum_bucket_weight.max():.3%}; maximum continuation exposure is "
            f"{primary.maximum_continuation_epochs.min():.2f}-{primary.maximum_continuation_epochs.max():.2f} epochs. "
            f"Across outer fits, maximum pairwise continuation TV at a fixed prefix is "
            f"{stability.max_refit_tv.min():.3f}-{stability.max_refit_tv.max():.3f}. "
            "TV is half the sum of absolute differences in mixture weights. The nearest observed continuation uses only "
            "already measured actions at the same prefix; tied-query prefix TV uses W's five training-prefix mixtures.",
            "",
            "Every raw optimum is unmeasured. `unmeasured_gap_below_best_local` is a predicted extrapolation gap, "
            "not measured optimism or improvement. The six outer fits share training data and are not "
            "independent repeats. "
            "The raw audit optimizes only the continuation, holding each prefix fixed, with eight "
            "deterministic starts on "
            "the full simplex and no KL constraint or epoch cap. All 36 fit/prefix cells and all 288 "
            "starts are retained. "
            "Only six cells use the fit that excludes the queried prefix; the remaining cells describe fit stability.",
            "",
            "Tied parity is algebraic: F(w,w)=A(w). The subtraction can still extrapolate because "
            "W(aggregate,aggregate) "
            "queries a prefix mixture absent from W's training data. `finite_composite_terms.csv`, "
            "`raw_metrics.csv`, and "
            "`raw_mtp002_rescore.csv` retain both W terms, their correction and cancellation ratio, "
            "and support distances. "
            "The ratio divides the sum of the absolute W terms by their absolute difference; tied "
            "corrections use a null ratio. "
            "A large ratio means most of the W level cancels, not that the remaining correction is accurate.",
            "",
            "A retains its frozen native inventory; W uses the realized physical prefix and continuation exposures "
            "at alpha=2400/3007. The verified inventory scale differs by approximately 0.5 ppm. "
            "A retains its frozen component weights, which differ from the prepared canonical weights by at most "
            f"{validation['maximum_component_weight_difference']:.3g}; replacing only those weights would change "
            f"the 66 anchor predictions by at most {validation['maximum_anchor_weight_sensitivity_bpb']:.3g} BPB. "
            "Both weight vectors are saved in `component_weights.csv`; no replacement is used in this comparison. "
            "The composite differs from the earlier HPR composite only in the source of its phase residual. "
            "It introduces no new temporal identification and does not identify a two-phase advantage without matched "
            "endpoint measurements. Neither finite-menu improvements nor a plausible raw optimum "
            "justify promotion alone.",
            "",
            "`selected_components.csv` audits all seven measured components of each selected finite "
            "candidate versus tied. "
            "`anchor_component_predictions.csv` contains A's component predictions. W predicts only aggregate BPB, "
            "so no composite component prediction is produced. Prefix replicas, the hardware bridge, "
            "and the outcome-selected "
            "prefix remain outside all primary scoring.",
            "",
            f"Validation: tied-parity error {validation['max_tied_parity_error']:.3g}, analytic-gradient error "
            f"{validation['max_gradient_error']:.3g}, frozen-prediction replay error "
            f"{validation['max_frozen_prediction_replay_error']:.3g}. "
            f"At least {validation['minimum_successful_starts']} of eight starts converged in every cell; "
            "selected-optimum stationarity residual is at most "
            f"{validation['maximum_selected_stationarity_residual']:.3g}. "
            "These are numerical checks and do not validate unmeasured policies.",
            "",
            "Reproduce with `uv run --offline audit_mariner_anchored_continuation_20260912.py` from "
            "the script directory. "
            "[PROTOCOL.md](PROTOCOL.md) fixes the comparison. `manifest.json` pins sources, inputs, and every output; "
            "per-cell results are resumable by those identities. No new parameters were fitted and no "
            "remote data, training, "
            "evaluation, registry, or ledger was changed by this script.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen(args.output)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] == frozen.identity and all(
            sha(args.output / p) == h for p, h in previous["outputs"].items()
        ):
            print("Existing anchored audit verified; no recomputation needed.")
            return
    with threadpool_limits(limits=1):
        validation = structural_checks(frozen, args.output)
        validation.update(finite_candidates(frozen, args.output))
        validation.update(raw_continuations(frozen, args.output))
    validation.update({"no_new_coefficients": True, "no_remote_calls": True, "identity": frozen.identity})
    write_json_atomic(args.output / "validation.json", validation)
    report(args.output, validation)
    snapshots = args.output / "sources"
    snapshots.mkdir(exist_ok=True)
    for path in frozen.identity:
        if path.endswith(".py"):
            shutil.copyfile(path, snapshots / Path(path).name)
    outputs = {
        str(path.relative_to(args.output)): sha(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    write_json_atomic(manifest_path, {"identity": frozen.identity, "outputs": outputs})
    print(pd.read_csv(args.output / "finite_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
