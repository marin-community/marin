# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Evaluate compact Weibull coverage with exact finite-potential transport."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from run_phase_transport_synthesis import (
    HELDOUT_PATH,
    RANDOM_SEEDS,
    RIDGE_GRID,
    TARGETS,
    Panel,
    PhaseFit,
    direction_splits,
    exact_fiber_predictions,
    fit_phase,
    load_heldouts,
    load_panel,
    metric_dict,
    plot_scatter,
    predict_phase,
    random_splits,
    summarize_fibers,
    summarize_heldouts,
)
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "reference_outputs" / "cross_session_compact_transport_20260723"
MODEL_IDS = ("compact_zero", "compact_fpt_global")
RATE_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
POWER_GRID = (0.35, 0.5, 0.7, 1.0)
AGGREGATE_RIDGE_GRID = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass(frozen=True)
class CompactShape:
    rate: float
    power: float
    ridge: float


@dataclass(frozen=True)
class CompactFit:
    shape: CompactShape
    intercept: float
    coefficients: np.ndarray
    effective_df: float


def compact_design(
    panel: Panel,
    weights: np.ndarray,
    shape: CompactShape,
) -> np.ndarray:
    mixture = np.asarray(weights, dtype=float)
    if mixture.ndim == 1:
        mixture = mixture[None, :]
    epochs = mixture * (panel.c0 + panel.c1)[None, :]
    benefit = -np.expm1(-(np.maximum(shape.rate * epochs, 0.0) ** shape.power))
    replay = np.sum(np.maximum(epochs - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return np.hstack([-benefit, replay])


def fit_compact(
    panel: Panel,
    weights: np.ndarray,
    target: np.ndarray,
    shape: CompactShape,
) -> CompactFit:
    design = compact_design(panel, weights, shape)
    mean = design.mean(axis=0)
    centered = design - mean[None, :]
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    scale = np.where(scale < 1e-10, 1.0, scale)
    standardized = centered / scale[None, :]
    target_mean = float(np.mean(target))
    target_centered = target - target_mean
    penalty = math.sqrt(shape.ridge) * np.eye(design.shape[1])
    augmented = np.vstack([standardized, penalty])
    rhs = np.concatenate([target_centered, np.zeros(design.shape[1])])
    standardized_coefficients, _ = nnls(
        augmented,
        rhs,
        maxiter=max(1000, 5 * design.shape[1]),
    )
    coefficients = standardized_coefficients / scale
    intercept = target_mean - float(mean @ coefficients)
    active = np.flatnonzero(standardized_coefficients > 1e-9)
    if len(active):
        active_design = standardized[:, active]
        gram = active_design.T @ active_design
        effective_df = 1.0 + float(np.trace(np.linalg.pinv(gram + shape.ridge * np.eye(len(active))) @ gram))
    else:
        effective_df = 1.0
    return CompactFit(shape, intercept, coefficients, effective_df)


def predict_compact(
    panel: Panel,
    fit: CompactFit,
    weights: np.ndarray,
) -> np.ndarray:
    return fit.intercept + compact_design(panel, weights, fit.shape) @ fit.coefficients


def compact_potential(
    panel: Panel,
    fit: CompactFit,
    weights: np.ndarray,
) -> np.ndarray:
    design = compact_design(panel, weights, fit.shape)
    return design @ fit.coefficients


def compact_transport_design(
    panel: Panel,
    fit: CompactFit,
    weights: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    phase0 = weights[:, 0]
    phase1 = weights[:, 1]
    aggregate = panel.alpha0 * phase0 + panel.alpha1 * phase1
    potential0 = compact_potential(panel, fit, phase0)
    potential1 = compact_potential(panel, fit, phase1)
    potential_aggregate = compact_potential(panel, fit, aggregate)
    odd = panel.alpha0 * panel.alpha1 * (potential0 - potential1)
    jensen = panel.alpha0 * potential0 + panel.alpha1 * potential1 - potential_aggregate
    return (
        np.column_stack([odd, jensen]),
        ("odd::compact_total", "jensen::compact_total"),
        np.asarray([False, True]),
    )


def select_compact_shape(panel: Panel, target_name: str) -> tuple[CompactShape, pd.DataFrame]:
    target = panel.one_targets[target_name]
    splits = random_splits(np.arange(len(target)), 20260730)
    rows: list[dict[str, float]] = []
    for rate in RATE_GRID:
        for power in POWER_GRID:
            for ridge in AGGREGATE_RIDGE_GRID:
                shape = CompactShape(rate, power, ridge)
                residuals: list[np.ndarray] = []
                for train, test in splits:
                    fit = fit_compact(
                        panel,
                        panel.one_weights[train, 0],
                        target[train],
                        shape,
                    )
                    residuals.append(predict_compact(panel, fit, panel.one_weights[test, 0]) - target[test])
                rows.append(
                    {
                        "rate": rate,
                        "power": power,
                        "ridge": ridge,
                        "oof_rmse": float(np.sqrt(np.mean(np.concatenate(residuals) ** 2))),
                    }
                )
    search = pd.DataFrame(rows).sort_values(["oof_rmse", "ridge", "rate", "power"])
    best = search.iloc[0]
    return (
        CompactShape(
            rate=float(best["rate"]),
            power=float(best["power"]),
            ridge=float(best["ridge"]),
        ),
        search.reset_index(drop=True),
    )


def select_phase_ridge(
    panel: Panel,
    target_name: str,
    shape: CompactShape,
    outer_train: np.ndarray,
    seed: int,
) -> float:
    errors: dict[float, list[np.ndarray]] = {ridge: [] for ridge in RIDGE_GRID}
    for inner_train, inner_test in random_splits(outer_train, seed, folds=4):
        aggregate_fit = fit_compact(
            panel,
            panel.one_weights[inner_train, 0],
            panel.one_targets[target_name][inner_train],
            shape,
        )
        train_design, names, constrained = compact_transport_design(
            panel,
            aggregate_fit,
            panel.two_weights[inner_train],
        )
        test_design, _, _ = compact_transport_design(
            panel,
            aggregate_fit,
            panel.two_weights[inner_test],
        )
        train_delta = panel.two_targets[target_name][inner_train] - panel.one_targets[target_name][inner_train]
        test_delta = panel.two_targets[target_name][inner_test] - panel.one_targets[target_name][inner_test]
        for ridge in RIDGE_GRID:
            phase_fit = fit_phase(
                panel,
                "compact_fpt_global",
                train_design,
                train_delta,
                constrained,
                ridge,
                names,
            )
            errors[ridge].append(predict_phase(phase_fit, test_design) - test_delta)
    return min(
        RIDGE_GRID,
        key=lambda ridge: (
            float(np.sqrt(np.mean(np.concatenate(errors[ridge]) ** 2))),
            ridge,
        ),
    )


def run_cv(
    panel: Panel,
    shapes: dict[str, CompactShape],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    schemes = [
        *[(f"random_seed_{seed}", seed, random_splits(np.arange(280), seed)) for seed in RANDOM_SEEDS],
        ("direction", 20260728, direction_splits(panel)),
    ]
    for target_name in TARGETS:
        shape = shapes[target_name]
        for model_id in MODEL_IDS:
            for scheme, seed, splits in schemes:
                for fold, (train, test) in enumerate(splits):
                    compact_fit = fit_compact(
                        panel,
                        panel.one_weights[train, 0],
                        panel.one_targets[target_name][train],
                        shape,
                    )
                    predicted_one = predict_compact(
                        panel,
                        compact_fit,
                        panel.one_weights[test, 0],
                    )
                    if model_id == "compact_zero":
                        ridge = 0.0
                        phase_fit = PhaseFit(
                            model_id,
                            np.zeros(0),
                            (),
                            0.0,
                            0.0,
                            0,
                            1.0,
                        )
                        predicted_delta = np.zeros(len(test))
                    else:
                        ridge = select_phase_ridge(
                            panel,
                            target_name,
                            shape,
                            train,
                            seed + fold + 211,
                        )
                        train_design, names, constrained = compact_transport_design(
                            panel,
                            compact_fit,
                            panel.two_weights[train],
                        )
                        test_design, _, _ = compact_transport_design(
                            panel,
                            compact_fit,
                            panel.two_weights[test],
                        )
                        phase_fit = fit_phase(
                            panel,
                            model_id,
                            train_design,
                            panel.two_targets[target_name][train] - panel.one_targets[target_name][train],
                            constrained,
                            ridge,
                            names,
                        )
                        predicted_delta = predict_phase(phase_fit, test_design)
                    observed_delta = panel.two_targets[target_name][test] - panel.one_targets[target_name][test]
                    for local, row_index in enumerate(test):
                        predictions.append(
                            {
                                "target": target_name,
                                "model_id": model_id,
                                "scheme": scheme,
                                "fold": fold,
                                "row_index": int(row_index),
                                "group_id": panel.group_ids[row_index],
                                "observed_one": panel.one_targets[target_name][row_index],
                                "predicted_one": predicted_one[local],
                                "observed_delta": observed_delta[local],
                                "predicted_delta": predicted_delta[local],
                                "observed_two": panel.two_targets[target_name][row_index],
                                "predicted_two": predicted_one[local] + predicted_delta[local],
                            }
                        )
                    fits.append(
                        {
                            "target": target_name,
                            "model_id": model_id,
                            "scheme": scheme,
                            "fold": fold,
                            "selected_phase_ridge": ridge,
                            "aggregate_rate": shape.rate,
                            "aggregate_power": shape.power,
                            "aggregate_ridge": shape.ridge,
                            "aggregate_effective_df": compact_fit.effective_df,
                            "phase_effective_df": phase_fit.effective_df,
                            "phase_rank": phase_fit.rank,
                            "phase_condition_number": phase_fit.condition_number,
                        }
                    )
                    for name, coefficient in zip(
                        phase_fit.feature_names,
                        phase_fit.coefficients,
                        strict=True,
                    ):
                        coefficients.append(
                            {
                                "target": target_name,
                                "model_id": model_id,
                                "scheme": scheme,
                                "fold": fold,
                                "feature": name,
                                "coefficient": coefficient,
                            }
                        )
    return pd.DataFrame(predictions), pd.DataFrame(fits), pd.DataFrame(coefficients)


def summarize_cv(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model_id, scheme), group in predictions.groupby(
        ["target", "model_id", "scheme"],
        sort=False,
    ):
        for response in ("one", "delta", "two"):
            rows.append(
                {
                    "target": target,
                    "model_id": model_id,
                    "scheme": scheme,
                    "response": response,
                    **metric_dict(
                        group[f"observed_{response}"].to_numpy(float),
                        group[f"predicted_{response}"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def full_phase_fit(
    panel: Panel,
    target_name: str,
    shape: CompactShape,
    aggregate_fit: CompactFit,
) -> tuple[PhaseFit, float]:
    ridge = select_phase_ridge(
        panel,
        target_name,
        shape,
        np.arange(280),
        20260731,
    )
    design, names, constrained = compact_transport_design(
        panel,
        aggregate_fit,
        panel.two_weights,
    )
    fit = fit_phase(
        panel,
        "compact_fpt_global",
        design,
        panel.two_targets[target_name] - panel.one_targets[target_name],
        constrained,
        ridge,
        names,
    )
    return fit, ridge


def run_heldouts(
    panel: Panel,
    shapes: dict[str, CompactShape],
    heldouts: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    parameters: list[dict[str, object]] = []
    aggregate_weights = panel.alpha0 * heldout_weights[:, 0] + panel.alpha1 * heldout_weights[:, 1]
    for target_name in TARGETS:
        shape = shapes[target_name]
        aggregate_fit = fit_compact(
            panel,
            panel.one_weights[:, 0],
            panel.one_targets[target_name],
            shape,
        )
        predicted_aggregate = predict_compact(
            panel,
            aggregate_fit,
            aggregate_weights,
        )
        phase_fit, phase_ridge = full_phase_fit(
            panel,
            target_name,
            shape,
            aggregate_fit,
        )
        heldout_phase_design, _, _ = compact_transport_design(
            panel,
            aggregate_fit,
            heldout_weights,
        )
        for model_id in MODEL_IDS:
            if model_id == "compact_zero":
                predicted_phase = np.zeros(len(heldouts))
                selected_phase_fit = PhaseFit(
                    model_id,
                    np.zeros(0),
                    (),
                    0.0,
                    0.0,
                    0,
                    1.0,
                )
            else:
                predicted_phase = predict_phase(
                    phase_fit,
                    heldout_phase_design,
                )
                selected_phase_fit = phase_fit
            result = heldouts.copy()
            result["fit_target"] = target_name
            result["model_id"] = model_id
            result["predicted_aggregate"] = predicted_aggregate
            result["predicted_phase_delta"] = predicted_phase
            result["predicted_target"] = predicted_aggregate + predicted_phase
            result["observed_target"] = heldouts[target_name].to_numpy(float)
            result["residual"] = result["predicted_target"] - result["observed_target"]
            result["optimism"] = result["observed_target"] - result["predicted_target"]
            result["phase_tv"] = 0.5 * np.sum(
                np.abs(heldout_weights[:, 1] - heldout_weights[:, 0]),
                axis=1,
            )
            result["aggregate_hash"] = [
                sha256(np.round(weights, 10).astype(np.float64).tobytes()).hexdigest() for weights in aggregate_weights
            ]
            frames.append(result)
            implied_recency = (
                panel.alpha0 + selected_phase_fit.coefficients[0] * panel.alpha0 * panel.alpha1
                if len(selected_phase_fit.coefficients)
                else panel.alpha0
            )
            parameters.append(
                {
                    "target": target_name,
                    "model_id": model_id,
                    "aggregate_rate": shape.rate,
                    "aggregate_power": shape.power,
                    "aggregate_ridge": shape.ridge,
                    "aggregate_effective_df": aggregate_fit.effective_df,
                    "phase_ridge": phase_ridge if model_id == "compact_fpt_global" else 0.0,
                    "phase_effective_df": selected_phase_fit.effective_df,
                    "phase_condition_number": selected_phase_fit.condition_number,
                    "implied_recency_share": implied_recency,
                    "phase_coefficients_json": json.dumps(
                        dict(
                            zip(
                                selected_phase_fit.feature_names,
                                selected_phase_fit.coefficients.tolist(),
                                strict=True,
                            )
                        ),
                        sort_keys=True,
                    ),
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(parameters)


def write_report(
    shape_search: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
    fiber_metrics: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    cv = cv_metrics.loc[cv_metrics["scheme"].str.startswith("random_seed") & (cv_metrics["response"] == "delta")]
    cv_summary = (
        cv.groupby(["target", "model_id"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            regret_at_1=("regret_at_1", "mean"),
            calibration_slope=("observed_on_predicted_slope", "mean"),
        )
        .sort_values(["target", "rmse"])
    )
    heldout = heldout_metrics.loc[heldout_metrics["slice"] == "coordinate_disjoint_target_matched"].sort_values(
        ["target", "rmse"]
    )
    fibers = fiber_metrics.loc[fiber_metrics["slice"] == "all_exact_aggregate_fibers"].sort_values(["target", "rmse"])
    selected_shapes = shape_search.loc[shape_search["selected"]]
    lines = [
        "# Compact finite-potential transport",
        "",
        "This is exposed local development evidence, not confirmation.",
        "",
        "## Selected one-phase shapes",
        "",
        selected_shapes.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired phase-delta CV",
        "",
        cv_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Coordinate-disjoint target-matched heldouts",
        "",
        heldout.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Exact aggregate fibers",
        "",
        fibers.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full-fit parameters",
        "",
        parameters.to_markdown(index=False, floatfmt=".6f"),
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    shapes: dict[str, CompactShape] = {}
    searches: list[pd.DataFrame] = []
    for target_name in TARGETS:
        shape, search = select_compact_shape(panel, target_name)
        shapes[target_name] = shape
        search = search.copy()
        search["target"] = target_name
        search["selected"] = False
        search.loc[0, "selected"] = True
        searches.append(search)
    shape_search = pd.concat(searches, ignore_index=True)

    cv_predictions, cv_fits, cv_coefficients = run_cv(panel, shapes)
    cv_metrics = summarize_cv(cv_predictions)
    heldouts, heldout_weights = load_heldouts(panel)
    heldout_predictions, parameters = run_heldouts(
        panel,
        shapes,
        heldouts,
        heldout_weights,
    )
    heldout_metrics = summarize_heldouts(heldout_predictions)
    fibers = exact_fiber_predictions(heldout_predictions)
    fiber_metrics = summarize_fibers(fibers)

    shape_search.to_csv(OUTPUT / "aggregate_shape_search.csv", index=False)
    cv_predictions.to_csv(OUTPUT / "paired_cv_predictions.csv", index=False)
    cv_metrics.to_csv(OUTPUT / "paired_cv_metrics.csv", index=False)
    cv_fits.to_csv(OUTPUT / "paired_cv_fit_audit.csv", index=False)
    cv_coefficients.to_csv(OUTPUT / "paired_cv_coefficients.csv", index=False)
    heldout_predictions.to_csv(OUTPUT / "heldout_predictions.csv", index=False)
    heldout_metrics.to_csv(OUTPUT / "heldout_metrics.csv", index=False)
    parameters.to_csv(OUTPUT / "full_fit_parameters.csv", index=False)
    fibers.to_csv(OUTPUT / "exact_fiber_predictions.csv", index=False)
    fiber_metrics.to_csv(OUTPUT / "exact_fiber_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "evaluation_round": "compact_finite_potential_transport_batch_2",
                "candidate_ids": json.dumps(list(MODEL_IDS)),
                "fit_data": "Delphi 3e18 paired one/two-phase swarms",
                "heldout_data": str(HELDOUT_PATH),
                "heldout_status": "exposed development",
                "inspiration": "Batch-1 aggregate-spine failure and prior compact raw-optimum health",
                "freeze_artifact": str(HERE / "BATCH_2_PROTOCOL.json"),
            }
        ]
    ).to_csv(OUTPUT / "data_use_ledger.csv", index=False)
    write_report(
        shape_search,
        cv_metrics,
        heldout_metrics,
        fiber_metrics,
        parameters,
    )
    plot_scatter(
        cv_predictions.loc[cv_predictions["scheme"].str.startswith("random_seed")],
        "observed_delta",
        "predicted_delta",
        "Compact finite-potential transport paired OOF",
        "paired_delta_oof_scatter.html",
    )
    plot_scatter(
        heldout_predictions,
        "observed_target",
        "predicted_target",
        "Compact finite-potential transport heldouts",
        "heldout_calibration_scatter.html",
    )
    plot_scatter(
        fibers.rename(columns={"fit_target": "target"}),
        "observed_fiber_delta",
        "predicted_fiber_delta",
        "Compact finite-potential transport exact fibers",
        "exact_fiber_delta_scatter.html",
    )
    print(OUTPUT / "report.md")


if __name__ == "__main__":
    main()
