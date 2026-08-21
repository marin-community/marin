# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Falsify pointwise-authority saturating phase control in staged gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    saturating_phase_control_20260730 as saturating,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "saturating_phase_control_20260730"
MODEL = "pointwise_authority_saturation"
ABLATION = "linear_control"
MODELS = (MODEL, ABLATION)
DESIGN_MINIMUM_P95_RATIO = 0.3
DESIGN_MINIMUM_SIGN_FRACTION = 0.25
EVEN_CONTRIBUTION_FLOOR = 1e-6
WSD_MAXIMUM_OPTIMUM_DISTANCE = 0.0707
WSD_MINIMUM_GAIN = 0.0072
WSD_MAXIMUM_GAIN = 0.0120
WSD_MAXIMUM_TIED_FIBER_GAIN = 0.004633
WSD_MAXIMUM_BLOCKED_SIGMA = 51.93
TRUE_OPTIMUM = (0.10, 0.50)
BOOTSTRAP_REPLICATES = 20_000
PAIR_THRESHOLDS = {
    "uncheatable": {
        "delta_rmse": 0.009243,
        "delta_bias": 0.000324,
        "sign_accuracy": 201 / 238,
        "asymmetric_rmse": 0.009402,
    },
    "table9": {
        "delta_rmse": 0.020460,
        "delta_bias": 0.000283,
        "sign_accuracy": 200 / 238,
        "asymmetric_rmse": 0.018696,
    },
}


def fit_full_aggregate_300m(target: str) -> tuple[benchmark.Dataset, replay_control.AggregateFitted]:
    dataset = benchmark.load_300m(target)
    geometry = benchmark.geometry_300m(dataset)
    folds = benchmark.grouped_folds(dataset.frame, seed=20_000, n_splits=3)
    aggregate = replay_control.fit_aggregate(dataset.weights, dataset.y, geometry, folds)
    return dataset, aggregate


def fit_full_aggregate_wsd() -> tuple[wsd80.Panel, replay_control.AggregateFitted]:
    panel = wsd80.load_surface()
    geometry = replay_control.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    tied = np.flatnonzero(replay_control.tied_rows(panel.weights))
    folds = benchmark.wsd_folds(panel.weights, tied, 3, seed=20_000, protocol="blocked")
    aggregate = replay_control.fit_aggregate(panel.weights, panel.y, geometry, folds)
    return panel, aggregate


def design_row(
    panel: str,
    weights: np.ndarray,
    aggregate: replay_control.AggregateFitted,
) -> dict[str, float | int | str | bool]:
    tied = replay_control.tied_rows(weights)
    control, _authority, ratio = saturating.control_statistics(weights, aggregate)
    asymmetric_control = control[~tied]
    asymmetric_ratio = ratio[~tied]
    tied_design = saturating.phase_design_matrix(weights[tied], aggregate, tau=0.5)
    positive = float(np.mean(asymmetric_control > 0.0))
    negative = float(np.mean(asymmetric_control < 0.0))
    p95 = float(np.quantile(asymmetric_ratio, 0.95))
    tied_maximum = float(np.max(np.abs(tied_design)))
    return {
        "panel": panel,
        "n_tied": int(tied.sum()),
        "n_asymmetric": int((~tied).sum()),
        "p95_absolute_control_fraction": p95,
        "positive_control_fraction": positive,
        "negative_control_fraction": negative,
        "maximum_tied_phase_column": tied_maximum,
        "passes_ratio": p95 >= DESIGN_MINIMUM_P95_RATIO,
        "passes_signs": min(positive, negative) >= DESIGN_MINIMUM_SIGN_FRACTION,
        "passes_tied_zero": tied_maximum <= 1e-12,
    }


def run_design_screen(output_dir: Path) -> bool:
    rows = []
    panel, aggregate = fit_full_aggregate_wsd()
    rows.append(design_row("wsd80", panel.weights, aggregate))
    for target in benchmark.TARGETS:
        dataset, aggregate = fit_full_aggregate_300m(target)
        rows.append(design_row(f"300m_{target}", dataset.weights, aggregate))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "design_screen.csv", index=False)
    passed = bool(frame[["passes_ratio", "passes_signs", "passes_tied_zero"]].to_numpy(dtype=bool).all())
    (output_dir / "design_gate.json").write_text(json.dumps({"passed": passed, "rows": rows}, indent=2) + "\n")
    return passed


def fit_wsd_model(
    panel: wsd80.Panel,
    indices: np.ndarray,
    seed: int,
    protocol: str,
    linear: bool,
) -> saturating.Fitted:
    weights = panel.weights[indices]
    target = panel.y[indices]
    tied = np.flatnonzero(replay_control.tied_rows(weights))
    aggregate_folds = benchmark.wsd_folds(weights, tied, 3, seed, protocol)
    phase_folds = benchmark.wsd_folds(weights, np.arange(len(weights)), 3, seed, protocol)
    geometry = replay_control.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    aggregate = replay_control.fit_aggregate(weights, target, geometry, aggregate_folds)
    if linear:
        return saturating.fit_fixed_tau(aggregate, weights, target, np.inf)
    return saturating.fit(aggregate, weights, target, phase_folds)


def wsd_metric_row(
    protocol: str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: float,
) -> dict[str, float | str]:
    residual = predicted - observed
    return {
        "protocol": protocol,
        "model": model,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rmse_sigma": float(np.sqrt(np.mean(residual**2)) / sigma),
        "median_absolute_sigma": float(np.median(np.abs(residual)) / sigma),
    }


def fitted_parameters(
    protocol: str,
    model: str,
    fold: int,
    fitted: saturating.Fitted,
) -> dict[str, float | int | str]:
    return {
        "protocol": protocol,
        "model": model,
        "fold": fold,
        "tau": fitted.tau,
        "phase_control": fitted.phase_control,
        "phase_information": fitted.phase_information,
        "replay_jensen": fitted.replay_jensen,
    }


def full_diagnostic(
    protocol: str,
    model: str,
    fitted: saturating.Fitted,
    panel: wsd80.Panel,
) -> dict[str, float | str | bool]:
    optimum = benchmark.predicted_optimum(
        fitted.predict,
        benchmark.OPTIMUM_GRID,
        wsd80.REALIZED_PHASE_0_FRACTION,
    )
    advantage = benchmark.two_phase_advantage(fitted.predict, benchmark.OPTIMUM_GRID)
    fiber = benchmark.predicted_phase_gain(
        fitted.predict,
        aggregate=0.30,
        resolution=benchmark.OPTIMUM_GRID,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    design = saturating.phase_design_matrix(panel.weights, fitted.aggregate, fitted.tau)
    even_contribution = design[:, 1:] @ fitted.phase_coefficients[1:]
    return {
        "protocol": protocol,
        "model": model,
        "tau": fitted.tau,
        **{f"optimum_{key}": value for key, value in optimum.items()},
        **{f"advantage_{key}": value for key, value in advantage.items()},
        "optimum_distance": float(
            np.hypot(
                optimum["phase_0"] - TRUE_OPTIMUM[0],
                optimum["phase_1"] - TRUE_OPTIMUM[1],
            )
        ),
        "phase_gain_at_0.30": fiber["phase_gain"],
        "maximum_even_contribution": float(np.max(np.abs(even_contribution))),
        "even_costs_active": bool(np.max(np.abs(even_contribution)) >= EVEN_CONTRIBUTION_FLOOR),
    }


def run_wsd_screen(output_dir: Path) -> bool:
    panel = wsd80.load_surface()
    sigma = wsd80.training_seed_sigma(wsd80.load_fiber_replicates())
    metrics = []
    parameters = []
    diagnostics = []
    predictions = []
    for protocol in ("random", "blocked"):
        outer = benchmark.wsd_folds(
            panel.weights,
            np.arange(len(panel.y)),
            3,
            seed=0,
            protocol=protocol,
        )
        oof = {model: np.full(len(panel.y), np.nan) for model in MODELS}
        for fold, (train, test) in enumerate(outer):
            for model in MODELS:
                fitted = fit_wsd_model(panel, train, 10_000 + fold, protocol, linear=model == ABLATION)
                oof[model][test] = fitted.predict(panel.weights[test])
                parameters.append(fitted_parameters(protocol, model, fold, fitted))
        for model in MODELS:
            metrics.append(wsd_metric_row(protocol, model, panel.y, oof[model], sigma))
            predictions.append(
                pd.DataFrame(
                    {
                        "protocol": protocol,
                        "model": model,
                        "row": np.arange(len(panel.y)),
                        "observed": panel.y,
                        "predicted": oof[model],
                    }
                )
            )
            fitted = fit_wsd_model(
                panel,
                np.arange(len(panel.y)),
                seed=20_000,
                protocol=protocol,
                linear=model == ABLATION,
            )
            parameters.append(fitted_parameters(protocol, model, -1, fitted))
            diagnostics.append(full_diagnostic(protocol, model, fitted, panel))
    metrics_frame = pd.DataFrame(metrics)
    diagnostics_frame = pd.DataFrame(diagnostics)
    pd.DataFrame(parameters).to_csv(output_dir / "parameters_wsd80.csv", index=False)
    metrics_frame.to_csv(output_dir / "metrics_wsd80.csv", index=False)
    diagnostics_frame.to_csv(output_dir / "diagnostics_wsd80.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions_wsd80.csv", index=False)

    candidate = diagnostics_frame.loc[
        diagnostics_frame["protocol"].eq("blocked") & diagnostics_frame["model"].eq(MODEL)
    ].iloc[0]
    ablation = diagnostics_frame.loc[
        diagnostics_frame["protocol"].eq("blocked") & diagnostics_frame["model"].eq(ABLATION)
    ].iloc[0]
    blocked_sigma = float(
        metrics_frame.loc[
            metrics_frame["protocol"].eq("blocked") & metrics_frame["model"].eq(MODEL),
            "rmse_sigma",
        ].iloc[0]
    )
    interior = 0.0 < float(candidate["optimum_phase_0"]) < 1.0 and 0.0 < float(candidate["optimum_phase_1"]) < 1.0
    no_worse = (
        float(candidate["optimum_distance"]) <= float(ablation["optimum_distance"])
        and abs(float(candidate["advantage_two_phase_gain"]) - 0.009594)
        <= abs(float(ablation["advantage_two_phase_gain"]) - 0.009594)
        and blocked_sigma
        <= float(
            metrics_frame.loc[
                metrics_frame["protocol"].eq("blocked") & metrics_frame["model"].eq(ABLATION),
                "rmse_sigma",
            ].iloc[0]
        )
    )
    improved = (
        float(candidate["optimum_distance"]) < float(ablation["optimum_distance"])
        or abs(float(candidate["advantage_two_phase_gain"]) - 0.009594)
        < abs(float(ablation["advantage_two_phase_gain"]) - 0.009594)
        or blocked_sigma
        < float(
            metrics_frame.loc[
                metrics_frame["protocol"].eq("blocked") & metrics_frame["model"].eq(ABLATION),
                "rmse_sigma",
            ].iloc[0]
        )
    )
    clauses = {
        "interior_optimum": interior,
        "optimum_distance": float(candidate["optimum_distance"]) <= WSD_MAXIMUM_OPTIMUM_DISTANCE,
        "two_phase_gain": WSD_MINIMUM_GAIN <= float(candidate["advantage_two_phase_gain"]) <= WSD_MAXIMUM_GAIN,
        "tied_fiber_gain": float(candidate["phase_gain_at_0.30"]) <= WSD_MAXIMUM_TIED_FIBER_GAIN,
        "blocked_rmse": blocked_sigma <= WSD_MAXIMUM_BLOCKED_SIGMA,
        "even_costs_active": bool(candidate["even_costs_active"]),
        "nested_ablation": no_worse and improved,
    }
    passed = all(clauses.values())
    (output_dir / "wsd_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "clauses": clauses,
                "candidate": candidate.to_dict(),
                "ablation": ablation.to_dict(),
                "blocked_rmse_sigma": blocked_sigma,
            },
            indent=2,
        )
        + "\n"
    )
    return passed


def fit_300m_model(
    dataset: benchmark.Dataset,
    train: np.ndarray,
    seed: int,
    linear: bool,
) -> saturating.Fitted:
    training_frame = dataset.frame.iloc[train].reset_index(drop=True)
    weights = dataset.weights[train]
    target = dataset.y[train]
    folds = benchmark.local_folds(training_frame, seed, 3)
    aggregate = replay_control.fit_aggregate(
        weights,
        target,
        benchmark.geometry_300m(dataset),
        folds,
    )
    paired = benchmark.paired_tied_target(dataset.frame, dataset.y)[train]
    if linear:
        return saturating.fit_fixed_tau(
            aggregate,
            weights,
            target,
            np.inf,
            paired_tied_target=paired,
        )
    return saturating.fit(
        aggregate,
        weights,
        target,
        folds,
        paired_tied_target=paired,
    )


def paired_arrays(
    dataset: benchmark.Dataset,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indexed = dataset.frame.reset_index().set_index(["phase_correspondence_key", "policy_family"])["index"]
    keys = sorted(
        set(dataset.frame.loc[dataset.frame["policy_family"].eq("single_phase"), "phase_correspondence_key"])
        & set(dataset.frame.loc[dataset.frame["policy_family"].eq("two_phase"), "phase_correspondence_key"])
    )
    one = np.asarray([indexed.loc[(key, "single_phase")] for key in keys], dtype=int)
    two = np.asarray([indexed.loc[(key, "two_phase")] for key in keys], dtype=int)
    asymmetric = ~replay_control.tied_rows(dataset.weights[two])
    one = one[asymmetric]
    two = two[asymmetric]
    return dataset.y[two] - dataset.y[one], predicted[two] - predicted[one]


def bootstrap_rmse_improvement(
    observed: np.ndarray,
    candidate: np.ndarray,
    ablation: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(observed), size=(BOOTSTRAP_REPLICATES, len(observed)))
    candidate_error = candidate - observed
    ablation_error = ablation - observed
    candidate_rmse = np.sqrt(np.mean(candidate_error[draws] ** 2, axis=1))
    ablation_rmse = np.sqrt(np.mean(ablation_error[draws] ** 2, axis=1))
    improvement = ablation_rmse - candidate_rmse
    return {
        "improvement": float(np.sqrt(np.mean(ablation_error**2)) - np.sqrt(np.mean(candidate_error**2))),
        "bootstrap_se": float(np.std(improvement, ddof=1)),
        "probability_positive": float(np.mean(improvement > 0.0)),
    }


def run_300m_gate(output_dir: Path) -> bool:
    metric_rows = []
    pair_rows = []
    parameter_rows = []
    prediction_frames = []
    averaged_predictions: dict[tuple[str, str], list[np.ndarray]] = {}
    for target in benchmark.TARGETS:
        dataset = benchmark.load_300m(target)
        for seed in (0, 1):
            outer = benchmark.grouped_folds(dataset.frame, seed, 3)
            oof = {model: np.full(dataset.n, np.nan) for model in MODELS}
            for fold, (train, test) in enumerate(outer):
                for model in MODELS:
                    fitted = fit_300m_model(
                        dataset,
                        train,
                        seed=10_000 + seed * 100 + fold,
                        linear=model == ABLATION,
                    )
                    oof[model][test] = fitted.predict(dataset.weights[test])
                    parameter_rows.append(
                        {
                            "target": target,
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "tau": fitted.tau,
                            "phase_control": fitted.phase_control,
                            "phase_information": fitted.phase_information,
                            "replay_jensen": fitted.replay_jensen,
                        }
                    )
            for model in MODELS:
                metric_rows.append(
                    benchmark.metric_row(
                        f"300m_{target}",
                        model,
                        seed,
                        dataset.y,
                        oof[model],
                        dataset.weights,
                        outer,
                    )
                )
                pair_rows.append(
                    benchmark.paired_metric_row(
                        f"300m_{target}",
                        model,
                        seed,
                        dataset.y,
                        oof[model],
                        dataset.frame,
                        dataset.weights,
                    )
                )
                averaged_predictions.setdefault((target, model), []).append(oof[model])
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "model": model,
                            "seed": seed,
                            "row": np.arange(dataset.n),
                            "run_name": dataset.frame["run_name"].astype(str),
                            "observed": dataset.y,
                            "predicted": oof[model],
                        }
                    )
                )
    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    parameters = pd.DataFrame(parameter_rows)
    metrics.to_csv(output_dir / "metrics_300m.csv", index=False)
    pairs.to_csv(output_dir / "paired_metrics_300m.csv", index=False)
    parameters.to_csv(output_dir / "parameters_300m.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions_300m.csv", index=False)

    gate_rows = []
    improvement_passes = []
    threshold_passes = []
    for target in benchmark.TARGETS:
        dataset = benchmark.load_300m(target)
        candidate = np.mean(averaged_predictions[(target, MODEL)], axis=0)
        ablation = np.mean(averaged_predictions[(target, ABLATION)], axis=0)
        observed_delta, candidate_delta = paired_arrays(dataset, candidate)
        _observed_delta, ablation_delta = paired_arrays(dataset, ablation)
        bootstrap = bootstrap_rmse_improvement(
            observed_delta,
            candidate_delta,
            ablation_delta,
            seed=30_000,
        )
        asymmetric = ~replay_control.tied_rows(dataset.weights)
        candidate_rmse = float(np.sqrt(np.mean((candidate[asymmetric] - dataset.y[asymmetric]) ** 2)))
        delta_rmse = float(np.sqrt(np.mean((candidate_delta - observed_delta) ** 2)))
        delta_bias = float(np.mean(candidate_delta - observed_delta))
        sign_accuracy = float(np.mean(np.sign(candidate_delta) == np.sign(observed_delta)))
        selected_tau = parameters.loc[
            parameters["target"].eq(target) & parameters["model"].eq(MODEL),
            "tau",
        ].to_numpy()
        finite_grid = sorted(value for value in saturating.TAU_GRID if np.isfinite(value))
        interior = np.isin(selected_tau, finite_grid[1:]).all() and np.isfinite(selected_tau).all()
        finite_indices = (
            np.asarray([finite_grid.index(value) for value in selected_tau], dtype=int) if interior else None
        )
        stable = bool(interior and np.ptp(finite_indices) <= 1)
        threshold = PAIR_THRESHOLDS[target]
        target_pass = (
            delta_rmse <= threshold["delta_rmse"]
            and abs(delta_bias) <= threshold["delta_bias"]
            and sign_accuracy >= threshold["sign_accuracy"]
            and candidate_rmse <= threshold["asymmetric_rmse"]
            and stable
        )
        improvement_pass = bootstrap["improvement"] >= bootstrap["bootstrap_se"]
        threshold_passes.append(target_pass)
        improvement_passes.append(improvement_pass)
        gate_rows.append(
            {
                "target": target,
                "asymmetric_rmse": candidate_rmse,
                "delta_rmse": delta_rmse,
                "delta_bias": delta_bias,
                "sign_accuracy": sign_accuracy,
                "tau_interior_and_stable": stable,
                **bootstrap,
                "passes_thresholds": target_pass,
                "passes_improvement": improvement_pass,
            }
        )
    passed = all(threshold_passes) and any(improvement_passes)
    pd.DataFrame(gate_rows).to_csv(output_dir / "gate_300m.csv", index=False)
    (output_dir / "gate_300m.json").write_text(json.dumps({"passed": passed, "rows": gate_rows}, indent=2) + "\n")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        choices=("design", "wsd", "300m", "all"),
        default="all",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"design", "all"}:
        passed_design = run_design_screen(args.output_dir)
        print(f"design gate: {'PASS' if passed_design else 'REJECT'}", flush=True)
        if not passed_design:
            return
    if args.stage in {"wsd", "all"}:
        passed_wsd = run_wsd_screen(args.output_dir)
        print(f"WSD gate: {'PASS' if passed_wsd else 'REJECT'}", flush=True)
        if not passed_wsd:
            return
    if args.stage in {"300m", "all"}:
        passed_300m = run_300m_gate(args.output_dir)
        print(f"300M gate: {'PASS' if passed_300m else 'REJECT'}", flush=True)


if __name__ == "__main__":
    main()
