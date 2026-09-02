# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "pyarrow>=20.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Test shrinkage-pooled bucket state parameters in Compact Retained State.

The two nested candidates replace one shared state-transition parameter at a
time with a shrinkage-pooled vector:

    z_i = exp(-lambda_i (1 - w_i^1)) e_i^0 + eta e_i^1,
    z_i = exp(-lambda (1 - w_i^1)) e_i^0 + eta_i e_i^1.

The response-curvature candidate instead learns one Weibull exponent per
bucket,

    F_i(z_i) = 1 - exp(-(rho z_i)^p_i),

with every ``log p_i`` shrunk toward the fitted shared ``log p``. This tests
whether buckets have distinct distributions of learning timescales while
retaining the same state transition and replay mechanism.

The Weibull response, shared learning clock, amplitudes, and literal shared
replay channel remain unchanged. Log-parameter shrinkage toward the shared
Compact fit is selected inside each outer fold. This isolates heterogeneous
retention or late plasticity without adding an output calibration layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_compact_nonlinear_solver_3e18_20260721 as solver_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_compact_bucket_mechanisms_3e18_20260721 as mechanisms,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_compact_learned_bucket_rates_3e18_20260721 as learned_rates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_delphi_expanded_fit_raw_optima_20260721 as endpoint,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_learned_state_vectors_3e18_20260721"
TARGETS = ("uncheatable", "table9")
SHRINK_GRID = learned_rates.RATE_SHRINK_GRID
INNER_SEED = mechanisms.INNER_SEED
OUTER_SEED = mechanisms.OUTER_SEED


class StateParameter(StrEnum):
    LATE_MULTIPLIER = "late_multiplier"
    FORGETTING_RATE = "forgetting_rate"
    RESPONSE_POWER = "response_power"


@dataclass(frozen=True)
class LearnedStateModel:
    parameter: StateParameter
    shape: compact.Shape
    shrink: float
    l2: float
    intercept: float
    signal_coef: np.ndarray
    replay_coef: np.ndarray
    log_values: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    objective: float
    iterations: int
    converged: bool

    def predict(self, weights: np.ndarray) -> np.ndarray:
        signal, replay, _derivative = features(
            np.asarray(weights, dtype=float),
            self.c0,
            self.c1,
            self.shape,
            self.parameter,
            self.log_values,
        )
        return np.asarray(self.intercept - signal @ self.signal_coef + replay @ self.replay_coef, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--parameters", default=",".join(parameter.value for parameter in StateParameter))
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--baseline-solver", choices=("deployed", "tight"), default="deployed")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def anchor_value(shape: compact.Shape, parameter: StateParameter) -> float:
    if parameter is StateParameter.LATE_MULTIPLIER:
        return shape.late_multiplier
    if parameter is StateParameter.FORGETTING_RATE:
        return shape.forgetting_rate
    if parameter is StateParameter.RESPONSE_POWER:
        return shape.power
    raise ValueError(f"Unsupported state parameter {parameter}")


def parameter_bounds(parameter: StateParameter) -> tuple[float, float]:
    if parameter is StateParameter.LATE_MULTIPLIER:
        return 0.05, 20.0
    if parameter is StateParameter.FORGETTING_RATE:
        return 0.01, 8.0
    if parameter is StateParameter.RESPONSE_POWER:
        return compact.POWER_BOUNDS
    raise ValueError(f"Unsupported state parameter {parameter}")


def features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    shape: compact.Shape,
    parameter: StateParameter,
    log_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase0_weight = weights[:, 0, :]
    phase1_weight = weights[:, 1, :]
    early = phase0_weight * c0[None, :]
    late = phase1_weight * c1[None, :]
    values = np.exp(log_values)

    if parameter is StateParameter.LATE_MULTIPLIER:
        retained_early = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * early
        late_contribution = values[None, :] * late
        retained = retained_early + late_contribution
        retained_log_derivative = late_contribution
    elif parameter is StateParameter.FORGETTING_RATE:
        retained_early = np.exp(-values[None, :] * (1.0 - phase1_weight)) * early
        retained = retained_early + shape.late_multiplier * late
        retained_log_derivative = -values[None, :] * (1.0 - phase1_weight) * retained_early
    elif parameter is StateParameter.RESPONSE_POWER:
        retained_early = np.exp(-shape.forgetting_rate * (1.0 - phase1_weight)) * early
        retained = retained_early + shape.late_multiplier * late
        retained_log_derivative = np.zeros_like(retained)
    else:
        raise ValueError(f"Unsupported state parameter {parameter}")

    retained = np.maximum(retained, 0.0)
    response_power = values[None, :] if parameter is StateParameter.RESPONSE_POWER else shape.power
    scaled_base = shape.rate * retained
    scaled_power = np.maximum(scaled_base, 0.0) ** response_power
    signal = -np.expm1(-scaled_power)
    if parameter is StateParameter.RESPONSE_POWER:
        log_base = np.log(np.maximum(scaled_base, 1e-300))
        signal_log_parameter_derivative = scaled_power * np.exp(-scaled_power) * response_power * log_base
        signal_log_parameter_derivative = np.where(retained > 1e-12, signal_log_parameter_derivative, 0.0)
    else:
        signal_exposure_derivative = np.divide(
            shape.power * scaled_power * np.exp(-scaled_power),
            retained,
            out=np.zeros_like(retained),
            where=retained > 1e-12,
        )
        signal_log_parameter_derivative = signal_exposure_derivative * retained_log_derivative
    total = early + late
    replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return signal, replay, signal_log_parameter_derivative


def fit_head(
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    parameter: StateParameter,
    log_values: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal, replay, derivative = features(
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        shape,
        parameter,
        log_values,
    )
    design = np.hstack([-signal, replay])
    intercept, coefficients = compact.fit_nonnegative_head(design, dataset.y[indices], l2)
    return intercept, coefficients[: dataset.m], coefficients[dataset.m :], signal, derivative


def objective_and_gradient(
    log_values: np.ndarray,
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    parameter: StateParameter,
    l2: float,
    shrink: float,
) -> tuple[float, np.ndarray]:
    intercept, signal_coef, replay_coef, signal, derivative = fit_head(
        dataset,
        indices,
        shape,
        parameter,
        log_values,
        l2,
    )
    _signal, replay, _derivative = features(
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        shape,
        parameter,
        log_values,
    )
    prediction = intercept - signal @ signal_coef + replay @ replay_coef
    residual = prediction - dataset.y[indices]
    prediction_derivative = -derivative * signal_coef[None, :]
    data_gradient = 2.0 * np.mean(residual[:, None] * prediction_derivative, axis=0)

    anchor = math.log(anchor_value(shape, parameter))
    displacement = log_values - anchor
    shrink_loss = shrink * float(np.mean(displacement**2))
    shrink_gradient = 2.0 * shrink * displacement / len(displacement)
    ridge_loss = l2 * float(np.sum(np.concatenate([signal_coef, replay_coef]) ** 2)) / len(indices)
    loss = float(np.mean(residual**2)) + ridge_loss + shrink_loss
    return loss, data_gradient + shrink_gradient


def fit_model(
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    parameter: StateParameter,
    l2: float,
    shrink: float,
    maxiter: int,
) -> LearnedStateModel:
    anchor = math.log(anchor_value(shape, parameter))
    start = np.full(dataset.m, anchor, dtype=float)
    low, high = parameter_bounds(parameter)
    result = minimize(
        objective_and_gradient,
        start,
        args=(dataset, indices, shape, parameter, l2, shrink),
        method="L-BFGS-B",
        jac=True,
        bounds=[(math.log(low), math.log(high))] * dataset.m,
        options={"maxiter": maxiter, "ftol": 1e-11, "maxls": 30},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(f"Non-finite optimization for {parameter}")
    log_values = np.asarray(result.x, dtype=float)
    intercept, signal_coef, replay_coef, _signal, _derivative = fit_head(
        dataset,
        indices,
        shape,
        parameter,
        log_values,
        l2,
    )
    return LearnedStateModel(
        parameter=parameter,
        shape=shape,
        shrink=shrink,
        l2=l2,
        intercept=intercept,
        signal_coef=np.asarray(signal_coef, dtype=float),
        replay_coef=np.asarray(replay_coef, dtype=float),
        log_values=log_values,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        objective=float(result.fun),
        iterations=int(result.nit),
        converged=bool(result.success),
    )


def fit_baseline(
    dataset: Any,
    indices: np.ndarray,
    l2: float,
    maxiter: int,
    baseline_solver: str,
) -> compact.FittedModel:
    config = observatory.COMPACT_TWO_PHASE_CONFIG
    if baseline_solver == "deployed":
        return compact.fit_model(dataset, indices, config, l2, maxiter=maxiter, top_k=2)
    if baseline_solver != "tight":
        raise ValueError(f"Unsupported baseline solver {baseline_solver}")
    problem = solver_audit.ProfiledProblem(dataset, indices, config, l2)
    bounds = compact.shape_bounds(config)
    scored = sorted(
        ((problem.objective(start), start) for start in compact.shape_starts(config)),
        key=lambda candidate: candidate[0],
    )
    theta, _objective, _converged = solver_audit.refine_starts(
        problem,
        tuple(start for _score, start in scored[:2]),
        bounds,
        max(200, maxiter),
        tight=True,
    )
    return problem.fitted_model(theta)


def select_shrink(
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    parameter: StateParameter,
    l2: float,
    seed: int,
    maxiter: int,
) -> tuple[float, pd.DataFrame]:
    local = mechanisms.subset(dataset, indices, f"{dataset.name}_inner_{parameter.value}")
    rows = []
    for shrink in SHRINK_GRID:
        prediction = np.full(local.n, np.nan, dtype=float)
        for inner_train, inner_test in observatory.folds(local, seed):
            model = fit_model(
                local,
                inner_train,
                shape,
                parameter,
                l2,
                shrink,
                max(12, maxiter // 2),
            )
            prediction[inner_test] = model.predict(local.weights[inner_test])
        metrics = analysis.metrics(local.y, prediction)
        rows.append(
            {
                "shrink": shrink,
                "rmse": metrics["rmse"],
                "regret_at_1": metrics["regret_at_1"],
                "calibration_error": metrics["calibration_error"],
            }
        )
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(["rmse", "regret_at_1", "shrink"], ascending=[True, True, False]).iloc[0]
    return float(selected["shrink"]), frame


def run_target(
    target: str,
    parameters: tuple[StateParameter, ...],
    output_dir: Path,
    maxiter: int,
    baseline_solver: str,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_dir = output_dir / target
    metrics_path = target_dir / "metrics.csv"
    selections_path = target_dir / "selections.csv"
    if not force and metrics_path.exists() and selections_path.exists():
        return pd.read_csv(metrics_path), pd.read_csv(selections_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    train, evaluation, provenance = endpoint.endpoint_training_set(target)
    provenance.to_csv(target_dir / "training_provenance.csv", index=False)
    l2 = float(analysis.frozen_spec(target, observatory.TWO_PHASE, "compact_retained_state").tuning["l2"])
    outer_folds = observatory.folds(train, OUTER_SEED)
    predictions = {
        "shared_compact": np.full(train.n, np.nan, dtype=float),
        **{parameter.value: np.full(train.n, np.nan, dtype=float) for parameter in parameters},
    }
    selection_rows: list[dict[str, Any]] = []

    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        baseline_model = fit_baseline(
            train,
            outer_train,
            l2,
            maxiter,
            baseline_solver,
        )
        shape = baseline_model.shape
        predictions["shared_compact"][outer_test] = baseline_model.predict(train.weights[outer_test])
        selection_rows.append(
            {
                "target": target,
                "parameter": "shared",
                "fit_scope": "outer_fold",
                "fold": fold_index,
                "selected_shrink": float("nan"),
                "converged": True,
                "iterations": 0,
                **{f"shape_{key}": value for key, value in asdict(shape).items()},
            }
        )
        for parameter in parameters:
            shrink, sweep = select_shrink(
                train,
                outer_train,
                shape,
                parameter,
                l2,
                INNER_SEED + fold_index,
                maxiter,
            )
            model = fit_model(train, outer_train, shape, parameter, l2, shrink, maxiter)
            predictions[parameter.value][outer_test] = model.predict(train.weights[outer_test])
            selection_rows.append(
                {
                    "target": target,
                    "parameter": parameter.value,
                    "fit_scope": "outer_fold",
                    "fold": fold_index,
                    "selected_shrink": shrink,
                    "value_min": float(np.exp(model.log_values).min()),
                    "value_median": float(np.median(np.exp(model.log_values))),
                    "value_max": float(np.exp(model.log_values).max()),
                    "converged": model.converged,
                    "iterations": model.iterations,
                    "inner_sweep_json": sweep.to_json(orient="records"),
                    "log_values_json": json.dumps(model.log_values.tolist(), separators=(",", ":")),
                    **{f"shape_{key}": value for key, value in asdict(shape).items()},
                }
            )
        pd.DataFrame(selection_rows).to_csv(selections_path, index=False)

    full_baseline = fit_baseline(
        train,
        np.arange(train.n),
        l2,
        maxiter,
        baseline_solver,
    )
    full_shape = full_baseline.shape
    observed_heldout = evaluation.frame[analysis.TARGET_COLUMNS[target]].to_numpy(dtype=float)
    baseline_heldout_prediction = full_baseline.predict(evaluation.weights)
    metric_rows = [
        {
            "target": target,
            "variant": "shared_compact",
            "parameter": "shared",
            "selected_shrink": float("nan"),
            **mechanisms.prefixed_metrics("oof", train.y, predictions["shared_compact"]),
            **mechanisms.prefixed_metrics("heldout", observed_heldout, baseline_heldout_prediction),
        }
    ]
    selection_rows.append(
        {
            "target": target,
            "parameter": "shared",
            "fit_scope": "full",
            "fold": -1,
            "selected_shrink": float("nan"),
            "converged": True,
            "iterations": 0,
            **{f"shape_{key}": value for key, value in asdict(full_shape).items()},
        }
    )
    for parameter in parameters:
        shrink, sweep = select_shrink(
            train,
            np.arange(train.n),
            full_shape,
            parameter,
            l2,
            INNER_SEED + len(outer_folds),
            maxiter,
        )
        model = fit_model(train, np.arange(train.n), full_shape, parameter, l2, shrink, maxiter)
        heldout_prediction = model.predict(evaluation.weights)
        anchor = math.log(anchor_value(full_shape, parameter))
        low, high = parameter_bounds(parameter)
        values = np.exp(model.log_values)
        retained, _total = mechanisms.retained_exposure(train.weights, train.c0, train.c1, full_shape)
        pd.DataFrame(
            {
                "domain": train.domain_names,
                "value": values,
                "shared_anchor": math.exp(anchor),
                "log_displacement": model.log_values - anchor,
                "signal_coefficient": model.signal_coef,
                "median_positive_retained_exposure": np.nanmedian(np.where(retained > 1e-10, retained, np.nan), axis=0),
                "at_lower_bound": values <= low + 1e-6,
                "at_upper_bound": values >= high - 1e-6,
            }
        ).to_csv(target_dir / f"parameters_{parameter.value}.csv", index=False)
        metric_rows.append(
            {
                "target": target,
                "variant": f"learned_{parameter.value}",
                "parameter": parameter.value,
                "selected_shrink": shrink,
                "log_displacement_rms": float(np.sqrt(np.mean((model.log_values - anchor) ** 2))),
                "value_min": float(np.exp(model.log_values).min()),
                "value_median": float(np.median(np.exp(model.log_values))),
                "value_max": float(np.exp(model.log_values).max()),
                "lower_bound_count": int(np.sum(values <= low + 1e-6)),
                "upper_bound_count": int(np.sum(values >= high - 1e-6)),
                "active_signal_count": int(np.sum(model.signal_coef > 1e-10)),
                **mechanisms.prefixed_metrics("oof", train.y, predictions[parameter.value]),
                **mechanisms.prefixed_metrics("heldout", observed_heldout, heldout_prediction),
            }
        )
        selection_rows.append(
            {
                "target": target,
                "parameter": parameter.value,
                "fit_scope": "full",
                "fold": -1,
                "selected_shrink": shrink,
                "value_min": float(np.exp(model.log_values).min()),
                "value_median": float(np.median(np.exp(model.log_values))),
                "value_max": float(np.exp(model.log_values).max()),
                "converged": model.converged,
                "iterations": model.iterations,
                "inner_sweep_json": sweep.to_json(orient="records"),
                "log_values_json": json.dumps(model.log_values.tolist(), separators=(",", ":")),
                **{f"shape_{key}": value for key, value in asdict(full_shape).items()},
            }
        )
    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    metrics.to_csv(metrics_path, index=False)
    selections.to_csv(selections_path, index=False)
    np.savez_compressed(target_dir / "oof_predictions.npz", **predictions)
    return metrics, selections


def write_report(metrics: pd.DataFrame, selections: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "target",
        "variant",
        "selected_shrink",
        "oof_rmse",
        "heldout_rmse",
        "heldout_calibration_slope",
        "heldout_regret_at_1",
        "heldout_optimism_gt_0p05",
        "heldout_worst_optimism",
    ]
    fold_stability = (
        selections[selections["fit_scope"] == "outer_fold"]
        .groupby(["target", "parameter"])["selected_shrink"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    path = output_dir / "report.md"
    path.write_text(
        "\n".join(
            [
                "# Compact Retained State bucket transition-vector screen",
                "",
                metrics[columns].to_markdown(index=False, floatfmt=".6f"),
                "",
                "## Fold-level shrinkage stability",
                "",
                fold_stability.to_markdown(index=False, floatfmt=".6g"),
                "",
            ]
        )
    )
    return path


def main() -> None:
    args = parse_args()
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown targets: {sorted(unknown)}")
    requested_parameters = tuple(value.strip() for value in args.parameters.split(",") if value.strip())
    known_parameters = {parameter.value: parameter for parameter in StateParameter}
    unknown_parameters = set(requested_parameters) - set(known_parameters)
    if unknown_parameters:
        raise ValueError(f"Unknown parameters: {sorted(unknown_parameters)}")
    parameters = tuple(known_parameters[value] for value in requested_parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1,
        "targets": list(targets),
        "parameters": [parameter.value for parameter in parameters],
        "shrink_grid": list(SHRINK_GRID),
        "fit_rows": 998,
        "fit_scale": "Delphi 3e18 only",
        "selection": "inner-fold RMSE only",
        "baseline_shape": "shared Compact shape refit independently inside every outer fold",
        "baseline_solver": args.baseline_solver,
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and not args.force and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory or --force")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    metrics = []
    selections = []
    for target in targets:
        target_metrics, target_selections = run_target(
            target,
            parameters,
            args.output_dir,
            args.maxiter,
            args.baseline_solver,
            args.force,
        )
        metrics.append(target_metrics)
        selections.append(target_selections)
    combined_metrics = pd.concat(metrics, ignore_index=True)
    combined_selections = pd.concat(selections, ignore_index=True)
    combined_metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    combined_selections.to_csv(args.output_dir / "selections.csv", index=False)
    report = write_report(combined_metrics, combined_selections, args.output_dir)
    print(report.read_text())


if __name__ == "__main__":
    main()
