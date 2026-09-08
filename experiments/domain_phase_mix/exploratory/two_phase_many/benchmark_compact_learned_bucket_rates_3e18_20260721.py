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
"""Test shrinkage-pooled, target-learned bucket clocks in Compact Retained State.

This is the final nested check after exposure-derived family and bucket clocks
collapsed to the shared-clock baseline. It learns one log learning rate per
bucket while shrinking every rate toward the shared Compact rate:

    L = b - sum_i a_i (1 - exp(-(rho_i z_i)^p)) + c sum_i [q_i - 1]_+^2
        + kappa mean_i (log rho_i - log rho)^2.

The retained state, response shape, and shared replay channel are unchanged.
``kappa`` is selected inside each outer fold. The exact shared Compact model is
reported beside the learned-rate candidate; heldout outcomes never select the
shrinkage strength.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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
    benchmark_compact_bucket_mechanisms_3e18_20260721 as mechanisms,
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_learned_bucket_rates_3e18_20260721"
MECHANISM_OUTPUT = SCRIPT_DIR / "reference_outputs/compact_bucket_mechanisms_3e18_20260721"
TARGETS = ("uncheatable", "table9")
RATE_SHRINK_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
RATE_BOUNDS = (1e-4, 100.0)
OUTER_SEED = mechanisms.OUTER_SEED
INNER_SEED = mechanisms.INNER_SEED


@dataclass(frozen=True)
class LearnedRateModel:
    shape: compact.Shape
    rate_shrink: float
    l2: float
    intercept: float
    signal_coef: np.ndarray
    replay_coef: np.ndarray
    log_rates: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    objective: float
    iterations: int
    converged: bool

    def predict(self, weights: np.ndarray) -> np.ndarray:
        signal, replay = mechanisms.features(
            np.asarray(weights, dtype=float),
            self.c0,
            self.c1,
            self.shape,
            np.exp(self.log_rates),
            mechanisms.ReplayScope.SHARED,
            (),
        )
        return np.asarray(self.intercept - signal @ self.signal_coef + replay @ self.replay_coef, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--rate-maxiter", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def shape_from_row(row: pd.Series) -> compact.Shape:
    return compact.Shape(
        rate=float(row["shape_rate"]),
        late_rate=float(row["shape_late_rate"]),
        power=float(row["shape_power"]),
        late_multiplier=float(row["shape_late_multiplier"]),
        forgetting_rate=float(row["shape_forgetting_rate"]),
    )


def fit_head(
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    log_rates: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    signal, replay = mechanisms.features(
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        shape,
        np.exp(log_rates),
        mechanisms.ReplayScope.SHARED,
        (),
    )
    design = np.hstack([-signal, replay])
    intercept, coefficients = compact.fit_nonnegative_head(design, dataset.y[indices], l2)
    return intercept, coefficients[: dataset.m], coefficients[dataset.m :], signal


def objective_and_gradient(
    log_rates: np.ndarray,
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    l2: float,
    rate_shrink: float,
) -> tuple[float, np.ndarray]:
    intercept, signal_coef, replay_coef, signal = fit_head(dataset, indices, shape, log_rates, l2)
    _signal, replay = mechanisms.features(
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        shape,
        np.exp(log_rates),
        mechanisms.ReplayScope.SHARED,
        (),
    )
    prediction = intercept - signal @ signal_coef + replay @ replay_coef
    residual = prediction - dataset.y[indices]

    retained, _total = mechanisms.retained_exposure(
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        shape,
    )
    scaled_power = (np.maximum(retained, 0.0) * np.exp(log_rates)[None, :]) ** shape.power
    signal_derivative = shape.power * scaled_power * np.exp(-scaled_power)
    prediction_derivative = -signal_derivative * signal_coef[None, :]
    data_gradient = 2.0 * np.mean(residual[:, None] * prediction_derivative, axis=0)

    anchor = math.log(shape.rate)
    displacement = log_rates - anchor
    shrink_loss = rate_shrink * float(np.mean(displacement**2))
    shrink_gradient = 2.0 * rate_shrink * displacement / len(displacement)
    ridge_loss = l2 * float(np.sum(np.concatenate([signal_coef, replay_coef]) ** 2)) / len(indices)
    loss = float(np.mean(residual**2)) + ridge_loss + shrink_loss
    return loss, data_gradient + shrink_gradient


def fit_model(
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    l2: float,
    rate_shrink: float,
    maxiter: int,
    multistart: bool,
) -> LearnedRateModel:
    anchor = math.log(shape.rate)
    starts = [np.full(dataset.m, anchor, dtype=float)]
    if multistart:
        retained, _total = mechanisms.retained_exposure(
            dataset.weights[indices],
            dataset.c0,
            dataset.c1,
            shape,
        )
        prior = mechanisms.centered_inverse_median(retained)
        starts.append(np.clip(anchor + 0.5 * prior, math.log(RATE_BOUNDS[0]), math.log(RATE_BOUNDS[1])))
    bounds = [(math.log(RATE_BOUNDS[0]), math.log(RATE_BOUNDS[1]))] * dataset.m
    results = [
        minimize(
            objective_and_gradient,
            start,
            args=(dataset, indices, shape, l2, rate_shrink),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-11, "maxls": 30},
        )
        for start in starts
    ]
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("No finite learned-rate optimization")
    result = min(finite, key=lambda candidate: float(candidate.fun))
    log_rates = np.asarray(result.x, dtype=float)
    intercept, signal_coef, replay_coef, _signal = fit_head(dataset, indices, shape, log_rates, l2)
    return LearnedRateModel(
        shape=shape,
        rate_shrink=rate_shrink,
        l2=l2,
        intercept=intercept,
        signal_coef=np.asarray(signal_coef, dtype=float),
        replay_coef=np.asarray(replay_coef, dtype=float),
        log_rates=log_rates,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        objective=float(result.fun),
        iterations=int(result.nit),
        converged=bool(result.success),
    )


def select_shrink(
    dataset: Any,
    indices: np.ndarray,
    shape: compact.Shape,
    l2: float,
    seed: int,
    maxiter: int,
) -> tuple[float, pd.DataFrame]:
    local = mechanisms.subset(dataset, indices, f"{dataset.name}_inner_learned_rates")
    rows = []
    for rate_shrink in RATE_SHRINK_GRID:
        prediction = np.full(local.n, np.nan, dtype=float)
        for inner_train, inner_test in observatory.folds(local, seed):
            model = fit_model(
                local,
                inner_train,
                shape,
                l2,
                rate_shrink,
                max(12, maxiter // 2),
                False,
            )
            prediction[inner_test] = model.predict(local.weights[inner_test])
        metrics = analysis.metrics(local.y, prediction)
        rows.append(
            {
                "rate_shrink": rate_shrink,
                "rmse": metrics["rmse"],
                "regret_at_1": metrics["regret_at_1"],
                "calibration_error": metrics["calibration_error"],
            }
        )
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(["rmse", "regret_at_1", "rate_shrink"], ascending=[True, True, False]).iloc[0]
    return float(selected["rate_shrink"]), frame


def run_target(target: str, output_dir: Path, rate_maxiter: int, force: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    target_dir = output_dir / target
    metrics_path = target_dir / "metrics.json"
    selections_path = target_dir / "selections.csv"
    if not force and metrics_path.exists() and selections_path.exists():
        return json.loads(metrics_path.read_text()), pd.read_csv(selections_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    train, evaluation, provenance = endpoint.endpoint_training_set(target)
    provenance.to_csv(target_dir / "training_provenance.csv", index=False)
    baseline_metrics = pd.read_csv(MECHANISM_OUTPUT / target / "metrics.csv")
    baseline = baseline_metrics[baseline_metrics["variant"] == mechanisms.BASELINE.name].iloc[0].to_dict()
    baseline_selections = pd.read_csv(MECHANISM_OUTPUT / target / "selections.csv")
    spec = analysis.frozen_spec(target, observatory.TWO_PHASE, "compact_retained_state")
    l2 = float(spec.tuning["l2"])
    prediction = np.full(train.n, np.nan, dtype=float)
    selection_rows: list[dict[str, Any]] = []
    outer_folds = observatory.folds(train, OUTER_SEED)

    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        shape_row = baseline_selections[
            (baseline_selections["fit_scope"] == "outer_fold")
            & (baseline_selections["fold"] == fold_index)
            & (baseline_selections["variant"] == mechanisms.BASELINE.name)
        ].iloc[0]
        shape = shape_from_row(shape_row)
        rate_shrink, sweep = select_shrink(
            train,
            outer_train,
            shape,
            l2,
            INNER_SEED + fold_index,
            rate_maxiter,
        )
        model = fit_model(train, outer_train, shape, l2, rate_shrink, rate_maxiter, True)
        prediction[outer_test] = model.predict(train.weights[outer_test])
        selection_rows.append(
            {
                "target": target,
                "fit_scope": "outer_fold",
                "fold": fold_index,
                "selected_rate_shrink": rate_shrink,
                "rate_min": float(np.exp(model.log_rates).min()),
                "rate_median": float(np.median(np.exp(model.log_rates))),
                "rate_max": float(np.exp(model.log_rates).max()),
                "converged": model.converged,
                "iterations": model.iterations,
                "inner_sweep_json": sweep.to_json(orient="records"),
                **{f"shape_{key}": value for key, value in asdict(shape).items()},
            }
        )
        pd.DataFrame(selection_rows).to_csv(selections_path, index=False)
        np.save(target_dir / "oof_prediction.npy", prediction)

    full_shape_row = baseline_selections[
        (baseline_selections["fit_scope"] == "full") & (baseline_selections["variant"] == mechanisms.BASELINE.name)
    ].iloc[0]
    full_shape = shape_from_row(full_shape_row)
    rate_shrink, sweep = select_shrink(
        train,
        np.arange(train.n),
        full_shape,
        l2,
        INNER_SEED + len(outer_folds),
        rate_maxiter,
    )
    model = fit_model(
        train,
        np.arange(train.n),
        full_shape,
        l2,
        rate_shrink,
        rate_maxiter,
        True,
    )
    heldout_prediction = model.predict(evaluation.weights)
    np.save(target_dir / "heldout_prediction.npy", heldout_prediction)
    observed_heldout = evaluation.frame[analysis.TARGET_COLUMNS[target]].to_numpy(dtype=float)
    learned = {
        "target": target,
        "variant": "learned_bucket_rates",
        "selected_rate_shrink": rate_shrink,
        "l2": l2,
        "nominal_parameter_count": 84,
        "effective_rate_displacement_rms": float(np.sqrt(np.mean((model.log_rates - math.log(full_shape.rate)) ** 2))),
        "rate_min": float(np.exp(model.log_rates).min()),
        "rate_median": float(np.median(np.exp(model.log_rates))),
        "rate_max": float(np.exp(model.log_rates).max()),
        "rate_boundary_count": int(
            np.sum(
                (np.exp(model.log_rates) <= RATE_BOUNDS[0] * 1.001) | (np.exp(model.log_rates) >= RATE_BOUNDS[1] / 1.001)
            )
        ),
        **mechanisms.prefixed_metrics("oof", train.y, prediction),
        **mechanisms.prefixed_metrics("heldout", observed_heldout, heldout_prediction),
    }
    comparisons = {
        "baseline": baseline,
        "learned_bucket_rates": learned,
        "delta": {
            key: float(learned[key]) - float(baseline[key])
            for key in (
                "oof_rmse",
                "heldout_rmse",
                "heldout_regret_at_1",
                "heldout_calibration_error",
                "heldout_optimism_gt_0p05",
                "heldout_worst_optimism",
            )
        },
    }
    selection_rows.append(
        {
            "target": target,
            "fit_scope": "full",
            "fold": -1,
            "selected_rate_shrink": rate_shrink,
            "rate_min": learned["rate_min"],
            "rate_median": learned["rate_median"],
            "rate_max": learned["rate_max"],
            "converged": model.converged,
            "iterations": model.iterations,
            "inner_sweep_json": sweep.to_json(orient="records"),
            "log_rates_json": json.dumps(model.log_rates.tolist(), separators=(",", ":")),
            **{f"shape_{key}": value for key, value in asdict(full_shape).items()},
        }
    )
    pd.DataFrame(selection_rows).to_csv(selections_path, index=False)
    metrics_path.write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")
    return comparisons, pd.DataFrame(selection_rows)


def write_report(results: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = []
    for result in results:
        for label in ("baseline", "learned_bucket_rates"):
            row = result[label]
            rows.append(
                {
                    "target": row["target"],
                    "variant": label,
                    "rate_shrink": row.get("selected_rate_shrink", float("nan")),
                    "oof_rmse": row["oof_rmse"],
                    "heldout_rmse": row["heldout_rmse"],
                    "heldout_calibration_slope": row["heldout_calibration_slope"],
                    "heldout_regret_at_1": row["heldout_regret_at_1"],
                    "heldout_optimism_gt_0p05": row["heldout_optimism_gt_0p05"],
                    "heldout_worst_optimism": row["heldout_worst_optimism"],
                }
            )
    frame = pd.DataFrame(rows)
    sections = [
        "# Compact Retained State learned bucket-rate screen",
        "",
        (
            "The response shape and replay mechanism are fixed to Compact Retained State. "
            "Only shrinkage-pooled bucket learning rates are added, with shrinkage selected inside each outer fold."
        ),
        "",
        frame.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    for result in results:
        sections.extend(
            [
                f"## {result['learned_bucket_rates']['target']}",
                "",
                f"Deltas relative to shared Compact: `{json.dumps(result['delta'], sort_keys=True)}`",
                "",
            ]
        )
    path = output_dir / "report.md"
    path.write_text("\n".join(sections) + "\n")
    return path


def main() -> None:
    args = parse_args()
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown targets: {sorted(unknown)}")
    if not (MECHANISM_OUTPUT / "metrics.csv").exists():
        raise FileNotFoundError("Run benchmark_compact_bucket_mechanisms_3e18_20260721.py first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1,
        "targets": list(targets),
        "rate_shrink_grid": list(RATE_SHRINK_GRID),
        "fit_rows": 998,
        "fit_scale": "Delphi 3e18 only",
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "selection": "inner-fold RMSE only",
        "heldout_use": "evaluation after form and hyperparameters are frozen",
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and not args.force and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory or --force")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    results = []
    selections = []
    for target in targets:
        result, selection = run_target(target, args.output_dir, args.rate_maxiter, args.force)
        results.append(result)
        selections.append(selection)
    pd.concat(selections, ignore_index=True).to_csv(args.output_dir / "selections.csv", index=False)
    write_report(results, args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
