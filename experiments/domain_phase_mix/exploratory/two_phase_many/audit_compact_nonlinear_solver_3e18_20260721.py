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
"""Audit whether Compact Retained State is limited by its nonlinear solver.

Compact is separable: for any nonlinear retained-state shape, its intercept
and nonnegative linear amplitudes are refit exactly. This script preserves that
profiled objective and compares the deployed top-two multistart L-BFGS-B fit
with exhaustive coarse-grid refinement, Sobol multistart, basin hopping, and
differential evolution. No response labels or hyperparameters differ across
solvers.

The audit is deliberately full-fit first. If a global method cannot improve
the profiled training objective or change predictions materially, refitting
all OOF folds would add cost without evidence of an optimization bottleneck.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, differential_evolution, minimize
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as analysis,
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_nonlinear_solver_audit_3e18_20260721"
TARGETS = ("uncheatable", "table9")
SEED = 20260721
OUTER_SEED = 20260721


@dataclass(frozen=True)
class SolverResult:
    solver: str
    theta: np.ndarray
    objective: float
    evaluations: int
    elapsed_seconds: float
    converged: bool


@dataclass
class ProfiledProblem:
    dataset: object
    indices: np.ndarray
    config: compact.ModelConfig
    l2: float
    evaluations: int = 0

    def objective(self, theta: np.ndarray) -> float:
        self.evaluations += 1
        shape = compact.decode_shape(np.asarray(theta, dtype=float), self.config)
        design = compact.design_matrix(
            self.dataset.weights[self.indices],
            self.dataset.c0,
            self.dataset.c1,
            self.config,
            shape,
            compact.family_members(self.dataset, self.config),
        )
        intercept, coefficients = compact.fit_nonnegative_head(design, self.dataset.y[self.indices], self.l2)
        residual = intercept + design @ coefficients - self.dataset.y[self.indices]
        return float(np.mean(residual**2))

    def fitted_model(self, theta: np.ndarray) -> compact.FittedModel:
        shape = compact.decode_shape(np.asarray(theta, dtype=float), self.config)
        families = compact.family_members(self.dataset, self.config)
        design = compact.design_matrix(
            self.dataset.weights[self.indices],
            self.dataset.c0,
            self.dataset.c1,
            self.config,
            shape,
            families,
        )
        intercept, coefficients = compact.fit_nonnegative_head(design, self.dataset.y[self.indices], self.l2)
        split = compact.signal_width(self.dataset, self.config)
        return compact.FittedModel(
            config=self.config,
            shape=shape,
            intercept=intercept,
            signal_coef=np.asarray(coefficients[:split], dtype=float),
            replay_coef=np.asarray(coefficients[split:], dtype=float),
            c0=np.asarray(self.dataset.c0, dtype=float),
            c1=np.asarray(self.dataset.c1, dtype=float),
            family_members=families,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sobol-power", type=int, default=6)
    parser.add_argument("--basin-iterations", type=int, default=64)
    parser.add_argument("--de-iterations", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def refine_starts(
    problem: ProfiledProblem,
    starts: tuple[np.ndarray, ...],
    bounds: list[tuple[float, float]],
    maxiter: int,
    *,
    tight: bool,
) -> tuple[np.ndarray, float, bool]:
    candidates: list[tuple[float, np.ndarray, bool]] = []
    for start in starts:
        start_value = problem.objective(start)
        if np.isfinite(start_value):
            candidates.append((start_value, np.asarray(start, dtype=float), True))
        options = {"maxiter": maxiter, "ftol": 1e-10, "maxls": 30}
        if tight:
            options.update({"ftol": 1e-12, "gtol": 1e-9, "maxls": 40})
        result = minimize(
            problem.objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append((float(result.fun), np.asarray(result.x, dtype=float), bool(result.success)))
    if not candidates:
        raise RuntimeError("No finite local refinement")
    best = min(candidates, key=lambda candidate: candidate[0])
    return best[1], best[0], best[2]


def scaled_points(points: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    return qmc.scale(np.asarray(points, dtype=float), lower, upper)


def as_unit(theta: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    return (np.asarray(theta, dtype=float) - lower) / (upper - lower)


def from_unit(unit: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    return scaled_points(np.asarray(unit, dtype=float)[None, :], bounds)[0]


def timed_solver(
    name: str,
    problem: ProfiledProblem,
    solve: Callable[[], tuple[np.ndarray, float, bool]],
) -> SolverResult:
    before = problem.evaluations
    started = time.perf_counter()
    theta, objective, converged = solve()
    return SolverResult(
        solver=name,
        theta=np.asarray(theta, dtype=float),
        objective=float(objective),
        evaluations=problem.evaluations - before,
        elapsed_seconds=time.perf_counter() - started,
        converged=converged,
    )


def solve_target(
    target: str,
    output_dir: Path,
    seed: int,
    sobol_power: int,
    basin_iterations: int,
    de_iterations: int,
) -> pd.DataFrame:
    train, evaluation, provenance = endpoint.endpoint_training_set(target)
    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    provenance.to_csv(target_dir / "training_provenance.csv", index=False)
    evaluation.frame.to_csv(target_dir / "evaluation_provenance.csv", index=False)

    config = observatory.COMPACT_TWO_PHASE_CONFIG
    frozen = analysis.frozen_spec(target, observatory.TWO_PHASE, "compact_retained_state")
    l2 = float(frozen.tuning["l2"])
    problem = ProfiledProblem(train, np.arange(train.n), config, l2)
    bounds = compact.shape_bounds(config)
    coarse = compact.shape_starts(config)
    scored = sorted(((problem.objective(start), start) for start in coarse), key=lambda candidate: candidate[0])

    results: list[SolverResult] = []
    results.append(
        timed_solver(
            "deployed_top2_lbfgsb_24",
            problem,
            lambda: refine_starts(
                problem,
                tuple(start for _score, start in scored[:2]),
                bounds,
                24,
                tight=False,
            ),
        )
    )
    results.append(
        timed_solver(
            "tight_top2_lbfgsb_200",
            problem,
            lambda: refine_starts(
                problem,
                tuple(start for _score, start in scored[:2]),
                bounds,
                200,
                tight=True,
            ),
        )
    )
    results.append(
        timed_solver(
            "all_grid_lbfgsb_200",
            problem,
            lambda: refine_starts(
                problem,
                tuple(start for _score, start in scored),
                bounds,
                200,
                tight=True,
            ),
        )
    )

    sobol = qmc.Sobol(d=len(bounds), scramble=True, seed=seed)
    sobol_starts = tuple(scaled_points(sobol.random_base2(sobol_power), bounds))
    results.append(
        timed_solver(
            f"sobol_{len(sobol_starts)}_lbfgsb_200",
            problem,
            lambda: refine_starts(problem, sobol_starts, bounds, 200, tight=True),
        )
    )

    current = results[0]
    current_unit = as_unit(current.theta, bounds)

    def solve_basin() -> tuple[np.ndarray, float, bool]:
        def unit_objective(unit: np.ndarray) -> float:
            return problem.objective(from_unit(unit, bounds))

        result = basinhopping(
            unit_objective,
            current_unit,
            niter=basin_iterations,
            T=1e-5,
            stepsize=0.2,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": [(0.0, 1.0)] * len(bounds),
                "options": {"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9, "maxls": 40},
            },
            seed=seed,
        )
        theta = from_unit(np.asarray(result.x, dtype=float), bounds)
        return theta, problem.objective(theta), bool(result.lowest_optimization_result.success)

    results.append(timed_solver(f"basinhopping_{basin_iterations}", problem, solve_basin))

    def solve_de() -> tuple[np.ndarray, float, bool]:
        def unit_objective(unit: np.ndarray) -> float:
            return problem.objective(from_unit(unit, bounds))

        result = differential_evolution(
            unit_objective,
            [(0.0, 1.0)] * len(bounds),
            maxiter=de_iterations,
            popsize=12,
            tol=1e-9,
            atol=1e-12,
            polish=False,
            seed=seed,
            updating="immediate",
            workers=1,
        )
        theta, objective, converged = refine_starts(
            problem,
            (from_unit(result.x, bounds),),
            bounds,
            300,
            tight=True,
        )
        return theta, objective, bool(result.success and converged)

    results.append(timed_solver(f"differential_evolution_{de_iterations}", problem, solve_de))

    current_model = problem.fitted_model(current.theta)
    current_train_prediction = current_model.predict(train.weights)
    current_heldout_prediction = current_model.predict(evaluation.weights)
    heldout_observed = evaluation.frame[analysis.TARGET_COLUMNS[target]].to_numpy(dtype=float)
    oof_predictions = {
        "deployed_top2_lbfgsb_24": np.full(train.n, np.nan, dtype=float),
        "tight_top2_lbfgsb_200": np.full(train.n, np.nan, dtype=float),
    }
    for outer_train, outer_test in observatory.folds(train, OUTER_SEED):
        fold_problem = ProfiledProblem(train, outer_train, config, l2)
        fold_scored = sorted(
            ((fold_problem.objective(start), start) for start in coarse),
            key=lambda candidate: candidate[0],
        )
        fold_starts = tuple(start for _score, start in fold_scored[:2])
        for solver, maxiter, tight in (
            ("deployed_top2_lbfgsb_24", 24, False),
            ("tight_top2_lbfgsb_200", 200, True),
        ):
            theta, _objective, _converged = refine_starts(
                fold_problem,
                fold_starts,
                bounds,
                maxiter,
                tight=tight,
            )
            oof_predictions[solver][outer_test] = fold_problem.fitted_model(theta).predict(train.weights[outer_test])
    if not all(np.isfinite(prediction).all() for prediction in oof_predictions.values()):
        raise ValueError(f"Incomplete solver OOF predictions for {target}")
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for result in results:
        model = problem.fitted_model(result.theta)
        train_prediction = model.predict(train.weights)
        heldout_prediction = model.predict(evaluation.weights)
        predictions[result.solver] = heldout_prediction
        oof_metrics = (
            {f"oof_{key}": value for key, value in analysis.metrics(train.y, oof_predictions[result.solver]).items()}
            if result.solver in oof_predictions
            else {}
        )
        rows.append(
            {
                "target": target,
                "solver": result.solver,
                "profiled_mse": result.objective,
                "profiled_rmse": math.sqrt(result.objective),
                "objective_delta_vs_current": result.objective - current.objective,
                "train_prediction_max_abs_delta_vs_current": float(
                    np.max(np.abs(train_prediction - current_train_prediction))
                ),
                "heldout_prediction_max_abs_delta_vs_current": float(
                    np.max(np.abs(heldout_prediction - current_heldout_prediction))
                ),
                "evaluations": result.evaluations,
                "elapsed_seconds": result.elapsed_seconds,
                "converged": result.converged,
                "nonzero_signal_coefficients": int(np.sum(model.signal_coef > 1e-10)),
                "nonzero_replay_coefficients": int(np.sum(model.replay_coef > 1e-10)),
                **oof_metrics,
                **{f"shape_{key}": value for key, value in asdict(model.shape).items()},
                **{
                    f"heldout_{key}": value
                    for key, value in analysis.metrics(heldout_observed, heldout_prediction).items()
                },
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(target_dir / "solver_metrics.csv", index=False)
    np.savez_compressed(target_dir / "heldout_predictions.npz", **predictions)
    return frame


def write_report(metrics: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "target",
        "solver",
        "profiled_rmse",
        "oof_rmse",
        "oof_regret_at_1",
        "objective_delta_vs_current",
        "heldout_prediction_max_abs_delta_vs_current",
        "heldout_rmse",
        "heldout_calibration_slope",
        "heldout_regret_at_1",
        "heldout_optimism_gt_0p05",
        "evaluations",
        "elapsed_seconds",
    ]
    path = output_dir / "report.md"
    path.write_text(
        "\n".join(
            [
                "# Compact nonlinear solver audit",
                "",
                "All methods profile out the same ridge-NNLS linear head and differ only in nonlinear search.",
                "",
                metrics[columns].to_markdown(index=False, floatfmt=".9g"),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1,
        "targets": list(targets),
        "fit_rows": 998,
        "fit_scale": "Delphi 3e18 only",
        "objective": "unpenalized training MSE after the frozen ridge-NNLS inner solve",
        "seed": args.seed,
        "sobol_power": args.sobol_power,
        "basin_iterations": args.basin_iterations,
        "differential_evolution_iterations": args.de_iterations,
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and not args.force and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory or --force")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    frames = [
        solve_target(
            target,
            args.output_dir,
            args.seed,
            args.sobol_power,
            args.basin_iterations,
            args.de_iterations,
        )
        for target in targets
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "solver_metrics.csv", index=False)
    report = write_report(metrics, args.output_dir)
    print(report.read_text())


if __name__ == "__main__":
    main()
