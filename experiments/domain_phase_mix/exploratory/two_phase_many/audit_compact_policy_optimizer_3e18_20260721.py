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
"""Audit multistart policy optimization for deployed and tightly fit Compact.

This script consumes the frozen nonlinear-solver audit. It does not refit a
surrogate. For each target and fit basin, it repeats the existing raw-policy
optimizer under several start seeds and reports objective agreement, policy
distance, exposure, phase divergence, and distance from fit support.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import softmax

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
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_delphi_expanded_fit_raw_optima_20260721 as endpoint,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SOLVER_OUTPUT = SCRIPT_DIR / "reference_outputs/compact_nonlinear_solver_audit_3e18_20260721"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_policy_optimizer_audit_3e18_20260721"
TARGETS = ("uncheatable", "table9")
FIT_SOLVERS = ("deployed_top2_lbfgsb_24", "tight_top2_lbfgsb_200")
DEFAULT_SEEDS = (20260721, 20260722, 20260723)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def theta_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            math.log(float(row["shape_rate"])),
            float(row["shape_power"]),
            math.log(float(row["shape_late_multiplier"])),
            float(row["shape_forgetting_rate"]),
        ],
        dtype=float,
    )


def compact_prediction_and_weight_gradient(model: object, weights: np.ndarray) -> tuple[float, np.ndarray]:
    phase0 = np.asarray(weights[0], dtype=float)
    phase1 = np.asarray(weights[1], dtype=float)
    early = phase0 * model.c0
    late = phase1 * model.c1
    decay = np.exp(-model.shape.forgetting_rate * (1.0 - phase1))
    retained_early = decay * early
    retained = np.maximum(retained_early + model.shape.late_multiplier * late, 0.0)
    scaled_power = (model.shape.rate * retained) ** model.shape.power
    signal = -np.expm1(-scaled_power)
    prediction = float(model.intercept - signal @ model.signal_coef)

    signal_exposure_derivative = np.divide(
        model.shape.power * scaled_power * np.exp(-scaled_power),
        retained,
        out=np.zeros_like(retained),
        where=retained > 1e-12,
    )
    benefit_gradient = -model.signal_coef * signal_exposure_derivative
    phase0_gradient = benefit_gradient * decay * model.c0
    phase1_gradient = benefit_gradient * (
        model.shape.forgetting_rate * retained_early + model.shape.late_multiplier * model.c1
    )

    total = early + late
    repeated = np.maximum(total - 1.0, 0.0)
    if len(model.replay_coef):
        replay_coefficient = float(model.replay_coef[0])
        prediction += replay_coefficient * float(np.sum(repeated**2))
        replay_gradient = 2.0 * replay_coefficient * repeated
        phase0_gradient += replay_gradient * model.c0
        phase1_gradient += replay_gradient * model.c1
    return prediction, np.stack([phase0_gradient, phase1_gradient], axis=0)


def optimize_contrast_model(
    dataset: object,
    model: object,
    frozen: analysis.FrozenSpec,
    seed: int,
    count: int,
) -> tuple[analysis.RawOptimum, dict[str, int]]:
    basis = helmert(dataset.m, full=False).T

    def weights_to_contrasts(weights: np.ndarray) -> np.ndarray:
        return (np.log(np.maximum(weights, 1e-12)) @ basis).ravel()

    def contrasts_to_weights(contrasts: np.ndarray) -> np.ndarray:
        logits = np.asarray(contrasts, dtype=float).reshape(2, dataset.m - 1) @ basis.T
        return softmax(logits, axis=1)

    def objective_and_gradient(contrasts: np.ndarray) -> tuple[float, np.ndarray]:
        weights = contrasts_to_weights(contrasts)
        prediction, weight_gradient = compact_prediction_and_weight_gradient(model, weights)
        logit_gradient = weights * (weight_gradient - np.sum(weight_gradient * weights, axis=1, keepdims=True))
        contrast_gradient = logit_gradient @ basis
        return prediction, contrast_gradient.ravel()

    candidates: list[tuple[float, np.ndarray, bool]] = []
    messages: dict[str, int] = {}
    starts = analysis.optimum_starts(dataset, model, frozen, seed, count, None)
    for start in starts:
        start_weights = analysis.logits_to_weights(start, dataset.m)
        result = minimize(
            objective_and_gradient,
            weights_to_contrasts(start_weights),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 800, "maxfun": 100000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        message = str(result.message)
        messages[message] = messages.get(message, 0) + 1
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append((float(result.fun), np.asarray(result.x, dtype=float), bool(result.success)))
    if not candidates:
        raise RuntimeError(f"No finite contrast optimum for {dataset.name}")
    best = min(candidates, key=lambda candidate: candidate[0])
    return (
        analysis.RawOptimum(
            weights=contrasts_to_weights(best[1]),
            predicted_bpb=best[0],
            optimizer_converged=best[2],
            successful_starts=sum(candidate[2] for candidate in candidates),
            finite_starts=len(candidates),
        ),
        messages,
    )


def run_target(target: str, seeds: tuple[int, ...], starts: int) -> pd.DataFrame:
    train, evaluation, _provenance = endpoint.endpoint_training_set(target)
    solver_metrics = pd.read_csv(SOLVER_OUTPUT / target / "solver_metrics.csv").set_index("solver")
    frozen = analysis.frozen_spec(target, observatory.TWO_PHASE, "compact_retained_state")
    l2 = float(frozen.tuning["l2"])
    config = observatory.COMPACT_TWO_PHASE_CONFIG
    rows = []
    previous_by_solver: dict[tuple[str, str], np.ndarray] = {}
    for fit_solver in FIT_SOLVERS:
        problem = solver_audit.ProfiledProblem(train, np.arange(train.n), config, l2)
        model = problem.fitted_model(theta_from_row(solver_metrics.loc[fit_solver]))
        for policy_optimizer in ("full_logits", "orthonormal_contrasts_analytic"):
            key = (fit_solver, policy_optimizer)
            for seed in seeds:
                if policy_optimizer == "full_logits":
                    optimum = analysis.optimize_raw_model(
                        train,
                        model,
                        frozen,
                        seed=seed,
                        count=starts,
                        previous=None,
                    )
                    messages: dict[str, int] = {}
                else:
                    optimum, messages = optimize_contrast_model(train, model, frozen, seed, starts)
                record = analysis.raw_optimum_record(
                    optimum,
                    train,
                    evaluation,
                    target,
                    "compact_retained_state",
                    f"{fit_solver}/{policy_optimizer}",
                    endpoint.EXPECTED_EXTENSION_ROWS,
                    endpoint.EXPECTED_TIED_ROWS,
                    seed,
                    previous_by_solver.get(key),
                )
                record["fit_solver"] = fit_solver
                record["policy_optimizer"] = policy_optimizer
                record["optimizer_starts"] = starts
                record["optimizer_messages_json"] = json.dumps(messages, sort_keys=True)
                rows.append(record)
                previous_by_solver[key] = optimum.weights
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "target",
        "fit_solver",
        "policy_optimizer",
        "seed",
        "predicted_bpb",
        "max_bucket_weight",
        "max_simulated_epochs",
        "phase_total_variation",
        "aggregate_tv_to_proportional",
        "standardized_fit_support_distance",
        "nearest_evaluation_observed",
        "successive_optimum_tv",
        "successful_starts",
        "finite_starts",
    ]
    stability = (
        metrics.groupby(["target", "fit_solver", "policy_optimizer"])
        .agg(
            predicted_bpb_range=("predicted_bpb", lambda values: float(values.max() - values.min())),
            max_successive_tv=("successive_optimum_tv", "max"),
            max_epoch=("max_simulated_epochs", "max"),
            max_weight=("max_bucket_weight", "max"),
        )
        .reset_index()
    )
    path = output_dir / "report.md"
    path.write_text(
        "\n".join(
            [
                "# Compact policy-optimizer audit",
                "",
                metrics[columns].to_markdown(index=False, floatfmt=".6f"),
                "",
                "## Cross-seed stability",
                "",
                stability.to_markdown(index=False, floatfmt=".6g"),
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
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 3,
        "targets": list(targets),
        "fit_solvers": list(FIT_SOLVERS),
        "seeds": list(seeds),
        "starts_per_seed": args.starts,
        "policy_optimizers": ["full_logits", "orthonormal_contrasts_analytic"],
        "fit_scale": "Delphi 3e18 only",
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and not args.force and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory or --force")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    metrics_path = args.output_dir / "metrics.csv"
    if metrics_path.exists() and not args.force:
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.concat([run_target(target, seeds, args.starts) for target in targets], ignore_index=True)
        metrics.to_csv(metrics_path, index=False)
    report = write_report(metrics, args.output_dir)
    print(report.read_text())


if __name__ == "__main__":
    main()
