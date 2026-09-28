# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy>=1.5", "numpy>=2.0", "pandas>=2.2", "scipy>=1.14"]
# ///
"""Reproduce the paper's Olmix optima (Table 2) from the frozen 3e18 swarm with the reference Olmix fitter.

Fits one positive log-linear law per task on the 280-run panel (Huber delta 0.01, 48 multistarts, seed 0, the
settings of the July KL sweep), solves the exact KL-regularized problem of Olmix's exact proposer with cvxpy
(objective = evaluation-weighted mean of the task predictions, KL(w || natural) with the paper's coefficients,
per-bucket caps at repetition 4), and compares the solutions with the mixtures that were trained
(`olmix_onephase_table9_d001_kl0p005_cap4`, `olmix_onephase_uncheatable_d001_kl005_cap4`, held-out registry).
Also writes the swarm as the two CSVs allenai/olmix's `olmix fit` reads (ratios and metrics), for the
reference-implementation check.

usage: uv run reproduce_delphi_olmix_optima_20260907.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix  # noqa: E402
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)

FROZEN = SCRIPT_DIR / "reference_outputs" / "delphi_offline_selection_20260906"
HELDOUT = SCRIPT_DIR / "reference_outputs" / "single_phase_heldout_benchmark_20260902" / "heldout_runs.csv"
OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_olmix_reproduction_20260907"
TARGET_BUDGET = 6_325_183_647_689
HUBER_DELTA = 0.01
N_STARTS = 48
SEED = 0
REPETITION_FACTOR = 4.0
CASES = {
    "table9": ("olmix_onephase_table9_d001_kl0p005_cap4", 0.005),
    "uncheatable": ("olmix_onephase_uncheatable_d001_kl005_cap4", 0.05),
}
SOLVERS = ("CLARABEL", "ECOS", "SCS")


def read_panel() -> dict[str, np.ndarray]:
    return benchmark.read_npz(FROZEN / "inputs" / "panel.npz")


def write_swarm_csvs(panel: dict[str, np.ndarray], output: Path) -> None:
    buckets = [str(b) for b in panel["buckets"]]
    runs = [str(r) for r in panel["runs"]]
    ratios = pd.DataFrame(np.asarray(panel["weights"], float), columns=buckets)
    ratios.insert(0, "name", runs)
    ratios.insert(0, "run", range(len(runs)))
    ratios.to_csv(output / "swarm_ratios.csv", index=False)
    metrics = pd.DataFrame(index=range(len(runs)))
    metrics["run"] = range(len(runs))
    metrics["name"] = runs
    for target in ("uncheatable", "table9"):
        for j, component in enumerate(str(c) for c in panel[f"{target}_components"]):
            metrics[component] = np.asarray(panel[f"{target}_outcomes"], float)[:, j]
    metrics.to_csv(output / "swarm_metrics.csv", index=False)


def fit_laws(weights: np.ndarray, outcomes: np.ndarray) -> list[olmix.OlmixLoglinearFit]:
    return [
        olmix.fit_olmix_loglinear_model(weights, outcomes[:, j], delta=HUBER_DELTA, seed=SEED, n_starts=N_STARTS)
        for j in range(outcomes.shape[1])
    ]


def solve_exact(
    fits: list[olmix.OlmixLoglinearFit],
    objective_weights: np.ndarray,
    natural: np.ndarray,
    caps: np.ndarray,
    kl_reg: float,
) -> tuple[np.ndarray, float, str]:
    weights = cp.Variable(len(natural))
    predicted = sum(
        float(objective_weights[i])
        * (float(np.exp(fit.log_c)) + cp.exp(cp.sum(cp.multiply(np.asarray(fit.coefficients, float), weights))))
        for i, fit in enumerate(fits)
    )
    kl = cp.sum(cp.rel_entr(weights, natural))
    problem = cp.Problem(cp.Minimize(predicted + kl_reg * kl), [weights >= 0, cp.sum(weights) == 1, weights <= caps])
    for solver in SOLVERS:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            solution = np.clip(np.asarray(weights.value, float), 0.0, None)
            return solution / solution.sum(), float(predicted.value), f"{solver}:{problem.status}"
    raise RuntimeError("no solver converged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = read_panel()
    write_swarm_csvs(panel, args.output_dir)
    buckets = [str(b) for b in panel["buckets"]]
    tokens = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b] for b in buckets], float)
    natural = tokens / tokens.sum()
    caps = np.minimum(1.0, tokens * REPETITION_FACTOR / TARGET_BUDGET)
    registry = pd.read_csv(HELDOUT, low_memory=False)
    weights = np.asarray(panel["weights"], float)
    summary = {}
    for target, (run_id, kl_reg) in CASES.items():
        outcomes = np.asarray(panel[f"{target}_outcomes"], float)
        objective_weights = np.asarray(panel[f"{target}_aggregation_weights"], float)
        fits = fit_laws(weights, outcomes)
        solution, predicted, status = solve_exact(fits, objective_weights, natural, caps, kl_reg)
        trained = registry[registry.source_row_id.fillna("").astype(str).str.startswith(run_id)].iloc[0]
        trained_weights = np.asarray([float(trained[f"weight::{b}"]) for b in buckets])
        tv = float(np.abs(solution - trained_weights).sum() / 2)
        trained_prediction = float(
            sum(
                objective_weights[i] * (np.exp(f.log_c) + np.exp(trained_weights @ np.asarray(f.coefficients)))
                for i, f in enumerate(fits)
            )
        )
        summary[target] = {
            "kl_reg": kl_reg,
            "solver": status,
            "tv_to_trained_mixture": tv,
            "max_abs_weight_difference": float(np.abs(solution - trained_weights).max()),
            "predicted_at_solution": predicted,
            "predicted_at_trained_mixture": trained_prediction,
            "measured_trained_mixture": float(trained["table9_macro_bpb" if target == "table9" else "uncheatable_bpb"]),
            "max_epochs_solution": float((solution * TARGET_BUDGET / tokens).max()),
            "max_epochs_trained": float((trained_weights * TARGET_BUDGET / tokens).max()),
        }
        pd.DataFrame({"bucket": buckets, "reproduced": solution, "trained": trained_weights}).to_csv(
            args.output_dir / f"{target}_weights.csv", index=False
        )
        (args.output_dir / f"{target}_laws.json").write_text(
            json.dumps(
                [{"log_c": f.log_c, "coefficients": list(f.coefficients), "huber_loss": f.huber_loss} for f in fits]
            )
        )
        print(target, json.dumps(summary[target], indent=1), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
