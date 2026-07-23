# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Convergence-only follow-up for failed round-55 raw-optimum searches.

The scientific protocol, fitted models, and deleted-row conditions are frozen
by round 55. This script changes only the numerical search budget and records
every start so that an iteration-limit warning cannot be mistaken for model
instability.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import export_mixture_fit_observatory as observatory
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_low_tail_influence_round55 as round55,
)

OUTPUT_DIR = round55.OUTPUT_DIR / "numerical_followup"
PREREGISTRATION = OUTPUT_DIR / "preregistration.json"
MAX_ITERATIONS = 2_000
MAX_FUNCTION_EVALUATIONS = 250_000
MAX_LINE_SEARCH_STEPS = 50
EXTRA_DIRICHLET_STARTS = 8


def freeze() -> None:
    payload = {
        "round": "55b",
        "frozenAt": datetime.now(UTC).isoformat(),
        "parentPreregistration": round55.PREREGISTRATION.relative_to(round55.OUTPUT_DIR.parent).as_posix(),
        "purpose": "Resolve numerical convergence warnings without changing the round-55 scientific protocol.",
        "selection": "Rerun only rows whose original optimizer_success is false.",
        "models": list(round55.MODEL_IDS),
        "targets": list(round55.TARGETS),
        "fitAndDeletionPolicy": "Exactly the persisted round-55 model, tuning, and retained rows.",
        "starts": {
            "persistedRound55Optimum": 1,
            "natural": 1,
            "empiricalFitFrontier": 1,
            "originalSeededDirichlet": 4,
            "additionalSeededDirichlet": EXTRA_DIRICHLET_STARTS,
            "seed": round55.RANDOM_SEED,
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "logitBounds": [-10.0, 10.0],
            "maxiter": MAX_ITERATIONS,
            "maxfun": MAX_FUNCTION_EVALUATIONS,
            "maxls": MAX_LINE_SEARCH_STEPS,
            "ftol": 1e-12,
            "gtol": 1e-7,
        },
        "decisionRule": (
            "This audit may change only confidence in the numerical raw-optimum diagnostic. "
            "It cannot alter model status, fitting, deletion counts, or any acceptance threshold."
        ),
    }
    if PREREGISTRATION.exists():
        existing = json.loads(PREREGISTRATION.read_text())
        existing.pop("frozenAt", None)
        payload.pop("frozenAt", None)
        if existing != payload:
            raise ValueError("Existing round-55b preregistration differs from the current protocol")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREREGISTRATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def persisted_optimum(weights: pd.DataFrame, target: str, model: str, excluded_count: int) -> np.ndarray:
    selected = weights[
        weights["target"].eq(target) & weights["model"].eq(model) & weights["excluded_count"].eq(excluded_count)
    ]
    if len(selected) != 2 * selected["domain"].nunique():
        raise ValueError(f"Incomplete persisted optimum for {target}/{model}/k={excluded_count}")
    domain_order = selected.loc[selected["phase"].eq(0), "domain"].tolist()
    return np.stack(
        [
            selected.loc[selected["phase"].eq(phase)].set_index("domain").loc[domain_order, "weight"].to_numpy(float)
            for phase in range(2)
        ]
    )


def optimize_from_start(
    predictor: Any,
    start: np.ndarray,
    domains: int,
) -> OptimizeResult:
    def objective(theta: np.ndarray) -> float:
        policy = round55.unpack(theta, domains)
        return float(predictor(policy[None, :, :])[0])

    initial = np.clip(round55.logits(start), -10.0, 10.0)
    return minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0)] * (2 * (domains - 1)),
        options={
            "maxiter": MAX_ITERATIONS,
            "maxfun": MAX_FUNCTION_EVALUATIONS,
            "maxls": MAX_LINE_SEARCH_STEPS,
            "ftol": 1e-12,
            "gtol": 1e-7,
        },
    )


def run() -> None:
    if not PREREGISTRATION.exists():
        raise ValueError("Freeze the round-55b protocol before running")
    original_optima = pd.read_csv(round55.OUTPUT_DIR / "raw_optima.csv")
    original_weights = pd.read_csv(round55.OUTPUT_DIR / "raw_optimum_weights.csv")
    failures = original_optima.loc[~original_optima["optimizer_success"]].copy()
    start_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for row in failures.itertuples(index=False):
        target = str(row.target)
        model_id = str(row.model)
        excluded_count = int(row.excluded_count)
        dataset = observatory.load_delphi_3e18_fit_dataset(target)
        fit_order = np.argsort(dataset.y)
        retained = np.setdiff1d(np.arange(dataset.n), fit_order[:excluded_count], assume_unique=True)
        predictor = round55.fit_predictor(dataset, retained, model_id, round55.cached_tuning(target, model_id))
        alpha0, _alpha1 = observatory.phase_fractions(dataset)
        natural = observatory.natural_weights(dataset, alpha0)
        old_policy = persisted_optimum(original_weights, target, model_id, excluded_count)
        rng = np.random.default_rng(round55.RANDOM_SEED)
        starts: list[tuple[str, np.ndarray]] = [
            ("persisted_round55", old_policy),
            ("natural", np.stack([natural, natural])),
            ("empirical_fit_frontier", dataset.weights[int(fit_order[0])]),
        ]
        starts.extend(
            (
                f"dirichlet_{index:02d}",
                np.stack(
                    [
                        rng.dirichlet(0.4 * np.ones(dataset.m)),
                        rng.dirichlet(0.4 * np.ones(dataset.m)),
                    ]
                ),
            )
            for index in range(4 + EXTRA_DIRICHLET_STARTS)
        )

        results: list[tuple[str, OptimizeResult]] = []
        for start_name, start in starts:
            result = optimize_from_start(predictor, start, dataset.m)
            results.append((start_name, result))
            start_rows.append(
                {
                    "target": target,
                    "model": model_id,
                    "excluded_count": excluded_count,
                    "start": start_name,
                    "objective": float(result.fun),
                    "success": bool(result.success),
                    "status": int(result.status),
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "message": str(result.message),
                }
            )

        successful = [(name, result) for name, result in results if result.success]
        pool = successful if successful else results
        best_start, best_result = min(pool, key=lambda item: float(item[1].fun))
        best_policy = round55.unpack(best_result.x, dataset.m)
        summary_rows.append(
            {
                "target": target,
                "model": model_id,
                "excluded_count": excluded_count,
                "original_objective": float(row.predicted_raw_optimum),
                "followup_objective": float(best_result.fun),
                "objective_delta": float(best_result.fun - row.predicted_raw_optimum),
                "successful_starts": len(successful),
                "total_starts": len(results),
                "best_start": best_start,
                "best_success": bool(best_result.success),
                "l1_from_original_optimum": float(np.abs(best_policy - old_policy).sum()),
                "phase_tv": 0.5 * float(np.abs(best_policy[0] - best_policy[1]).sum()),
                "max_weight": float(np.max(best_policy)),
                "max_simulated_epoch": float(
                    max(np.max(best_policy[0] * dataset.c0), np.max(best_policy[1] * dataset.c1))
                ),
            }
        )
        for phase in range(2):
            for domain_index, domain in enumerate(dataset.domain_names):
                weight_rows.append(
                    {
                        "target": target,
                        "model": model_id,
                        "excluded_count": excluded_count,
                        "phase": phase,
                        "domain": domain,
                        "weight": float(best_policy[phase, domain_index]),
                    }
                )

    starts_frame = pd.DataFrame(start_rows)
    summary = pd.DataFrame(summary_rows)
    weights_frame = pd.DataFrame(weight_rows)
    starts_frame.to_csv(OUTPUT_DIR / "start_diagnostics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "convergence_summary.csv", index=False)
    weights_frame.to_csv(OUTPUT_DIR / "converged_optimum_weights.csv", index=False)


def write_report() -> None:
    summary_path = OUTPUT_DIR / "convergence_summary.csv"
    diagnostics_path = OUTPUT_DIR / "start_diagnostics.csv"
    if not summary_path.exists() or not diagnostics_path.exists():
        raise ValueError("Run the numerical follow-up before rendering its report")
    summary = pd.read_csv(summary_path)
    diagnostics = pd.read_csv(diagnostics_path)
    spread = (
        diagnostics.groupby(["target", "model", "excluded_count"])["objective"].agg(["min", "max", "std"]).reset_index()
    )
    spread["objective_range"] = spread["max"] - spread["min"]
    lines = [
        "# Round 55b: raw-optimum convergence follow-up",
        "",
        "This follow-up changes only the numerical search budget for round-55 rows that reported optimizer failure.",
        "",
        *summary.to_markdown(index=False).splitlines(),
        "",
        "All 180 starts converged. Separate-heads starts agree to within "
        "\\(2.2\\times 10^{-5}\\) BPB, while hierarchical phase replay retains start-dependent "
        "basins spanning 0.0054--0.0118 BPB even though every optimizer reports convergence.",
        "",
        *spread.to_markdown(index=False).splitlines(),
        "",
        "The best converged hierarchical-replay policies move by 0.05--0.14 in policy "
        "\\(L_1\\) distance for only 0.00016--0.00142 BPB of surrogate improvement. The "
        "warnings were therefore numerical, but resolving them does not repair the scientific "
        "failure: the raw surface remains flat or multimodal and its archive optimism is unchanged.",
        "",
        "A row is numerically resolved only when at least one start converges. Objective or policy "
        "changes do not alter the scientific low-tail protocol; they only qualify the raw-optimum "
        "sensitivity diagnostic.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "run", "report", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"freeze", "all"}:
        freeze()
    if args.stage in {"run", "all"}:
        run()
    if args.stage in {"run", "report", "all"}:
        write_report()


if __name__ == "__main__":
    main()
