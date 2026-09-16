# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy>=1.5", "numpy>=2.0", "pandas>=2.2", "scipy>=1.14"]
# ///
"""Materialize matched Olmix policies: the reference per-task log-linear law fitted on the frozen Qwen3 3e18 swarm.

The trained Olmix comparators of the paper were fitted in June and July on the earlier Llama 200M/6B data
(`revision_notes/20260908_reader_pi_revision/CHANGES.md`, item 11). This script produces the matched policies
for validation: one positive log-linear law per task fitted on the 280-run Qwen swarm (Huber delta 0.01, 48
multistarts, seed 0, the settings of the July sweep and of `reproduce_delphi_olmix_optima_20260907.py`), the exact
KL-regularized proposer with per-bucket caps at repetition 4 solved with cvxpy, and Marin's 1/2048 runtime grid.
The runtime rounding keeps every bucket under the cap so the candidate ids can carry `cap04` honestly.

usage: uv run materialize_delphi_matched_olmix_20260908.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base  # noqa: E402
from experiments.domain_phase_mix import olmix_loglinear_fit as olmix  # noqa: E402
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)

FROZEN_PANEL = SCRIPT_DIR / "reference_outputs" / "delphi_offline_selection_20260906" / "inputs" / "panel.npz"
HELDOUT = SCRIPT_DIR / "reference_outputs" / "single_phase_heldout_benchmark_20260902" / "heldout_runs.csv"
OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_3e18_20260908"
TARGET_BUDGET = base.SIMULATED_EPOCH_TARGET_BUDGET
MIXTURE_BLOCK_SIZE = 2048
HUBER_DELTA = 0.01
N_STARTS = 48
SEED = 0
EPOCH_CAP = 4
SURROGATE = "olmix_loglinear_d001_qwen3e18"
SOLVERS = ("CLARABEL", "ECOS", "SCS")
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "Table 9"}
# Candidate order is frozen: it is the order the launcher's candidate loader checks.
CASES = (
    ("olmixq_u_kl0p05_cap04", "uncheatable", 0.05, "olmix_onephase_uncheatable_d001_kl005_cap4"),
    ("olmixq_u_kl0p1_cap04", "uncheatable", 0.1, "olmix_onephase_uncheatable_d001_kl0p1_cap4"),
    ("olmixq_u_kl0_cap04", "uncheatable", 0.0, "olmix_onephase_uncheatable_d001_kl0_cap4"),
    ("olmixq_t9_kl0p005_cap04", "table9", 0.005, "olmix_onephase_table9_d001_kl0p005_cap4"),
    ("olmixq_t9_kl0_cap04", "table9", 0.0, "olmix_onephase_table9_d001_kl0_cap4"),
)


@dataclass(frozen=True)
class Solution:
    candidate_id: str
    target: str
    kl_reg: float
    solver: str
    continuous: np.ndarray
    counts: np.ndarray


def fit_laws(weights: np.ndarray, outcomes: np.ndarray) -> list[olmix.OlmixLoglinearFit]:
    return [
        olmix.fit_olmix_loglinear_model(weights, outcomes[:, j], delta=HUBER_DELTA, seed=SEED, n_starts=N_STARTS)
        for j in range(outcomes.shape[1])
    ]


def predict_objective(fits: list[olmix.OlmixLoglinearFit], objective_weights: np.ndarray, w: np.ndarray) -> float:
    return float(
        sum(
            objective_weights[i] * (np.exp(fit.log_c) + np.exp(w @ np.asarray(fit.coefficients, float)))
            for i, fit in enumerate(fits)
        )
    )


def solve_exact(
    fits: list[olmix.OlmixLoglinearFit],
    objective_weights: np.ndarray,
    natural: np.ndarray,
    caps: np.ndarray,
    kl_reg: float,
) -> tuple[np.ndarray, str]:
    weights = cp.Variable(len(natural))
    predicted = sum(
        float(objective_weights[i])
        * (float(np.exp(fit.log_c)) + cp.exp(cp.sum(cp.multiply(np.asarray(fit.coefficients, float), weights))))
        for i, fit in enumerate(fits)
    )
    objective = predicted + kl_reg * cp.sum(cp.rel_entr(weights, natural)) if kl_reg > 0 else predicted
    problem = cp.Problem(cp.Minimize(objective), [weights >= 0, cp.sum(weights) == 1, weights <= caps])
    for solver in SOLVERS:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            solution = np.clip(np.asarray(weights.value, float), 0.0, None)
            return solution / solution.sum(), f"{solver}:{problem.status}"
    raise RuntimeError("no solver converged")


def quantize_under_cap(weights: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Round to the 1/2048 runtime grid, giving the remainder to the largest buckets with cap slack."""
    maximum = np.floor(EPOCH_CAP * tokens / TARGET_BUDGET * MIXTURE_BLOCK_SIZE + 1e-9).astype(np.int64)
    maximum = np.minimum(maximum, MIXTURE_BLOCK_SIZE)
    counts = np.minimum(np.floor(weights * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64), maximum)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    if remaining < 0:
        raise RuntimeError("floor rounding exceeded the block size")
    for bucket in np.argsort(-weights):
        if remaining == 0:
            break
        room = int(maximum[bucket] - counts[bucket])
        if room <= 0:
            continue
        added = min(room, remaining)
        counts[bucket] += added
        remaining -= added
    if remaining != 0 or int(counts.sum()) != MIXTURE_BLOCK_SIZE or np.any(counts > maximum):
        raise RuntimeError("runtime quantization failed under the epoch cap")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = benchmark.read_npz(FROZEN_PANEL)
    buckets = [str(b) for b in panel["buckets"]]
    if set(buckets) != set(base.DOMAIN_NAMES) or len(buckets) != len(base.DOMAIN_NAMES):
        raise ValueError("Frozen panel buckets differ from the launcher's domains")
    launcher_order = [buckets.index(domain) for domain in base.DOMAIN_NAMES]
    tokens = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b] for b in buckets], float)
    natural = tokens / tokens.sum()
    caps = np.minimum(1.0, tokens * EPOCH_CAP / TARGET_BUDGET)
    swarm_weights = np.asarray(panel["weights"], float)
    registry = pd.read_csv(HELDOUT, low_memory=False)

    fits: dict[str, list[olmix.OlmixLoglinearFit]] = {}
    objective_weights: dict[str, np.ndarray] = {}
    for target in TARGET_LABELS:
        outcomes = np.asarray(panel[f"{target}_outcomes"], float)
        objective_weights[target] = np.asarray(panel[f"{target}_aggregation_weights"], float)
        fits[target] = fit_laws(swarm_weights, outcomes)
        (args.output_dir / f"laws_{target}.json").write_text(
            json.dumps(
                [
                    {
                        "component": str(c),
                        "log_c": f.log_c,
                        "coefficients": list(f.coefficients),
                        "huber_loss": f.huber_loss,
                    }
                    for c, f in zip(panel[f"{target}_components"], fits[target], strict=True)
                ]
            )
        )
        print(f"fitted {len(fits[target])} {target} laws", flush=True)

    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = {}
    solution_frames = []
    for candidate_id, target, kl_reg, native_run_id in CASES:
        continuous, status = solve_exact(fits[target], objective_weights[target], natural, caps, kl_reg)
        counts = quantize_under_cap(continuous, tokens)
        runtime = counts / MIXTURE_BLOCK_SIZE
        epochs = TARGET_BUDGET * runtime / tokens
        native = registry[registry.source_row_id.fillna("").astype(str).str.startswith(native_run_id)]
        native_weights = np.asarray([float(native.iloc[0][f"weight::{b}"]) for b in buckets]) if len(native) else None
        summary[candidate_id] = {
            "target": target,
            "kl_reg": kl_reg,
            "solver": status,
            "predicted_continuous": predict_objective(fits[target], objective_weights[target], continuous),
            "predicted_runtime": predict_objective(fits[target], objective_weights[target], runtime),
            "tv_runtime_to_continuous": float(np.abs(runtime - continuous).sum() / 2),
            "max_epochs_runtime": float(epochs.max()),
            "active_buckets_runtime": int((counts > 0).sum()),
            "buckets_at_cap_runtime": int((epochs > EPOCH_CAP - 0.05).sum()),
            "native_run": native_run_id,
            "tv_runtime_to_native": (
                None if native_weights is None else float(np.abs(runtime - native_weights).sum() / 2)
            ),
            "predicted_at_native": (
                None
                if native_weights is None
                else predict_objective(fits[target], objective_weights[target], native_weights)
            ),
            "measured_native": (
                None
                if native_weights is None
                else float(native.iloc[0]["table9_macro_bpb" if target == "table9" else "uncheatable_bpb"])
            ),
        }
        solution_frames.append(
            pd.DataFrame(
                {
                    "candidate_id": candidate_id,
                    "bucket": buckets,
                    "continuous": continuous,
                    "runtime": runtime,
                    "native": native_weights if native_weights is not None else np.nan,
                }
            )
        )
        for index in launcher_order:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "epoch_cap": EPOCH_CAP,
                    "surrogate": SURROGATE,
                    "domain": buckets[index],
                    "runtime_count": int(counts[index]),
                    "weight": float(runtime[index]),
                    "materialized_epochs": float(epochs[index]),
                }
            )
        print(candidate_id, json.dumps(summary[candidate_id], indent=1), flush=True)

    with (args.output_dir / "candidate_weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pd.concat(solution_frames).to_csv(args.output_dir / "solutions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
