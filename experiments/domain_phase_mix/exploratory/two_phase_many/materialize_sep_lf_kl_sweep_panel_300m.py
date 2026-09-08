# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "fsspec", "gcsfs"]
# ///
# ruff: noqa: E402, E501
"""Materialize the separate-heads KL sweep (+ Approach-A baselines) for 3e18 validation, both targets.

Separate-heads is the one genuinely-distinct argmin (LEARN-FORGET is ~0.02 from eff-exp on the 300M swarm).
Its raw optimum (KL=0) is a sensible curriculum (code-heavy aggregate, fresh-value late, broad-early) with a
localized over-epoching fantasy on a few tiny synthetic domains and tolerable max epochs (15-17), so KL=0 is
included in the sweep. Per target: separate_heads at KL {0,0.1,0.2,0.3,0.4}, plus eff-exp two-phase, eff-exp
one-phase, and one LEARN-FORGET (confirming ~eff-exp) at KL=0.2. All bowl-link; sep L2=0.1 (pinned locally).
Weights via per_component.mixture_frame (consistent simulated_epochs, cache-safe distinct weights)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_olmix_reference_deletion_augmented_300m as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_two_phase_canonical_bowl_candidates_300m as C,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_lf_sepheads_kl_sweep_300m import build_predictors
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_two_phase_canonical_bowl_kl_sweep_300m import (
    optimize_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "sep_lf_kl_sweep_panel_20260706"
SEP_KLS = (0.0, 0.1, 0.2, 0.3, 0.4)
BASELINE_KL = 0.2
TARGET_ABBR = {"uncheatable": "unch", "table9": "t9"}


def kl_tag(kl: float) -> str:
    return f"kl{kl:g}".replace(".", "p")


def one_phase_argmin(
    predict_fn, m: int, natural: np.ndarray, kl: float, starts: list[np.ndarray], maxiter: int
) -> np.ndarray:
    """Argmin restricted to one-phase mixtures (w0 == w1); the transferable one-phase reference."""

    def to_p(lg: np.ndarray) -> np.ndarray:
        e = np.exp(lg - lg.max())
        return e / e.sum()

    def obj(lg: np.ndarray) -> float:
        p = to_p(lg)
        w = np.stack([p, p])
        val = float(predict_fn(w))
        if kl > 0:
            val += kl * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS))
        return val

    best_val, best_p = np.inf, None
    for st in starts:
        res = minimize(obj, np.asarray(st[:m], float), method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
        if float(res.fun) < best_val:
            best_val, best_p = float(res.fun), to_p(np.asarray(res.x, float))
    assert best_p is not None
    return np.stack([best_p, best_p])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--maxiter", type=int, default=250)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for target in ("uncheatable", "table9"):
        print(f"==== {target} ====", flush=True)
        packet, domains, natural, token_counts, target_budget, _ = C.load_objective(target)
        preds = build_predictors(packet)
        starts = C.opt_starts(packet, packet.m, k=4)
        # (candidate_suffix, model_key, kl, one_phase)
        jobs: list[tuple[str, str, float, bool]] = [(f"sep_{kl_tag(kl)}", "separate_heads", kl, False) for kl in SEP_KLS]
        jobs += [
            (f"effexp2p_{kl_tag(BASELINE_KL)}", "eff_exp", BASELINE_KL, False),
            (f"effexp1p_{kl_tag(BASELINE_KL)}", "eff_exp", BASELINE_KL, True),
            (f"lf2p_{kl_tag(BASELINE_KL)}", "learn_forget", BASELINE_KL, False),
        ]
        for suffix, model_key, kl, one_phase in jobs:
            fn = preds[model_key]
            weights = (
                one_phase_argmin(fn, packet.m, natural, kl, starts, args.maxiter)
                if one_phase
                else optimize_fast(fn, packet.m, natural, kl, starts, args.maxiter)
            )
            frame = per_component.mixture_frame(
                domains=domains, natural=natural, weights=weights, token_counts=token_counts, target_budget=target_budget
            )
            cand = f"seplf_{TARGET_ABBR[target]}_{suffix}"
            cand_dir = args.output_dir / cand
            cand_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cand_dir / "proposed_mixture_weights.csv", index=False)
            agg = base.aggregate_phase_weights(weights)
            sim = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
            rows.append(
                {
                    "candidate": cand,
                    "target": target,
                    "model": model_key,
                    "kl": kl,
                    "one_phase": one_phase,
                    "predicted_bpb": float(fn(weights)),
                    "agg_tv_to_prop": float(0.5 * np.abs(agg - natural).sum()),
                    "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
                    "max_sim_epoch": float(sim.max()),
                    "q95_sim_epoch": float(np.quantile(sim, 0.95)),
                }
            )
            print(
                f"  {cand:28s} pred={rows[-1]['predicted_bpb']:.4f} aggTV={rows[-1]['agg_tv_to_prop']:.3f} phaseTV={rows[-1]['phase_tv']:.3f} maxep={rows[-1]['max_sim_epoch']:.1f}",
                flush=True,
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"\nWrote {len(rows)} mixtures ({len(SEP_KLS)} sep KL x 2 targets + 3 baselines x 2) to {args.output_dir}")


if __name__ == "__main__":
    main()
