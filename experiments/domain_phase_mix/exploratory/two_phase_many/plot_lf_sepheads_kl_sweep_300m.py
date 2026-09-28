# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
# ruff: noqa: E402
"""Local KL sweep for LEARN-FORGET and separate-heads vs eff-exp / asym-bowl.

Companion to `plot_two_phase_canonical_bowl_kl_sweep_300m.py` (reuses its plotting). Overlays four
surrogates' KL-regularized argmin diagnostics (predicted BPB, TV-to-proportional, max epochs, max weight)
so the sane KL range for separate-heads (the one genuinely-distinct argmin) can be read off before choosing
the 3e18 sweep bracket. All bowl-link; LINEAR_REG=0.01 except separate-heads (L2=0.1, pinned locally)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_olmix_reference_deletion_augmented_300m as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_two_phase_canonical_bowl_candidates_300m as C,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_two_phase_canonical_bowl_kl_sweep_300m import (
    kl_label,
    optimize_fast,
    plot_metric,
    weights_to_logits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "lf_sepheads_kl_sweep_20260706"
DEFAULT_KL_REGS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
SEP_L2 = 0.1  # pinned locally (controls the 4m-param concentration)


def _gridmu(w, cc, gamma, c1, y, l2):
    E = w[:, 0, :] * cc[None, :] + gamma * w[:, 1, :] * c1[None, :]
    b = np.clip(np.median(np.log1p(np.where(E > 1e-8, E, np.nan)), 0), -2, 8)
    b = np.where(np.isfinite(b), b, 2.0)
    best = None
    for s in np.linspace(-2, 2, 13):
        mu = np.clip(b + s, -2, 8)
        D = C.abowl_design(w, cc, c1, mu, gamma)
        b0, co = C.fit_head(D, y, l2)
        r = float(np.sqrt(np.mean((y - (b0 + D @ co)) ** 2)))
        if best is None or r < best[0]:
            best = (r, mu)
    return best[1]


def build_predictors(packet):
    """Return {model_name: predict_fn(w[2,m])} for eff-exp / asym-bowl / learn_forget / separate_heads."""
    c0, c1, w, y = packet.c0, packet.c1, packet.w, packet.y
    zero_c0 = np.zeros_like(c0)  # phase-0 zeroed -> bowl on e1 alone
    dsp.LINEAR_REG = C.LINEAR_REG
    effexp, _ = C.phase_dsp.fit_variant_with_l2(
        packet, "effective_exposure", C.LINEAR_REG, maxiter=40, coarse_top_k=3, basin_hopping_iters=0
    )
    bowl = C.fit_asymmetric_bowl(packet, C.LINEAR_REG)
    mus, gam = bowl["mu"], bowl["gamma"]
    # LEARN-FORGET: asym-bowl design + per-mixture forget column (phase-0 exposure of dropped domains)
    W0, W1 = w[:, 0, :], w[:, 1, :]
    forg = np.sum((W0 * c0[None, :]) * np.maximum(W0 - W1, 0.0), axis=1, keepdims=True)
    D_lf = np.hstack([C.abowl_design(w, c0, c1, mus, gam), forg])
    b_lf, co_lf = C.fit_head(D_lf, y, C.LINEAR_REG)
    # separate-heads: bowl(e0; mu0) + bowl(e1; mu1), L2=SEP_L2
    mu0 = _gridmu(w, c0, 0.0, c1, y, SEP_L2)
    mu1 = _gridmu(w, zero_c0, 1.0, c1, y, SEP_L2)
    D_sep = np.hstack([C.abowl_design(w, c0, c1, mu0, 0.0), C.abowl_design(w, zero_c0, c1, mu1, 1.0)])
    b_sep, co_sep = C.fit_head(D_sep, y, SEP_L2)

    def p_eff(ww):
        return float(dsp.predict(effexp, ww[None, :, :])[0])

    def p_abowl(ww):
        return float(C.abowl_predict(ww[None, :, :], c0, c1, bowl)[0])

    def p_lf(ww):
        e0 = ww[0] * c0
        d = np.hstack(
            [C.abowl_design(ww[None, :, :], c0, c1, mus, gam), [[float(np.sum(e0 * np.maximum(ww[0] - ww[1], 0.0)))]]]
        )
        return float((b_lf + d @ co_lf).item())

    def p_sep(ww):
        d = np.hstack(
            [C.abowl_design(ww[None, :, :], c0, c1, mu0, 0.0), C.abowl_design(ww[None, :, :], zero_c0, c1, mu1, 1.0)]
        )
        return float((b_sep + d @ co_sep).item())

    return {"eff_exp": p_eff, "asym_bowl": p_abowl, "learn_forget": p_lf, "separate_heads": p_sep}


def sweep(objective, kl_regs, maxiter):
    packet, _domains, natural, token_counts, target_budget, _ = C.load_objective(objective)
    preds = build_predictors(packet)
    starts = C.opt_starts(packet, packet.m, k=2)
    ref = np.stack([natural, natural], axis=0)
    rows = []
    for name, fn in preds.items():
        warm = list(starts)
        for kl in sorted(kl_regs, reverse=True):
            print(f"  {objective}/{name} KL={kl:g}", flush=True)
            wts = optimize_fast(fn, packet.m, natural, kl, warm, maxiter)
            warm = [weights_to_logits(wts), *starts]
            sim = base.simulated_epochs(wts, token_counts, target_budget=target_budget)
            rows.append(
                {
                    "objective": objective,
                    "model": name,
                    "kl_reg": kl,
                    "kl_label": kl_label(kl),
                    "predicted_bpb": float(fn(wts)),
                    "tv_to_proportional": float(0.5 * np.abs(wts - ref).sum(1).mean()),
                    "max_weight": float(np.max(wts)),
                    "max_simulated_epoch": float(np.max(sim)),
                    "phase_tv": float(0.5 * np.abs(wts[0] - wts[1]).sum()),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--kl-regs", default=",".join(str(k) for k in DEFAULT_KL_REGS))
    ap.add_argument("--maxiter", type=int, default=120)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kl_regs = [float(k) for k in args.kl_regs.split(",") if k.strip()]
    rows = []
    for obj in ("table9", "uncheatable"):
        print(f"==== {obj} ====", flush=True)
        rows.extend(sweep(obj, kl_regs, args.maxiter))
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "kl_sweep_diagnostics.csv", index=False)
    for metric in ("predicted_bpb", "tv_to_proportional", "max_simulated_epoch", "phase_tv"):
        plot_metric(df, metric, args.output_dir / f"kl_sweep_{metric}.html", kl_regs)
    print(df.to_string(index=False))
    print(f"Wrote plots + CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
