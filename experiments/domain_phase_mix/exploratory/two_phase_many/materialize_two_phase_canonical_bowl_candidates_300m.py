# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Materialize 3e18 validation candidates for canonical / effective-exposure / asymmetric-bowl DSP.

Records the local three-model comparison as concrete two-phase mixtures so Codex
can validate the *optima* at 3e18. For each objective (Table-9 macro BPB and
Uncheatable BPB) and each model, writes the proposed optimum weights at two
trust-region settings:

  - kl_0    : raw unconstrained optimum (diagnostic; expected to be a fantasy /
              degenerate corner -- canonical goes to a single-domain corner).
  - kl_0p1  : deployment optimum (the defensible candidate to actually validate).

Models
------
  canonical            : z = e0 + e1 ; benefit a(1+gamma*r)(1-e^-rho z), r=e1/z ;
                         penalty on raw z. Phase premium on benefit only.
  effective_exposure   : z = e0 + gamma*e1 ; benefit a(1-e^-rho z) ; penalty on same z.
  asymmetric_bowl      : z = e0 + gamma*e1 ; L = b0 + sum c-_i min(log1p z - mu,0)^2
                         + sum c+_i max(log1p z - mu,0)^2 (target-exposure U).

Local finding (300M, deletion-augmented panel; see the Fieldbook note): all three
MATCH-or-progress on OOF fit (canonical 0.865/0.896 < eff-exp 0.897/0.914 ~=
bowl 0.899/0.913 Spearman t9/unc) but differ sharply at the optimum. Raw-optimum
optimism (best_observed - predicted) decreases strictly canonical (0.56/0.21) ->
eff-exp (0.24/0.10) -> bowl (0.12/0.09), and the bowl proposes the most
conservative deployment tilt (kl0.1 TV-to-proportional 0.36/0.34 vs 0.50/0.36 vs
0.61/0.48). Expected 3e18 ordering therefore: bowl <= eff-exp < canonical.

Output layout (under --output-dir):
  candidate_manifest.csv
  bowl_model_<objective>.json
  <objective>/<model>/kl_<klstr>/proposed_mixture_weights.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "two_phase_dsp_canonical_bowl_candidates_20260703"
LINEAR_REG = 0.01
LOWER_TAIL_FRAC = 0.15
KL_REGS = (0.0, 0.1)


def load_objective(objective: str):
    if objective == "table9":
        _s, columns, domains, natural = base.load_raw_signal_panel()
        token_counts = base.load_domain_token_counts(domains)
        panel, _m = paper_olmix.build_fit_panel(columns)
        tb = base.load_target_budget()
        packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, "table9_macro_bpb")
        folds = component_dsp.panel_stratified_folds(panel, n_splits=5, seed=0)
        return packet, list(domains), np.asarray(natural, float), np.asarray(token_counts, float), int(tb), folds
    if objective == "uncheatable":
        packet, panel, domains, natural, token_counts, tb = eta_diag.load_packet()
        folds = component_dsp.panel_stratified_folds(panel, n_splits=5, seed=0)
        return packet, list(domains), np.asarray(natural, float), np.asarray(token_counts, float), int(tb), folds
    raise ValueError(f"unknown objective {objective!r}")


# ---------------- asymmetric bowl ----------------
def abowl_design(weights, c0, c1, mu, gamma):
    e0 = weights[:, 0, :] * c0[None, :]
    e1 = weights[:, 1, :] * c1[None, :]
    d = np.log1p(e0 + gamma * e1) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


def fit_head(design, y, l2):
    dmean = design.mean(0, keepdims=True)
    ymean = float(y.mean())
    cd, ct = design - dmean, y - ymean
    if l2 > 0:
        cd = np.vstack([cd, np.sqrt(l2) * np.eye(cd.shape[1])])
        ct = np.concatenate([ct, np.zeros(cd.shape[1])])
    coef, _ = nnls(cd, ct)
    return ymean - float((dmean @ coef).item()), coef


def abowl_predict(weights, c0, c1, model):
    return model["b0"] + abowl_design(weights, c0, c1, model["mu"], model["gamma"]) @ model["coef"]


def _abowl_profile(theta, packet, l2):
    m = packet.m
    mu, gamma = theta[:m], float(np.exp(theta[m]))
    design = abowl_design(packet.w, packet.c0, packet.c1, mu, gamma)
    b0, coef = fit_head(design, packet.y, l2)
    pred = b0 + design @ coef
    rmse = float(np.sqrt(np.mean((pred - packet.y) ** 2)))
    tail = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    idx = np.argsort(pred)[:tail]
    return rmse + 0.5 * float(np.mean(np.maximum(packet.y[idx] - pred[idx], 0.0)))


def fit_asymmetric_bowl(packet, l2):
    m = packet.m
    z0 = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    base_mu = np.clip(np.median(np.log1p(np.where(z0 > 1e-8, z0, np.nan)), axis=0), -2.0, 8.0)
    base_mu = np.where(np.isfinite(base_mu), base_mu, 2.0)
    bounds = [(-2.0, 8.0)] * m + [(np.log(1e-4), np.log(100.0))]
    best = None
    for gamma in (1.0, 4.0, 16.0):
        for shift in (-1.5, -0.5, 0.5):
            s = np.concatenate([np.clip(base_mu + shift, -2, 8), [np.log(gamma)]])
            res = minimize(lambda t: _abowl_profile(t, packet, l2), s, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 80, "ftol": 1e-8})
            if best is None or float(res.fun) < float(best.fun):
                best = res
    theta = np.asarray(best.x, float)
    mu, gamma = theta[:m], float(np.exp(theta[m]))
    b0, coef = fit_head(abowl_design(packet.w, packet.c0, packet.c1, mu, gamma), packet.y, l2)
    return {"mu": mu, "gamma": gamma, "b0": b0, "coef": coef}


# ---------------- optimum with KL ----------------
def optimize(predict_fn, m, natural, kl_reg, starts):
    def to_w(logits):
        out = np.zeros((2, m))
        for ph in range(2):
            zz = logits[ph * m:(ph + 1) * m]
            e = np.exp(zz - zz.max())
            out[ph] = e / e.sum()
        return out

    def obj(logits):
        w = to_w(logits)
        pred = float(predict_fn(w))
        if kl_reg <= 0:
            return pred
        return pred + kl_reg * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS))

    best_v, best_w = np.inf, None
    for s in starts:
        res = minimize(obj, s, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-9})
        if float(res.fun) < best_v:
            best_v, best_w = float(res.fun), to_w(res.x)
    return best_w


def opt_starts(packet, m, k=16):
    starts = [np.zeros(2 * m)]
    for idx in np.argsort(packet.y)[:k]:
        starts.append(np.log(np.clip(packet.w[int(idx)], 1e-12, 1.0)).reshape(-1))
    return starts


def kl_str(kl):
    return "0" if kl == 0 else str(kl).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="table9,uncheatable")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dsp.LINEAR_REG = LINEAR_REG

    manifest_rows = []
    for objective in [o.strip() for o in args.objectives.split(",") if o.strip()]:
        print(f"==== {objective} ====", flush=True)
        packet, domains, natural, token_counts, tb, folds = load_objective(objective)
        m = packet.m
        canonical, _ = phase_dsp.fit_variant_with_l2(packet, "canonical", LINEAR_REG, maxiter=40, coarse_top_k=3, basin_hopping_iters=0)
        effexp, _ = phase_dsp.fit_variant_with_l2(packet, "effective_exposure", LINEAR_REG, maxiter=40, coarse_top_k=3, basin_hopping_iters=0)
        bowl = fit_asymmetric_bowl(packet, LINEAR_REG)
        (args.output_dir / f"bowl_model_{objective}.json").write_text(json.dumps(
            {"objective": objective, "gamma": bowl["gamma"], "b0": bowl["b0"], "domains": domains,
             "mu": bowl["mu"].tolist(), "c_under": bowl["coef"][:m].tolist(), "c_over": bowl["coef"][m:].tolist()},
            indent=2) + "\n")

        predict_fns = {
            "canonical": lambda w: float(dsp.predict(canonical, w[None, :, :])[0]),
            "effective_exposure": lambda w: float(dsp.predict(effexp, w[None, :, :])[0]),
            "asymmetric_bowl": lambda w: float(abowl_predict(w[None, :, :], packet.c0, packet.c1, bowl)[0]),
        }
        best_observed = float(np.min(packet.y))
        starts = opt_starts(packet, m)
        reference = np.stack([natural, natural], axis=0)
        for model_name, fn in predict_fns.items():
            for kl in KL_REGS:
                weights = optimize(fn, m, natural, kl, starts)
                frame = per_component.mixture_frame(domains=domains, natural=natural, weights=weights,
                                                    token_counts=token_counts, target_budget=tb)
                out_dir = args.output_dir / objective / model_name / f"kl_{kl_str(kl)}"
                out_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(out_dir / "proposed_mixture_weights.csv", index=False)
                dists = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
                nidx = int(np.argmin(dists))
                sim = base.simulated_epochs(weights, token_counts, target_budget=tb)
                pred = float(fn(weights))
                manifest_rows.append({
                    "objective": objective, "model": model_name, "kl_reg": kl,
                    "candidate": "raw_diagnostic" if kl == 0 else "deployment",
                    "weights_csv": str((out_dir / "proposed_mixture_weights.csv").relative_to(args.output_dir)),
                    "predicted_bpb": pred, "best_observed_bpb": best_observed,
                    "optimism_vs_best_observed": best_observed - pred,
                    "nearest_observed_bpb": float(packet.y[nidx]), "nearest_observed_tv": float(dists[nidx]),
                    "tv_to_proportional": float(0.5 * np.abs(weights - reference).sum(axis=1).mean()),
                    "max_weight": float(np.max(weights)), "max_simulated_epoch": float(np.max(sim)),
                    "q95_simulated_epoch": float(np.quantile(sim, 0.95)),
                })
                print(f"  {model_name:20s} kl={kl}: pred={pred:.4f} tv_prop={manifest_rows[-1]['tv_to_proportional']:.3f} "
                      f"maxw={manifest_rows[-1]['max_weight']:.3f} maxepoch={manifest_rows[-1]['max_simulated_epoch']:.1f}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    print("\n=== candidate_manifest.csv ===")
    pd.set_option("display.width", 220)
    print(manifest[["objective", "model", "kl_reg", "candidate", "predicted_bpb", "tv_to_proportional",
                    "max_weight", "q95_simulated_epoch"]].round(4).to_string(index=False))
    print(f"\nWrote candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
