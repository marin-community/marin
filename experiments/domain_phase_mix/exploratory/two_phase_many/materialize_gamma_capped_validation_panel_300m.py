# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
# ruff: noqa: E501
"""Materialize the 3e18 validation panel for the gamma-capped effective-exposure bowl.

Spends the (expensive) 3e18 budget only on the anti-transferable axis -- the things the
300M panel cannot select. Domain-response knobs (linear_reg, link, mu/c) are settled
locally; this panel tests only:

  gamma-cliff  : does gamma_eff ~= 8-12 beat gamma_eff ~= 16 (~= DSP's RMSE fit) and gamma=1
                 (aggregate) at 3e18? (bowl, two-phase, KL=0.2, gamma in {1,8,10,12,16})
  KL sweep     : trust-region for the gamma-capped model (bowl, two-phase, gamma=10, KL in {0.1,0.3})
  phase        : one-phase vs two-phase; the one-phase gamma-capped mix is the best shot to
                 beat the current best one-phase dsp_onephase = 1.070728 (bowl, gamma=10, 1phase)
  link ablation: eff-exp vs bowl domain response at matched gamma (does the bowl link help at 3e18?)
  cross-objective: the recommended config on Uncheatable BPB

z_i = e0_i + gamma_eff*e1_i. Weights written in per_component.mixture_frame format.
Compare against already-validated anchors (no re-run): dsp_onephase 1.0707, prior asym-bowl
kl0.2 1.0909, proportional 1.1987. Budget 2-3 seed repeats on the observed winner (noise floor
Table-9 macro sd = 0.0038 -> two-run difference-sd 0.0053; gaps < ~0.008 are not resolvable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "gamma_capped_bowl_3e18_validation_panel_20260704"
LINEAR_REG = 0.01
LTF = 0.15

# (objective, link, gamma_eff, kl, phase, sweep_role)
PANEL = [
    ("table9", "bowl", 1.0, 0.2, "two", "gamma_cliff:aggregate(g1)"),
    ("table9", "bowl", 8.0, 0.2, "two", "gamma_cliff:sweetspot_low"),
    ("table9", "bowl", 10.0, 0.2, "two", "anchor:recommended"),
    ("table9", "bowl", 12.0, 0.2, "two", "gamma_cliff:sweetspot_high"),
    ("table9", "bowl", 16.0, 0.2, "two", "gamma_cliff:over_cliff(~DSP)"),
    ("table9", "bowl", 10.0, 0.1, "two", "kl_sweep:0.1"),
    ("table9", "bowl", 10.0, 0.3, "two", "kl_sweep:0.3"),
    ("table9", "bowl", 10.0, 0.1, "one", "one_phase:kl0.1"),
    ("table9", "bowl", 10.0, 0.2, "one", "one_phase:kl0.2"),
    ("table9", "effexp", 10.0, 0.2, "two", "link_ablation:effexp"),
    ("uncheatable", "bowl", 10.0, 0.2, "two", "cross_objective:two_phase"),
    ("uncheatable", "bowl", 10.0, 0.2, "one", "cross_objective:one_phase"),
]


def load(objective):
    if objective == "table9":
        _s, columns, domains, natural = base.load_raw_signal_panel()
        tc = base.load_domain_token_counts(domains)
        panel, _m = paper_olmix.build_fit_panel(columns)
        tb = base.load_target_budget()
        packet = top_level_dsp.build_dsp_packet(panel, columns, domains, tc, "table9_macro_bpb")
        return packet, list(domains), np.asarray(natural, float), np.asarray(tc, float), int(tb)
    packet, panel, domains, natural, tc, tb = eta_diag.load_packet()
    return packet, list(domains), np.asarray(natural, float), np.asarray(tc, float), int(tb)


# ---- asymmetric bowl on z = e0 + gamma*e1 ----
def bowl_design(z, mu):
    d = np.log1p(z) - mu[None, :]
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


def fit_bowl(packet, gamma, l2):
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    z = e0 + gamma * e1
    m = packet.m
    base_mu = np.clip(np.median(np.log1p(np.where(z > 1e-8, z, np.nan)), axis=0), -2.0, 8.0)
    base_mu = np.where(np.isfinite(base_mu), base_mu, 2.0)

    def prof(mu):
        de = bowl_design(z, mu)
        b0, c = fit_head(de, packet.y, l2)
        p = b0 + de @ c
        r = float(np.sqrt(np.mean((p - packet.y) ** 2)))
        idx = np.argsort(p)[: max(5, int(np.ceil(LTF * len(packet.y))))]
        return r + 0.5 * float(np.mean(np.maximum(packet.y[idx] - p[idx], 0.0)))

    best = None
    for sh in (-1.5, -0.5, 0.5):
        rr = minimize(
            prof,
            np.clip(base_mu + sh, -2, 8),
            method="L-BFGS-B",
            bounds=[(-2.0, 8.0)] * m,
            options={"maxiter": 80, "ftol": 1e-8},
        )
        if best is None or float(rr.fun) < float(best.fun):
            best = rr
    mu = np.asarray(best.x, float)
    b0, c = fit_head(bowl_design(z, mu), packet.y, l2)

    def predict(w, c0=packet.c0, c1=packet.c1, gamma=gamma, mu=mu, b0=b0, c=c):
        z = w[0] * c0 + gamma * (w[1] * c1)
        return float((b0 + bowl_design(z[None, :], mu) @ c)[0])

    return predict


# ---- eff-exp DSP with gamma pinned ----
def fit_effexp_pinned(packet, gamma, l2):
    variant = dsp.VARIANTS["effective_exposure"]
    m = packet.m
    free_bounds = dsp.bounds(variant, m)[: 2 * m]
    log_g = float(np.log(gamma))
    starts = dsp.start_bank(packet, variant)
    orig = dsp.LINEAR_REG
    dsp.LINEAR_REG = l2
    try:
        best = None
        for s in starts[:4]:
            res = minimize(
                lambda th: dsp.profile_objective(packet, variant, np.concatenate([np.asarray(th, float), [log_g]])),
                s[: 2 * m],
                method="L-BFGS-B",
                bounds=free_bounds,
                options={"maxiter": 40, "ftol": 1e-7},
            )
            if best is None or float(res.fun) < float(best.fun):
                best = res
        params = dsp.unpack_theta(np.concatenate([np.asarray(best.x, float), [log_g]]), variant, m)
        model = dsp.fit_linear_head(packet.w, packet.y, packet, variant, params)
    finally:
        dsp.LINEAR_REG = orig
    return lambda w: float(dsp.predict(model, w[None, :, :])[0])


def optimize(pred, m, natural, kl, one_phase):
    def to_w(lg):
        if one_phase:
            e = np.exp(lg - lg.max())
            p = e / e.sum()
            return np.stack([p, p])
        out = np.zeros((2, m))
        for ph in range(2):
            z = lg[ph * m : (ph + 1) * m]
            e = np.exp(z - z.max())
            out[ph] = e / e.sum()
        return out

    def obj(lg):
        w = to_w(lg)
        return pred(w) + (kl * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS)) if kl > 0 else 0.0)

    dim = m if one_phase else 2 * m
    seed = np.log(np.clip(natural, 1e-9, 1))
    starts = [np.zeros(dim), (seed if one_phase else np.concatenate([seed, seed]))]
    bv, bw = np.inf, None
    for s in starts:
        r = minimize(obj, s, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-10})
        if float(r.fun) < bv:
            bv, bw = float(r.fun), to_w(r.x)
    return bw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packets = {}
    predictors = {}
    rows = []
    for objective, link, gamma, kl, phase, role in PANEL:
        if objective not in packets:
            packets[objective] = load(objective)
        packet, domains, natural, tc, tb = packets[objective]
        key = (objective, link, gamma)
        if key not in predictors:
            predictors[key] = (
                fit_bowl(packet, gamma, LINEAR_REG) if link == "bowl" else fit_effexp_pinned(packet, gamma, LINEAR_REG)
            )
        pred = predictors[key]
        w = optimize(pred, packet.m, natural, kl, one_phase=(phase == "one"))
        frame = per_component.mixture_frame(
            domains=domains, natural=natural, weights=w, token_counts=tc, target_budget=tb
        )
        name = f"{link}_g{gamma:g}_kl{str(kl).replace('.', 'p')}_{phase}phase"
        d = args.output_dir / objective / name
        d.mkdir(parents=True, exist_ok=True)
        frame.to_csv(d / "proposed_mixture_weights.csv", index=False)
        ref = np.stack([natural, natural])
        sim = base.simulated_epochs(w, tc, target_budget=tb)
        rows.append(
            {
                "candidate": f"{objective}/{name}",
                "objective": objective,
                "link": link,
                "gamma_eff": gamma,
                "kl_reg": kl,
                "phase": phase,
                "sweep_role": role,
                "weights_csv": str((d / "proposed_mixture_weights.csv").relative_to(args.output_dir)),
                "predicted_bpb_300m": pred(w),
                "tv_to_proportional": float(0.5 * np.abs(w - ref).sum(axis=1).mean()),
                "max_weight": float(w.max()),
                "max_simulated_epoch": float(sim.max()),
                "q95_simulated_epoch": float(np.quantile(sim, 0.95)),
            }
        )
        print(
            f"  {name:38s} [{objective:11s}] tv={rows[-1]['tv_to_proportional']:.3f} maxw={rows[-1]['max_weight']:.3f} q95ep={rows[-1]['q95_simulated_epoch']:.2f}",
            flush=True,
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== candidate_manifest.csv ===")
    print(
        manifest[
            ["candidate", "sweep_role", "predicted_bpb_300m", "tv_to_proportional", "max_weight", "q95_simulated_epoch"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nWrote {len(rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
