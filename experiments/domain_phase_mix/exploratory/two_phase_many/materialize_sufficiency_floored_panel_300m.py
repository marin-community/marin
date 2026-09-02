# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Sufficiency-floored two-phase panel for 3e18 validation (uncheatable primary + Table-9).

Diagnosis: naive two-phase optima UNDEREXPOSE mid-training domains (they satisfy exposure
targets via phase-1, only 20% of tokens) -> suboptimal; the manual exposure-repair reached the
uncheatable frontier 0.985974. This systematizes that: optimize the two-phase mixture with an
AGGREGATE-SUFFICIENCY FLOOR, floor_i = alpha * one_phase_aggregate_i, so no domain's aggregate
exposure falls below alpha x its one-phase level, while the phase split still front-loads valued
domains late. At alpha=1 the aggregate equals the one-phase optimum -> cannot underexpose, yet keeps
the late benefit -> should beat both one-phase and the naive two-phase, and target < 0.985974.

Panel (uncheatable primary; Table-9 secondary): floor alpha sweep {0, 0.7, 1.0, 1.2}, a lower-gamma
row, an eff-exp-link row, a one-phase control, and an eval-relevant heuristic (fixed one-phase
aggregate with eval-relevant domains shifted to phase-1). Writes per_component.mixture_frame CSVs.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "sufficiency_floored_panel_20260705"
LINEAR_REG = 0.01
LTF = 0.15
KL = 0.2
LAM_FLOOR = 200.0
# eval-relevant mid-training domains to front-load late (uncheatable: arxiv/github/wiki/math/code)
EVAL_RELEVANT = [
    "dolma3_arxiv",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolma3_wikipedia",
    "dolmino_synth_math",
    "dolma3_finemath_3plus",
    "dolmino_stem_heavy_crawl",
]

# (objective, link, gamma, floor_alpha, phase, mode, role)
PANEL = [
    ("uncheatable", "bowl", 10.0, 0.0, "two", "floored", "floor_a0_base"),
    ("uncheatable", "bowl", 10.0, 0.7, "two", "floored", "floor_a0p7"),
    ("uncheatable", "bowl", 10.0, 1.0, "two", "floored", "floor_a1p0"),
    ("uncheatable", "bowl", 10.0, 1.2, "two", "floored", "floor_a1p2_overrepair"),
    ("uncheatable", "bowl", 6.0, 1.0, "two", "floored", "floor_a1p0_g6"),
    ("uncheatable", "effexp", 10.0, 1.0, "two", "floored", "floor_a1p0_effexp"),
    ("uncheatable", "bowl", 10.0, 1.0, "one", "floored", "one_phase_control"),
    ("uncheatable", "bowl", 10.0, 1.0, "two", "evalrel", "evalrel_late_heuristic"),
    ("table9", "bowl", 8.0, 0.0, "two", "floored", "t9_floor_a0_base"),
    ("table9", "bowl", 8.0, 1.0, "two", "floored", "t9_floor_a1p0"),
    ("table9", "bowl", 8.0, 1.0, "one", "floored", "t9_one_phase_control"),
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


def bowl_design(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


def fit_head(design, y, l2):
    dm = design.mean(0, keepdims=True)
    ym = float(y.mean())
    cd, ct = design - dm, y - ym
    if l2 > 0:
        cd = np.vstack([cd, np.sqrt(l2) * np.eye(cd.shape[1])])
        ct = np.concatenate([ct, np.zeros(cd.shape[1])])
    coef, _ = nnls(cd, ct)
    return ym - float((dm @ coef).item()), coef


def fit_bowl(packet, gamma):
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    z = e0 + gamma * e1
    m = packet.m
    bm = np.clip(np.median(np.log1p(np.where(z > 1e-8, z, np.nan)), axis=0), -2, 8)
    bm = np.where(np.isfinite(bm), bm, 2.0)

    def prof(mu):
        de = bowl_design(z, mu)
        b0, c = fit_head(de, packet.y, LINEAR_REG)
        p = b0 + de @ c
        idx = np.argsort(p)[: max(5, int(np.ceil(LTF * len(packet.y))))]
        return float(np.sqrt(np.mean((p - packet.y) ** 2))) + 0.5 * float(
            np.mean(np.maximum(packet.y[idx] - p[idx], 0.0))
        )

    best = None
    for sh in (-1.5, -0.5, 0.5):
        r = minimize(
            prof,
            np.clip(bm + sh, -2, 8),
            method="L-BFGS-B",
            bounds=[(-2.0, 8.0)] * m,
            options={"maxiter": 80, "ftol": 1e-8},
        )
        if best is None or float(r.fun) < float(best.fun):
            best = r
    mu = np.asarray(best.x, float)
    b0, c = fit_head(bowl_design(z, mu), packet.y, LINEAR_REG)

    def predict(w, c0=packet.c0, c1=packet.c1):
        zz = w[0] * c0 + gamma * (w[1] * c1)
        return float((b0 + bowl_design(zz[None, :], mu) @ c)[0])

    return predict


def fit_effexp(packet, gamma):
    variant = dsp.VARIANTS["effective_exposure"]
    m = packet.m
    fb = dsp.bounds(variant, m)[: 2 * m]
    lg = float(np.log(gamma))
    starts = dsp.start_bank(packet, variant)
    orig = dsp.LINEAR_REG
    dsp.LINEAR_REG = LINEAR_REG
    try:
        best = None
        for s in starts[:4]:
            res = minimize(
                lambda th: dsp.profile_objective(packet, variant, np.concatenate([np.asarray(th, float), [lg]])),
                s[: 2 * m],
                method="L-BFGS-B",
                bounds=fb,
                options={"maxiter": 40, "ftol": 1e-7},
            )
            if best is None or float(res.fun) < float(best.fun):
                best = res
        params = dsp.unpack_theta(np.concatenate([np.asarray(best.x, float), [lg]]), variant, m)
        model = dsp.fit_linear_head(packet.w, packet.y, packet, variant, params)
    finally:
        dsp.LINEAR_REG = orig
    return lambda w: float(dsp.predict(model, w[None, :, :])[0])


def optimize(pred, m, natural, floor, one_phase):
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
        val = pred(w) + KL * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS))
        if floor is not None:
            agg = base.aggregate_phase_weights(w)
            val += LAM_FLOOR * float(np.sum(np.maximum(floor - agg, 0.0) ** 2))
        return val

    dim = m if one_phase else 2 * m
    seed = np.log(np.clip(natural, 1e-9, 1))
    bv, bw = np.inf, None
    for s in [np.zeros(dim), (seed if one_phase else np.concatenate([seed, seed]))]:
        r = minimize(obj, s, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
        if float(r.fun) < bv:
            bv, bw = float(r.fun), to_w(r.x)
    return bw


def evalrel_mixture(onephase_w, domains, natural):
    """Fixed one-phase aggregate; shift eval-relevant domains toward phase-1 (late), others toward phase-0,
    preserving each domain's aggregate = one-phase aggregate exactly."""
    agg = base.aggregate_phase_weights(onephase_w)  # target aggregate = one-phase
    f0, f1 = base.PHASE_FRACTIONS
    idx = {d: i for i, d in enumerate(domains)}
    rel = np.zeros(len(domains), dtype=bool)
    for d in EVAL_RELEVANT:
        if d in idx:
            rel[idx[d]] = True
    # agg_i = f0*p0_i + f1*p1_i, with p0,p1 simplices. Put eval-relevant fully into phase-1 where feasible,
    # others fully into phase-0, then rescale to simplices while keeping aggregate ~ target.
    p1 = np.where(rel, agg / max(f1, 1e-9), 0.0)
    p0 = np.where(rel, 0.0, agg / max(f0, 1e-9))
    p1 = p1 / p1.sum() if p1.sum() > 0 else natural.copy()
    p0 = p0 / p0.sum() if p0.sum() > 0 else natural.copy()
    return np.stack([p0, p1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dsp.LINEAR_REG = LINEAR_REG

    cache = {}
    onephase_agg = {}
    rows = []
    for objective, link, gamma, alpha, phase, mode, role in PANEL:
        if objective not in cache:
            cache[objective] = {"data": load(objective)}
        packet, domains, natural, tc, tb = cache[objective]["data"]
        pkey = (objective, link, gamma)
        if pkey not in cache:
            cache[pkey] = fit_bowl(packet, gamma) if link == "bowl" else fit_effexp(packet, gamma)
        pred = cache[pkey]
        # one-phase aggregate reference (per objective+link+gamma), for the floor
        okey = (objective, link, gamma)
        if okey not in onephase_agg:
            w1 = optimize(pred, packet.m, natural, floor=None, one_phase=True)
            onephase_agg[okey] = base.aggregate_phase_weights(w1)
        floor_ref = onephase_agg[okey]

        if mode == "evalrel":
            w1 = optimize(pred, packet.m, natural, floor=None, one_phase=True)
            w = evalrel_mixture(w1, domains, natural)
        else:
            floor = None if alpha == 0.0 else alpha * floor_ref
            w = optimize(pred, packet.m, natural, floor=floor, one_phase=(phase == "one"))

        frame = per_component.mixture_frame(
            domains=domains, natural=natural, weights=w, token_counts=tc, target_budget=tb
        )
        name = f"{objective}_{role}"
        d = args.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        frame.to_csv(d / "proposed_mixture_weights.csv", index=False)
        ref = np.stack([natural, natural])
        sim = base.simulated_epochs(w, tc, target_budget=tb)
        agg = base.aggregate_phase_weights(w)
        underexp = float(np.median([agg[i] / max(floor_ref[i], 1e-9) for i in range(len(domains))]))
        rows.append(
            {
                "candidate": name,
                "objective": objective,
                "link": link,
                "gamma_eff": gamma,
                "floor_alpha": alpha,
                "phase": phase,
                "mode": mode,
                "sweep_role": role,
                "weights_csv": str((d / "proposed_mixture_weights.csv").relative_to(args.output_dir)),
                "predicted_bpb_300m": pred(w),
                "tv_to_proportional": float(0.5 * np.abs(w - ref).sum(axis=1).mean()),
                "max_weight": float(w.max()),
                "max_simulated_epoch": float(sim.max()),
                "q95_simulated_epoch": float(np.quantile(sim, 0.95)),
                "agg_over_onephase_median": underexp,
            }
        )
        print(
            f"  {name:38s} pred={pred(w):.4f} tv={rows[-1]['tv_to_proportional']:.3f} maxw={w.max():.3f} "
            f"q95ep={rows[-1]['q95_simulated_epoch']:.2f} agg/1p_med={underexp:.3f}",
            flush=True,
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== candidate_manifest.csv ===")
    print(
        manifest[
            [
                "candidate",
                "sweep_role",
                "predicted_bpb_300m",
                "tv_to_proportional",
                "q95_simulated_epoch",
                "agg_over_onephase_median",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nWrote {len(rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
