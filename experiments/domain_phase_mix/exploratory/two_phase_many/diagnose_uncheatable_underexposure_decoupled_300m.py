# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "scipy", "scikit-learn", "plotly"]
# ///
# ruff: noqa: RUF059
"""Fix the underexposure pathology by DECOUPLING exposure-sufficiency from late-value.

Current gamma-capped bowl: z = e0 + gamma*e1 feeds both bowl arms -> the model satisfies a
domain's exposure target via phase-1 (gamma-amplified, but only 20% of tokens) -> underexposes
mid-training domains (stack_edu, code, wiki at 0.64-0.78x the one-phase aggregate).

Decoupled model: the exposure-sufficiency axis (under/over arms) uses TRUE epochs e0+e1 (so
phase-1 hiding cannot satisfy it), and phase-1's extra value is a SEPARATE additive bonus:
    z_true_i = e0_i + e1_i
    late_i   = log1p(e1_i)
    L = b0 + sum c-_i min(log1p(z_true_i)-mu_i,0)^2 + sum c+_i max(...,0)^2 - sum beta_i * late_i
    c-, c+, beta >= 0 (NNLS)
No gamma in the exposure axis. Tests on Uncheatable: fit vs current bowl, and does the optimum
stop underexposing stack_edu/code/wiki?
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)

LINEAR_REG = 0.01
LTF = 0.15
MID = [
    "dolma3_arxiv",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolma3_wikipedia",
    "dolmino_synth_math",
    "dolma3_finemath_3plus",
]


def fit_head(design, y, l2):
    dm = design.mean(0, keepdims=True)
    ym = float(y.mean())
    cd, ct = design - dm, y - ym
    if l2 > 0:
        cd = np.vstack([cd, np.sqrt(l2) * np.eye(cd.shape[1])])
        ct = np.concatenate([ct, np.zeros(cd.shape[1])])
    coef, _ = nnls(cd, ct)
    return ym - float((dm @ coef).item()), coef


# ---------- current gamma-capped bowl (z = e0+gamma*e1, both arms) ----------
def bowl_design(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


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
        return float(np.sqrt(np.mean((p - packet.y) ** 2))) + 0.5 * _opt(p, packet.y)

    mu = _minmu(prof, bm, m)
    b0, c = fit_head(bowl_design(z, mu), packet.y, LINEAR_REG)

    def predict(w, c0=packet.c0, c1=packet.c1):
        z = w[0] * c0 + gamma * (w[1] * c1)
        return float((b0 + bowl_design(z[None, :], mu) @ c)[0])

    return predict, mu


# ---------- decoupled: exposure on true epochs + separate late bonus ----------
def decoupled_design(z_true, late, mu):
    d = np.log1p(z_true) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2, -late])


def fit_decoupled(packet):
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    z_true = e0 + e1
    late = np.log1p(e1)
    m = packet.m
    bm = np.clip(np.median(np.log1p(np.where(z_true > 1e-8, z_true, np.nan)), axis=0), -2, 8)
    bm = np.where(np.isfinite(bm), bm, 2.0)

    def prof(mu):
        de = decoupled_design(z_true, late, mu)
        b0, c = fit_head(de, packet.y, LINEAR_REG)
        p = b0 + de @ c
        return float(np.sqrt(np.mean((p - packet.y) ** 2))) + 0.5 * _opt(p, packet.y)

    mu = _minmu(prof, bm, m)
    b0, c = fit_head(decoupled_design(z_true, late, mu), packet.y, LINEAR_REG)

    def predict(w, c0=packet.c0, c1=packet.c1):
        zt = w[0] * c0 + w[1] * c1
        lt = np.log1p(w[1] * c1)
        return float((b0 + decoupled_design(zt[None, :], lt[None, :], mu) @ c)[0])

    return predict, mu


def _opt(p, y):
    idx = np.argsort(p)[: max(5, int(np.ceil(LTF * len(y))))]
    return float(np.mean(np.maximum(y[idx] - p[idx], 0.0)))


def _minmu(prof, bm, m):
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
    return np.asarray(best.x, float)


def oof(packet, fitfn):
    out = np.zeros_like(packet.y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(packet.y))):
        subpacket = type(packet)(
            frame=packet.frame.iloc[tr].reset_index(drop=True),
            name_col=packet.name_col,
            y=packet.y[tr],
            w=packet.w[tr],
            m=packet.m,
            c0=packet.c0,
            c1=packet.c1,
            domain_names=packet.domain_names,
        )
        pred, _ = fitfn(subpacket)
        out[te] = [pred(packet.w[i]) for i in te]
    return out


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
        return pred(w) + (kl * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS)) if kl > 0 else 0)

    dim = m if one_phase else 2 * m
    seed = np.log(np.clip(natural, 1e-9, 1))
    bv, bw = np.inf, None
    for s in [np.zeros(dim), (seed if one_phase else np.concatenate([seed, seed]))]:
        r = minimize(obj, s, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-10})
        if float(r.fun) < bv:
            bv, bw = float(r.fun), to_w(r.x)
    return bw


def underexposure(w2, w1, domains, natural):
    agg2 = base.aggregate_phase_weights(w2)
    agg1 = base.aggregate_phase_weights(w1)
    idx = {d: i for i, d in enumerate(domains)}
    print(f"    {'domain':26s} {'2p_agg/1p_agg':>13s} {'2p_agg/prop':>12s} {'p1/p0':>7s}")
    for d in MID:
        if d in idx:
            i = idx[d]
            ratio = agg2[i] / max(agg1[i], 1e-9)
            rprop = agg2[i] / max(natural[i], 1e-9)
            p1p0 = w2[1][i] / max(w2[0][i], 1e-9)
            print(f"    {d:26s} {ratio:13.3f} {rprop:12.3f} {p1p0:7.2f}")


def main():
    packet, panel, domains, natural, tc, tb = eta_diag.load_packet()
    natural = np.asarray(natural, float)
    print("=== FIT (uncheatable 300M, 5-fold OOF) ===")
    for name, fitfn in (
        ("gamma_bowl(g10)", lambda p: fit_bowl(p, 10.0)),
        ("decoupled(true-epochs+late)", fit_decoupled),
    ):
        o = oof(packet, fitfn)
        rmse = float(np.sqrt(np.mean((o - packet.y) ** 2)))
        sp = float(spearmanr(packet.y, o).statistic)
        print(f"  {name:32s} OOF_rmse={rmse:.5f} OOF_spearman={sp:.4f}")

    print("\n=== OPTIMUM underexposure (2phase vs 1phase aggregate), KL=0.2 ===")
    for name, fitfn in (("gamma_bowl(g10)", lambda p: fit_bowl(p, 10.0)), ("decoupled", fit_decoupled)):
        pred, _ = fitfn(packet)
        w2 = optimize(pred, packet.m, natural, 0.2, one_phase=False)
        w1 = optimize(pred, packet.m, natural, 0.2, one_phase=True)
        print(f"  --- {name} ---")
        underexposure(w2, w1, domains, natural)
        # summary: median underexposure over MID domains
        idx = {d: i for i, d in enumerate(domains)}
        agg2 = base.aggregate_phase_weights(w2)
        agg1 = base.aggregate_phase_weights(w1)
        ratios = [agg2[idx[d]] / max(agg1[idx[d]], 1e-9) for d in MID if d in idx]
        print(f"    median MID 2p_agg/1p_agg = {np.median(ratios):.3f}  (1.0 = no underexposure)")


if __name__ == "__main__":
    main()
