# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
# ruff: noqa: E402, E501, F841, RUF059
"""Can an EVAL-RELEVANCE-AUGMENTED, TILT-DECOUPLED surrogate reach the winner's recipe (high aggregate
overweight of eval-relevant domains + MODEST tilt) that the plain bowl structurally could not?

Form:  L(w) = bowl_gamma1(w)                        # transferable aggregate saturation (300M fit, gamma=1)
             - lam_rel * sum_i rel_i * log1p(z_agg_i)   # eval-relevance aggregate value (fixes Gap 2)
             - beta   * sum_i rel_i * log1p(e1_i)       # bounded, decoupled late bonus (modest tilt)
z_agg_i = e0_i+e1_i (true epochs), e1_i = phase-1 epochs. rel_i=1 for eval-matching domains.
Sweep (lam_rel, beta); measure the optimum's eval-relevant aggregate overweight vs proportional (winner
~4.25), median tilt (winner ~1.4), and underexposure. If some (lam_rel,beta) hits BOTH -> the form fixes
Gaps 1+2 and panel 2 only calibrates the two coefficients."""

from __future__ import annotations

import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmix_reference_deletion_augmented_300m as base,
)

LINEAR_REG = 0.01
LTF = 0.15
KL = 0.1
EVAL_RELEVANT = [
    "dolma3_arxiv",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolma3_wikipedia",
    "dolmino_synth_math",
    "dolma3_finemath_3plus",
]
REPAIR_GCS = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_dsp_exposure_repair_validation_mixtures_20260702/mixtures"
)


def fit_head(design, y, l2):
    dm = design.mean(0, keepdims=True)
    ym = float(y.mean())
    cd, ct = design - dm, y - ym
    if l2 > 0:
        cd = np.vstack([cd, np.sqrt(l2) * np.eye(cd.shape[1])])
        ct = np.concatenate([ct, np.zeros(cd.shape[1])])
    coef, _ = nnls(cd, ct)
    return ym - float((dm @ coef).item()), coef


def bowl_design(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


def fit_bowl_gamma1(packet):
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    z = e0 + e1
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

    def predict(w):
        zz = w[0] * packet.c0 + w[1] * packet.c1
        return float((b0 + bowl_design(zz[None, :], mu) @ c)[0])

    return predict


def optimize_aug(pred, packet, natural, rel, lam_rel, beta, one_phase=False):
    m = packet.m
    c0, c1 = packet.c0, packet.c1

    def to_w(lg):
        if one_phase:
            e = np.exp(lg - lg.max())
            p = e / e.sum()
            return np.stack([p, p])
        out = np.zeros((2, m))
        for ph in range(2):
            zz = lg[ph * m : (ph + 1) * m]
            e = np.exp(zz - zz.max())
            out[ph] = e / e.sum()
        return out

    def obj(lg):
        w = to_w(lg)
        e0 = w[0] * c0
        e1 = w[1] * c1
        zagg = e0 + e1
        val = pred(w) + KL * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS))
        val -= lam_rel * float(np.sum(rel * np.log1p(zagg)))
        val -= beta * float(np.sum(rel * np.log1p(e1)))
        return val

    dim = m if one_phase else 2 * m
    seed = np.log(np.clip(natural, 1e-9, 1))
    bv, bw = np.inf, None
    for s in [np.zeros(dim), (seed if one_phase else np.concatenate([seed, seed]))]:
        r = minimize(obj, s, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
        if float(r.fun) < bv:
            bv, bw = float(r.fun), to_w(r.x)
    return bw


def main():
    packet, panel, domains, natural, tc, tb = eta_diag.load_packet()
    natural = np.asarray(natural, float)
    idx = {d: i for i, d in enumerate(domains)}
    rel = np.array([1.0 if d in EVAL_RELEVANT else 0.0 for d in domains])
    ridx = [idx[d] for d in EVAL_RELEVANT if d in idx]
    with fsspec.open(f"{REPAIR_GCS}/dsp_uncheatable_exposure_all_deficits.csv", "r") as fh:
        win = pd.read_csv(fh).set_index("domain").reindex(domains)
    win_ow = float(np.mean(win["aggregate_weight"].to_numpy(float)[ridx] / natural[ridx]))
    pred = fit_bowl_gamma1(packet)
    print(f"TARGET (winner): evalrel_ow(vs proportional) ~ {win_ow:.2f}, tilt ~1.4\n")
    print(
        f"{'lam_rel':>7s} {'beta':>6s} | {'evalrel_ow':>10s} {'tilt_med':>8s} {'underexp':>8s} {'tv_prop':>7s} {'pred':>8s}"
    )
    for lam_rel in [0.0, 0.01, 0.03, 0.06, 0.10]:
        for beta in [0.0, 0.01, 0.03]:
            w = optimize_aug(pred, packet, natural, rel, lam_rel, beta)
            e0 = w[0] * packet.c0
            e1 = w[1] * packet.c1
            zagg = e0 + e1
            agg = base.aggregate_phase_weights(w)
            evalrel_ow = float(np.mean(agg[ridx] / natural[ridx]))
            tilt = float(np.median(w[1][ridx] / np.clip(w[0][ridx], 1e-9, None)))
            # underexposure vs a lam=beta=0 one-phase reference
            w1 = optimize_aug(pred, packet, natural, rel, 0.0, 0.0, one_phase=True)
            one_agg = base.aggregate_phase_weights(w1)
            underexp = float(np.median(agg[ridx] / np.clip(one_agg[ridx], 1e-9, None)))
            tvp = float(0.5 * np.abs(agg - natural).sum())
            print(
                f"{lam_rel:7.2f} {beta:6.2f} | {evalrel_ow:10.2f} {tilt:8.2f} {underexp:8.2f} {tvp:7.3f} {pred(w):8.4f}"
            )


if __name__ == "__main__":
    main()
