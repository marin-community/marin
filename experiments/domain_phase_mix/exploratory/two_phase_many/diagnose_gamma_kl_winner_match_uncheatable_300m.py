# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
# ruff: noqa: E402, E501, RUF059, UP031
"""Can the gamma-capped bowl's OPTIMUM reproduce the proven winner's recipe (high aggregate overweight
of eval-relevant domains + modest tilt) under some (gamma_opt, KL)? Thesis: underexposure and
not-beating-frontier share one root cause = phase-1 over-crediting. The winner buys domain value via
AGGREGATE (transferable); our optima buy it via phase-1 TILT (anti-transferable) -> low aggregate.

Sweep (gamma, KL); for each, refit bowl on 300M uncheatable, optimize two-phase, and measure how close
the optimum's eval-relevant AGGREGATE profile gets to the winner's, plus underexposure and tilt."""

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


def optimize(pred, m, natural, kl, one_phase=False):
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
        return pred(w) + kl * float(base.weighted_multiclass_kl(w, natural, base.PHASE_FRACTIONS))

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
    rel = [idx[d] for d in EVAL_RELEVANT if d in idx]
    with fsspec.open(f"{REPAIR_GCS}/dsp_uncheatable_exposure_all_deficits.csv", "r") as fh:
        win = pd.read_csv(fh).set_index("domain").reindex(domains)
    win_agg = win["aggregate_weight"].to_numpy(float)

    # one-phase reference aggregate per gamma (each gamma has its own one-phase optimum)
    print("WINNER eval-relevant aggregate overweight (vs its own one-phase not available; vs proportional):")
    print(
        "  " + " ".join(f"{d.split('_')[-1]}={win_agg[idx[d]] / natural[idx[d]]:.2f}" for d in EVAL_RELEVANT if d in idx)
    )
    print(f"  winner tv(agg,proportional)={0.5 * np.abs(win_agg - natural).sum():.3f}\n")

    print(
        f"{'gamma':>5s} {'KL':>5s} | {'underexp_med':>12s} {'evalrel_ow_1p':>13s} {'tilt_med':>8s} {'tv_agg_1p':>9s} {'winnerdist':>10s} {'pred_opt':>9s}"
    )
    results = []
    for gamma in [1.0, 2.0, 4.0, 8.0, 12.0]:
        pred = fit_bowl(packet, gamma)
        w1 = optimize(pred, packet.m, natural, kl=0.2, one_phase=True)  # one-phase ref at moderate KL
        one_agg = base.aggregate_phase_weights(w1)
        for kl in [0.05, 0.1, 0.2]:
            w2 = optimize(pred, packet.m, natural, kl=kl, one_phase=False)
            agg2 = base.aggregate_phase_weights(w2)
            ow_1p = np.array([agg2[i] / max(one_agg[i], 1e-9) for i in rel])
            underexp = float(np.median(ow_1p))  # 1.0 = no underexposure of eval-relevant
            evalrel_ow = float(np.mean(agg2[rel] / natural[rel]))  # overweight vs proportional (winner ~1.5-1.9)
            tilt = float(np.median(w2[1][rel] / np.clip(w2[0][rel], 1e-9, None)))
            tv1p = float(0.5 * np.abs(agg2 - one_agg).sum())
            # distance of eval-relevant aggregate profile (vs proportional) to the winner's, in log space
            wdist = float(np.mean(np.abs(np.log(agg2[rel] / natural[rel]) - np.log(win_agg[rel] / natural[rel]))))
            po = pred(w2)
            print(
                f"{gamma:5.0f} {kl:5.2f} | {underexp:12.3f} {evalrel_ow:13.2f} {tilt:8.2f} {tv1p:9.3f} {wdist:10.3f} {po:9.4f}"
            )
            results.append((gamma, kl, underexp, evalrel_ow, tilt, wdist))
    best = min(results, key=lambda r: r[5])
    print(
        f"\nCLOSEST to winner aggregate profile: gamma={best[0]:.0f} KL={best[1]:.2f} winnerdist={best[5]:.3f} "
        f"(underexp_med={best[2]:.3f}, evalrel_ow={best[3]:.2f}, tilt={best[4]:.2f})"
    )
    print("winner evalrel_ow (vs proportional) ~ %.2f; winner tilt ~1.4" % float(np.mean(win_agg[rel] / natural[rel])))


if __name__ == "__main__":
    main()
