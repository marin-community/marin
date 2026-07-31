# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly"]
# ///
# ruff: noqa: E402
"""Does the coverage-augmented form generalize to the 300M uncheatable swarm? If OOF improves on BOTH
swarms, phase-divergence/concentration is a genuine, transferable functional-form improvement over the
additive DSP (not production-overfit)."""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_dsp_uncheatable_eta_heldout as eta_diag

LREG = 0.01
LTF = 0.15
packet, panel, domains, natural, tc, tb = eta_diag.load_packet()
w = packet.w  # (n,2,m)
w0 = w[:, 0, :] / w[:, 0, :].sum(1, keepdims=True)
w1 = w[:, 1, :] / w[:, 1, :].sum(1, keepdims=True)
y = packet.y
c0 = packet.c0
c1 = packet.c1
n, m = w0.shape
E0 = w0 * c0[None, :]
E1 = w1 * c1[None, :]


def cov_feats(a0, a1):
    agg = 0.8 * a0 + 0.2 * a1
    if a0.ndim == 1:
        return np.array([0.5 * np.abs(a0 - a1).sum(), np.sum(agg**2), np.sum(a1**2)])
    return np.stack([0.5 * np.abs(a0 - a1).sum(1), np.sum(agg**2, 1), np.sum(a1**2, 1)], 1)


def bd(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


def fit(design, yy):
    dm = design.mean(0, keepdims=True)
    ym = float(yy.mean())
    cd = np.vstack([design - dm, np.sqrt(LREG) * np.eye(design.shape[1])])
    ct = np.concatenate([yy - ym, np.zeros(design.shape[1])])
    co, _ = nnls(cd, ct)
    return ym - float((dm @ co).item()), co


CF = cov_feats(w0, w1)


def run(gamma, use_cov):
    Z = E0 + gamma * E1
    bm = np.clip(np.median(np.log1p(np.where(Z > 1e-8, Z, np.nan)), 0), -2, 8)
    bm = np.where(np.isfinite(bm), bm, 2.0)

    def prof(mu):
        de = bd(Z, mu)
        b0, co = fit(de, y)
        p = b0 + de @ co
        i = np.argsort(p)[: max(5, int(np.ceil(LTF * n)))]
        return float(np.sqrt(np.mean((p - y) ** 2))) + 0.5 * float(np.mean(np.maximum(y[i] - p[i], 0.0)))

    best_mu, bf = None, np.inf
    for sh in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
        mm = np.clip(bm + sh, -2, 8)
        f = prof(mm)
        if f < bf:
            bf, best_mu = f, mm
    mu = best_mu
    D0 = bd(Z, mu)
    D = np.hstack([D0, CF]) if use_cov else D0
    oof = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
        b0, co = fit(D[tr], y[tr])
        oof[te] = b0 + D[te] @ co
    return (
        float(spearmanr(y, oof).statistic),
        float(np.sqrt(np.mean((oof - y) ** 2))),
        (co[2 * m :] if use_cov else None),
    )


print(f"300M uncheatable swarm: n={n} buckets={m}")
print(f"{'form':24s} {'oof_spear':>9s} {'oof_rmse':>8s}")
for g in [5.0, 10.0]:
    sp, rmse, _ = run(g, False)
    spc, rmsec, cc = run(g, True)
    print(f"bowl g={g:<4.0f} baseline    {sp:9.4f} {rmse:8.4f}")
    print(f"bowl g={g:<4.0f} + coverage  {spc:9.4f} {rmsec:8.4f}   cov_coefs[tv,hhi_agg,hhi_p1]={np.round(cc, 4)}")
