# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy","pandas","scipy","scikit-learn"]
# ///
# ruff: noqa: SIM115
"""Test a coverage-augmented DSP form on the production swarm. The additive bowl misses phase-divergence
/ concentration structure (residual corr -0.24 with tv(p0,p1)). Add per-row coverage penalties as extra
linear-head features (NNLS positive coef -> they RAISE predicted BPB for divergent/concentrated mixtures):
    y_hat = b0 + bowl_design(z) . c  +  lam_div*tv(p0,p1) + lam_hhi*hhi_agg + lam_p1*hhi_p1
Does it (a) improve OOF Spearman/RMSE, and (b) produce a CREDIBLE optimum (lower predicted-BPB optimism +
lower TV-to-observed), since the optimizer now pays for divergence/concentration the fantasy exploited?"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

R = Path(__file__).resolve().parent / "reference_outputs"
df = pd.read_csv(R / "grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv")
model = json.load(open(R / "grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"))
buckets = model["domain_names"]
c0 = np.array(model["c0"])
c1 = np.array(model["c1"])
w0 = df[[f"phase_0/{b}" for b in buckets]].to_numpy(float)
w1 = df[[f"phase_1/{b}" for b in buckets]].to_numpy(float)
w0 /= w0.sum(1, keepdims=True)
w1 /= w1.sum(1, keepdims=True)
y = df["eval/uncheatable_eval/bpb"].to_numpy(float)
n, m = w0.shape
E0 = w0 * c0[None, :]
E1 = w1 * c1[None, :]
GAMMA = 8.0
LREG = 0.01


def cov_feats(a0, a1):
    """per-mixture coverage scalars: phase-divergence TV, aggregate HHI, phase-1 HHI."""
    agg = 0.8 * a0 + 0.2 * a1
    if a0.ndim == 1:
        return np.array([0.5 * np.abs(a0 - a1).sum(), np.sum(agg**2), np.sum(a1**2)])
    return np.stack([0.5 * np.abs(a0 - a1).sum(1), np.sum(agg**2, 1), np.sum(a1**2, 1)], 1)


def bd(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


Z = E0 + GAMMA * E1
bm = np.clip(np.median(np.log1p(np.where(Z > 1e-8, Z, np.nan)), 0), -2, 8)
bm = np.where(np.isfinite(bm), bm, 2.0)


def fit_mu_grid(D0feat_none=None):
    def prof(mu):
        de = bd(Z, mu)
        b0, co = fit(de, y)
        p = b0 + de @ co
        i = np.argsort(p)[: max(5, int(np.ceil(0.15 * n)))]
        return float(np.sqrt(np.mean((p - y) ** 2))) + 0.5 * float(np.mean(np.maximum(y[i] - p[i], 0.0)))

    best_mu, best_f = None, np.inf
    for sh in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
        mm = np.clip(bm + sh, -2, 8)
        f = prof(mm)
        if f < best_f:
            best_f, best_mu = f, mm
    return best_mu


CF = cov_feats(w0, w1)  # (n,3)


def fit(design, yy):
    dm = design.mean(0, keepdims=True)
    ym = float(yy.mean())
    cd = np.vstack([design - dm, np.sqrt(LREG) * np.eye(design.shape[1])])
    ct = np.concatenate([yy - ym, np.zeros(design.shape[1])])
    co, _ = nnls(cd, ct)
    return ym - float((dm @ co).item()), co


mu = fit_mu_grid()  # coarse-start mu (grid shift), shared by both forms for a fair comparison


def run(use_cov):
    D0 = bd(Z, mu)
    D = np.hstack([D0, CF]) if use_cov else D0
    oof = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
        b0, co = fit(D[tr], y[tr])
        oof[te] = b0 + D[te] @ co
    sp = float(spearmanr(y, oof).statistic)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    regret = float(y[int(np.argmin(oof))] - y.min())
    b0, co = fit(D, y)

    def predict(a0, a1):
        z = a0 * c0 + GAMMA * (a1 * c1)
        base = float((b0 + bd(z[None, :], mu) @ co[: 2 * m])[0])
        return base + (float(cov_feats(a0, a1) @ co[2 * m :]) if use_cov else 0.0)

    def obj(lg):
        a0 = np.exp(lg[:m] - lg[:m].max())
        a0 /= a0.sum()
        a1 = np.exp(lg[m:] - lg[m:].max())
        a1 /= a1.sum()
        return predict(a0, a1)

    r = minimize(obj, np.zeros(2 * m), method="L-BFGS-B", options={"maxiter": 400})
    a0 = np.exp(r.x[:m] - r.x[:m].max())
    a0 /= a0.sum()
    a1 = np.exp(r.x[m:] - r.x[m:].max())
    a1 /= a1.sum()
    tv = ((0.5 * np.abs(w0 - a0[None, :]).sum(1) + 0.5 * np.abs(w1 - a1[None, :]).sum(1)) / 2).min()
    effN = 1.0 / np.sum((0.8 * a0 + 0.2 * a1) ** 2)
    covcoef = co[2 * m :] if use_cov else None
    return sp, rmse, regret, predict(a0, a1), tv, effN, covcoef


print(
    f"{'form':22s} {'oof_spear':>9s} {'oof_rmse':>8s} {'regret':>7s} | {'opt_pred':>8s} {'opt_TV':>6s} {'opt_effN':>8s}"
)
for tag, uc in [("bowl g=8 (baseline)", False), ("bowl + coverage", True)]:
    sp, rmse, reg, op, tv, en, cc = run(uc)
    print(f"{tag:22s} {sp:9.4f} {rmse:8.4f} {reg:7.4f} | {op:8.4f} {tv:6.3f} {en:8.1f}")
    if cc is not None:
        print(f"    coverage coefs [tv_p0p1, hhi_agg, hhi_p1] = {np.round(cc, 4)}")
print(f"\nobserved best BPB = {y.min():.4f}; eff-exp baseline opt_pred=0.669 TV=0.806 effN~11.")
