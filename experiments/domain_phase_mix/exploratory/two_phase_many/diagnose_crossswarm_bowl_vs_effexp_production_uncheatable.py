# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy","pandas","scipy","scikit-learn"]
# ///
# ruff: noqa: E501, SIM115
"""Cross-swarm functional-form test on the 840-row Grug-MoE production swarm (uncheatable BPB).
Tests whether 300M-swarm conclusions transfer:
  (1) gamma-capped BOWL vs eff-exp DSP on OOF ranking/regret (bowl won on 300M);
  (2) does the bowl's preferred gamma agree with eff-exp's 5.25 (300M wanted 10-16)?
  (3) does every high-gamma variant produce a FANTASY raw optimum (far from observed), while
      aggregate-anchoring gamma=1 stays credible? -> the phase-1 over-crediting thesis, cross-swarm.
Protocol matches Codex's coarse-start: fit nonlinear mu once per gamma on full data, OOF the NNLS head."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

R = Path(__file__).resolve().parent / "reference_outputs"
DATA = R / "grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv"
MODEL = R / "grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"
LINEAR_REG = 0.01
LTF = 0.15

df = pd.read_csv(DATA)
model = json.load(open(MODEL))
buckets = model["domain_names"]
c0 = np.array(model["c0"], float)
c1 = np.array(model["c1"], float)
w0 = df[[f"phase_0/{b}" for b in buckets]].to_numpy(float)
w1 = df[[f"phase_1/{b}" for b in buckets]].to_numpy(float)
w0 = w0 / w0.sum(1, keepdims=True)
w1 = w1 / w1.sum(1, keepdims=True)
y = df["eval/uncheatable_eval/bpb"].to_numpy(float)
n, m = w0.shape
E0 = w0 * c0[None, :]
E1 = w1 * c1[None, :]
print(f"n={n} buckets={m} y[min={y.min():.4f} mean={y.mean():.4f}]")


def fit_head(design, yy, l2):
    dm = design.mean(0, keepdims=True)
    ym = float(yy.mean())
    cd, ct = design - dm, yy - ym
    if l2 > 0:
        cd = np.vstack([cd, np.sqrt(l2) * np.eye(cd.shape[1])])
        ct = np.concatenate([ct, np.zeros(cd.shape[1])])
    coef, _ = nnls(cd, ct)
    return ym - float((dm @ coef).item()), coef


def bowl_design(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


def fit_mu(z):
    # Coarse-start (Codex protocol): per-bucket median ideal exposure + one global shift over a grid.
    bm = np.clip(np.median(np.log1p(np.where(z > 1e-8, z, np.nan)), 0), -2, 8)
    bm = np.where(np.isfinite(bm), bm, 2.0)

    def prof(mu):
        de = bowl_design(z, mu)
        b0, c = fit_head(de, y, LINEAR_REG)
        p = b0 + de @ c
        i = np.argsort(p)[: max(5, int(np.ceil(LTF * n)))]
        return float(np.sqrt(np.mean((p - y) ** 2))) + 0.5 * float(np.mean(np.maximum(y[i] - p[i], 0.0)))

    best_mu, best_f = None, np.inf
    for sh in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
        mu = np.clip(bm + sh, -2, 8)
        f = prof(mu)
        if f < best_f:
            best_f, best_mu = f, mu
    return best_mu


def oof_and_optimum(gamma):
    Z = E0 + gamma * E1
    mu = fit_mu(Z)  # coarse: mu once on full data (Codex protocol)
    D = bowl_design(Z, mu)
    oof = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
        b0, c = fit_head(D[tr], y[tr], LINEAR_REG)
        oof[te] = b0 + D[te] @ c
    sp = float(spearmanr(y, oof).statistic)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    regret = float(y[int(np.argmin(oof))] - y.min())
    # raw unconstrained optimum (2-phase softmax, no KL) + TV to nearest observed row
    b0, c = fit_head(D, y, LINEAR_REG)

    def predict(w0v, w1v):
        z = w0v * c0 + gamma * (w1v * c1)
        return float((b0 + bowl_design(z[None, :], mu) @ c)[0])

    def obj(lg):
        a0 = np.exp(lg[:m] - lg[:m].max())
        a0 /= a0.sum()
        a1 = np.exp(lg[m:] - lg[m:].max())
        a1 /= a1.sum()
        return predict(a0, a1)

    r = minimize(obj, np.zeros(2 * m), method="L-BFGS-B", options={"maxiter": 300})
    a0 = np.exp(r.x[:m] - r.x[:m].max())
    a0 /= a0.sum()
    a1 = np.exp(r.x[m:] - r.x[m:].max())
    a1 /= a1.sum()
    tv0 = 0.5 * np.abs(w0 - a0[None, :]).sum(1)
    tv1 = 0.5 * np.abs(w1 - a1[None, :]).sum(1)
    tv = (tv0 + tv1) / 2
    nearest = int(np.argmin(tv))
    return sp, rmse, regret, float(predict(a0, a1)), float(tv[nearest]), float(y[nearest])


print(
    f"\n{'variant':16s} {'oof_spear':>9s} {'oof_rmse':>8s} {'regret@1':>8s} | {'opt_pred':>8s} {'opt_TV':>6s} {'near_obs':>8s}"
)
print(
    f"{'eff-exp(Codex)':16s} {0.8608:9.4f} {0.010636:8.4f} {0.005233:8.4f} | {'0.6689':>8s} {'0.806':>6s} {'0.8887':>8s}"
)
for g in [1.0, 3.0, 5.25, 8.0, 12.0]:
    sp, rmse, reg, op, tv, nb = oof_and_optimum(g)
    print(f"bowl g={g:<5.2f}     {sp:9.4f} {rmse:8.4f} {reg:8.4f} | {op:8.4f} {tv:6.3f} {nb:8.4f}")
print(f"\nobserved best BPB = {y.min():.4f}. opt_pred << observed + high opt_TV => fantasy optimum.")
