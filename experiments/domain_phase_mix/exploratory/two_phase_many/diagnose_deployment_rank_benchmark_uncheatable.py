# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
# ruff: noqa: E402, SIM115
"""THE benchmark that matters: fit each candidate form on the 300M swarm, then rank the VALIDATED 3e18
uncheatable mixtures by predicted BPB and measure Spearman/regret vs their ACTUAL 3e18 BPB. Proxy OOF
and optimum-TV are surrogates; this measures whether a form predicts DEPLOYMENT. Also reports proxy OOF
(300M) so we can see fit-vs-deployment divergence (anti-transferability)."""

import json
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

S = Path(
    "/private/tmp/claude-501/-Users-calvinxu-Projects-Work-Marin-marin/c910cd66-2969-442b-8c13-9b7c3936a61b/scratchpad"
)
REPO = Path("/Users/calvinxu/Projects/Work/Marin/marin")
sys.path.insert(0, str(REPO))
from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_dsp_uncheatable_eta_heldout as eta

LREG = 0.01
LTF = 0.15
GCS = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture"
DIRS = {  # name-prefix -> GCS mixtures dir
    "suff_": f"{GCS}/delphi_sufficiency_floored_mixtures_20260705/mixtures",
    "wnbr_": f"{GCS}/delphi_winner_neighborhood_mixtures_20260705/mixtures",
    "dsp_gamma_": f"{GCS}/delphi_dsp_gamma_capped_bowl_validation_mixtures_20260704/mixtures",
    "dsp_uncheatable_exposure_": f"{GCS}/delphi_dsp_exposure_repair_validation_mixtures_20260702/mixtures",
    "dsp_table9_exposure_": f"{GCS}/delphi_dsp_exposure_repair_validation_mixtures_20260702/mixtures",
}
EVAL_RELEVANT = [
    "dolma3_arxiv",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolma3_wikipedia",
    "dolmino_synth_math",
    "dolma3_finemath_3plus",
]

# ---- training swarm (300M) ----
pk, panel, domains, natural, tc, tb = eta.load_packet()
w = pk.w
w0 = w[:, 0, :] / w[:, 0, :].sum(1, keepdims=True)
w1 = w[:, 1, :] / w[:, 1, :].sum(1, keepdims=True)
y = pk.y
c0 = pk.c0
c1 = pk.c1
n, m = w0.shape
E0 = w0 * c0[None, :]
E1 = w1 * c1[None, :]
rel = np.array([1.0 if d in EVAL_RELEVANT else 0.0 for d in domains])

# ---- deployment test set (validated 3e18) ----
bpb = json.load(open(S / "deploy_bpb.json"))
fs = fsspec.filesystem("gcs")
T0, T1, TY, TN = [], [], [], []
for nm, val in bpb.items():
    d = next((v for k, v in DIRS.items() if nm.startswith(k)), None)
    if d is None:
        continue
    p = f"{d}/{nm}.csv"
    try:
        with fs.open(p, "r") as fh:
            df = pd.read_csv(fh).set_index("domain").reindex(domains)
        a0 = df["phase_0_weight"].to_numpy(float)
        a1 = df["phase_1_weight"].to_numpy(float)
        if np.isnan(a0).any():
            continue
        T0.append(a0 / a0.sum())
        T1.append(a1 / a1.sum())
        TY.append(val)
        TN.append(nm)
    except Exception:
        continue
T0 = np.array(T0)
T1 = np.array(T1)
TY = np.array(TY)
print(
    f"train(300M)={n}x{m} buckets | deployment test set={len(TY)} validated 3e18 mixtures "
    f"(BPB {TY.min():.4f}-{TY.max():.4f})"
)


# ---- forms ----
def fit_head(D, yy, extra=None):
    Dd = np.hstack([D, extra]) if extra is not None else D
    dm = Dd.mean(0, keepdims=True)
    ym = float(yy.mean())
    cd = np.vstack([Dd - dm, np.sqrt(LREG) * np.eye(Dd.shape[1])])
    ct = np.concatenate([yy - ym, np.zeros(Dd.shape[1])])
    co, _ = nnls(cd, ct)
    return ym - float((dm @ co).item()), co


def bd(z, mu):
    dd = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(dd, 0.0) ** 2, np.maximum(dd, 0.0) ** 2])


def cov(a0, a1):
    agg = 0.8 * a0 + 0.2 * a1
    ax = 1 if a0.ndim == 2 else 0
    return np.stack([0.5 * np.abs(a0 - a1).sum(ax), (agg**2).sum(ax), (a1**2).sum(ax)], -1)


def fit_mu(Z):
    bm = np.clip(np.median(np.log1p(np.where(Z > 1e-8, Z, np.nan)), 0), -2, 8)
    bm = np.where(np.isfinite(bm), bm, 2.0)

    def prof(mu):
        de = bd(Z, mu)
        b0, co = fit_head(de, y)
        p = b0 + de @ co
        i = np.argsort(p)[: max(5, int(np.ceil(LTF * n)))]
        return float(np.sqrt(np.mean((p - y) ** 2))) + 0.5 * float(np.mean(np.maximum(y[i] - p[i], 0.0)))

    bmu, bf = None, np.inf
    for sh in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
        mm = np.clip(bm + sh, -2, 8)
        f = prof(mm)
        if f < bf:
            bf, bmu = f, mm
    return bmu


CF_tr = cov(w0, w1)
CF_te = cov(T0, T1)


def evaluate(gamma, use_cov, lam_rel=0.0, tag=""):
    Z = E0 + gamma * E1
    mu = fit_mu(Z)
    D = bd(Z, mu)
    extra = CF_tr if use_cov else None
    # proxy OOF on 300M
    oof = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
        b0, co = fit_head(D[tr], y[tr], CF_tr[tr] if use_cov else None)
        oof[te] = b0 + (np.hstack([D[te], CF_tr[te]]) @ co if use_cov else D[te] @ co)
    proxy_sp = float(spearmanr(y, oof).statistic)
    # fit full, predict deployment test set
    b0, co = fit_head(D, y, extra)
    Zt = T0 * c0[None, :] + gamma * (T1 * c1[None, :])
    Dt = bd(Zt, mu)
    predt = b0 + (np.hstack([Dt, CF_te]) @ co if use_cov else Dt @ co)
    if lam_rel:
        predt = predt - lam_rel * (rel[None, :] * np.log1p(Zt)).sum(1)
    dep_sp = float(spearmanr(TY, predt).statistic)
    regret = float(TY[int(np.argmin(predt))] - TY.min())  # actual BPB of the form's top pick minus best
    return tag, proxy_sp, dep_sp, regret


def CF_te_none(te, use_cov):
    return None


print("\n=== gamma sweep: where does PROXY fit peak vs DEPLOYMENT rank? ===")
print(f"{'form':26s} {'proxyOOF':>8s} {'DEPLOY_rank':>11s} {'deploy_regret':>13s}")
for g in [1.0, 3.0, 5.0, 7.0, 10.0, 14.0, 20.0]:
    t, ps, ds, rg = evaluate(g, False, 0.0, f"bowl g={g:g}")
    print(f"{t:26s} {ps:8.3f} {ds:11.3f} {rg:13.5f}")
print("--- at best-deploy gamma, does coverage help deployment? ---")
for tag, g, uc in [("bowl g=5 + coverage", 5.0, True), ("bowl g=7 + coverage", 7.0, True)]:
    t, ps, ds, rg = evaluate(g, uc, 0.0, tag)
    print(f"{t:26s} {ps:8.3f} {ds:11.3f} {rg:13.5f}")
print(
    f"\nn_test={len(TY)} (BPB {TY.min():.4f}-{TY.max():.4f}); DEPLOY_rank=Spearman(pred,actual 3e18); "
    f"regret=actual BPB of #1 pick minus best. NOTE: test set includes model-proposed mixtures (mild circularity)."
)

# ===== DEPLOYMENT RESIDUAL ANALYSIS: what does the 300M bowl mis-predict at 3e18? =====
print("\n=== DEPLOYMENT RESIDUAL (bowl g=10): what structure does the 300M fit miss at 3e18? ===")
Zt10 = T0 * c0[None, :] + 10.0 * (T1 * c1[None, :])
Ztr10 = E0 + 10.0 * E1
mu10 = fit_mu(Ztr10)
b0, co = fit_head(bd(Ztr10, mu10), y)
predt = b0 + bd(Zt10, mu10) @ co
zp = (predt - predt.mean()) / predt.std()
za = (TY - TY.mean()) / TY.std()
resid = zp - za  # >0: bowl predicts BETTER (lower BPB) than actual -> over-optimistic
# per-mixture deployment features
e0t = T0 * c0[None, :]
e1t = T1 * c1[None, :]
zt = e0t + e1t
feat = {
    "phase1_epoch_share": e1t.sum(1) / (e0t + e1t).sum(1),
    "tv_p0_p1": 0.5 * np.abs(T0 - T1).sum(1),
    "hhi_agg": ((0.8 * T0 + 0.2 * T1) ** 2).sum(1),
    "evalrel_expo": (rel[None, :] * np.log1p(zt)).sum(1),
    "max_epoch": zt.max(1),
    "q95_epoch": np.quantile(zt, 0.95, axis=1),
}
for k, v in feat.items():
    print(f"  corr(resid, {k:18s}) = {spearmanr(resid, v).statistic:+.3f}")
# which specific mixtures are most mis-ranked?
order = np.argsort(np.abs(resid))[::-1][:6]
print("  most mis-predicted mixtures (name: resid, actual_BPB):")
for i in order:
    print(f"    {TN[i]:38s} resid={resid[i]:+.2f} actual={TY[i]:.4f}")

# ===== does adding coverage lift DEPLOYMENT rank? (honest LOO over the 32) =====
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

print("\n=== bowl + coverage on DEPLOYMENT (LOO over 32; coverage features standardized) ===")
tvd = 0.5 * np.abs(T0 - T1).sum(1)
hhid = ((0.8 * T0 + 0.2 * T1) ** 2).sum(1)


def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-9)


Xbowl = zsc(predt).reshape(-1, 1)  # bowl g=10 300M prediction
Xcov = np.column_stack([zsc(predt), zsc(tvd), zsc(hhid)])  # + coverage


def loo_rank(X):
    p = np.zeros(len(TY))
    for tr, te in LeaveOneOut().split(X):
        r = Ridge(alpha=1.0).fit(X[tr], TY[tr])
        p[te] = r.predict(X[te])
    return float(spearmanr(TY, p).statistic), float(TY[int(np.argmin(p))] - TY.min())


sb, rb = loo_rank(Xbowl)
sc, rc = loo_rank(Xcov)
print(f"  bowl g=10 alone (LOO)      deploy_rank={sb:.3f}  regret={rb:.5f}")
print(f"  bowl g=10 + coverage (LOO) deploy_rank={sc:.3f}  regret={rc:.5f}")
print(f"  ceiling(top-competitive)~0.978; delta from coverage = {sc - sb:+.3f}")
