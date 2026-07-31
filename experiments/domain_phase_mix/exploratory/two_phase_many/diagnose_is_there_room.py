# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy","pandas","scipy","scikit-learn"]
# ///
# ruff: noqa: E402, SIM115
"""Is there room to improve beyond the additive DSP (~0.86 OOF, fantasy optimum)?
(1) NOISE FLOOR: nearest-neighbor mixture pairs -> BPB spread -> irreducible error. Compare to OOF RMSE.
(2) RESIDUAL STRUCTURE: do the additive-DSP OOF residuals correlate with COVERAGE/CONCENTRATION features
    (HHI, entropy, effective buckets, phase balance)? The additive per-bucket form CANNOT see coverage;
    if residuals carry it, a better form exists AND capturing it would tame the fantasy optimum."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

R = Path(__file__).resolve().parent / "reference_outputs"
df = pd.read_csv(R / "grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv")
rp = pd.read_csv(R / "grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/row_predictions.csv")
model = json.load(open(R / "grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"))
buckets = model["domain_names"]
w0 = df[[f"phase_0/{b}" for b in buckets]].to_numpy(float)
w1 = df[[f"phase_1/{b}" for b in buckets]].to_numpy(float)
w0 /= w0.sum(1, keepdims=True)
w1 /= w1.sum(1, keepdims=True)
y = df["eval/uncheatable_eval/bpb"].to_numpy(float)
agg = 0.8 * w0 + 0.2 * w1
agg /= agg.sum(1, keepdims=True)
n = len(y)

# ---- (1) NOISE FLOOR via nearest-neighbor mixture pairs ----
# aggregate-space TV distance; find each row's nearest neighbor, collect |dy| for the closest pairs
from scipy.spatial.distance import cdist

Dtv = cdist(agg, agg, metric="cityblock") * 0.5  # TV on aggregate
np.fill_diagonal(Dtv, np.inf)
nn = Dtv.argmin(1)
nn_tv = Dtv[np.arange(n), nn]
nn_dy = np.abs(y - y[nn])
order = np.argsort(nn_tv)
close = order[:40]  # 40 closest pairs
print("=== NOISE FLOOR (nearest-neighbor pairs) ===")
print(f"  closest 40 pairs: median TV={np.median(nn_tv[close]):.4f}, median |dBPB|={np.median(nn_dy[close]):.5f}")
print(f"  -> noise_std lower bound ~ median|dBPB|/sqrt(2) = {np.median(nn_dy[close]) / np.sqrt(2):.5f}")
print("  Codex OOF RMSE = 0.01064, target std = 0.02057")
print("  => if OOF RMSE >> noise_std, there is fittable signal left.\n")

# ---- (2) RESIDUAL STRUCTURE: coverage/concentration features ----
rp = rp.sort_values("experiment_index").reset_index(drop=True)
resid = rp["oof_residual_prediction_minus_actual"].to_numpy(float)  # pred - actual


# coverage features per row
def hhi(w):
    return np.sum(w**2, 1)


def entropy(w):
    p = np.clip(w, 1e-12, 1)
    return -np.sum(p * np.log(p), 1)


def effN(w):
    return 1.0 / np.sum(w**2, 1)


feats = pd.DataFrame(
    {
        "hhi_agg": hhi(agg),
        "entropy_agg": entropy(agg),
        "effN_agg": effN(agg),
        "max_agg": agg.max(1),
        "hhi_p1": hhi(w1),
        "phase1_conc": hhi(w1) - hhi(w0),
        "tv_p0_p1": 0.5 * np.abs(w0 - w1).sum(1),
    }
)
print("=== RESIDUAL ~ coverage features (Spearman corr of OOF residual with each) ===")
for col in feats.columns:
    rho = float(spearmanr(feats[col].to_numpy(), resid).statistic)
    print(f"  {col:14s} corr(resid, feat) = {rho:+.3f}")
# how much residual variance can coverage features explain? (5-fold CV R2)
X = feats.to_numpy()
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
pred_resid = cross_val_predict(LinearRegression(), X, resid, cv=5)
ss_res = np.sum((resid - pred_resid) ** 2)
ss_tot = np.sum((resid - resid.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\n  5-fold CV R2 of coverage-features -> OOF residual: {r2:.3f}")
print(f"  residual std = {resid.std():.5f}; if R2>0 the additive DSP misses coverage-structure it CANNOT")
print("  represent (per-bucket additive) -> a coverage-aware form should fit better AND tame the optimum.")
