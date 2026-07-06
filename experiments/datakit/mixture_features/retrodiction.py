# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "joblib"]
# ///
"""H2b held-out-dose retrodiction suite + H4 ablations (mixing-via-embeddings).

Pre-registered protocol (logbook 2026-07-05 entry; thresholds there OVERRIDE the spec
defaults, recalibrated before any model fitting because the swarm has no vertex runs):

- PhaseReducer MAX: dose(run, k) = max over the 2 phases of w_k. Train: dose <= 0.02.
  Test: dose >= 0.06 (primary) / 0.08 (sensitivity). The middle band is discarded.
- Per scale, never pooled. Eligible domains: n_train >= 60 AND n_test >= 20, recomputed
  from the data. Target `eval/uncheatable_eval/bpb`, lower-better; Spearman is reported on
  (-metric) vs (-prediction), which equals spearman(prediction, metric).
- Predictors fit on identical train rows, scored on identical test rows; hyperparameters by
  inner 5-fold CV on train only; seed 0. Content predictors run three featurization arms:
  semantic, shuffled-columns (N_CONTROL_SEEDS seeds, averaged), matched-random per-column
  cell permutation (N_CONTROL_SEEDS seeds, averaged), each at the predictor's granularity.
- Exposure features are SKIPPED: within a scale all runs share the budget and 80/20 phase
  structure, so EXPOSURE_GLOBAL is constant across rows and carries no information.
- Paired bootstrap (N_BOOT resamples of the test rows) CIs of semantic minus each control
  and semantic minus WEIGHTS_RIDGE. Diagnostics: content_novelty(k) (nonneg-lsq residual of
  sqrt V_k onto the cone of sqrt V_{-k} at K=1000) and per-test-run design_support (residual
  of the test h to the convex hull of train h's at K=40, nonneg lsq with sum-to-1 penalty).
- Pre-registered SUCCESS readout, per scale at the primary threshold: a semantic predictor
  beats BOTH controls with CI separation on more than half the eligible domains AND its
  median per-domain Spearman >= WEIGHTS_RIDGE's.

H4 ablations (60M, primary threshold only): HIST_RIDGE granularity {40, 1000, 5000};
per-phase vs pooled (phase-token-weighted) for HIST_K1000 and RFF; Hellinger vs Euclidean
kernel at K=1000; KME vs HIST_K1000 vs concat.

Outputs under scratch/mixture_features/: h2b/{results.parquet, predictions.parquet,
diagnostics.parquet, verdict.json, results_partial.parquet} and h4/ablations.parquet.
"""

import json
import logging
import time
import zlib
from enum import StrEnum
from pathlib import Path

import featurize
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from h1_audit import json_default, load_artifacts, run_weight_matrix
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.model_selection import GridSearchCV, KFold

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
HIST_DIR = SCRATCH / "domain_histograms"
OUT_DIR = SCRATCH / "h2b"
H4_DIR = SCRATCH / "h4"

SEED = 0
TARGET = "eval/uncheatable_eval/bpb"
PRIMARY_SCALE = "60m_1p2b"
TRAIN_MAX_DOSE = 0.02
TEST_MIN_DOSES = (0.06, 0.08)  # (primary, sensitivity)
PRIMARY_TEST_MIN_DOSE = 0.06
MIN_TRAIN, MIN_TEST = 60, 20
N_BOOT = 500
N_CONTROL_SEEDS = 10
N_INNER_FOLDS = 5
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
KR_GAMMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)  # x 1/median(train squared-distance)
KR_ALPHAS = np.logspace(-3, 2, 6)
LGBM_GRID = {"num_leaves": [7, 15, 31], "min_child_samples": [5, 10]}
HULL_PENALTY = 1e3  # weight of the sum-to-1 row in the design-support nnls


class PredictorKind(StrEnum):
    MEAN_BASELINE = "mean_baseline"
    DOSE_LINEAR = "dose_linear"
    WEIGHTS_RIDGE = "weights_ridge"
    WEIGHTS_LGBM = "weights_lgbm"
    HIST_RIDGE_K40 = "hist_ridge_k40"
    HIST_RIDGE_K1000 = "hist_ridge_k1000"
    KME_RIDGE = "kme_ridge"
    RFF_RIDGE = "rff_ridge"
    KERNEL_HELLINGER = "kernel_hellinger"
    NN_HIST = "nn_hist"


# Content predictors get the three featurization arms; ridge ones read a bundle feature key.
CONTENT_RIDGE_FEATURE = {
    PredictorKind.HIST_RIDGE_K40: "h40",
    PredictorKind.HIST_RIDGE_K1000: "h1000",
    PredictorKind.KME_RIDGE: "kme",
    PredictorKind.RFF_RIDGE: "rff",
}
CONTENT_PREDICTORS = (*tuple(CONTENT_RIDGE_FEATURE), PredictorKind.KERNEL_HELLINGER, PredictorKind.NN_HIST)
# Predictors eligible to trigger the pre-registered success readout (NN_HIST is a floor).
VERDICT_PREDICTORS = (*tuple(CONTENT_RIDGE_FEATURE), PredictorKind.KERNEL_HELLINGER)
ARMS = ("semantic", "shuffled", "matched")


# ---------------------------------------------------------------------------
# Exact ridge with inner 5-fold CV (SVD path: handles p >> n cheaply)
# ---------------------------------------------------------------------------


def _ridge_solve(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Standardize on train, fit ridge (intercept via centering) for every alpha; (n_te, n_alphas)."""
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (y_tr - ym)
    bv = b @ vt.T  # (n_te, r)
    out = np.empty((x_te.shape[0], len(alphas)))
    for i, al in enumerate(alphas):
        out[:, i] = bv @ (uty * s / (s**2 + al)) + ym
    return out


def ridge_cv_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """Ridge predictions with alpha chosen by inner 5-fold CV on the train rows only."""
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    sse = np.zeros(len(RIDGE_ALPHAS))
    for tr, va in kf.split(x_tr):
        p = _ridge_solve(x_tr[tr], y_tr[tr], x_tr[va], RIDGE_ALPHAS)
        sse += ((p - y_tr[va][:, None]) ** 2).sum(axis=0)
    best = int(np.argmin(sse))
    return _ridge_solve(x_tr, y_tr, x_te, RIDGE_ALPHAS[best : best + 1])[:, 0]


# ---------------------------------------------------------------------------
# Dual kernel ridge on a precomputed squared-distance matrix
# ---------------------------------------------------------------------------


def _kr_fit_predict(k_tr: np.ndarray, y_tr: np.ndarray, k_te_tr: np.ndarray, alpha: float) -> np.ndarray:
    ym = y_tr.mean()
    dual = np.linalg.solve(k_tr + alpha * np.eye(len(y_tr)), y_tr - ym)
    return k_te_tr @ dual + ym


def kernel_cv_predict(d2: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kernel ridge k(a,b)=exp(-gamma d2); gamma x alpha by inner 5-fold CV on train."""
    d_tr = d2[np.ix_(tr, tr)]
    med = float(np.median(d_tr[~np.eye(len(tr), dtype=bool)]))
    gammas = np.asarray(KR_GAMMA_FACTORS) / max(med, 1e-12)
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(len(tr))))
    best, best_sse = None, np.inf
    for g in gammas:
        k_full = np.exp(-g * d_tr)
        for al in KR_ALPHAS:
            sse = 0.0
            for itr, iva in folds:
                p = _kr_fit_predict(k_full[np.ix_(itr, itr)], y[tr][itr], k_full[np.ix_(iva, itr)], al)
                sse += ((p - y[tr][iva]) ** 2).sum()
            if sse < best_sse:
                best, best_sse = (g, al), sse
    g, al = best
    return _kr_fit_predict(np.exp(-g * d_tr), y[tr], np.exp(-g * d2[np.ix_(te, tr)]), al)


def nn_predict(d2: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y[tr[np.argmin(d2[np.ix_(te, tr)], axis=1)]]


# ---------------------------------------------------------------------------
# Vectorized Spearman + diagnostics
# ---------------------------------------------------------------------------


def spearman_cols(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman of each column of p (n, m) against y (n,); NaN for constant columns."""
    rp = rankdata(p, axis=0).astype(np.float64)
    ry = rankdata(y).astype(np.float64)
    rp -= rp.mean(axis=0)
    ry -= ry.mean()
    denom = np.sqrt((rp**2).sum(axis=0) * (ry**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, rp.T @ ry / denom, np.nan)


def hull_residual(train_feats: np.ndarray, h: np.ndarray) -> float:
    """L2 residual of h to the convex hull of train_feats rows (nnls with sum-to-1 penalty row)."""
    a = np.vstack([train_feats.T, HULL_PENALTY * np.ones((1, train_feats.shape[0]))])
    b = np.concatenate([h, [HULL_PENALTY]])
    coef, _ = nnls(a, b)
    return float(np.linalg.norm(train_feats.T @ coef - h))


def content_novelty(v1000: np.ndarray) -> np.ndarray:
    """Per-domain nonneg-lsq residual of sqrt(V_k) onto the cone of sqrt(V_{-k}) at K=1000."""
    s = np.sqrt(np.asarray(v1000))
    out = np.empty(s.shape[1])
    for j in range(s.shape[1]):
        _, out[j] = nnls(np.delete(s, j, axis=1), s[:, j])
    return out


# ---------------------------------------------------------------------------
# Per-scale feature bundle (all arms precomputed; split-independent)
# ---------------------------------------------------------------------------


def _sq_hellinger(h_phases: np.ndarray) -> np.ndarray:
    """Mean over phases of the squared Hellinger distance; h_phases is (n, 2, k)."""
    n = h_phases.shape[0]
    d = np.zeros((n, n))
    for p in range(h_phases.shape[1]):
        s = np.sqrt(np.clip(h_phases[:, p, :], 0.0, None))
        d += np.clip(1.0 - s @ s.T, 0.0, None)
    return d / h_phases.shape[1]


def _sq_euclid(h_phases: np.ndarray) -> np.ndarray:
    n = h_phases.shape[0]
    d = np.zeros((n, n))
    for p in range(h_phases.shape[1]):
        g = h_phases[:, p, :] @ h_phases[:, p, :].T
        sq = np.diag(g)
        d += np.clip(sq[:, None] + sq[None, :] - 2 * g, 0.0, None)
    return d / h_phases.shape[1]


def _arm_features(
    w: np.ndarray,  # (n, 2, 39)
    v40: np.ndarray,
    v1000: np.ndarray,
    v5000: np.ndarray,
    rff: np.ndarray,  # (39, 2048)
    centroids: np.ndarray,  # (5000, 192)
) -> dict[str, np.ndarray]:
    n = w.shape[0]
    h40 = np.stack([w[:, p, :] @ v40.T for p in range(2)], axis=1)  # (n, 2, 40)
    h1000 = np.stack([w[:, p, :] @ v1000.T for p in range(2)], axis=1)
    h5000 = np.stack([w[:, p, :] @ v5000.T for p in range(2)], axis=1)
    return {
        "h40": h40.reshape(n, -1),
        "h1000": h1000.reshape(n, -1),
        "kme": (h5000 @ centroids).reshape(n, -1),  # (n, 384)
        "rff": np.concatenate([w[:, p, :] @ rff for p in range(2)], axis=1),  # (n, 4096)
        "d2_hell": _sq_hellinger(h1000),
        "h40_phases": h40,  # kept for design-support (semantic arm only ever used)
    }


def build_bundle(
    sub: pd.DataFrame,
    domains: list[str],
    v40: np.ndarray,
    v1000: np.ndarray,
    v5000: np.ndarray,
    rff: np.ndarray,
    centroids: np.ndarray,
    with_h4: bool,
) -> dict:
    w = run_weight_matrix(sub, domains)  # (n, 2, 39)
    n = len(sub)
    doses = np.maximum(w[:, 0, :], w[:, 1, :])  # PhaseReducer.MAX
    arms: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    arms[("semantic", -1)] = _arm_features(w, v40, v1000, v5000, rff, centroids)
    for s in range(N_CONTROL_SEEDS):
        # One domain permutation per seed, shared across granularities (shuffled_columns_v
        # derives it from default_rng(seed).permutation(n_domains) for every input).
        arms[("shuffled", s)] = _arm_features(
            w,
            featurize.shuffled_columns_v(v40, s),
            featurize.shuffled_columns_v(v1000, s),
            featurize.shuffled_columns_v(v5000, s),
            featurize.shuffled_columns_v(rff.T, s).T,
            centroids,
        )
        arms[("matched", s)] = _arm_features(
            w,
            featurize.matched_random_v(v40, s),
            featurize.matched_random_v(v1000, s),
            featurize.matched_random_v(v5000, s),
            featurize.matched_random_v(rff.T, s).T,  # per-domain permutation of the 2048 coords
            centroids,
        )
    bundle = {
        "y": sub[TARGET].to_numpy(dtype=np.float64),
        "run_names": sub["run_name"].tolist(),
        "doses": doses,
        "x_weights": w.reshape(n, -1),
        "arms": arms,
    }
    if with_h4:
        sem = arms[("semantic", -1)]
        t = sub[["phase_0_tokens", "phase_1_tokens"]].to_numpy(dtype=np.float64)
        w_pool = (w[:, 0, :] * t[:, :1] + w[:, 1, :] * t[:, 1:]) / t.sum(axis=1, keepdims=True)
        h5000 = np.stack([w[:, p, :] @ v5000.T for p in range(2)], axis=1)
        h1000 = np.stack([w[:, p, :] @ v1000.T for p in range(2)], axis=1)
        bundle["h4"] = {
            "h5000": h5000.reshape(n, -1),
            "h1000_pooled": w_pool @ v1000.T,
            "rff_pooled": w_pool @ rff,
            "kme_hist_concat": np.concatenate([sem["kme"], sem["h1000"]], axis=1),
            "d2_euclid": _sq_euclid(h1000),
        }
    return bundle


# ---------------------------------------------------------------------------
# One (scale, threshold, domain) split
# ---------------------------------------------------------------------------


def run_split(bundle: dict, scale: str, tmin: float, dom_j: int, domain: str, novelty: float) -> dict:
    t0 = time.monotonic()
    y, doses = bundle["y"], bundle["doses"]
    dose = doses[:, dom_j]
    tr = np.flatnonzero(dose <= TRAIN_MAX_DOSE)
    te = np.flatnonzero(dose >= tmin)
    y_tr, y_te = y[tr], y[te]

    preds: dict[tuple[str, str, int], np.ndarray] = {}
    preds[(PredictorKind.MEAN_BASELINE, "none", -1)] = np.full(len(te), y_tr.mean())
    x_dose = dose[:, None]
    coef = np.polyfit(x_dose[tr, 0], y_tr, 1)
    preds[(PredictorKind.DOSE_LINEAR, "none", -1)] = np.polyval(coef, x_dose[te, 0])
    preds[(PredictorKind.WEIGHTS_RIDGE, "none", -1)] = ridge_cv_predict(
        bundle["x_weights"][tr], y_tr, bundle["x_weights"][te]
    )
    gs = GridSearchCV(
        lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED, verbosity=-1, n_jobs=1),
        LGBM_GRID,
        cv=KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    gs.fit(bundle["x_weights"][tr], y_tr)
    preds[(PredictorKind.WEIGHTS_LGBM, "none", -1)] = gs.predict(bundle["x_weights"][te])

    for (arm, s), feats in bundle["arms"].items():
        for kind, key in CONTENT_RIDGE_FEATURE.items():
            preds[(kind, arm, s)] = ridge_cv_predict(feats[key][tr], y_tr, feats[key][te])
        preds[(PredictorKind.KERNEL_HELLINGER, arm, s)] = kernel_cv_predict(feats["d2_hell"], tr, te, y)
        preds[(PredictorKind.NN_HIST, arm, s)] = nn_predict(feats["d2_hell"], tr, te, y)

    # --- metrics -----------------------------------------------------------
    keys = list(preds)
    p_mat = np.column_stack([preds[k] for k in keys])
    sp_point = spearman_cols(p_mat, y_te)
    rmse_point = np.sqrt(((p_mat - y_te[:, None]) ** 2).mean(axis=0))
    col = {k: i for i, k in enumerate(keys)}

    rng = np.random.default_rng(zlib.crc32(f"{scale}|{tmin}|{domain}".encode()))
    bidx = rng.integers(0, len(te), (N_BOOT, len(te)))
    sp_boot = np.stack([spearman_cols(p_mat[bi], y_te[bi]) for bi in bidx])  # (N_BOOT, m)

    def ctrl_cols(kind: PredictorKind, arm: str) -> list[int]:
        return [col[(kind, arm, s)] for s in range(N_CONTROL_SEEDS)]

    wr_col = col[(PredictorKind.WEIGHTS_RIDGE, "none", -1)]
    rows = []
    for kind in (
        PredictorKind.MEAN_BASELINE,
        PredictorKind.DOSE_LINEAR,
        PredictorKind.WEIGHTS_RIDGE,
        PredictorKind.WEIGHTS_LGBM,
    ):
        i = col[(kind, "none", -1)]
        rows.append(
            {
                "predictor": str(kind),
                "arm": "none",
                "spearman": float(sp_point[i]) if kind is not PredictorKind.MEAN_BASELINE else float("nan"),
                "rmse": float(rmse_point[i]),
            }
        )
    for kind in CONTENT_PREDICTORS:
        sem_i = col[(kind, "semantic", -1)]
        row = {
            "predictor": str(kind),
            "arm": "semantic",
            "spearman": float(sp_point[sem_i]),
            "rmse": float(rmse_point[sem_i]),
        }
        sem_b = sp_boot[:, sem_i]
        for ctrl in ("shuffled", "matched"):
            cols = ctrl_cols(kind, ctrl)
            ctrl_b = np.nanmean(sp_boot[:, cols], axis=1)
            diff = sem_b - ctrl_b
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            row[f"d_vs_{ctrl}_mean"] = float(np.nanmean(diff))
            row[f"d_vs_{ctrl}_lo"], row[f"d_vs_{ctrl}_hi"] = float(lo), float(hi)
            row[f"beats_{ctrl}"] = bool(lo > 0)
        diff = sem_b - sp_boot[:, wr_col]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        row["d_vs_weights_ridge_mean"] = float(np.nanmean(diff))
        row["d_vs_weights_ridge_lo"], row["d_vs_weights_ridge_hi"] = float(lo), float(hi)
        row["beats_weights_ridge"] = bool(lo > 0)
        rows.append(row)
        for ctrl in ("shuffled", "matched"):
            cols = ctrl_cols(kind, ctrl)
            rows.append(
                {
                    "predictor": str(kind),
                    "arm": ctrl,
                    "spearman": float(np.nanmean(sp_point[cols])),
                    "rmse": float(np.nanmean(rmse_point[cols])),
                }
            )

    # --- design support (semantic K=40 per-phase features) ------------------
    sem_h40 = bundle["arms"][("semantic", -1)]["h40"]
    support = np.array([hull_residual(sem_h40[tr], sem_h40[i]) for i in te])

    base = {
        "scale": scale,
        "test_min_dose": tmin,
        "domain": domain,
        "n_train": len(tr),
        "n_test": len(te),
        "content_novelty_k1000": float(novelty),
        "design_support_median": float(np.median(support)),
        "design_support_p90": float(np.quantile(support, 0.9)),
        "design_support_max": float(support.max()),
    }
    results = pd.DataFrame([base | r for r in rows])

    pred_rows = []
    run_names = [bundle["run_names"][i] for i in te]
    for (kind, arm, s), p in preds.items():
        pred_rows.append(
            pd.DataFrame(
                {
                    "scale": scale,
                    "test_min_dose": tmin,
                    "domain": domain,
                    "run_name": run_names,
                    "predictor": str(kind),
                    "arm": arm,
                    "control_seed": s,
                    "y_true": y_te,
                    "y_pred": p,
                }
            )
        )
    predictions = pd.concat(pred_rows, ignore_index=True)

    diagnostics = pd.DataFrame(
        {
            "scale": scale,
            "test_min_dose": tmin,
            "domain": domain,
            "run_name": run_names,
            "dose": dose[te],
            "y_true": y_te,
            "design_support_residual": support,
            "content_novelty_k1000": float(novelty),
            "n_train": len(tr),
        }
    )
    logger.info(
        "split %s tmin=%.2f %s: n_tr=%d n_te=%d %.1fs", scale, tmin, domain, len(tr), len(te), time.monotonic() - t0
    )
    return {"results": results, "predictions": predictions, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# H4 ablations (60M, primary threshold, semantic arm only)
# ---------------------------------------------------------------------------

H4_CONFIGS = (
    ("granularity", "hist_ridge_k40_per_phase"),
    ("granularity", "hist_ridge_k1000_per_phase"),
    ("granularity", "hist_ridge_k5000_per_phase"),
    ("phase_handling", "hist_ridge_k1000_pooled"),
    ("phase_handling", "rff_ridge_per_phase"),
    ("phase_handling", "rff_ridge_pooled"),
    ("kernel", "kernel_hellinger_k1000"),
    ("kernel", "kernel_euclidean_k1000"),
    ("representation", "kme_ridge"),
    ("representation", "hist_ridge_k1000"),
    ("representation", "kme_plus_hist_k1000_concat"),
)


def run_h4_split(bundle: dict, dom_j: int, domain: str) -> pd.DataFrame:
    y, dose = bundle["y"], bundle["doses"][:, dom_j]
    tr = np.flatnonzero(dose <= TRAIN_MAX_DOSE)
    te = np.flatnonzero(dose >= PRIMARY_TEST_MIN_DOSE)
    sem, h4 = bundle["arms"][("semantic", -1)], bundle["h4"]
    feats = {
        "hist_ridge_k40_per_phase": sem["h40"],
        "hist_ridge_k1000_per_phase": sem["h1000"],
        "hist_ridge_k5000_per_phase": h4["h5000"],
        "hist_ridge_k1000_pooled": h4["h1000_pooled"],
        "rff_ridge_per_phase": sem["rff"],
        "rff_ridge_pooled": h4["rff_pooled"],
        "kme_ridge": sem["kme"],
        "hist_ridge_k1000": sem["h1000"],
        "kme_plus_hist_k1000_concat": h4["kme_hist_concat"],
    }
    rows = []
    for group, name in H4_CONFIGS:
        if name == "kernel_hellinger_k1000":
            p = kernel_cv_predict(sem["d2_hell"], tr, te, y)
        elif name == "kernel_euclidean_k1000":
            p = kernel_cv_predict(h4["d2_euclid"], tr, te, y)
        else:
            x = feats[name]
            p = ridge_cv_predict(x[tr], y[tr], x[te])
        sp = spearman_cols(p[:, None], y[te])[0]
        rows.append(
            {
                "ablation": group,
                "config": name,
                "domain": domain,
                "spearman": float(sp),
                "rmse": float(np.sqrt(((p - y[te]) ** 2).mean())),
                "n_train": len(tr),
                "n_test": len(te),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation + verdict
# ---------------------------------------------------------------------------


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scale, tmin, pred, arm), g in results.groupby(["scale", "test_min_dose", "predictor", "arm"]):
        row = {
            "scale": scale,
            "test_min_dose": tmin,
            "predictor": pred,
            "arm": arm,
            "n_domains": len(g),
            "spearman_median": float(g["spearman"].median()),
            "spearman_iqr_lo": float(g["spearman"].quantile(0.25)),
            "spearman_iqr_hi": float(g["spearman"].quantile(0.75)),
            "rmse_median": float(g["rmse"].median()),
        }
        if arm == "semantic":
            row["n_beats_both_controls"] = int((g["beats_shuffled"] & g["beats_matched"]).sum())
            row["n_beats_weights_ridge"] = int(g["beats_weights_ridge"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def verdicts(results: pd.DataFrame, agg: pd.DataFrame) -> dict:
    out: dict = {}
    for (scale, tmin), _ in results.groupby(["scale", "test_min_dose"]):
        sub = agg[(agg["scale"] == scale) & (agg["test_min_dose"] == tmin)]
        wr_median = float(sub[(sub["predictor"] == str(PredictorKind.WEIGHTS_RIDGE))]["spearman_median"].iloc[0])
        per_pred = {}
        for kind in CONTENT_PREDICTORS:
            sem = sub[(sub["predictor"] == str(kind)) & (sub["arm"] == "semantic")]
            if sem.empty:
                continue
            r = sem.iloc[0]
            per_pred[str(kind)] = {
                "n_eligible": int(r["n_domains"]),
                "n_beats_both_controls": int(r["n_beats_both_controls"]),
                "n_beats_weights_ridge": int(r["n_beats_weights_ridge"]),
                "spearman_median": r["spearman_median"],
                "weights_ridge_median": wr_median,
                "beats_controls_on_majority": bool(r["n_beats_both_controls"] > r["n_domains"] / 2),
                "ge_weights_ridge_overall": bool(r["spearman_median"] >= wr_median),
                "rule_pass": bool(r["n_beats_both_controls"] > r["n_domains"] / 2 and r["spearman_median"] >= wr_median),
            }
        out.setdefault(scale, {})[f"tmin_{tmin}"] = {
            "per_predictor": per_pred,
            "pass": bool(any(per_pred[str(k)]["rule_pass"] for k in VERDICT_PREDICTORS if str(k) in per_pred)),
        }
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    H4_DIR.mkdir(parents=True, exist_ok=True)

    hists, views, centroids = load_artifacts()
    domains = [h.domain for h in hists]
    runs = pd.read_parquet(SCRATCH / "runs.parquet")
    v40, order = featurize.composition_matrix(hists, k=40, views=views)
    v1000, _ = featurize.composition_matrix(hists, k=1000, views=views)
    v5000, _ = featurize.composition_matrix(hists, k=5000, views=views)
    assert order == domains
    npz = np.load(HIST_DIR / "rff_means.npz")
    rff_order = {d: i for i, d in enumerate(npz["domains"].tolist())}
    rff = np.asarray(npz["rff_means"], dtype=np.float64)[[rff_order[d] for d in domains]]

    novelty = content_novelty(np.asarray(v1000))
    v40a, v1000a, v5000a = (np.asarray(v) for v in (v40, v1000, v5000))

    bundles, tasks = {}, []
    for scale in sorted(runs["scale"].unique()):
        sub = runs[runs["scale"] == scale].reset_index(drop=True)
        bundles[scale] = build_bundle(
            sub, domains, v40a, v1000a, v5000a, rff, centroids, with_h4=(scale == PRIMARY_SCALE)
        )
        doses = bundles[scale]["doses"]
        for tmin in TEST_MIN_DOSES:
            n_tr = (doses <= TRAIN_MAX_DOSE).sum(axis=0)
            n_te = (doses >= tmin).sum(axis=0)
            eligible = [j for j in range(len(domains)) if n_tr[j] >= MIN_TRAIN and n_te[j] >= MIN_TEST]
            logger.info("%s tmin=%.2f: %d eligible domains", scale, tmin, len(eligible))
            tasks.extend((scale, tmin, j) for j in eligible)

    n_jobs = min(8, joblib.cpu_count())
    logger.info("running %d splits on %d workers", len(tasks), n_jobs)
    partial_frames: list[pd.DataFrame] = []
    outputs = []
    gen = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        joblib.delayed(run_split)(bundles[scale], scale, tmin, j, domains[j], novelty[j]) for scale, tmin, j in tasks
    )
    for out in gen:
        outputs.append(out)
        partial_frames.append(out["results"])
        pd.concat(partial_frames, ignore_index=True).to_parquet(OUT_DIR / "results_partial.parquet", index=False)

    results = pd.concat([o["results"] for o in outputs], ignore_index=True)
    predictions = pd.concat([o["predictions"] for o in outputs], ignore_index=True)
    diagnostics = pd.concat([o["diagnostics"] for o in outputs], ignore_index=True)
    results.to_parquet(OUT_DIR / "results.parquet", index=False)
    predictions.to_parquet(OUT_DIR / "predictions.parquet", index=False)
    diagnostics.to_parquet(OUT_DIR / "diagnostics.parquet", index=False)

    agg = aggregate(results)
    verdict = {
        "protocol": {
            "phase_reducer": "max",
            "train_max_dose": TRAIN_MAX_DOSE,
            "test_min_doses": list(TEST_MIN_DOSES),
            "eligibility": {"min_train": MIN_TRAIN, "min_test": MIN_TEST},
            "target": TARGET,
            "seed": SEED,
            "n_boot": N_BOOT,
            "n_control_seeds": N_CONTROL_SEEDS,
            "inner_cv_folds": N_INNER_FOLDS,
            "exposure_features": "skipped: constant within a scale (shared budget + 80/20 phase split)",
            "rule": (
                "pass iff a semantic predictor beats BOTH controls with paired-bootstrap CI "
                "excluding 0 on >half the eligible domains AND has median per-domain Spearman >= WEIGHTS_RIDGE"
            ),
            "verdict_predictors": [str(k) for k in VERDICT_PREDICTORS],
        },
        "verdicts": verdicts(results, agg),
        "aggregates": agg.to_dict("records"),
    }
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=json_default))

    # --- H4 ------------------------------------------------------------------
    doses60 = bundles[PRIMARY_SCALE]["doses"]
    n_tr = (doses60 <= TRAIN_MAX_DOSE).sum(axis=0)
    n_te = (doses60 >= PRIMARY_TEST_MIN_DOSE).sum(axis=0)
    eligible60 = [j for j in range(len(domains)) if n_tr[j] >= MIN_TRAIN and n_te[j] >= MIN_TEST]
    h4_frames = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_h4_split)(bundles[PRIMARY_SCALE], j, domains[j]) for j in eligible60
    )
    ablations = pd.concat(h4_frames, ignore_index=True)
    ablations.to_parquet(H4_DIR / "ablations.parquet", index=False)

    # --- report ---------------------------------------------------------------
    pd.set_option("display.width", 240, "display.max_rows", 400)
    print("\n=== H2b aggregates (median Spearman per predictor x arm) ===")
    print(
        agg.pivot_table(
            index=["scale", "test_min_dose", "predictor"], columns="arm", values="spearman_median"
        ).to_string()
    )
    print("\n=== H2b domain-win counts (semantic arm) ===")
    sem = agg[agg["arm"] == "semantic"]
    print(
        sem[
            [
                "scale",
                "test_min_dose",
                "predictor",
                "n_domains",
                "spearman_median",
                "n_beats_both_controls",
                "n_beats_weights_ridge",
            ]
        ].to_string(index=False)
    )
    print("\n=== H2b verdicts ===")
    print(json.dumps(verdict["verdicts"], indent=2))
    print("\n=== H4 ablations (60M, tmin=0.06, semantic arm) ===")
    h4_agg = ablations.groupby(["ablation", "config"])["spearman"].agg(["median", "count"]).reset_index()
    print(h4_agg.to_string(index=False))
    print(f"\nwrote {OUT_DIR} and {H4_DIR}")


if __name__ == "__main__":
    main()
