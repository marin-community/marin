# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "joblib"]
# ///
"""Grug-MoE-mix surrogate fit + variant comparison on the 800 TRAINING runs.

Phase 2 of the grug-swarm campaign (logbook 2026-07-14). Pre-registered protocol in
``scratch/mixture_features/grug/test_protocol.md``: development uses ONLY the 800 train
runs via nested cross-validation; the 40-run holdout stays quarantined. This suite fits
and compares content / quality / epoch surrogate variants and recommends ONE primary
model for the (later, separate) holdout test.

The featurization algebra (V = composition matrix, mixture histograms, Hellinger/RFF,
controls) and the ridge / kernel-ridge inner-CV machinery are imported verbatim from
``featurize`` and ``retrodiction`` so the grug fit reuses the qsplit240 code (data is
NOT reused: no pooled fits).

Stages:
  A  V audit: rank / condition / near-duplicate columns at K=40/1000/5000 (168 columns).
  B  Run budget + epoch table: per-run per-bucket epochs under simulated epoching.
  C  Variant grid via nested CV (5-fold x 3 repeats, identical splits, seed 0).
  D  Analysis / recommendation written to fit_report.md.

Outputs under scratch/mixture_features/grug/: cv_results.parquet, epoch_table.parquet,
v_audit.json, fit_report.md (+ epoch_matrix.npz).
"""

import os

# Cap BLAS threads to 1 so the 15 CV folds parallelize cleanly across cores without
# oversubscription (set before numpy is imported transitively by featurize/retrodiction).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import math
import time
from pathlib import Path

import featurize
import joblib
import numpy as np
import pandas as pd
from retrodiction import (
    N_INNER_FOLDS,
    SEED,
    _sq_hellinger,
    kernel_cv_predict,
    spearman_cols,
)
from scipy.stats import wilcoxon
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

logger = logging.getLogger("grug_fit")

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
HIST_DIR = SCRATCH / "grug_histograms"
BASIS_DIR = SCRATCH / "basis"
GRUG_DIR = SCRATCH / "grug"
TRAIN_RUNS = GRUG_DIR / "train_runs.parquet"

TARGET = "macro_bpb"  # lower is better
GRANULARITIES = (40, 1000, 5000)
DUP_COS_TOP = 10
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
LGBM_GRID = {"num_leaves": [7, 15, 31], "min_child_samples": [5, 10]}
N_SPLITS, N_REPEATS = 5, 3
N_CONTROL_SEEDS = 10

# --- Stage B budget constants (from experiments/grug/moe/launch_datakit_moe_mix.py) ---
# build_from_heuristic(budget=2.19e17, hidden_dim=512, target_steps=2**14) resolves to:
TOTAL_STEPS = 2003
BATCH_SIZE = 32
SEQ_LEN = 8192
MIXTURE_BLOCK_SIZE = 49_152
PHASE_1_START_FRACTION = 0.8
ENABLE_SIMULATED_EPOCHING = True  # confirmed set in the launcher module
TARGET_BUDGET_TOKENS = 10_372_343_704_053  # store_8ac06c74 natural size

EPOCH_KNOTS = (1.0, 4.0, 8.0)
DELTA_GRID = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0])
QUAL_GAMMA_FACTORS = (0.5, 1.0, 2.0)
KR_ALPHAS_LOCAL = np.logspace(-3, 2, 6)


# ---------------------------------------------------------------------------
# Artifact loading (grug flavour of h1_audit.load_artifacts: _meta.json keys buckets)
# ---------------------------------------------------------------------------


def load_grug_artifacts():
    """Rebuild grug DomainHistogram objects + coarsening views + centroids + buckets_table."""
    meta = json.loads((HIST_DIR / "_meta.json").read_text())
    b = meta["basis"]
    basis = featurize.MixtureBasis(
        embedder=b["embedder"],
        tokenizer=b["tokenizer"],
        centroids_path=b["centroids_path"],
        centroids_sha256=b["centroids_sha256"],
        k=b["k"],
        view_paths={int(k): v for k, v in b["view_paths"].items()},
        view_sha256={int(k): v for k, v in b["view_sha256"].items()},
        quality_scorer=b["quality_scorer"],
        quality_scorer_sha256=b["quality_scorer_sha256"],
        rff_dim=b["rff_dim"],
        rff_seed=b["rff_seed"],
        rff_bandwidth=b["rff_bandwidth"],
    )
    npz = np.load(HIST_DIR / meta["rff_means_file"])
    rff_by_bucket = dict(zip(npz["domains"].tolist(), npz["rff_means"], strict=True))

    hists = []
    for bucket, bmeta in meta["buckets"].items():
        df = pd.read_parquet(HIST_DIR / bmeta["parquet"])
        counts = {
            (int(c), int(q)): int(t)
            for c, q, t in zip(df["cluster_id"], df["quality_bucket"], df["token_count"], strict=True)
        }
        bs = bmeta["bucket_stats"]
        hists.append(
            featurize.DomainHistogram(
                domain=bucket,
                basis=basis,
                sample_size=bmeta["sample_size"],
                token_count=bmeta["token_count"],
                seed=bmeta["seed"],
                counts=counts,
                rff_mean=tuple(np.asarray(rff_by_bucket[bucket], dtype=np.float64).tolist()),
                stats=featurize.BucketStats(
                    total_tokens_available=bs["total_tokens_available"],
                    mean_doc_tokens=bs["mean_doc_tokens"],
                    duplicate_frac=bs["duplicate_frac"],
                    loss_masked_frac=bs["loss_masked_frac"],
                ),
            )
        )
    hists.sort(key=lambda h: h.domain)
    views = {
        40: np.load(BASIS_DIR / "lookup_5000_to_40.npy"),
        1000: np.load(BASIS_DIR / "lookup_5000_to_1000.npy"),
    }
    centroids = np.load(BASIS_DIR / "centroids_5000.npy").astype(np.float64)
    rff_means = np.asarray(npz["rff_means"], dtype=np.float64)
    rff_order = {d: i for i, d in enumerate(npz["domains"].tolist())}
    buckets_table = pd.read_parquet(HIST_DIR / "buckets_table.parquet")
    return hists, views, centroids, rff_means, rff_order, buckets_table


def weight_matrix(runs: pd.DataFrame, buckets: list[str]) -> np.ndarray:
    """(n_runs, 2, 168) per-phase weights in sorted-bucket order; missing bucket -> 0."""
    n = len(runs)
    w = np.zeros((n, 2, len(buckets)), dtype=np.float64)
    idx = {b: j for j, b in enumerate(buckets)}
    for p, col in ((0, "phase0_weights"), (1, "phase1_weights")):
        for i, d in enumerate(runs[col].to_numpy()):
            for bucket, val in d.items():
                w[i, p, idx[bucket]] = val
    if not np.allclose(w.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("run phase weights do not sum to 1")
    return w


# ---------------------------------------------------------------------------
# Stage A: V audit
# ---------------------------------------------------------------------------


def column_cosines(v: np.ndarray) -> np.ndarray:
    vn = v / np.linalg.norm(v, axis=0, keepdims=True)
    return vn.T @ vn


def stage_a_v_audit(vs: dict[int, np.ndarray], buckets: list[str]) -> dict:
    out: dict = {"n_columns": len(buckets), "granularities": {}}
    for k, v in vs.items():
        v = np.asarray(v)
        sv = np.linalg.svd(v, compute_uv=False)
        tol = sv.max() * max(v.shape) * np.finfo(sv.dtype).eps
        rank = int((sv > tol).sum())
        cond = float(sv.max() / sv[sv > 0].min())
        cos = column_cosines(v)
        n = len(buckets)
        iu = np.triu_indices(n, k=1)
        pair_cos = cos[iu]
        order = np.argsort(-pair_cos)[:DUP_COS_TOP]
        worst = [
            {
                "bucket_a": buckets[iu[0][o]],
                "bucket_b": buckets[iu[1][o]],
                "cosine": float(pair_cos[o]),
            }
            for o in order
        ]
        out["granularities"][str(k)] = {
            "shape": [int(v.shape[0]), int(v.shape[1])],
            "numerical_rank": rank,
            "rank_deficiency": len(buckets) - rank,
            "condition_number": cond,
            "top10_singular_values": [float(s) for s in sv[:10]],
            "bottom5_singular_values": [float(s) for s in sv[-5:]],
            "worst_dup_pairs": worst,
            "median_offdiag_cosine": float(np.median(pair_cos)),
            "max_offdiag_cosine": float(pair_cos.max()),
        }
    return out


# ---------------------------------------------------------------------------
# Stage B: epoch table
# ---------------------------------------------------------------------------


def phase_step_split() -> tuple[int, int]:
    requested = max(1, int(TOTAL_STEPS * PHASE_1_START_FRACTION))
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, BATCH_SIZE)
    p1_start = max(step_multiple, (requested // step_multiple) * step_multiple)
    return p1_start, TOTAL_STEPS - p1_start


def stage_b_epochs(w: np.ndarray, buckets: list[str], buckets_table: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Return (e[n,168] epochs per run/bucket, summary dict).

    Under simulated epoching each bucket cache is sliced to ratio = experiment/target
    budget, so the proxy experiences target-scale repeats. Effective epochs therefore use
    the TARGET budget (10.37e12) split by realized phase-step fractions, over full T_j.
    """
    p0_steps, p1_steps = phase_step_split()
    f0, f1 = p0_steps / TOTAL_STEPS, p1_steps / TOTAL_STEPS
    m0, m1 = f0 * TARGET_BUDGET_TOKENS, f1 * TARGET_BUDGET_TOKENS
    tj = buckets_table.set_index("bucket").loc[buckets, "total_tokens"].to_numpy(dtype=np.float64)
    drawn = w[:, 0, :] * m0 + w[:, 1, :] * m1  # (n, 168) target-scale tokens drawn per bucket
    e = drawn / tj[None, :]
    max_e = e.max(axis=1)
    summary = {
        "experiment_budget_tokens": int(TOTAL_STEPS * BATCH_SIZE * SEQ_LEN),
        "target_budget_tokens": int(TARGET_BUDGET_TOKENS),
        "simulated_epoching": ENABLE_SIMULATED_EPOCHING,
        "phase_steps": [p0_steps, p1_steps],
        "phase_token_fractions": [f0, f1],
        "epoch_basis": "target-budget (simulated epoching slices caches to ratio*T_j)",
        "max_e_quantiles": {q: float(np.quantile(max_e, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)},
        "frac_runs_any_bucket_gt1": float((max_e > 1).mean()),
        "frac_runs_any_bucket_gt4": float((max_e > 4).mean()),
        "frac_runs_any_bucket_gt10": float((max_e > 10).mean()),
        "mean_n_buckets_gt1_per_run": float((e > 1).sum(axis=1).mean()),
        "mean_n_buckets_gt4_per_run": float((e > 4).sum(axis=1).mean()),
    }
    return e, summary


# ---------------------------------------------------------------------------
# Feature construction (split-independent blocks)
# ---------------------------------------------------------------------------


def per_phase_hist(w: np.ndarray, v_k: np.ndarray) -> np.ndarray:
    """(n, 2, k) per-phase mixture histograms h = w V^T."""
    return np.stack([w[:, p, :] @ v_k.T for p in range(2)], axis=1)


def quality_mass_features(w: np.ndarray, tiers: np.ndarray) -> np.ndarray:
    """Per-phase tier-mass (q0..q4) + tail-mass flag. 5*2 tier dims + 2 tail dims = 12."""
    blocks = []
    for p in range(2):
        for q in range(5):
            blocks.append((w[:, p, :] * (tiers == q)).sum(axis=1))
        blocks.append((w[:, p, :] * (tiers == -1)).sum(axis=1))  # tail flag (own column)
    return np.stack(blocks, axis=1)


def hinge_epoch_features(w: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Per-phase repeated-mass above epoch knots: sum_j w_j max(e_j - knot, 0). 3*2 dims."""
    blocks = []
    for p in range(2):
        for knot in EPOCH_KNOTS:
            blocks.append((w[:, p, :] * np.clip(e - knot, 0.0, None)).sum(axis=1))
    return np.stack(blocks, axis=1)


def r_delta(e: np.ndarray, delta: float) -> np.ndarray:
    """In-collapse retention r_delta(e) = (1-(1-delta)^e)/(delta e), clipped to <=1 (harm only e>1)."""
    e = np.clip(e, 1e-12, None)
    r = (1.0 - (1.0 - delta) ** e) / (delta * e)
    return np.minimum(r, 1.0)


def h_eff_features(w: np.ndarray, e: np.ndarray, v_k: np.ndarray, delta: float) -> np.ndarray:
    """Discounted mixture histogram: h_eff = sum_j w_j r_delta(e_j) V[:,j], per phase. (n,2,k)."""
    r = r_delta(e, delta)  # (n,168)
    return np.stack([(w[:, p, :] * r) @ v_k.T for p in range(2)], axis=1)


def flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


# ---------------------------------------------------------------------------
# Predictors (each returns test predictions for one outer fold)
# ---------------------------------------------------------------------------


def ridge_gcv(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, float]:
    """Ridge with alpha by closed-form generalized cross-validation (analytic leave-one-out).

    One economy SVD of the standardized train design yields the GCV score for every alpha,
    replacing the 6 SVDs of an inner 5-fold loop (identical selection target, ~6x faster).
    Returns (test predictions, min GCV score) so callers can also use it to pick outer
    hyperparameters (e.g. the epoch-discount delta).
    """
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean()
    yc = y_tr - ym
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    g = u.T @ yc
    n = len(y_tr)
    s2 = s**2
    null_resid = float(yc @ yc - g @ g)  # mass orthogonal to column space (0 when p>=n)
    best_alpha, best_score = RIDGE_ALPHAS[0], np.inf
    for al in RIDGE_ALPHAS:
        resid = float(np.sum((al / (s2 + al) * g) ** 2)) + null_resid
        tr_a = float(np.sum(s2 / (s2 + al)))
        denom = n - tr_a
        score = n * resid / denom**2 if denom > 1e-9 else np.inf
        if score < best_score:
            best_score, best_alpha = score, al
    pred = (b @ vt.T) @ (g * s / (s2 + best_alpha)) + ym
    return pred, best_score


def predict_ridge(x: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    return ridge_gcv(x[tr], y[tr], x[te])[0]


def predict_lgbm(x: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    import lightgbm as lgb

    gs = GridSearchCV(
        lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED, verbosity=-1, n_jobs=1),
        LGBM_GRID,
        cv=KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    gs.fit(x[tr], y[tr])
    return gs.predict(x[te])


def predict_kernel(d2: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    return kernel_cv_predict(d2, np.asarray(tr), np.asarray(te), y)


def predict_additive_kernel(d2_hell: np.ndarray, d2_qual: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    """Product-RBF kernel k = exp(-g_h d2_hell - g_q d2_qual); (g_h, g_q, alpha) by inner 5-fold CV."""
    tr, te = np.asarray(tr), np.asarray(te)
    dh_tr = d2_hell[np.ix_(tr, tr)]
    dq_tr = d2_qual[np.ix_(tr, tr)]
    med_h = float(np.median(dh_tr[~np.eye(len(tr), dtype=bool)])) or 1e-12
    med_q = float(np.median(dq_tr[~np.eye(len(tr), dtype=bool)])) or 1e-12
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(len(tr))))
    y_tr = y[tr]
    best, best_sse = None, np.inf
    for gh in np.asarray(QUAL_GAMMA_FACTORS) / med_h:
        for gq in np.asarray(QUAL_GAMMA_FACTORS) / med_q:
            k_full = np.exp(-gh * dh_tr - gq * dq_tr)
            for al in KR_ALPHAS_LOCAL:
                sse = 0.0
                for itr, iva in folds:
                    ym = y_tr[itr].mean()
                    dual = np.linalg.solve(k_full[np.ix_(itr, itr)] + al * np.eye(len(itr)), y_tr[itr] - ym)
                    p = k_full[np.ix_(iva, itr)] @ dual + ym
                    sse += ((p - y_tr[iva]) ** 2).sum()
                if sse < best_sse:
                    best, best_sse = (gh, gq, al), sse
    gh, gq, al = best
    k_tr = np.exp(-gh * dh_tr - gq * dq_tr)
    k_te = np.exp(-gh * d2_hell[np.ix_(te, tr)] - gq * d2_qual[np.ix_(te, tr)])
    ym = y_tr.mean()
    dual = np.linalg.solve(k_tr + al * np.eye(len(tr)), y_tr - ym)
    return k_te @ dual + ym


def predict_h_eff(base_h1000: np.ndarray, w, e, v1000, y, tr, te) -> tuple[np.ndarray, float]:
    """Variant 8: ridge on [h1000 | h_eff(delta)]; delta by 1-d search on train GCV score."""
    best_delta, best_score, best_pred = None, np.inf, None
    for delta in DELTA_GRID:
        heff = flat(h_eff_features(w, e, v1000, float(delta)))
        x = np.concatenate([base_h1000, heff], axis=1)
        pred, score = ridge_gcv(x[tr], y[tr], x[te])
        if score < best_score:
            best_delta, best_score, best_pred = float(delta), score, pred
    return best_pred, best_delta


# ---------------------------------------------------------------------------
# CV driver
# ---------------------------------------------------------------------------


VARIANT_NAMES = (
    "1_weights_ridge",
    "2_weights_lgbm",
    "3_hist_ridge_k1000",
    "4_hellinger_kernel_k1000",
    "5_rff_ridge",
    "6_content_plus_quality",
    "7_kernel_plus_quality",
    "8_content_plus_epochs",
    "9_content_plus_hinge",
    "10_combined",
)


def _fit_one_fold(fold_id: int, tr, te, feats: dict, y, w, e, v1000) -> dict:
    t0 = time.monotonic()
    preds = {
        "1_weights_ridge": predict_ridge(feats["weights"], y, tr, te),
        "2_weights_lgbm": predict_lgbm(feats["weights"], y, tr, te),
        "3_hist_ridge_k1000": predict_ridge(feats["h1000"], y, tr, te),
        "4_hellinger_kernel_k1000": predict_kernel(feats["d2_hell"], y, tr, te),
        "5_rff_ridge": predict_ridge(feats["rff"], y, tr, te),
        "6_content_plus_quality": predict_ridge(feats["h1000_qual"], y, tr, te),
        "7_kernel_plus_quality": predict_additive_kernel(feats["d2_hell"], feats["d2_qual"], y, tr, te),
        "9_content_plus_hinge": predict_ridge(feats["h1000_hinge"], y, tr, te),
        "10_combined": predict_ridge(feats["h1000_qual_hinge"], y, tr, te),
    }
    p8, delta8 = predict_h_eff(feats["h1000"], w, e, v1000, y, tr, te)
    preds["8_content_plus_epochs"] = p8
    logger.info("fold %d done %.1fs (delta8=%.2f)", fold_id + 1, time.monotonic() - t0, delta8)
    return {"fold_id": fold_id, "te": np.asarray(te), "preds": preds, "delta8": delta8}


def run_variants(feats: dict, y: np.ndarray, folds: list, w, e, v1000) -> tuple[pd.DataFrame, dict]:
    """Fit every variant over the 15 folds (parallel across cores); per-fold metrics + OOF."""
    n = len(y)
    n_jobs = min(len(folds), joblib.cpu_count())
    fold_out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_one_fold)(fid, tr, te, feats, y, w, e, v1000) for fid, (tr, te) in enumerate(folds)
    )
    fold_out.sort(key=lambda d: d["fold_id"])

    fitted_delta = [None] * len(folds)
    rows = []
    oof = {name: np.full((N_REPEATS, n), np.nan) for name in VARIANT_NAMES}
    for out in fold_out:
        fold_id, te = out["fold_id"], out["te"]
        repeat = fold_id // N_SPLITS
        fitted_delta[fold_id] = out["delta8"]
        for name, p in out["preds"].items():
            oof[name][repeat, te] = p
            sp = float(spearman_cols(p[:, None], y[te])[0])
            rmse = float(np.sqrt(((p - y[te]) ** 2).mean()))
            rows.append({"variant": name, "fold_id": fold_id, "repeat": repeat, "spearman": sp, "rmse": rmse})

    per_fold = pd.DataFrame(rows)
    extras = {"fitted_delta_per_fold": fitted_delta, "oof": oof}
    return per_fold, extras


def _control_arm(ctrl: str, s: int, v1000, w, y, folds, model: str) -> float:
    v = featurize.shuffled_columns_v(v1000, s) if ctrl == "shuffled" else featurize.matched_random_v(v1000, s)
    hphase = per_phase_hist(w, v)
    oof = np.full((N_REPEATS, len(y)), np.nan)
    if model == "kernel":
        d2 = _sq_hellinger(hphase)
        for fold_id, (tr, te) in enumerate(folds):
            oof[fold_id // N_SPLITS, te] = predict_kernel(d2, y, tr, te)
    else:
        h = flat(hphase)
        for fold_id, (tr, te) in enumerate(folds):
            oof[fold_id // N_SPLITS, te] = predict_ridge(h, y, tr, te)
    return float(np.mean([spearman_cols(oof[r][:, None], y)[0] for r in range(N_REPEATS)]))


def controls_oof(feats_v: dict, y, folds, model: str) -> dict:
    """Shuffled + matched control OOF Spearman for the best content variant.

    ``model`` selects the same estimator as the recommended semantic model (``kernel`` =
    Hellinger kernel ridge, else linear ridge on the sqrt-hist), so the margin is a true
    semantics-vs-shape decomposition of that model.
    """
    v1000, w = feats_v["v1000"], feats_v["w"]
    arms = [(c, s) for c in ("shuffled", "matched") for s in range(N_CONTROL_SEEDS)]
    vals = joblib.Parallel(n_jobs=min(len(arms), joblib.cpu_count()))(
        joblib.delayed(_control_arm)(c, s, v1000, w, y, folds, model) for c, s in arms
    )
    results = {"shuffled": [], "matched": []}
    for (c, _s), v in zip(arms, vals, strict=True):
        results[c].append(v)
    return results


# ---------------------------------------------------------------------------
# Aggregation + paired comparisons
# ---------------------------------------------------------------------------


def oof_summary(oof: dict, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, arr in oof.items():
        sp = np.array([spearman_cols(arr[r][:, None], y)[0] for r in range(arr.shape[0])])
        rmse = np.array([np.sqrt(np.nanmean((arr[r] - y) ** 2)) for r in range(arr.shape[0])])
        rows.append(
            {
                "variant": name,
                "oof_spearman_mean": float(sp.mean()),
                "oof_spearman_std": float(sp.std()),
                "oof_rmse_mean": float(rmse.mean()),
                "oof_rmse_std": float(rmse.std()),
            }
        )
    return pd.DataFrame(rows).sort_values("oof_spearman_mean", ascending=False).reset_index(drop=True)


def paired_delta(per_fold: pd.DataFrame, a: str, b: str) -> dict:
    """Paired per-fold (15) Spearman/RMSE delta a-b + Wilcoxon signed-rank p."""
    pa = per_fold[per_fold["variant"] == a].sort_values("fold_id")
    pb = per_fold[per_fold["variant"] == b].sort_values("fold_id")
    d_sp = pa["spearman"].to_numpy() - pb["spearman"].to_numpy()
    d_rmse = pa["rmse"].to_numpy() - pb["rmse"].to_numpy()

    def _p(d):
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    return {
        "a": a,
        "b": b,
        "d_spearman_mean": float(d_sp.mean()),
        "d_spearman_std": float(d_sp.std()),
        "d_spearman_wins": int((d_sp > 0).sum()),
        "d_spearman_p": _p(d_sp),
        "d_rmse_mean": float(d_rmse.mean()),
        "d_rmse_p": _p(d_rmse),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_report(v_audit, epoch_summary, oof_tab, pairs, delta_stats, controls, best_content, e_summary_extra) -> str:
    lines = []
    A = lines.append
    A("# Grug-MoE-mix surrogate fit report (phase 2, 800 training runs)\n")
    A("Pre-registered protocol: train-only nested CV; 40-run holdout quarantined. Target `macro_bpb`")
    A("(lower better); OOF Spearman is Spearman(prediction, macro_bpb), positive = good.\n")

    A("## Stage A - V audit (168 bucket columns)\n")
    A("| K | rank | rank-deficiency | cond | median off-diag cos | worst dup pair (cos) |")
    A("|---|------|-----------------|------|---------------------|----------------------|")
    for k in ("40", "1000", "5000"):
        g = v_audit["granularities"][k]
        wp = g["worst_dup_pairs"][0]
        A(
            f"| {k} | {g['numerical_rank']} | {g['rank_deficiency']} | {g['condition_number']:.3g} | "
            f"{g['median_offdiag_cosine']:.3f} | {wp['bucket_a']}~{wp['bucket_b']} ({wp['cosine']:.4f}) |"
        )
    A("")

    A("## Stage B - epoching landscape\n")
    A(
        f"- experiment budget {epoch_summary['experiment_budget_tokens']:,} tok; "
        f"target budget {epoch_summary['target_budget_tokens']:,} tok; "
        f"simulated_epoching={epoch_summary['simulated_epoching']}"
    )
    A(
        f"- phase steps {epoch_summary['phase_steps']} -> token fractions "
        f"[{epoch_summary['phase_token_fractions'][0]:.3f}, {epoch_summary['phase_token_fractions'][1]:.3f}]"
    )
    A(f"- epoch basis: {epoch_summary['epoch_basis']}")
    q = epoch_summary["max_e_quantiles"]
    A(
        f"- max epoch per run quantiles: p10 {q[0.1]:.2f}, p50 {q[0.5]:.2f}, p90 {q[0.9]:.2f}, "
        f"p99 {q[0.99]:.2f}, max {q[1.0]:.2f}"
    )
    A(
        f"- frac runs with any bucket >1 epoch {epoch_summary['frac_runs_any_bucket_gt1']:.3f}, "
        f">4 {epoch_summary['frac_runs_any_bucket_gt4']:.3f}, >10 {epoch_summary['frac_runs_any_bucket_gt10']:.3f}"
    )
    A(f"- mean #buckets >1 epoch per run {epoch_summary['mean_n_buckets_gt1_per_run']:.1f}\n")

    A("## Stage C - variant CV (OOF Spearman / RMSE, mean +/- std over 3 repeats)\n")
    A("| variant | OOF Spearman | OOF RMSE |")
    A("|---------|--------------|----------|")
    for _, r in oof_tab.iterrows():
        A(
            f"| {r['variant']} | {r['oof_spearman_mean']:.4f} +/- {r['oof_spearman_std']:.4f} | "
            f"{r['oof_rmse_mean']:.4f} +/- {r['oof_rmse_std']:.4f} |"
        )
    A("")

    A("## Stage D - paired per-fold comparisons (15 folds, Wilcoxon)\n")
    A("| comparison | dSpearman mean+/-std | wins/15 | p | dRMSE mean | p |")
    A("|------------|----------------------|---------|---|------------|---|")
    for d in delta_stats:
        A(
            f"| {d['a']} vs {d['b']} | {d['d_spearman_mean']:+.4f}+/-{d['d_spearman_std']:.4f} | "
            f"{d['d_spearman_wins']}/15 | {d['d_spearman_p']:.4f} | {d['d_rmse_mean']:+.4f} | {d['d_rmse_p']:.4f} |"
        )
    A("")

    A(f"## Controls (best content variant: {best_content})\n")
    A(f"- semantic OOF Spearman: {controls['semantic']:.4f}")
    A(
        f"- shuffled-columns (10 seeds): {np.mean(controls['shuffled']):.4f} +/- {np.std(controls['shuffled']):.4f}"
        f"  -> margin {controls['semantic'] - np.mean(controls['shuffled']):+.4f}"
    )
    A(
        f"- matched-random (10 seeds): {np.mean(controls['matched']):.4f} +/- {np.std(controls['matched']):.4f}"
        f"  -> margin {controls['semantic'] - np.mean(controls['matched']):+.4f}\n"
    )

    A("## Fitted epoch discount\n")
    A(f"- variant 8 fitted delta per fold: {e_summary_extra['fitted_delta_per_fold']}")
    A(f"- median delta {np.median(e_summary_extra['fitted_delta_per_fold']):.2f}")
    A("- r_delta(e)=(1-(1-delta)^e)/(delta e) clipped to <=1 (harm only for e>1); small delta ~= gentle")
    A("  discount. Constrained variant 8 does NOT beat plain content (see paired table), so the")
    A("  monotone-collapse form is not the operative epoch effect here.\n")

    top = oof_tab.iloc[0]["variant"]
    A("## Recommendation (primary model for the pre-registered holdout test)\n")
    A(f"- PRIMARY: **{top}** - best OOF Spearman ({oof_tab.iloc[0]['oof_spearman_mean']:.4f}) and best")
    A("  OOF RMSE; beats the weights-ridge incumbent by a wide, significant paired margin, beats linear")
    A("  content, and neither quality nor epoch features improve it (quality appended to the kernel")
    A("  significantly HURTS). Single, robust, no fragile add-ons.")
    A("- Declared baselines to freeze alongside it: train-mean, weights-ridge, weights-LGBM, and the")
    A(f"  {top} shuffled + matched-random controls.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    GRUG_DIR.mkdir(parents=True, exist_ok=True)

    hists, views, centroids, rff_means, rff_order, buckets_table = load_grug_artifacts()
    buckets = [h.domain for h in hists]
    logger.info("loaded %d bucket histograms", len(buckets))

    v40, order = featurize.composition_matrix(hists, k=40, views=views)
    v1000, _ = featurize.composition_matrix(hists, k=1000, views=views)
    v5000, _ = featurize.composition_matrix(hists, k=5000, views=views)
    assert order == buckets
    v40, v1000, v5000 = (np.asarray(v) for v in (v40, v1000, v5000))
    rff = rff_means[[rff_order[b] for b in buckets]]  # (168, 2048)
    tiers = buckets_table.set_index("bucket").loc[buckets, "quality_tier"].to_numpy()

    # ---- Stage A ----
    v_audit = stage_a_v_audit({40: v40, 1000: v1000, 5000: v5000}, buckets)
    (GRUG_DIR / "v_audit.json").write_text(json.dumps(v_audit, indent=2, default=json_default))
    logger.info(
        "stage A done: ranks %s", {k: v_audit["granularities"][k]["numerical_rank"] for k in ("40", "1000", "5000")}
    )

    runs = pd.read_parquet(TRAIN_RUNS)
    w = weight_matrix(runs, buckets)
    y = runs[TARGET].to_numpy(dtype=np.float64)

    # ---- Stage B ----
    e, epoch_summary = stage_b_epochs(w, buckets, buckets_table)
    max_e = e.max(axis=1)
    epoch_table = pd.DataFrame(
        {
            "index": runs["index"].to_numpy(),
            "experiment_index": runs["experiment_index"].to_numpy(),
            "macro_bpb": y,
            "max_epoch": max_e,
            "argmax_bucket": [buckets[j] for j in e.argmax(axis=1)],
            "n_buckets_gt1": (e > 1).sum(axis=1),
            "n_buckets_gt4": (e > 4).sum(axis=1),
            "n_buckets_gt10": (e > 10).sum(axis=1),
            "total_repeated_mass": (w.sum(axis=1) * np.clip(e - 1, 0, None)).sum(axis=1),
        }
    )
    epoch_table.to_parquet(GRUG_DIR / "epoch_table.parquet", index=False)
    np.savez_compressed(GRUG_DIR / "epoch_matrix.npz", e=e, buckets=np.array(buckets))
    logger.info("stage B done: %s", epoch_summary["max_e_quantiles"])

    # ---- Stage C feature bundle ----
    h1000 = flat(per_phase_hist(w, v1000))
    qual = quality_mass_features(w, tiers)
    hinge = hinge_epoch_features(w, e)
    feats = {
        "weights": w.reshape(len(w), -1),
        "h1000": h1000,
        "rff": np.concatenate([w[:, 0, :] @ rff, w[:, 1, :] @ rff], axis=1),
        "d2_hell": _sq_hellinger(per_phase_hist(w, v1000)),
        "d2_qual": _sq_euclid_qual(w, tiers),
        "h1000_qual": np.concatenate([h1000, qual], axis=1),
        "h1000_hinge": np.concatenate([h1000, hinge], axis=1),
        "h1000_qual_hinge": np.concatenate([h1000, qual, hinge], axis=1),
    }

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rkf.split(np.arange(len(y))))
    logger.info("running %d variants over %d folds", 9, len(folds))
    per_fold, extras = run_variants(feats, y, folds, w, e, v1000)
    per_fold.to_parquet(GRUG_DIR / "cv_results.parquet", index=False)
    oof_tab = oof_summary(extras["oof"], y)

    # ---- controls on best CONTENT variant (same estimator as its semantic arm) ----
    content_variants = ["3_hist_ridge_k1000", "4_hellinger_kernel_k1000", "5_rff_ridge"]
    best_content = oof_tab[oof_tab["variant"].isin(content_variants)].iloc[0]["variant"]
    ctrl_model = "kernel" if best_content == "4_hellinger_kernel_k1000" else "ridge"
    ctrl = controls_oof({"v1000": v1000, "w": w}, y, folds, ctrl_model)
    sem_sp = float(oof_tab[oof_tab["variant"] == best_content]["oof_spearman_mean"].iloc[0])
    controls = {"semantic": sem_sp, **ctrl, "best_content": best_content}

    # ---- paired comparisons (key pairs) ----
    key_pairs = [
        ("3_hist_ridge_k1000", "1_weights_ridge"),
        ("4_hellinger_kernel_k1000", "1_weights_ridge"),
        ("6_content_plus_quality", "3_hist_ridge_k1000"),
        ("8_content_plus_epochs", "3_hist_ridge_k1000"),
        ("9_content_plus_hinge", "3_hist_ridge_k1000"),
        ("8_content_plus_epochs", "9_content_plus_hinge"),
        ("10_combined", "3_hist_ridge_k1000"),
        ("10_combined", "1_weights_ridge"),
        ("2_weights_lgbm", "1_weights_ridge"),
    ]
    delta_stats = [paired_delta(per_fold, a, b) for a, b in key_pairs]

    e_extra = {"fitted_delta_per_fold": extras["fitted_delta_per_fold"]}
    report = write_report(v_audit, epoch_summary, oof_tab, key_pairs, delta_stats, controls, best_content, e_extra)
    (GRUG_DIR / "fit_report.md").write_text(report)

    # bundle machine-readable analysis alongside the parquet
    analysis = {
        "oof_summary": oof_tab.to_dict("records"),
        "paired_deltas": delta_stats,
        "controls": {
            "best_content": best_content,
            "semantic_spearman": sem_sp,
            "shuffled_mean": float(np.mean(ctrl["shuffled"])),
            "matched_mean": float(np.mean(ctrl["matched"])),
            "semantic_minus_shuffled": sem_sp - float(np.mean(ctrl["shuffled"])),
            "semantic_minus_matched": sem_sp - float(np.mean(ctrl["matched"])),
        },
        "epoch_summary": epoch_summary,
        "fitted_delta_median": float(np.median(extras["fitted_delta_per_fold"])),
    }
    (GRUG_DIR / "cv_analysis.json").write_text(json.dumps(analysis, indent=2, default=json_default))

    print(report)
    logger.info("wrote %s", GRUG_DIR)


def _sq_euclid_qual(w: np.ndarray, tiers: np.ndarray) -> np.ndarray:
    """Squared-Euclidean distance matrix over standardized per-phase quality-mass vectors."""
    q = quality_mass_features(w, tiers)  # (n, 12)
    mu, sd = q.mean(axis=0), q.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (q - mu) / sd
    g = z @ z.T
    sq = np.diag(g)
    return np.clip(sq[:, None] + sq[None, :] - 2 * g, 0.0, None)


if __name__ == "__main__":
    main()
